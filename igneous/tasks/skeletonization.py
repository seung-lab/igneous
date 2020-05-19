import six

from functools import reduce
import itertools
import json
import pickle
import posixpath
import os
import re
from collections import defaultdict

from tqdm import tqdm

import numpy as np

import cloudvolume
from cloudvolume import CloudVolume, PrecomputedSkeleton, view
from cloudvolume.storage import Storage, SimpleStorage
from cloudvolume.lib import xyzrange, min2, max2, Vec, Bbox, mkdir, save_images, jsonify, scatter
from cloudvolume.datasource.precomputed.sharding import synthesize_shard_files

import fastremap
import kimimaro

from taskqueue import RegisteredTask

SEGIDRE = re.compile(r'/(\d+):.*?$')

def filename_to_segid(filename):
  matches = SEGIDRE.search(filename)
  if matches is None:
    raise ValueError("There was an issue with the fragment filename: " + filename)

  segid, = matches.groups()
  return int(segid)

def skeldir(cloudpath):
  with SimpleStorage(cloudpath) as storage:
    info = storage.get_json('info')

  skel_dir = 'skeletons/'
  if 'skeletons' in info:
    skel_dir = info['skeletons']
  return skel_dir

class SkeletonTask(RegisteredTask):
  """
  Stage 1 of skeletonization.

  Convert chunks of segmentation into chunked skeletons and point clouds.
  They will be merged in the stage 2 task SkeletonMergeTask.
  """
  def __init__(
    self, cloudpath, shape, offset, 
    mip, teasar_params, will_postprocess, 
    info=None, object_ids=None, mask_ids=None,
    fix_branching=True, fix_borders=True,
    dust_threshold=1000, progress=False,
    parallel=1, fill_missing=False, sharded=False,
    spatial_index=True, spatial_grid_shape=None,
    synapses=None
  ):
    super(SkeletonTask, self).__init__(
      cloudpath, shape, offset, mip, 
      teasar_params, will_postprocess, 
      info, object_ids, mask_ids,
      fix_branching, fix_borders,
      dust_threshold, progress, parallel,
      fill_missing, bool(sharded), bool(spatial_index),
      spatial_grid_shape, synapses
    )
    self.bounds = Bbox(offset, Vec(*shape) + Vec(*offset))
    self.index_bounds = Bbox(offset, Vec(*spatial_grid_shape) + Vec(*offset))

  def execute(self):
    vol = CloudVolume(
      self.cloudpath, mip=self.mip, 
      info=self.info, cdn_cache=False,
      parallel=self.parallel, 
      fill_missing=self.fill_missing,
    )
    bbox = Bbox.clamp(self.bounds, vol.bounds)
    index_bbox = Bbox.clamp(self.index_bounds, vol.bounds)

    path = skeldir(self.cloudpath)
    path = os.path.join(self.cloudpath, path)

    all_labels = vol[ bbox.to_slices() ]
    all_labels = all_labels[:,:,:,0]

    if self.mask_ids:
      all_labels = fastremap.mask(all_labels, self.mask_ids)

    extra_targets_after = {}
    if self.synapses:
      extra_targets_after = kimimaro.synapses_to_targets(
        all_labels, self.synapses
      )

    skeletons = kimimaro.skeletonize(
      all_labels, self.teasar_params, 
      object_ids=self.object_ids, 
      anisotropy=vol.resolution,
      dust_threshold=self.dust_threshold, 
      progress=self.progress, 
      fix_branching=self.fix_branching,
      fix_borders=self.fix_borders,
      parallel=self.parallel,
      extra_targets_after=extra_targets_after.keys(),
    )    

    for segid, skel in six.iteritems(skeletons):
      skel.vertices[:] += bbox.minpt * vol.resolution

    if self.synapses:
      for segid, skel in six.iteritems(skeletons):
        terminal_nodes = skel.vertices[ skel.terminals() ]

        for i, vert in enumerate(terminal_nodes):
          vert = vert / vol.resolution - self.bounds.minpt
          vert = tuple(np.round(vert).astype(int))
          if vert in extra_targets_after.keys():
            skel.vertex_types[i] = extra_targets_after[vert]
    
    if self.sharded:
      self.upload_batch(vol, path, index_bbox, skeletons)
    else:
      self.upload_individuals(vol, path, bbox, skeletons)

    if self.spatial_index:
      self.upload_spatial_index(vol, path, index_bbox, skeletons)
  
  def upload_batch(self, vol, path, bbox, skeletons):
    # neuroglancer doesn't support int attributes for shards
    for skel in skeletons.values():
      skel.extra_attributes = [ 
      attr for attr in skel.extra_attributes 
      if attr['data_type'] in ('float32', 'float64')
    ]

    with SimpleStorage(path, progress=vol.progress) as stor:
      # Create skeleton batch for postprocessing later
      stor.put_file(
        file_path="{}.frags".format(bbox.to_filename()),
        content=pickle.dumps(skeletons),
        compress='gzip',
        content_type="application/python-pickle",
        cache_control=False,
      )

  def upload_individuals(self, vol, path, bbox, skeletons):
    skeletons = skeletons.values()

    if not self.will_postprocess:
      vol.skeleton.upload(skeletons)
      return 

    bbox = bbox * vol.resolution
    with Storage(path, progress=vol.progress) as stor:
      for skel in skeletons:
        stor.put_file(
          file_path="{}:{}".format(skel.id, bbox.to_filename()),
          content=pickle.dumps(skel),
          compress='gzip',
          content_type="application/python-pickle",
          cache_control=False,
        )

  def upload_spatial_index(self, vol, path, bbox, skeletons):
    spatial_index = {}
    for segid, skel in tqdm(skeletons.items(), disable=(not vol.progress), desc="Extracting Bounding Boxes"):
      segid_bbx = Bbox.from_points( skel.vertices )
      spatial_index[segid] = segid_bbx.to_list()

    bbox = bbox * vol.resolution
    with SimpleStorage(path, progress=vol.progress) as stor:
      stor.put_file(
        file_path="{}.spatial".format(bbox.to_filename()),
        content=jsonify(spatial_index).encode('utf8'),
        compress='gzip',
        content_type="application/json",
        cache_control=False,
      )

class UnshardedSkeletonMergeTask(RegisteredTask):
  """
  Stage 2 of skeletonization.

  Merge chunked TEASAR skeletons into a single skeleton.

  If we parallelize using prefixes single digit prefixes ['0','1',..'9'] all meshes will
  be correctly processed. But if we do ['10','11',..'99'] meshes from [0,9] won't get
  processed and need to be handle specifically by creating tasks that will process
  a single mesh ['0:','1:',..'9:']
  """
  def __init__(
      self, cloudpath, prefix, 
      crop=0, mip=0, dust_threshold=4000, 
      tick_threshold=6000, delete_fragments=False
    ):
    super(UnshardedSkeletonMergeTask, self).__init__(
      cloudpath, prefix, crop, mip, 
      dust_threshold, tick_threshold, delete_fragments
    )

  def execute(self):
    self.vol = CloudVolume(self.cloudpath, mip=self.mip, cdn_cache=False)

    fragment_filenames = self.get_filenames()
    skels = self.get_skeletons_by_segid(fragment_filenames)

    skeletons = []
    for segid, frags in skels.items():
      skeleton = self.fuse_skeletons(frags)
      skeleton = kimimaro.postprocess(
        skeleton, self.dust_threshold, self.tick_threshold
      )
      skeleton.id = segid
      skeletons.append(skeleton)

    self.vol.skeleton.upload(skeletons)
    
    if self.delete_fragments:
      with Storage(self.cloudpath, progress=True) as stor:
        stor.delete_files(fragment_filenames)

  def get_filenames(self):
    prefix = '{}/{}'.format(self.vol.skeleton.path, self.prefix)

    with Storage(self.cloudpath, progress=True) as stor:
      return [ _ for _ in stor.list_files(prefix=prefix) ]

  def get_skeletons_by_segid(self, filenames):
    with Storage(self.cloudpath, progress=True) as stor:
      skels = stor.get_files(filenames)

    skeletons = defaultdict(list)
    for skel in skels:
      try:
        segid = filename_to_segid(skel['filename'])
      except ValueError:
        # Typically this is due to preexisting fully
        # formed skeletons e.g. skeletons_mip_3/1588494
        continue

      skeletons[segid].append( 
        (
          Bbox.from_filename(skel['filename']),
          pickle.loads(skel['content'])
        )
      )

    return skeletons

  def fuse_skeletons(self, skels):
    if len(skels) == 0:
      return PrecomputedSkeleton()

    bbxs = [ item[0] for item in skels ]
    skeletons = [ item[1] for item in skels ]

    skeletons = self.crop_skels(bbxs, skeletons)
    skeletons = [ s for s in skeletons if not s.empty() ]

    if len(skeletons) == 0:
      return PrecomputedSkeleton()

    return PrecomputedSkeleton.simple_merge(skeletons).consolidate()

  def crop_skels(self, bbxs, skeletons):
    cropped = [ s.clone() for s in skeletons ]

    if self.crop <= 0:
      return cropped
    
    for i in range(len(skeletons)):
      bbx = bbxs[i] 
      bbx.minpt += self.crop * self.vol.resolution
      bbx.maxpt -= self.crop * self.vol.resolution

      if bbx.volume() <= 0:
        continue

      cropped[i] = cropped[i].crop(bbx)

    return cropped

class ShardedSkeletonMergeTask(RegisteredTask):
  def __init__(
    self, cloudpath, shard_no, 
    dust_threshold=4000, tick_threshold=6000
  ):
    super(ShardedSkeletonMergeTask, self).__init__(
      cloudpath, shard_no,  
      dust_threshold, tick_threshold
    )
    self.progress = False

  def execute(self):
    # cache is necessary for local computation, but on GCE download is very fast
    # so cache isn't necessary.
    cv = CloudVolume(self.cloudpath, cache=False, progress=self.progress)
    labels = self.labels_for_shard(cv)
    locations = self.locations_for_labels(labels, cv)
    skeletons = self.process_skeletons(locations, cv)

    if len(skeletons) == 0:
      return

    shard_files = synthesize_shard_files(cv.skeleton.reader.spec, skeletons)

    if len(shard_files) != 1:
      raise ValueError(
        "Only one shard file should be generated per task. Expected: {} Got: {} ".format(
          str(self.shard_no), ", ".join(shard_files.keys())
      ))

    uploadable = [ (fname, data) for fname, data in shard_files.items() ]
    with Storage(cv.skeleton.meta.layerpath, progress=self.progress) as stor:
      stor.put_files(
        files=uploadable, 
        compress=False,
        content_type='application/octet-stream',
        cache_control='no-cache',
      )

  def process_skeletons(self, locations, cv):    
    filenames = set(itertools.chain(*locations.values()))
    labels = set(locations.keys())
    unfused_skeletons = self.get_unfused(labels, filenames, cv)

    skeletons = {}
    for label, skels in tqdm(unfused_skeletons.items(), desc="Postprocessing", disable=(not self.progress)):
      skel = PrecomputedSkeleton.simple_merge(skels)
      skel.id = label
      skel.extra_attributes = [ 
        attr for attr in skel.extra_attributes \
        if attr['data_type'] == 'float32' 
      ]      
      skeletons[label] = kimimaro.postprocess(
        skel, 
        dust_threshold=self.dust_threshold, # voxels 
        tick_threshold=self.tick_threshold, # nm
      ).to_precomputed()

    return skeletons

  def get_unfused(self, labels, filenames, cv):    
    skeldirfn = lambda loc: cv.meta.join(cv.skeleton.meta.skeleton_path, loc)
    filenames = [ skeldirfn(loc) for loc in filenames ]

    block_size = 50

    if len(filenames) < block_size:
      blocks = [ filenames ]
      n_blocks = 1
    else:
      n_blocks = max(len(filenames) // block_size, 1)
      blocks = scatter(filenames, n_blocks)

    all_skels = defaultdict(list)
    for filenames_block in tqdm(blocks, desc="Filename Block", total=n_blocks, disable=(not self.progress)):
      all_files = cv.skeleton.cache.download(filenames_block, progress=self.progress)
      
      for filename, content in tqdm(all_files.items(), desc="Unpickling Fragments", disable=(not self.progress)):
        fragment = pickle.loads(content)
        for label in labels:
          if label in fragment:
            all_skels[label].append(fragment[label])

    return all_skels

  def locations_for_labels(self, labels, cv):
    SPATIAL_EXT = re.compile(r'\.spatial$')
    index_filenames = cv.skeleton.spatial_index.file_locations_per_label(labels)
    for label, locations in index_filenames.items():
      for i, location in enumerate(locations):
        bbx = Bbox.from_filename(re.sub(SPATIAL_EXT, '', location))
        bbx /= cv.meta.resolution(cv.skeleton.meta.mip)
        index_filenames[label][i] = bbx.to_filename() + '.frags'
    return index_filenames

  def labels_for_shard(self, cv):
    """
    Try to fetch precalculated labels from `$shardno.labels` (faster) otherwise, 
    compute which labels are applicable to this shard from the shard index (much slower).
    """
    labels = SimpleStorage(cv.skeleton.meta.layerpath).get_json(self.shard_no + '.labels')
    if labels is not None:
      return labels

    labels = cv.skeleton.spatial_index.query(cv.bounds * cv.resolution)
    spec = cv.skeleton.reader.spec

    return [ 
      lbl for lbl in tqdm(labels, desc="Computing Shard Numbers", disable=(not self.progress))  \
      if spec.compute_shard_location(lbl).shard_number == self.shard_no 
    ]
