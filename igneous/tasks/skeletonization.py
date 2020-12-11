import six

from functools import reduce
import itertools
import json
import mmap
import pickle
import posixpath
import os
import re
from collections import defaultdict

from tqdm import tqdm

import numpy as np

from mapbuffer import MapBuffer
from cloudfiles import CloudFiles

import cloudvolume
from cloudvolume import CloudVolume, PrecomputedSkeleton
from cloudvolume.lib import Vec, Bbox, sip
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
  cf = CloudFiles(cloudpath)
  info = cf.get_json('info')

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
    fix_branching=True, fix_borders=True, fix_avocados=False,
    dust_threshold=1000, progress=False,
    parallel=1, fill_missing=False, sharded=False,
    spatial_index=True, spatial_grid_shape=None,
    synapses=None
  ):
    super(SkeletonTask, self).__init__(
      cloudpath, shape, offset, mip, 
      teasar_params, will_postprocess, 
      info, object_ids, mask_ids,
      fix_branching, fix_borders, fix_avocados,
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
      fix_avocados=self.fix_avocados,
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
    
    # neuroglancer doesn't support int attributes
    self.strip_integer_attributes(skeletons.values())

    if self.sharded:
      self.upload_batch(vol, path, index_bbox, skeletons)
    else:
      self.upload_individuals(vol, path, bbox, skeletons)

    if self.spatial_index:
      self.upload_spatial_index(vol, path, index_bbox, skeletons)
  
  def strip_integer_attributes(self, skeletons):
    for skel in skeletons:
      skel.extra_attributes = [ 
      attr for attr in skel.extra_attributes 
      if attr['data_type'] in ('float32', 'float64')
    ]
    return skeletons

  def upload_batch(self, vol, path, bbox, skeletons):
    mbuf = MapBuffer(
      skeletons, compress="br", 
      tobytesfn=lambda skel: skel.to_precomputed()
    )

    cf = CloudFiles(path, progress=vol.progress)
    cf.put(
      path="{}.frags".format(bbox.to_filename()),
      content=mbuf.tobytes(),
      compress=None,
      content_type="application/x-mapbuffer",
      cache_control=False,
    )

  def upload_individuals(self, vol, path, bbox, skeletons):
    skeletons = skeletons.values()

    if not self.will_postprocess:
      vol.skeleton.upload(skeletons)
      return 

    bbox = bbox * vol.resolution
    cf = CloudFiles(path, progress=vol.progress)
    cf.puts(
      (
        (
          f"{skel.id}:{bbox.to_filename()}",
          pickle.dumps(skel)
        )
        for skel in skeletons
      ),
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
    cf = CloudFiles(path, progress=vol.progress)
    cf.put_json(
      path=f"{bbox.to_filename()}.spatial",
      content=spatial_index,
      compress='gzip',
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
      cf = CloudFiles(self.cloudpath, progress=True)
      cf.delete(fragment_filenames)

  def get_filenames(self):
    prefix = '{}/{}'.format(self.vol.skeleton.path, self.prefix)

    cf = CloudFiles(self.cloudpath, progress=True)
    return [ _ for _ in cf.list(prefix=prefix) ]

  def get_skeletons_by_segid(self, filenames):
    cf = CloudFiles(self.cloudpath, progress=True)
    skels = cf.get(filenames)

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
    dust_threshold=4000, tick_threshold=6000,
    sqlite_db=None
  ):
    super(ShardedSkeletonMergeTask, self).__init__(
      cloudpath, shard_no,  
      dust_threshold, tick_threshold, sqlite_db
    )
    self.progress = False

  def execute(self):
    # cache is necessary for local computation, but on GCE download is very fast
    # so cache isn't necessary.
    cv = CloudVolume(self.cloudpath, cache=False, progress=self.progress)

    # This looks messy because we are trying to avoid retaining
    # unnecessary memory. In the original iteration, this was 
    # using 50 GB+ memory on minnie65. With changes to this
    # and the spatial_index, we are getting it down to something reasonable.
    locations = self.locations_for_labels(self.labels_for_shard(cv), cv)
    filenames = set(itertools.chain(*locations.values()))
    labels = set(locations.keys())
    del locations
    skeletons = self.get_unfused(labels, filenames, cv)
    del labels
    del filenames
    skeletons = self.process_skeletons(skeletons, in_place=True)

    if len(skeletons) == 0:
      return

    shard_files = synthesize_shard_files(cv.skeleton.reader.spec, skeletons)

    if len(shard_files) != 1:
      raise ValueError(
        "Only one shard file should be generated per task. Expected: {} Got: {} ".format(
          str(self.shard_no), ", ".join(shard_files.keys())
      ))

    cf = CloudFiles(cv.skeleton.meta.layerpath, progress=self.progress)
    cf.puts( 
      ( (fname, data) for fname, data in shard_files.items() ),
      compress=False,
      content_type='application/octet-stream',
      cache_control='no-cache',      
    )

  def process_skeletons(self, unfused_skeletons, in_place=False):
    skeletons = {}
    if in_place:
      skeletons = unfused_skeletons

    for label in tqdm(unfused_skeletons.keys(), desc="Postprocessing", disable=(not self.progress)):
      skels = unfused_skeletons[label]
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
      blocks = sip(filenames, block_size)

    all_skels = defaultdict(list)
    for filenames_block in tqdm(blocks, desc="Filename Block", total=n_blocks, disable=(not self.progress)):
      if cv.meta.path.protocol == "file":
        all_files = {}
        prefix = cv.cloudpath.replace("file://", "")
        for filename in filenames_block:
          f = open(os.path.join(prefix, filename), "rb")
          all_files[filename] = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
      else:
        all_files = cv.skeleton.cache.download(filenames_block, progress=self.progress)
      
      for filename, content in tqdm(all_files.items(), desc="Scanning Fragments", disable=(not self.progress)):
        try:
          fragment = pickle.loads(content)
        except pickle.UnpicklingError:
          fragment = MapBuffer(content, frombytesfn=PrecomputedSkeleton.from_precomputed)

        for label in labels:
          try:
            skel = fragment[label]
            skel.id = label
            all_skels[label].append(skel)
          except KeyError:
            continue

    return all_skels

  def locations_for_labels(self, labels, cv):
    SPATIAL_EXT = re.compile(r'\.spatial$')
    if self.sqlite_db:
      cv.skeleton.spatial_index.sqlite_db = self.sqlite_db
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
    labels = CloudFiles(cv.skeleton.meta.layerpath).get_json(self.shard_no + '.labels')
    if labels is not None:
      return labels

    labels = cv.skeleton.spatial_index.query(cv.bounds * cv.resolution)
    spec = cv.skeleton.reader.spec

    return [ 
      lbl for lbl in tqdm(labels, desc="Computing Shard Numbers", disable=(not self.progress))  \
      if spec.compute_shard_location(lbl).shard_number == self.shard_no 
    ]
