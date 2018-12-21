try:
    from StringIO import cStringIO as BytesIO
except ImportError:
    from io import BytesIO

from functools import reduce
import json
import pickle
import os
import re
from collections import defaultdict

import numpy as np
import scipy.ndimage
from tqdm import tqdm

import cloudvolume
from cloudvolume import CloudVolume, PrecomputedSkeleton
from cloudvolume.storage import Storage, SimpleStorage
from cloudvolume.lib import xyzrange, min2, max2, Vec, Bbox, mkdir, save_images

import edt # euclidean distance transform
import fastremap
from taskqueue import RegisteredTask

from igneous import chunks, downsample, downsample_scales
import igneous.cc3d as cc3d
import igneous.skeletontricks

from .skeletonization import TEASAR
from .postprocess import trim_skeleton

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
    info=None, object_ids=None
  ):
    super(SkeletonTask, self).__init__(
      cloudpath, shape, offset, mip, 
      teasar_params, will_postprocess, 
      info, object_ids
    )
    self.cloudpath = cloudpath
    self.bounds = Bbox(offset, Vec(*shape) + Vec(*offset))
    self.mip = mip
    self.teasar_params = teasar_params
    self.will_postprocess = will_postprocess
    self.cloudinfo = info
    self.object_ids = object_ids

  def execute(self):
    vol = CloudVolume(self.cloudpath, mip=self.mip, info=self.cloudinfo, cdn_cache=False)
    bbox = Bbox.clamp(self.bounds, vol.bounds)

    all_labels = vol[ bbox.to_slices() ]
    all_labels = all_labels[:,:,:,0]
    if self.object_ids is not None:
      all_labels[~np.isin(all_labels, self.object_ids)] = 0

    if not np.any(all_labels):
      return

    tmp_labels, remapping = fastremap.renumber(all_labels)
    cc_labels = cc3d.connected_components(tmp_labels, max_labels=int(bbox.volume() / 4))

    del tmp_labels
    remapping = igneous.skeletontricks.get_mapping(all_labels, cc_labels)
    del all_labels

    path = skeldir(self.cloudpath)
    path = os.path.join(self.cloudpath, path)

    all_dbf = edt.edt(cc_labels, 
      anisotropy=vol.resolution.tolist(),
      black_border=False,
      order='F',
    )
    # slows things down, but saves memory
    # max_all_dbf = np.max(all_dbf)
    # if max_all_dbf < np.finfo(np.float16).max:
    #   all_dbf = all_dbf.astype(np.float16)

    cc_segids, pxct = np.unique(cc_labels, return_counts=True)
    cc_segids = [ sid for sid, ct in zip(cc_segids, pxct) if ct > 1000 ]

    all_slices = scipy.ndimage.find_objects(cc_labels)

    skeletons = defaultdict(list)
    for segid in tqdm(cc_segids):
      if segid == 0:
        continue 

      # Crop DBF to ROI
      slices = all_slices[segid - 1]
      if slices is None:
        continue

      labels = cc_labels[slices]
      labels = (labels == segid)
      dbf = (labels * all_dbf[slices]).astype(np.float32)

      roi = Bbox.from_slices(slices)
      roi += bbox.minpt 

      skeleton = self.skeletonize(
        labels, dbf, roi, 
        anisotropy=vol.resolution.tolist()
      )

      if skeleton.empty():
        continue

      orig_segid = remapping[segid]
      skeleton.id = orig_segid
      skeleton.vertices *= vol.resolution
      skeletons[orig_segid].append(skeleton)

    self.upload(vol, path, bbox, skeletons)
      
  def upload(self, vol, path, bbox, skeletons):

    skel_lst = []
    for segid, skels in skeletons.items():
      skel = PrecomputedSkeleton.simple_merge(skels)
      skel_lst.append( skel.consolidate() )
      
    if not self.will_postprocess:
      vol.skeleton.upload(skel_lst)
      return 

    bbox = bbox * vol.resolution
    with Storage(path, progress=vol.progress) as stor:
      for skel in skel_lst:
        stor.put_file(
          file_path="{}:{}".format(skel.id, bbox.to_filename()),
          content=pickle.dumps(skel),
          compress='gzip',
          content_type="application/python-pickle",
          cache_control=False,
        )
      
  def skeletonize(self, labels, dbf, bbox, anisotropy):
    skeleton = TEASAR(labels, dbf, anisotropy=anisotropy, **self.teasar_params)

    skeleton.vertices[:,0] += bbox.minpt.x
    skeleton.vertices[:,1] += bbox.minpt.y
    skeleton.vertices[:,2] += bbox.minpt.z

    return skeleton

class SkeletonMergeTask(RegisteredTask):
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
    super(SkeletonMergeTask, self).__init__(
      cloudpath, prefix, crop, mip, 
      dust_threshold, tick_threshold, delete_fragments
    )
    self.cloudpath = cloudpath
    self.prefix = prefix
    self.crop_zone = crop # voxels
    self.mip = mip
    self.dust_threshold = dust_threshold # nm
    self.tick_threshold = tick_threshold # nm
    self.delete_fragments = delete_fragments

  def execute(self):
    self.vol = CloudVolume(self.cloudpath, mip=self.mip, cdn_cache=False)

    fragment_filenames = self.get_filenames()
    skels = self.get_skeletons_by_segid(fragment_filenames)

    skeletons = []
    for segid, frags in tqdm(skels.items(), desc='segid'):
      skeleton = self.fuse_skeletons(frags)
      skeleton = trim_skeleton(skeleton, self.dust_threshold, self.tick_threshold)
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

    skeletons = self.crop(bbxs, skeletons)
    skeletons = [ s for s in skeletons if not s.empty() ]

    if len(skeletons) == 0:
      return PrecomputedSkeleton()

    return PrecomputedSkeleton.simple_merge(skeletons).consolidate()

  def crop(self, bbxs, skeletons):
    cropped = [ s.clone() for s in skeletons ]

    if self.crop_zone <= 0:
      return cropped
    
    for i in range(len(skeletons)):
      bbx = bbxs[i] 
      bbx.minpt += self.crop_zone * self.vol.resolution
      bbx.maxpt -= self.crop_zone * self.vol.resolution

      if bbx.volume() <= 0:
        continue

      cropped[i] = cropped[i].crop(bbx)

    return cropped