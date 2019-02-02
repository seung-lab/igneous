import six

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

import cc3d # connected components
import edt # euclidean distance transform
import fastremap
from taskqueue import RegisteredTask

import igneous.skeletontricks

from .postprocess import trim_skeleton

import kimimaro

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
    info=None, object_ids=None, fix_branching=True
  ):
    super(SkeletonTask, self).__init__(
      cloudpath, shape, offset, mip, 
      teasar_params, will_postprocess, 
      info, object_ids, fix_branching
    )
    self.bounds = Bbox(offset, Vec(*shape) + Vec(*offset))


  def execute(self):
    vol = CloudVolume(self.cloudpath, mip=self.mip, info=self.cloudinfo, cdn_cache=False)
    bbox = Bbox.clamp(self.bounds, vol.bounds)

    path = skeldir(self.cloudpath)
    path = os.path.join(self.cloudpath, path)

    all_labels = vol[ bbox.to_slices() ]
    all_labels = all_labels[:,:,:,0]

    skeletons = kimimaro.skeletonize(
      all_labels, self.teasar_params, 
      object_ids=self.object_ids, anisotropy=vol.resolution,
      dust_threshold=1000, cc_safety_factor=0.25,
      progress=True, fix_branching=self.fix_branching
    )

    for segid, skel in six.iteritems(skeletons):
      skel.vertices[:,0] += bbox.minpt.x
      skel.vertices[:,1] += bbox.minpt.y
      skel.vertices[:,2] += bbox.minpt.z

    self.upload(vol, path, bbox, skeletons.values())
      
  def upload(self, vol, path, bbox, skeletons):

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