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
    if not self.will_postprocess:
      skels = []
      for segid, skel in skeletons.items():
        skel = PrecomputedSkeleton.simple_merge(skel)
        skel = skel.consolidate()
        skels.append(skel)
      vol.skeleton.upload_multiple(skels)
    else:
      with Storage(path, progress=True) as stor:
        for segid, skels in skeletons.items():
          for skeleton in skels:
            bbox *= vol.resolution
            stor.put_file(
              file_path="{}:{}".format(skeleton.id, bbox.to_filename()),
              content=pickle.dumps(skeleton),
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
  def __init__(self, cloudpath, prefix, crop=0):
    super(SkeletonMergeTask, self).__init__(cloudpath, prefix, crop)
    self.cloudpath = cloudpath
    self.prefix = prefix
    self.crop_zone = crop

  def execute(self):
    self.vol = CloudVolume(self.cloudpath, cdn_cache=False)

    with Storage(self.cloudpath) as stor:
      skels = self.get_filenames_subset(stor)

      vol = self.vol

      for segid, frags in tqdm(skels.items(), desc='segid'):
        skeleton = self.fuse_skeletons(frags, stor)
        # skeleton = trim_skeleton(skeleton)
        vol.skeleton.upload(
          segid, skeleton.vertices, skeleton.edges, skeleton.radii, skeleton.vertex_types
        )
      
      # for segid, frags in skels.items():
      #   stor.delete_files(frags)

  def get_filenames_subset(self, storage):
    prefix = '{}/{}'.format(self.vol.skeleton.path, self.prefix)
    skeletons = defaultdict(list)

    for filename in storage.list_files(prefix=prefix):
      # `match` implies the beginning (^). `search` matches whole string
      matches = re.search(r'(\d+):', filename)

      if not matches:
        continue

      segid, = matches.groups()
      segid = int(segid)
      skeletons[segid].append(filename)

    return skeletons

  def fuse_skeletons(self, filenames, storage):
    if len(filenames) == 0:
      return Skeleton()
    
    skldl = storage.get_files(filenames)
    bbxs = [ Bbox.from_filename(item['filename']) for item in skldl ]
    skeletons = [ pickle.loads(item['content']) for item in skldl ]

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
      bbx.minpt += self.crop_zone
      bbx.maxpt -= self.crop_zone

      if bbx.volume() <= 0:
        continue

      cropped[i] = cropped[i].crop(bbx)

    return cropped