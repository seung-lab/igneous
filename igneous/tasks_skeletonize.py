try:
    from StringIO import cStringIO as BytesIO
except ImportError:
    from io import BytesIO

import io
import itertools
import json
import math
import os
import random
import re

import numpy as np
from tqdm import tqdm

from cloudvolume import CloudVolume, Storage, SimpleStorage
from cloudvolume.lib import xyzrange, min2, max2, Vec, Bbox, mkdir
from taskqueue import RegisteredTask

from igneous import chunks, downsample, downsample_scales
from igneous import Mesher # broken out for ease of commenting out

def skeldir(cloudpath):
  with SimpleStorage(self.layer_path) as storage:
    info = storage.get_json('info')

  skel_dir = 'skeletons/points'
  if 'skeletons' in info:
    skel_dir = info['skeletons']
  return skel_dir

class PointCloudTask(RegisteredTask):
  """
  Stage 1 of skeletonization.

  Convert chunks of segmentation into point clouds.
  They will be agglomerated in the stage 2 task
  PointCloudAggregationTask.
  """
  def __init__(self, cloudpath, shape, offset, mip):
    super(PointCloudTask, self).__init__(cloudpath, shape, offset, mip)
    self.cloudpath = cloudpath
    self.bounds = Bbox(offset, Vec(*shape) + Vec(*offset))
    self.mip = mip

  def execute(self):
    vol = CloudVolume(self.cloudpath, mip=self.mip)
    bbox = Bbox.clamp(self.bounds, vol.bounds)

    image = vol[ bbox.to_slices() ]
    image = image[:,:,:,0]
    uniques = np.unique(image)

    point_path = os.path.join(skel_dir(self.cloudpath), 'points')

    with Storage(point_path) as storage:
      for segid in uniques:
        # type = int32 b/c coords can be less than zero after bbox adjustment
        # and int32 saves 2x over int64 default
        coords = np.argwhere( image == segid ).astype(np.int32)
        coords[:,0,0] += bbox.minpt.x
        coords[0,:,0] += bbox.minpt.y
        coords[0,0,:] += bbox.minpt.z
        storage.put_file(
          file_path="{}:{}".format(segid, bbox.to_filename()),
          content=coords.tostring('C'),
          compress='gzip',
        )

class PointCloudAggregationTask(RegisteredTask):
  """
  Stage 2 of skeletonization.

  Combine point cloud chunks into a single unified point cloud.

  If we parallelize using prefixes single digit prefixes ['0','1',..'9'] all meshes will
  be correctly processed. But if we do ['10','11',..'99'] meshes from [0,9] won't get
  processed and need to be handle specifically by creating tasks that will process
  a single mesh ['0:','1:',..'9:']
  """
  def __init__(self, cloudpath, prefix):
    super(PointCloudAggregationTask, self).__init__(cloudpath, prefix)
    self.cloudpath = cloudpath
    self.prefix = prefix

  def execute(self):
    self.pointdir = os.path.join(skeldir(self.cloudpath), 'points')

    with Storage(self.cloudpath) as storage:
      self._agglomerate(storage)

  def _get_filenames_subset(self, storage):
    prefix = '{}/{}'.format(self.pointdir, self.prefix)
    segids = defaultdict(list)

    for filename in storage.list_files(prefix=prefix):
      filename = os.path.basename(filename)
      # `match` implies the beginning (^). `search` matches whole string
      matches = re.search(r'(\d+):', filename)

      if not matches:
        continue

      segid, = matches.groups()
      segid = int(segid)
      segids[segid].append(filename)

    return segids

  def _agglomerate(self, storage):
    segids = self._get_filenames_subset(storage)
    for segid, frags in segids.items():
      points = [ f['content'] for f in storage.get_files(frags) ]

      for i in range(len(points)):
        buf = points[i]
        shape = (len(buf) // 3, 3) # 2d list of xyz
        points[i] = np.frombuffer(buf, dtype=np.int32).reshape(shape, order='C')

      points = np.concatenate(*points, axis=0)
      points.sort(axis=0) # sorts x column, but not y unfortunately

      storage.put_file(
        file_path='{}/{}'.format(self.pointdir, segid),
        content=points.tostring('C'),
        compress='gzip',
      ).wait()

      storage.delete_files(frags)























