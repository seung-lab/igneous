from collections import defaultdict
from itertools import product
from functools import reduce, partial
import operator

import copy
import json
import math
import os
import re
import subprocess
import time
from time import strftime

import numpy as np
from tqdm import tqdm

import cloudvolume
import cloudvolume.exceptions
from cloudvolume import CloudVolume
from cloudvolume.lib import Vec, Bbox, max2, min2, xyzrange, find_closest_divisor, yellow, jsonify
from cloudvolume.datasource.precomputed.sharding import ShardingSpecification
from cloudfiles import CloudFiles
from taskqueue import TaskQueue, MockTaskQueue 

from igneous import downsample_scales, chunks
import igneous.tasks
from igneous.tasks import (
  HyperSquareConsensusTask, 
  MeshTask, MeshManifestTask, GrapheneMeshTask,
  DownsampleTask, QuantizeTask, 
  TransferTask, WatershedRemapTask, DeleteTask, 
  LuminanceLevelsTask, ContrastNormalizationTask,
  SkeletonTask, UnshardedSkeletonMergeTask, ShardedSkeletonMergeTask, 
  MaskAffinitymapTask, InferenceTask
)
# from igneous.tasks import BigArrayTask

# for provenance files

OPERATOR_CONTACT = ''
if __name__ == '__main__':
  try:
    OPERATOR_CONTACT = subprocess.check_output("git config user.email", shell=True)
    OPERATOR_CONTACT = str(OPERATOR_CONTACT.rstrip())
  except:
    try:
      print(yellow('Unable to determine provenance contact email. Set "git config user.email". Using unix $USER instead.'))
      OPERATOR_CONTACT = os.environ['USER']
    except:
      print(yellow('$USER was not set. The "owner" field of the provenance file will be blank.'))
      OPERATOR_CONTACT = ''

def get_bounds(vol, bounds, mip, chunk_size=None):
  if bounds is None:
    bounds = vol.bounds.clone()
  else:
    bounds = Bbox.create(bounds)
    bounds = vol.bbox_to_mip(bounds, mip=0, to_mip=mip)
    if chunk_size is not None:
      bounds = bounds.expand_to_chunk_size(chunk_size, vol.meta.voxel_offset(mip))
    bounds = Bbox.clamp(bounds, vol.meta.bounds(mip))
  

  print("Volume Bounds: ", vol.meta.bounds(mip))
  print("Selected ROI:  ", bounds)

  return bounds

def num_tasks(bounds, shape):
  return int(reduce(operator.mul, np.ceil(bounds.size3() / shape)))

class FinelyDividedTaskIterator():
  """
  Parallelizes tasks that do not have overlap.

  Evenly splits tasks between processes without 
  regards to whether the dividing line lands in
  the middle of a slice. 
  """
  def __init__(self, bounds, shape):
    self.bounds = bounds 
    self.shape = Vec(*shape)
    self.start = 0
    self.end = num_tasks(bounds, shape)

  def __len__(self):
    return self.end - self.start
  
  def __getitem__(self, slc):
    itr = copy.deepcopy(self)
    itr.start = max(self.start + slc.start, self.start)
    itr.end = min(self.start + slc.stop, self.end)
    return itr

  def __iter__(self):
    for i in range(self.start, self.end):
      pt = self.to_coord(i)
      offset = pt * self.shape + self.bounds.minpt
      yield self.task(self.shape.clone(), offset.clone())

    self.on_finish()

  def to_coord(self, index):
    """Convert an index into a grid coordinate defined by the task shape."""
    sx, sy, sz = np.ceil(self.bounds.size3() / self.shape).astype(int)
    sxy = sx * sy
    z = index // sxy
    y = (index - (z * sxy)) // sx
    x = index - sx * (y + z * sy)
    return Vec(x,y,z)

  def task(self, shape, offset):
    raise NotImplementedError()

  def on_finish(self):
    pass

# def create_bigarray_task(cloudpath):
#   """
#   Creates one task for each bigarray chunk present in the bigarray folder.
#   These tasks will convert the bigarray chunks into chunks that ingest tasks are able to understand.
#   """
#   class BigArrayTaskIterator():
#     def __iter__(self):    
#       with Storage(cloudpath) as storage:
#         for filename in storage.list_blobs(prefix='bigarray/'):
#           yield BigArrayTask(
#             chunk_path=storage.get_path_to_file('bigarray/'+filename),
#             chunk_encoding='npz', # npz_uint8 to convert affinites float32 affinties to uint8
#             version='{}/{}'.format(storage._path.dataset_name, storage._path.layer_name)
#           )
#   return BigArrayTaskIterator()

def create_blackout_tasks(
    cloudpath, bounds, 
    mip=0, shape=(2048, 2048, 64), 
    value=0, non_aligned_writes=False
  ):

  vol = CloudVolume(cloudpath, mip=mip)

  shape = Vec(*shape)
  bounds = Bbox.create(bounds)
  bounds = vol.bbox_to_mip(bounds, mip=0, to_mip=mip)

  if not non_aligned_writes:
    bounds = bounds.expand_to_chunk_size(vol.chunk_size, vol.voxel_offset)
    
  bounds = Bbox.clamp(bounds, vol.mip_bounds(mip))

  class BlackoutTaskIterator(FinelyDividedTaskIterator):
    def task(self, shape, offset):
      bounded_shape = min2(shape, vol.bounds.maxpt - offset)
      return partial(igneous.tasks.BlackoutTask, 
        cloudpath=cloudpath, 
        mip=mip, 
        shape=shape.clone(), 
        offset=offset.clone(),
        value=value, 
        non_aligned_writes=non_aligned_writes,
      )

    def on_finish(self):
      vol.provenance.processing.append({
        'method': {
          'task': 'BlackoutTask',
          'cloudpath': cloudpath,
          'mip': mip,
          'non_aligned_writes': non_aligned_writes,
          'value': value,
          'shape': shape.tolist(),
          'bounds': [
            bounds.minpt.tolist(),
            bounds.maxpt.tolist(),
          ],
        },
        'by': OPERATOR_CONTACT,
        'date': strftime('%Y-%m-%d %H:%M %Z'),
      })

  return BlackoutTaskIterator(bounds, shape)
  
def create_touch_tasks(
    cloudpath, 
    mip=0, shape=(2048, 2048, 64),
    bounds=None
  ):

  vol = CloudVolume(cloudpath, mip=mip)

  shape = Vec(*shape)

  if bounds is None:
    bounds = vol.bounds.clone()

  bounds = Bbox.create(bounds)
  bounds = vol.bbox_to_mip(bounds, mip=0, to_mip=mip)
  bounds = Bbox.clamp(bounds, vol.mip_bounds(mip))

  class TouchTaskIterator(FinelyDividedTaskIterator):
    def task(self, shape, offset):
      bounded_shape = min2(shape, vol.bounds.maxpt - offset)
      return igneous.tasks.TouchTask(
        cloudpath=cloudpath,
        shape=bounded_shape.clone(),
        offset=offset.clone(),
        mip=mip,
      )

    def on_finish(self):
      vol.provenance.processing.append({
        'method': {
            'task': 'TouchTask',
            'mip': mip,
            'shape': shape.tolist(),
            'bounds': [
              bounds.minpt.tolist(),
              bounds.maxpt.tolist(),
            ],
          },
        'by': OPERATOR_CONTACT,
        'date': strftime('%Y-%m-%d %H:%M %Z'),
      })
      vol.commit_provenance()

  return TouchTaskIterator(bounds, shape)

def create_downsampling_tasks(
    layer_path, mip=0, fill_missing=False, 
    axis='z', num_mips=5, preserve_chunk_size=True,
    sparse=False, bounds=None, chunk_size=None,
    encoding=None, delete_black_uploads=False, 
    background_color=0, dest_path=None, compress=None,
    factor=None
  ):
    """
    mip: Download this mip level, writes to mip levels greater than this one.
    fill_missing: interpret missing chunks as black instead of issuing an EmptyVolumeException
    axis: technically 'x' and 'y' are supported, but no one uses them.
    num_mips: download a block chunk * 2**num_mips size and generate num_mips mips. If you have
      memory problems, try reducing this number.
    preserve_chunk_size: if true, maintain chunk size of starting mip, else, find the closest
      evenly divisible chunk size to 64,64,64 for this shape and use that. The latter can be
      useful when mip 0 uses huge chunks and you want to simply visualize the upper mips.
    chunk_size: (overrides preserve_chunk_size) force chunk size for new layers to be this.
    sparse: When downsampling segmentation, if true, don't count black pixels when computing
      the mode. Useful for e.g. synapses and point labels.
    bounds: By default, downsample everything, but you can specify restricted bounding boxes
      instead. The bounding box will be expanded to the nearest chunk. Bbox is specifed in mip 0
      coordinates.
    delete_black_uploads: issue delete commands instead of upload chunks
      that are all background.
    background_color: Designates which color should be considered background.
    dest_path: (optional) instead of writing downsamples to the existing 
      volume, write them somewhere else. This can be useful e.g. if someone 
      doesn't want you to touch the existing info file.
    compress: None, 'gzip', or 'br' Determines which compression algorithm to use 
      for new uploaded files.
    factor: (overrides axis) can manually specify what each downsampling round is
      supposed to do: e.g. (2,2,1), (2,2,2), etc
    """
    def ds_shape(mip, chunk_size=None, factor=None):
      if chunk_size:
        shape = Vec(*chunk_size)
      else:
        shape = vol.meta.chunk_size(mip)[:3]

      if factor is None:
        factor = downsample_scales.axis_to_factor(axis)

      shape.x *= factor[0] ** num_mips
      shape.y *= factor[1] ** num_mips
      shape.z *= factor[2] ** num_mips
      return shape

    vol = CloudVolume(layer_path, mip=mip)
    shape = ds_shape(mip, chunk_size, factor)

    vol = downsample_scales.create_downsample_scales(
      layer_path, mip, shape, 
      preserve_chunk_size=preserve_chunk_size, chunk_size=chunk_size,
      encoding=encoding, factor=factor
    )

    if not preserve_chunk_size or chunk_size:
      shape = ds_shape(mip + 1, chunk_size, factor)

    bounds = get_bounds(vol, bounds, mip, vol.chunk_size)
    
    class DownsampleTaskIterator(FinelyDividedTaskIterator):
      def task(self, shape, offset):
        return partial(DownsampleTask, 
          layer_path=layer_path,
          mip=vol.mip,
          shape=shape.clone(),
          offset=offset.clone(),
          axis=axis,
          fill_missing=fill_missing,
          sparse=sparse,
          delete_black_uploads=delete_black_uploads,
          background_color=background_color,
          dest_path=dest_path,
          compress=compress,
          factor=factor,
        )

      def on_finish(self):
        vol.provenance.processing.append({
          'method': {
            'task': 'DownsampleTask',
            'mip': mip,
            'num_mips': num_mips,
            'shape': shape.tolist(),
            'axis': axis,
            'method': 'downsample_with_averaging' if vol.layer_type == 'image' else 'downsample_segmentation',
            'sparse': sparse,
            'bounds': str(bounds),
            'chunk_size': (list(chunk_size) if chunk_size else None),
            'preserve_chunk_size': preserve_chunk_size,
            'encoding': encoding,
            'fill_missing': bool(fill_missing),
            'delete_black_uploads': bool(delete_black_uploads),
            'background_color': background_color,
            'dest_path': dest_path,
            'compress': compress,
            'factor': (tuple(factor) if factor else None),
          },
          'by': OPERATOR_CONTACT,
          'date': strftime('%Y-%m-%d %H:%M %Z'),
        })
        vol.commit_provenance()

    return DownsampleTaskIterator(bounds, shape)

def create_deletion_tasks(
    layer_path, mip=0, num_mips=5, 
    shape=None, bounds=None
  ):
  vol = CloudVolume(layer_path, max_redirects=0)
  
  if shape is None:
    shape = vol.mip_underlying(mip)[:3]
    shape.x *= 2 ** num_mips
    shape.y *= 2 ** num_mips
  else:
    shape = Vec(*shape)

  if not bounds:
    bounds = vol.mip_bounds(mip).clone()

  class DeleteTaskIterator(FinelyDividedTaskIterator):
    def task(self, shape, offset):
      bounded_shape = min2(shape, bounds.maxpt - offset)
      return partial(DeleteTask,
        layer_path=layer_path,
        shape=bounded_shape.clone(),
        offset=offset.clone(),
        mip=mip,
        num_mips=num_mips,
      )

    def on_finish(self):
      vol = CloudVolume(layer_path, max_redirects=0)
      vol.provenance.processing.append({
        'method': {
          'task': 'DeleteTask',
          'mip': mip,
          'num_mips': num_mips,
          'shape': shape.tolist(),
        },
        'by': OPERATOR_CONTACT,
        'date': strftime('%Y-%m-%d %H:%M %Z'),
      })
      vol.commit_provenance()

  return DeleteTaskIterator(bounds, shape)

def create_skeletonizing_tasks(
    cloudpath, mip, 
    shape=Vec(512, 512, 512),
    teasar_params={'scale':10, 'const': 10}, 
    info=None, object_ids=None, mask_ids=None,
    fix_branching=True, fix_borders=True, 
    fix_avocados=False, fill_holes=False,
    dust_threshold=1000, progress=False,
    parallel=1, fill_missing=False, 
    sharded=False, spatial_index=True,
    synapses=None, num_synapses=None
  ):
  """
  Assign tasks with one voxel overlap in a regular grid 
  to be densely skeletonized. The default shape (512,512,512)
  was designed to work within 6 GB of RAM on average at parallel=1 
  but can exceed this amount for certain objects such as glia. 
  4 GB is usually OK.

  When this run completes, you'll follow up with create_skeleton_merge_tasks
  to postprocess the generated fragments into single skeletons. 

  WARNING: If you are processing hundreds of millions of labels or more and
  are using Cloud Storage this can get expensive ($8 per million labels typically
  accounting for fragment generation and postprocessing)! This scale is when 
  the experimental sharded format generator becomes crucial to use.

  cloudpath: cloudvolume path
  mip: which mip level to skeletonize 
    For a 4x4x40 dataset, mip 3 is good. Mip 4 starts introducing 
    artifacts like snaking skeletons along the edge of thin objects.

  teasar_params: 
    NOTE: see github.com/seung-lab/kimimaro for an updated list
          see https://github.com/seung-lab/kimimaro/wiki/Intuition-for-Setting-Parameters-const-and-scale
          for help with setting these parameters.
    NOTE: DBF = Distance from Boundary Field (i.e. euclidean distance transform)

    scale: float, multiply invalidation radius by distance from boundary
    const: float, add this physical distance to the invalidation radius
    soma_detection_threshold: if object has a DBF value larger than this, 
        root will be placed at largest DBF value and special one time invalidation
        will be run over that root location (see soma_invalidation scale)
        expressed in chosen physical units (i.e. nm) 
    pdrf_scale: scale factor in front of dbf, used to weight DBF over euclidean distance (higher to pay more attention to dbf) 
    pdrf_exponent: exponent in dbf formula on distance from edge, faster if factor of 2 (default 16)
    soma_invalidation_scale: the 'scale' factor used in the one time soma root invalidation (default .5)
    soma_invalidation_const: the 'const' factor used in the one time soma root invalidation (default 0)
                           (units in chosen physical units (i.e. nm))    

  info: supply your own info file 
  object_ids: mask out all but these ids if specified
  mask_ids: mask out these ids if specified
  fix_branching: Trades speed for quality of branching at forks. You'll
    almost always want this set to True.
  fix_borders: Allows trivial merging of single overlap tasks. You'll only
    want to set this to false if you're working on single or non-overlapping
    volumes.
  dust_threshold: don't skeletonize labels smaller than this number of voxels
    as seen by a single task.
  progress: show a progress bar
  parallel: number of processes to deploy against a single task. parallelizes
    over labels, it won't speed up a single complex label. You can be slightly
    more memory efficient using a single big task with parallel than with seperate
    tasks that add up to the same volume. Unless you know what you're doing, stick
    with parallel=1 for cloud deployments.
  fill_missing: passthrough to CloudVolume, fill missing image tiles with zeros
    instead of throwing an error if True.
  sharded: (bool) if true, output a single mapbuffer dict containing all skeletons
    in a task, which will serve as input to a sharded format generator. You don't 
    want this unless you know what you're doing. If False, generate a skeleton fragment
    file per a label for later agglomeration using the SkeletonMergeTask.
  spatial_index: (bool) Concurrently generate a json file that describes which
    labels were skeletonized in a given task. This makes it possible to query for
    skeletons by bounding box later on using CloudVolume.
  synapses: If provided, after skeletonization of a label is complete, draw 
    additional paths to one of the nearest voxels to synapse centroids.
    (x,y,z) centroid is specified in physical coordinates.

    Iterable yielding ((x,y,z),segid,swc_label)

  num_synapses: If synapses is an iterator, you must provide the total number of synapses.
  """
  shape = Vec(*shape)
  vol = CloudVolume(cloudpath, mip=mip, info=info)

  kdtree, labelsmap = None, None
  if synapses:
    centroids, kdtree, labelsmap = synapses_in_space(synapses, N=num_synapses)
  if not 'skeletons' in vol.info:
    vol.info['skeletons'] = 'skeletons_mip_{}'.format(mip)
    vol.commit_info()

  if spatial_index:
    if 'spatial_index' not in vol.skeleton.meta.info or not vol.skeleton.meta.info['spatial_index']:
      vol.skeleton.meta.info['spatial_index'] = {}
    vol.skeleton.meta.info['@type'] = 'neuroglancer_skeletons'
    vol.skeleton.meta.info['spatial_index']['resolution'] = tuple(vol.resolution)
    vol.skeleton.meta.info['spatial_index']['chunk_size'] = tuple(shape * vol.resolution)
  
  vol.skeleton.meta.info['mip'] = int(mip)
  vol.skeleton.meta.info['vertex_attributes'] = vol.skeleton.meta.info['vertex_attributes'][:1]
  vol.skeleton.meta.commit_info()

  will_postprocess = bool(np.any(vol.bounds.size3() > shape))
  bounds = vol.bounds.clone()

  class SkeletonTaskIterator(FinelyDividedTaskIterator):
    def task(self, shape, offset):
      bbox_synapses = None
      if synapses:
        bbox_synapses = self.synapses_for_bbox(shape, offset)

      return SkeletonTask(
        cloudpath=cloudpath,
        shape=(shape + 1).clone(), # 1px overlap on the right hand side
        offset=offset.clone(),
        mip=mip,
        teasar_params=teasar_params,
        will_postprocess=will_postprocess,
        info=info,
        object_ids=object_ids,
        mask_ids=mask_ids,
        fix_branching=fix_branching,
        fix_borders=fix_borders,
        fix_avocados=fix_avocados,
        dust_threshold=dust_threshold,
        progress=progress,
        parallel=parallel,
        fill_missing=bool(fill_missing),
        sharded=bool(sharded),
        spatial_index=bool(spatial_index),
        spatial_grid_shape=shape.clone(), # used for writing index filenames
        synapses=bbox_synapses,
      )

    def synapses_for_bbox(self, shape, offset):
      """
      Returns { seigd: [ ((x,y,z), swc_label), ... ] 
      where x,y,z are in voxel coordinates with the
      origin set to the bottom left corner of this cutout.
      """
      bbox = Bbox( offset, shape + offset ) * vol.resolution
      center = bbox.center()
      diagonal = Vec(*((bbox.maxpt - center)))
      pts = [ centroids[i,:] for i in kdtree.query_ball_point(center, diagonal.length()) ]
      pts = [ tuple(Vec(*pt, dtype=int)) for pt in pts if bbox.contains(pt) ]

      synapses = defaultdict(list)
      for pt in pts:
        for label, swc_label in labelsmap[pt]:
          voxel_pt = Vec(*pt, dtype=np.float32) / vol.resolution - offset
          synapses[label].append(( tuple(voxel_pt.astype(int)), swc_label))
      return synapses

    def on_finish(self):
      vol.provenance.processing.append({
        'method': {
          'task': 'SkeletonTask',
          'cloudpath': cloudpath,
          'mip': mip,
          'shape': shape.tolist(),
          'dust_threshold': dust_threshold,
          'teasar_params': teasar_params,
          'object_ids': object_ids,
          'mask_ids': mask_ids,
          'will_postprocess': will_postprocess,
          'fix_branching': fix_branching,
          'fix_borders': fix_borders,
          'fix_avocados': fix_avocados,
          'progress': progress,
          'parallel': parallel,
          'fill_missing': bool(fill_missing),
          'sharded': bool(sharded),
          'spatial_index': bool(spatial_index),
          'synapses': bool(synapses),
        },
        'by': OPERATOR_CONTACT,
        'date': strftime('%Y-%m-%d %H:%M %Z'),
      }) 
      vol.commit_provenance()

  return SkeletonTaskIterator(bounds, shape)

def synapses_in_space(synapse_itr, N=None):
  """
  Compute a kD tree of synapse locations and 
  a dictionary mapping centroid => labels

  Input: [ ((x,y,z),segid,swc_label), ... ]
  Output: centroids, kdtree, { centroid: (segid, swc_label) }
  """
  from scipy.spatial import cKDTree

  if N is None:
    N = len(synapse_itr)

  centroids = np.zeros( (N+1,3), dtype=np.int32)
  labels = defaultdict(list)

  for idx, (centroid,segid,swc_label) in enumerate(synapse_itr):
    centroid = tuple(Vec(*centroid, dtype=int))
    labels[centroid].append((segid, swc_label))
    centroids[idx,:] = centroid

  return centroids, cKDTree(centroids), labels

def create_flat_graphene_skeleton_merge_tasks(    
    cloudpath, mip, crop=0,
    dust_threshold=4000, 
    tick_threshold=6000, 
    delete_fragments=False
  ):

  prefixes = graphene_prefixes()

  class GrapheneSkeletonMergeTaskIterator():
    def __len__(self):
      return len(prefixes)
    def __iter__(self):
      # For a prefix like 100, tasks 1-99 will be missed. Account for them by
      # enumerating them individually with a suffixed ':' to limit matches to
      # only those small numbers
      for prefix in prefixes:
        yield UnshardedSkeletonMergeTask(
          cloudpath=cloudpath, 
          prefix=str(prefix),
          crop=crop,
          mip=mip,
          dust_threshold=dust_threshold,
          tick_threshold=tick_threshold,
          delete_fragments=delete_fragments,
        )

  return GrapheneSkeletonMergeTaskIterator()

def create_sharded_skeleton_merge_tasks(
    layer_path, dust_threshold, tick_threshold,
    preshift_bits, minishard_bits, shard_bits,
    minishard_index_encoding='gzip', data_encoding='gzip',
    max_cable_length=None
  ): 
  spec = ShardingSpecification(
    type='neuroglancer_uint64_sharded_v1',
    preshift_bits=preshift_bits,
    hash='murmurhash3_x86_128',
    minishard_bits=minishard_bits,
    shard_bits=shard_bits,
    minishard_index_encoding=minishard_index_encoding,
    data_encoding=data_encoding,
  )

  cv = CloudVolume(layer_path)
  cv.skeleton.meta.info['sharding'] = spec.to_dict()
  cv.skeleton.meta.commit_info()

  cv = CloudVolume(layer_path, progress=True) # rebuild b/c sharding changes the skeleton object
  cv.mip = cv.skeleton.meta.mip

  # 17 sec to download for pinky100
  all_labels = cv.skeleton.spatial_index.query(cv.bounds * cv.resolution)
  # perf: ~36k hashes/sec
  shardfn = lambda lbl: cv.skeleton.reader.spec.compute_shard_location(lbl).shard_number

  shard_labels = defaultdict(list)
  for label in tqdm(all_labels, desc="Hashes"):
    shard_labels[shardfn(label)].append(label)

  cf = CloudFiles(cv.skeleton.meta.layerpath, progress=True)
  files = ( 
    (str(shardno) + '.labels', labels) 
    for shardno, labels in shard_labels.items() 
  )
  cf.put_jsons(
    files, compress="gzip", 
    cache_control="no-cache", total=len(shard_labels)
  )
  
  cv.provenance.processing.append({
    'method': {
      'task': 'ShardedSkeletonMergeTask',
      'cloudpath': layer_path,
      'mip': cv.skeleton.meta.mip,
      'dust_threshold': dust_threshold,
      'tick_threshold': tick_threshold,
      'max_cable_length': max_cable_length,
      'preshift_bits': preshift_bits, 
      'minishard_bits': minishard_bits, 
      'shard_bits': shard_bits,
    },
    'by': OPERATOR_CONTACT,
    'date': strftime('%Y-%m-%d %H:%M %Z'),
  }) 
  cv.commit_provenance()

  return (
    ShardedSkeletonMergeTask(
      layer_path, shard_no, 
      dust_threshold, tick_threshold,
      max_cable_length=max_cable_length
    )
    for shard_no in shard_labels.keys()
  )

# split the work up into ~1000 tasks (magnitude 3)
def create_unsharded_skeleton_merge_tasks(    
    layer_path, crop=0,
    magnitude=3, dust_threshold=4000, max_cable_length=None,
    tick_threshold=6000, delete_fragments=False
  ):
  assert int(magnitude) == magnitude

  start = 10 ** (magnitude - 1)
  end = 10 ** magnitude

  class UnshardedSkeletonMergeTaskIterator():
    def __len__(self):
      return 10 ** magnitude
    def __iter__(self):
      # For a prefix like 100, tasks 1-99 will be missed. Account for them by
      # enumerating them individually with a suffixed ':' to limit matches to
      # only those small numbers
      for prefix in range(1, start):
        yield UnshardedSkeletonMergeTask(
          cloudpath=layer_path, 
          prefix=str(prefix) + ':',
          crop=crop,
          dust_threshold=dust_threshold,
          max_cable_length=max_cable_length,
          tick_threshold=tick_threshold,
          delete_fragments=delete_fragments,
        )

      # enumerate from e.g. 100 to 999
      for prefix in range(start, end):
        yield UnshardedSkeletonMergeTask(
          cloudpath=layer_path, 
          prefix=prefix, 
          crop=crop,
          dust_threshold=dust_threshold, 
          max_cable_length=max_cable_length,
          tick_threshold=tick_threshold,
          delete_fragments=delete_fragments,
        )

      vol = CloudVolume(layer_path)
      vol.provenance.processing.append({
        'method': {
          'task': 'UnshardedSkeletonMergeTask',
          'cloudpath': layer_path,
          'crop': crop,
          'dust_threshold': dust_threshold,
          'tick_threshold': tick_threshold,
          'delete_fragments': delete_fragments,
          'max_cable_length': max_cable_length,
        },
        'by': OPERATOR_CONTACT,
        'date': strftime('%Y-%m-%d %H:%M %Z'),
      }) 
      vol.commit_provenance()

  return UnshardedSkeletonMergeTaskIterator()

def create_meshing_tasks(
    layer_path, mip, shape=(448, 448, 448), 
    simplification=True, max_simplification_error=40,
    mesh_dir=None, cdn_cache=False, dust_threshold=None,
    object_ids=None, progress=False, fill_missing=False,
    encoding='precomputed', spatial_index=True, sharded=False,
    compress='gzip'
  ):
  shape = Vec(*shape)

  vol = CloudVolume(layer_path, mip)

  if mesh_dir is None:
    mesh_dir = 'mesh_mip_{}_err_{}'.format(mip, max_simplification_error)

  if not 'mesh' in vol.info:
    vol.info['mesh'] = mesh_dir
    vol.commit_info()

  cf = CloudFiles(layer_path)
  info_filename = '{}/info'.format(mesh_dir)
  mesh_info = cf.get_json(info_filename) or {}
  mesh_info['@type'] = 'neuroglancer_legacy_mesh'
  mesh_info['mip'] = int(vol.mip)
  mesh_info['chunk_size'] = shape.tolist()
  if spatial_index:
    mesh_info['spatial_index'] = {
        'resolution': vol.resolution.tolist(),
        'chunk_size': (shape*vol.resolution).tolist(),
    }
  cf.put_json(info_filename, mesh_info)

  class MeshTaskIterator(FinelyDividedTaskIterator):
    def task(self, shape, offset):
      return MeshTask(
        shape=shape.clone(),
        offset=offset.clone(),
        layer_path=layer_path,
        mip=vol.mip,
        simplification_factor=(0 if not simplification else 100),
        max_simplification_error=max_simplification_error,
        mesh_dir=mesh_dir, 
        cache_control=('' if cdn_cache else 'no-cache'),
        dust_threshold=dust_threshold,
        progress=progress,
        object_ids=object_ids,
        fill_missing=fill_missing,
        encoding=encoding,
        spatial_index=spatial_index,
        sharded=sharded,
        compress=compress,
      )

    def on_finish(self):
      vol.provenance.processing.append({
        'method': {
          'task': 'MeshTask',
          'layer_path': layer_path,
          'mip': vol.mip,
          'shape': shape.tolist(),
          'simplification': simplification,
          'max_simplification_error': max_simplification_error,
          'mesh_dir': mesh_dir,
          'fill_missing': fill_missing,
          'cdn_cache': cdn_cache,
          'dust_threshold': dust_threshold,
          'encoding': encoding,
          'object_ids': object_ids,
          'spatial_index': spatial_index,
          'sharded': sharded,
          'compress': compress,
        },
        'by': OPERATOR_CONTACT,
        'date': strftime('%Y-%m-%d %H:%M %Z'),
      }) 
      vol.commit_provenance()

  return MeshTaskIterator(vol.mip_bounds(mip), shape)

def create_graphene_meshing_tasks(
  cloudpath, timestamp, mip,
  simplification=True, max_simplification_error=40,
  mesh_dir=None, cdn_cache=False, object_ids=None, 
  progress=False, fill_missing=False, sharding=None,
  draco_compression_level=1, bounds=None
):
  cv = CloudVolume(cloudpath, mip=mip)

  if mip < cv.meta.watershed_mip:
    raise ValueError("Must mesh at or above the watershed mip level. Watershed MIP: {} Got: {}".format(
      cv.meta.watershed_mip, mip
    ))

  if mesh_dir is None:
    mesh_dir = 'meshes'

  cv.info['mesh'] = mesh_dir # necessary to set the mesh.commit_info() dir right
  if not 'mesh' in cv.info:
    cv.commit_info()

  watershed_downsample_ratio = cv.resolution // cv.meta.resolution(cv.meta.watershed_mip)
  shape = Vec(*cv.meta.graph_chunk_size) // watershed_downsample_ratio

  cv.mesh.meta.info['@type'] = 'neuroglancer_legacy_mesh'
  cv.mesh.meta.info['mip'] = cv.mip
  cv.mesh.meta.info['chunk_size'] = list(shape)
  if sharding:
    cv.mesh.meta.info['sharding'] = sharding
  cv.mesh.meta.commit_info()

  simplification = (0 if not simplification else 100)

  class GrapheneMeshTaskIterator(FinelyDividedTaskIterator):
    def task(self, shape, offset):
      return GrapheneMeshTask(
        cloudpath=cloudpath,
        shape=shape.clone(),
        offset=offset.clone(),
        mip=int(mip),
        simplification_factor=simplification,
        max_simplification_error=max_simplification_error,
        draco_compression_level=draco_compression_level,
        mesh_dir=mesh_dir, 
        cache_control=('' if cdn_cache else 'no-cache'),
        progress=progress,
        fill_missing=fill_missing,
        timestamp=timestamp,
      )

    def on_finish(self):
      cv.provenance.processing.append({
        'method': {
          'task': 'GrapheneMeshTask',
          'cloudpath': cv.cloudpath,
          'shape': cv.meta.graph_chunk_size,
          'mip': int(mip),
          'simplification': simplification,
          'max_simplification_error': max_simplification_error,
          'mesh_dir': mesh_dir,
          'fill_missing': fill_missing,
          'cdn_cache': cdn_cache,
          'timestamp': timestamp,
          'draco_compression_level': draco_compression_level,
        },
        'by': OPERATOR_CONTACT,
        'date': strftime('%Y-%m-%d %H:%M %Z'),
      }) 
      cv.commit_provenance()

  if bounds is None:
    bounds = cv.meta.bounds(mip).clone()
  else:
    bounds = cv.bbox_to_mip(bounds, mip=0, to_mip=mip)
    bounds = Bbox.clamp(bounds, cv.bounds)

  bounds = bounds.expand_to_chunk_size(shape, cv.voxel_offset)

  return GrapheneMeshTaskIterator(bounds, shape)

def create_transfer_tasks(
    src_layer_path, dest_layer_path, 
    chunk_size=None, shape=None, 
    fill_missing=False, translate=None,
    bounds=None, mip=0, preserve_chunk_size=True,
    encoding=None, skip_downsamples=False,
    delete_black_uploads=False, background_color=0,
    agglomerate=False, timestamp=None, compress='gzip',
    factor=None, sparse=False, dest_voxel_offset=None,
    memory_target=3.5e9, max_mips=5
  ):
  """
  Transfer data to a new data layer. You can use this operation
  to make changes to the dataset representation as well. For 
  example, you can change the chunk size, compression, bounds,
  and offset.

  Downsamples will be automatically generated while transferring
  unless skip_downsamples is set. The number of downsamples will
  be determined by the chunk size and the task shape.

  bounds: Bbox specified in terms of the destination image and its
    highest resolution.
  translate: Vec3 pointing from source bounds to dest bounds
    and is in terms of the highest resolution of the source image.
    This allows you to compensate for differing voxel offsets
    or enables you to move part of the image to a new location.
  dest_voxel_offset: When creating a new image, move the 
    global coordinate origin to this point. This is commonly
    used to "zero" a newly aligned image (e.g. (0,0,0)) 

  background_color: Designates which color should be considered background.
  chunk_size: (overrides preserve_chunk_size) force chunk size for new layers to be this.
  compress: None, 'gzip', or 'br' Determines which compression algorithm to use 
    for new uploaded files.
  delete_black_uploads: issue delete commands instead of upload chunks
    that are all background.
  encoding: "raw", "jpeg", "compressed_segmentation", "compresso", "fpzip", or "kempressed"
    depending on which kind of data you're dealing with. raw works for everything (no compression) 
    but you might get better compression with another encoding. You can think of encoding as the
    image type-specific first stage of compression and the "compress" flag as the data
    agnostic second stage compressor. For example, compressed_segmentation and gzip work
    well together, but not jpeg and gzip.
  factor: (overrides axis) can manually specify what each downsampling round is
    supposed to do: e.g. (2,2,1), (2,2,2), etc
  fill_missing: Treat missing image tiles as zeroed for both src and dest.
  max_mips: (pairs with memory_target) maximum number of downsamples to generate even
    if the memory budget is large enough for more.
  memory_target: given a task size in bytes, pick the task shape that will produce the 
    maximum number of downsamples. Only works for (2,2,1) or (2,2,2).
  preserve_chunk_size: if true, maintain chunk size of starting mip, else, find the closest
    evenly divisible chunk size to 64,64,64 for this shape and use that. The latter can be
    useful when mip 0 uses huge chunks and you want to simply visualize the upper mips.
  shape: (overrides memory_target) The 3d size of each task. Choose a shape that meets 
    the following criteria unless you're doing something out of the ordinary.
    (a) 2^n multiple of destination chunk size (b) doesn't consume too much memory
    (c) n is related to the downsample factor for each axis, so for a factor of (2,2,1) (default)
      z only needs to be a single chunk, but x and y should be 2, 4, 8,or 16 times the chunk size.
    Remember to multiply 4/3 * shape.x * shape.y * shape.z * data_type to estimate how much memory 
    each task will require. If downsamples are off, you can skip the 4/3. In the future, if chunk
    sizes match we might be able to do a simple file transfer. The problem can be formulated as 
    producing the largest number of downsamples within a given memory target.

    EXAMPLE: destination is uint64 with chunk size (128, 128, 64) with a memory target of
      at most 3GB per task and a downsample factor of (2,2,1).

      The largest number of downsamples is 4 using 2048 * 2048 * 64 sized tasks which will
      use 2.9 GB of memory. The next size up would use 11.5GB and is too big. 

  sparse: When downsampling segmentation, if true, don't count black pixels when computing
    the mode. Useful for e.g. synapses and point labels.

  agglomerate: (graphene only) remap the watershed layer to a proofread segmentation.
  timestamp: (graphene only) integer UNIX timestamp indicating the proofreading state
    to represent.
  """
  src_vol = CloudVolume(src_layer_path, mip=mip)

  if dest_voxel_offset:
    dest_voxel_offset = Vec(*dest_voxel_offset, dtype=int)
  else:
    dest_voxel_offset = src_vol.voxel_offset.clone()

  if factor is None:
    factor = (2,2,1)

  if skip_downsamples:
    factor = (1,1,1)

  if not chunk_size:
    chunk_size = src_vol.info['scales'][mip]['chunk_sizes'][0]
  chunk_size = Vec(*chunk_size)

  try:
    dest_vol = CloudVolume(dest_layer_path, mip=mip)
  except cloudvolume.exceptions.InfoUnavailableError:
    info = copy.deepcopy(src_vol.info)
    dest_vol = CloudVolume(dest_layer_path, info=info, mip=mip)
    dest_vol.commit_info()

  if dest_voxel_offset is not None:
    dest_vol.scale["voxel_offset"] = dest_voxel_offset

  # If translate is not set, but dest_voxel_offset is then it should naturally be
  # only be the difference between datasets.
  if translate is None:
    translate = dest_vol.voxel_offset - src_vol.voxel_offset # vector pointing from src to dest
  else:
    translate = Vec(*translate) // src_vol.downsample_ratio

  if encoding is not None:
    dest_vol.info['scales'][mip]['encoding'] = encoding
    if encoding == 'compressed_segmentation' and 'compressed_segmentation_block_size' not in dest_vol.info['scales'][mip]:
      dest_vol.info['scales'][mip]['compressed_segmentation_block_size'] = (8,8,8)
  dest_vol.info['scales'] = dest_vol.info['scales'][:mip+1]
  dest_vol.info['scales'][mip]['chunk_sizes'] = [ chunk_size.tolist() ]
  dest_vol.commit_info()

  if shape is None:
    if memory_target is not None:
      shape = downsample_scales.downsample_shape_from_memory_target(
        np.dtype(src_vol.dtype).itemsize, 
        dest_vol.chunk_size.x, dest_vol.chunk_size.y, dest_vol.chunk_size.z, 
        factor, memory_target, max_mips
      )
    else:
      raise ValueError("Either shape or memory_target must be specified.")

  shape = Vec(*shape)

  if factor[2] == 1:
    shape.z = int(dest_vol.chunk_size.z * round(shape.z / dest_vol.chunk_size.z))

  if not skip_downsamples:
    downsample_scales.create_downsample_scales(dest_layer_path, 
      mip=mip, ds_shape=shape, 
      preserve_chunk_size=preserve_chunk_size,
      encoding=encoding
    )

  dest_bounds = get_bounds(dest_vol, bounds, mip, chunk_size)

  class TransferTaskIterator(FinelyDividedTaskIterator):
    def task(self, shape, offset):  
      return partial(TransferTask,
        src_path=src_layer_path,
        dest_path=dest_layer_path,
        shape=shape.clone(),
        offset=offset.clone(),
        fill_missing=fill_missing,
        translate=translate,
        mip=mip,
        skip_downsamples=skip_downsamples,
        delete_black_uploads=bool(delete_black_uploads),
        background_color=background_color,
        agglomerate=agglomerate,
        timestamp=timestamp,
        compress=compress,
        factor=factor,
        sparse=sparse,
      )

    def on_finish(self):
      job_details = {
        'method': {
          'task': 'TransferTask',
          'src': src_layer_path,
          'dest': dest_layer_path,
          'shape': list(map(int, shape)),
          'fill_missing': fill_missing,
          'translate': list(map(int, translate)),
          'skip_downsamples': skip_downsamples,
          'delete_black_uploads': bool(delete_black_uploads),
          'background_color': background_color,
          'bounds': [
            dest_bounds.minpt.tolist(),
            dest_bounds.maxpt.tolist()
          ],
          'mip': mip,
          'agglomerate': bool(agglomerate),
          'timestamp': timestamp,
          'compress': compress,
          'encoding': encoding,
          'memory_target': memory_target,
          'factor': (tuple(factor) if factor else None),
          'sparse': bool(sparse),
        },
        'by': OPERATOR_CONTACT,
        'date': strftime('%Y-%m-%d %H:%M %Z'),
      }

      dest_vol = CloudVolume(dest_layer_path)
      dest_vol.provenance.sources = [ src_layer_path ]
      dest_vol.provenance.processing.append(job_details) 
      dest_vol.commit_provenance()

      if src_vol.meta.path.protocol != 'boss':
        src_vol.provenance.processing.append(job_details)
        src_vol.commit_provenance()

  return TransferTaskIterator(dest_bounds, shape)

def create_contrast_normalization_tasks(
    src_path, dest_path, levels_path=None,
    shape=None, mip=0, clip_fraction=0.01, 
    fill_missing=False, translate=(0,0,0),
    minval=None, maxval=None, bounds=None
  ):

  srcvol = CloudVolume(src_path, mip=mip)
  
  try:
    dvol = CloudVolume(dest_path, mip=mip)
  except Exception: # no info file
    info = copy.deepcopy(srcvol.info)
    dvol = CloudVolume(dest_path, mip=mip, info=info)
    dvol.info['scales'] = dvol.info['scales'][:mip+1]
    dvol.commit_info()

  if shape == None:
    shape = Bbox( (0,0,0), (2048, 2048, 64) )
    shape = shape.shrink_to_chunk_size(dvol.underlying).size3()
    shape = Vec.clamp(shape, (1,1,1), bounds.size3() )
  
  shape = Vec(*shape)

  downsample_scales.create_downsample_scales(dest_path, mip=mip, ds_shape=shape, preserve_chunk_size=True)
  dvol.refresh_info()

  bounds = get_bounds(srcvol, bounds, mip)

  class ContrastNormalizationTaskIterator(FinelyDividedTaskIterator):
    def task(self, shape, offset):
      task_shape = min2(shape.clone(), srcvol.bounds.maxpt - offset)
      return ContrastNormalizationTask( 
        src_path=src_path, 
        dest_path=dest_path,
        levels_path=levels_path,
        shape=task_shape, 
        offset=offset.clone(), 
        clip_fraction=clip_fraction,
        mip=mip,
        fill_missing=fill_missing,
        translate=translate,
        minval=minval,
        maxval=maxval,
      )
    
    def on_finish(self):
      dvol.provenance.processing.append({
        'method': {
          'task': 'ContrastNormalizationTask',
          'src_path': src_path,
          'dest_path': dest_path,
          'shape': Vec(*shape).tolist(),
          'clip_fraction': clip_fraction,
          'mip': mip,
          'translate': Vec(*translate).tolist(),
          'minval': minval,
          'maxval': maxval,
          'bounds': [
            bounds.minpt.tolist(),
            bounds.maxpt.tolist()
          ],
        },
        'by': OPERATOR_CONTACT,
        'date': strftime('%Y-%m-%d %H:%M %Z'),
      }) 
      dvol.commit_provenance()

  return ContrastNormalizationTaskIterator(bounds, shape)

def create_luminance_levels_tasks(
    layer_path, levels_path=None, coverage_factor=0.01, 
    shape=None, offset=(0,0,0), mip=0, bounds=None
  ):
  """
  Compute per slice luminance level histogram and write them as
  $layer_path/levels/$z. Each z file looks like:

  {
    "levels": [ 0, 35122, 12, ... ], # 256 indices, index = luminance i.e. 0 is black, 255 is white 
    "patch_size": [ sx, sy, sz ], # metadata on how large the patches were
    "num_patches": 20, # metadata on
    "coverage_ratio": 0.011, # actual sampled area on this slice normalized by ROI size.
  }

  layer_path: source image to sample from
  levels_path: which path to write ./levels/ to (default: $layer_path)
  coverage_factor: what fraction of the image to sample

  offset & shape: Allows you to specify an ROI if much of
    the edges are black. Defaults to entire image.
  mip: int, which mip to work with, default maximum resolution
  """
  vol = CloudVolume(layer_path, mip=mip)

  if shape == None:
    shape = Vec(*vol.shape)
    shape.z = 1

  offset = Vec(*offset)
  zoffset = offset.clone()

  bounds = get_bounds(vol, bounds, mip)
  protocol = vol.meta.path.protocol

  class LuminanceLevelsTaskIterator(object):
    def __len__(self):
      return bounds.maxpt.z - bounds.minpt.z
    def __iter__(self):
      for z in range(bounds.minpt.z, bounds.maxpt.z + 1):
        zoffset.z = z
        yield LuminanceLevelsTask( 
          src_path=layer_path, 
          levels_path=levels_path,
          shape=shape, 
          offset=zoffset, 
          coverage_factor=coverage_factor,
          mip=mip,
        )

      if protocol == 'boss':
        raise StopIteration()

      if levels_path:
        try:
          vol = CloudVolume(levels_path)
        except cloudvolume.exceptions.InfoUnavailableError:
          vol = CloudVolume(levels_path, info=vol.info)
      else:
        vol = CloudVolume(layer_path, mip=mip)
      
      vol.provenance.processing.append({
        'method': {
          'task': 'LuminanceLevelsTask',
          'src': layer_path,
          'levels_path': levels_path,
          'shape': Vec(*shape).tolist(),
          'offset': Vec(*offset).tolist(),
          'bounds': [
            bounds.minpt.tolist(),
            bounds.maxpt.tolist()
          ],
          'coverage_factor': coverage_factor,
          'mip': mip,
        },
        'by': OPERATOR_CONTACT,
        'date': strftime('%Y-%m-%d %H:%M %Z'),
      }) 
      vol.commit_provenance()

  return LuminanceLevelsTaskIterator()

def create_watershed_remap_tasks(
    map_path, src_layer_path, dest_layer_path, 
    shape=Vec(2048, 2048, 64)
  ):
  shape = Vec(*shape)
  vol = CloudVolume(src_layer_path)

  downsample_scales.create_downsample_scales(dest_layer_path, mip=0, ds_shape=shape)

  class WatershedRemapTaskIterator(FinelyDividedTaskIterator):
    def task(self, shape, offset):
      return WatershedRemapTask(
        map_path=map_path,
        src_path=src_layer_path,
        dest_path=dest_layer_path,
        shape=shape.clone(),
        offset=offset.clone(),
      )
    
    def on_finish(self):
      dvol = CloudVolume(dest_layer_path)
      dvol.provenance.processing.append({
        'method': {
          'task': 'WatershedRemapTask',
          'src': src_layer_path,
          'dest': dest_layer_path,
          'remap_file': map_path,
          'shape': list(shape),
        },
        'by': OPERATOR_CONTACT,
        'date': strftime('%Y-%m-%d %H:%M %Z'),
      }) 
      dvol.commit_provenance()

  return WatershedRemapTaskIterator(vol.bounds, shape)

def compute_fixup_offsets(vol, points, shape):
  pts = map(np.array, points)

  # points are specified in high res coordinates 
  # because that's what people read off the screen.
  def nearest_offset(pt):
    mip0offset = (np.floor((pt - vol.mip_voxel_offset(0)) / shape) * shape) + vol.mip_voxel_offset(0)
    return mip0offset / vol.downsample_ratio

  return map(nearest_offset, pts)

def create_fixup_downsample_tasks(
    layer_path, points, 
    shape=Vec(2048, 2048, 64), mip=0, axis='z'
  ):
  """you can use this to fix black spots from when downsample tasks fail
  by specifying a point inside each black spot.
  """
  vol = CloudVolume(layer_path, mip)
  offsets = compute_fixup_offsets(vol, points, shape)

  class FixupDownsampleTasksIterator():
    def __len__(self):
      return len(offsets)
    def __iter__(self):
      for offset in offsets:
        yield DownsampleTask(
          layer_path=layer_path,
          mip=mip,
          shape=shape,
          offset=offset,
          axis=axis,
        )

  return FixupDownsampleTasksIterator()

def create_quantized_affinity_info(
    src_layer, dest_layer, shape, mip, 
    chunk_size, encoding
  ):
  srcvol = CloudVolume(src_layer)
  
  info = copy.deepcopy(srcvol.info)
  info['num_channels'] = 1
  info['data_type'] = 'uint8'
  info['type'] = 'image'
  info['scales'] = info['scales'][:mip+1]
  for i in range(mip+1):
    info['scales'][i]['encoding'] = encoding
    info['scales'][i]['chunk_sizes'] = [ chunk_size ]
  return info

def create_quantize_tasks(
    src_layer, dest_layer, shape, 
    mip=0, fill_missing=False, 
    chunk_size=(128, 128, 64), 
    encoding='raw', bounds=None
  ):

  shape = Vec(*shape)

  info = create_quantized_affinity_info(
    src_layer, dest_layer, shape, 
    mip, chunk_size, encoding
  )
  destvol = CloudVolume(dest_layer, info=info, mip=mip)
  destvol.commit_info()

  downsample_scales.create_downsample_scales(
    dest_layer, mip=mip, ds_shape=shape, 
    chunk_size=chunk_size, encoding=encoding
  )

  if bounds is None:
    bounds = destvol.mip_bounds(mip)
  else:
    bounds = destvol.bbox_to_mip(bounds, mip=0, to_mip=mip)
    bounds = bounds.expand_to_chunk_size(
      destvol.mip_chunk_size(mip), destvol.mip_voxel_offset(mip)
    )

  class QuantizeTasksIterator(FinelyDividedTaskIterator):
    def task(self, shape, offset):
      return QuantizeTask(
        source_layer_path=src_layer,
        dest_layer_path=dest_layer,
        shape=shape.tolist(),
        offset=offset.tolist(),
        fill_missing=fill_missing,
        mip=mip,
      )

    def on_finish(self):
      destvol.provenance.sources = [ src_layer ]
      destvol.provenance.processing.append({
        'method': {
          'task': 'QuantizeTask',
          'source_layer_path': src_layer,
          'dest_layer_path': dest_layer,
          'shape': shape.tolist(),
          'fill_missing': fill_missing,
          'mip': mip,
        },
        'by': OPERATOR_CONTACT,
        'date': strftime('%Y-%m-%d %H:%M %Z'),
      }) 
      destvol.commit_provenance()

  return QuantizeTasksIterator(bounds, shape)

# split the work up into ~1000 tasks (magnitude 3)
def create_mesh_manifest_tasks(layer_path, magnitude=3, mesh_dir=None):
  assert int(magnitude) == magnitude

  start = 10 ** (magnitude - 1)
  end = 10 ** magnitude

  class MeshManifestTaskIterator(object):
    def __len__(self):
      return 10 ** magnitude
    def __iter__(self):
      for prefix in range(1, start):
        yield MeshManifestTask(layer_path=layer_path, prefix=str(prefix) + ':', mesh_dir=mesh_dir)

      # enumerate from e.g. 100 to 999
      for prefix in range(start, end):
        yield MeshManifestTask(layer_path=layer_path, prefix=prefix, mesh_dir=mesh_dir)

  return MeshManifestTaskIterator()


def graphene_prefixes(
    mip=1, mip_bits=8, 
    coord_bits=(10, 10, 10), 
    prefix_length=6
  ):
  """
  Graphene structures segids as decimal numbers following
  the below format:

  mip x y z segid

  Typical parameter values are 
  mip_bits=4 or 8, x_bits=8 or 10, y_bits=8 or 10
  """
  coord_bits = Vec(*coord_bits)

  mip_shift = 64 - mip_bits
  x_shift = mip_shift - coord_bits.x
  y_shift = x_shift - coord_bits.y
  z_shift = y_shift - coord_bits.z

  x_range = 2 ** coord_bits.x 
  y_range = 2 ** coord_bits.y
  z_range = 2 ** coord_bits.z

  prefixes = set()
  for x in range(x_range):
    for y in range(y_range):
      num = (mip << mip_shift) + (x << x_shift) + (y << y_shift)
      num = str(num)[:prefix_length]
      prefixes.add(num)

  return prefixes

def create_graphene_hybrid_mesh_manifest_tasks(
  cloudpath, mip, mip_bits, x_bits, y_bits, z_bits
):
  prefixes = graphene_prefixes(mip, mip_bits, (x_bits, y_bits, z_bits))

  class GrapheneHybridMeshManifestTaskIterator(object):
    def __len__(self):
      return len(prefixes)
    def __iter__(self):
      for prefix in prefixes:
        yield MeshManifestTask(layer_path=cloudpath, prefix=str(prefix))

  return GrapheneHybridMeshManifestTaskIterator()

def create_hypersquare_ingest_tasks(
    hypersquare_bucket_name, dataset_name, 
    hypersquare_chunk_size, resolution, 
    voxel_offset, volume_size, overlap
  ):
  
  def crtinfo(layer_type, dtype, encoding):
    return CloudVolume.create_new_info(
      num_channels=1,
      layer_type=layer_type,
      data_type=dtype,
      encoding=encoding,
      resolution=resolution,
      voxel_offset=voxel_offset,
      volume_size=volume_size,
      chunk_size=[ 56, 56, 56 ],
    )

  imginfo = crtinfo('image', 'uint8', 'jpeg')
  seginfo = crtinfo('segmentation', 'uint16', 'raw')

  scales = downsample_scales.compute_plane_downsampling_scales(hypersquare_chunk_size)[1:] # omit (1,1,1)

  IMG_LAYER_NAME = 'image'
  SEG_LAYER_NAME = 'segmentation'

  imgvol = CloudVolume(dataset_name, IMG_LAYER_NAME, 0, info=imginfo)
  segvol = CloudVolume(dataset_name, SEG_LAYER_NAME, 0, info=seginfo)

  print("Creating info files for image and segmentation...")
  imgvol.commit_info()
  segvol.commit_info()

  def crttask(volname, tasktype, layer_name):
    return HyperSquareTask(
      bucket_name=hypersquare_bucket_name,
      dataset_name=dataset_name,
      layer_name=layer_name,
      volume_dir=volname,
      layer_type=tasktype,
      overlap=overlap,
      resolution=resolution,
    )

  print("Listing hypersquare bucket...")
  volumes_listing = lib.gcloud_ls('gs://{}/'.format(hypersquare_bucket_name))

  # download this from: 
  # with open('e2198_volumes.json', 'r') as f:
  #   volumes_listing = json.loads(f.read())

  volumes_listing = [ x.split('/')[-2] for x in volumes_listing ]

  class CreateHypersquareIngestTaskIterator(object):
    def __len__(self):
      return len(volumes_listing)
    def __iter__(self):
      for cloudpath in volumes_listing:
        # img_task = crttask(cloudpath, 'image', IMG_LAYER_NAME)
        yield crttask(cloudpath, 'segmentation', SEG_LAYER_NAME)
        # seg_task.execute()

  return CreateHypersquareIngestTaskIterator()

def create_hypersquare_consensus_tasks(
    src_path, dest_path, 
    volume_map_file, consensus_map_path
  ):
  """
  Transfer an Eyewire consensus into neuroglancer. This first requires
  importing the raw segmentation via a hypersquare ingest task. However,
  this can probably be streamlined at some point.

  The volume map file should be JSON encoded and 
  look like { "X-X_Y-Y_Z-Z": EW_VOLUME_ID }

  The consensus map file should look like:
  { VOLUMEID: { CELLID: [segids] } }
  """

  with open(volume_map_file, 'r') as f:
    volume_map = json.loads(f.read())

  vol = CloudVolume(dest_path)

  class HyperSquareConsensusTaskIterator(object):
    def __len__(self):
      return len(volume_map)
    def __iter__(self):
      for boundstr, volume_id in volume_map.items():
        bbox = Bbox.from_filename(boundstr)
        bbox.minpt = Vec.clamp(bbox.minpt, vol.bounds.minpt, vol.bounds.maxpt)
        bbox.maxpt = Vec.clamp(bbox.maxpt, vol.bounds.minpt, vol.bounds.maxpt)

        yield HyperSquareConsensusTask(
          src_path=src_path,
          dest_path=dest_path,
          ew_volume_id=int(volume_id),
          consensus_map_path=consensus_map_path,
          shape=bbox.size3(),
          offset=bbox.minpt.clone(),
        )

  return HyperSquareConsensusTaskIterator()

def create_mask_affinity_map_tasks(
    aff_input_layer_path, aff_output_layer_path, 
    aff_mip, mask_layer_path, mask_mip, output_block_start, 
    output_block_size, grid_size 
  ):
    """
    affinity map masking block by block. The block coordinates should be aligned with 
    cloud storage. 
    """

    class MaskAffinityMapTaskIterator():
      def __len__(self):
        return int(reduce(operator.mul, grid_size))
      def __iter__(self):
        for x, y, z in xyzrange(grid_size):
          output_bounds = Bbox.from_slices(tuple(slice(s+x*b, s+x*b+b)
                  for (s, x, b) in zip(output_block_start, (z, y, x), output_block_size)))
          yield MaskAffinitymapTask(
              aff_input_layer_path=aff_input_layer_path,
              aff_output_layer_path=aff_output_layer_path,
              aff_mip=aff_mip, 
              mask_layer_path=mask_layer_path,
              mask_mip=mask_mip,
              output_bounds=output_bounds,
          )

        vol = CloudVolume(output_layer_path, mip=aff_mip)
        vol.provenance.processing.append({
            'method': {
                'task': 'InferenceTask',
                'aff_input_layer_path': aff_input_layer_path,
                'aff_output_layer_path': aff_output_layer_path,
                'aff_mip': aff_mip,
                'mask_layer_path': mask_layer_path,
                'mask_mip': mask_mip,
                'output_block_start': output_block_start,
                'output_block_size': output_block_size, 
                'grid_size': grid_size,
            },
            'by': OPERATOR_CONTACT,
            'date': strftime('%Y-%m-%d %H:%M %Z'),
        })
        vol.commit_provenance()

    return MaskAffinityMapTaskIterator()

def create_inference_tasks(
    image_layer_path, convnet_path, 
    mask_layer_path, output_layer_path, output_block_start, output_block_size, 
    grid_size, patch_size, patch_overlap, cropping_margin_size,
    output_key='output', num_output_channels=3, 
    image_mip=1, output_mip=1, mask_mip=3
  ):
    """
    convnet inference block by block. The block coordinates should be aligned with 
    cloud storage. 
    """
    class InferenceTaskIterator():
      def __len__(self):
        return int(reduce(operator.mul, grid_size))
      def __iter__(self):
        for x, y, z in xyzrange(grid_size):
          output_offset = tuple(s+x*b for (s, x, b) in 
                                zip(output_block_start, (z, y, x), 
                                    output_block_size))
          yield InferenceTask(
              image_layer_path=image_layer_path,
              convnet_path=convnet_path,
              mask_layer_path=mask_layer_path,
              output_layer_path=output_layer_path,
              output_offset=output_offset,
              output_shape=output_block_size,
              patch_size=patch_size, 
              patch_overlap=patch_overlap,
              cropping_margin_size=cropping_margin_size,
              output_key=output_key,
              num_output_channels=num_output_channels,
              image_mip=image_mip,
              output_mip=output_mip,
              mask_mip=mask_mip
          )


        vol = CloudVolume(output_layer_path, mip=output_mip)
        vol.provenance.processing.append({
            'method': {
                'task': 'InferenceTask',
                'image_layer_path': image_layer_path,
                'convnet_path': convnet_path,
                'mask_layer_path': mask_layer_path,
                'output_layer_path': output_layer_path,
                'output_offset': output_offset,
                'output_shape': output_block_size,
                'patch_size': patch_size,
                'patch_overlap': patch_overlap,
                'cropping_margin_size': cropping_margin_size,
                'output_key': output_key,
                'num_output_channels': num_output_channels,
                'image_mip': image_mip,
                'output_mip': output_mip,
                'mask_mip': mask_mip,
            },
            'by': OPERATOR_CONTACT,
            'date': strftime('%Y-%m-%d %H:%M %Z'),
        })
        vol.commit_provenance()

    return InferenceTaskIterator()

def cascade(tq, fnlist):
  for fn in fnlist:
    fn(tq)
    N = tq.enqueued
    while N > 0:
      N = tq.enqueued
      print('\r {} remaining'.format(N), end='')
      time.sleep(2)
