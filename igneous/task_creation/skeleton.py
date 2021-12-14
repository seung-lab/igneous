from collections import defaultdict
from functools import reduce, partial
from typing import Any, Dict, Tuple, cast

from time import strftime

import numpy as np
from tqdm import tqdm

from cloudvolume import CloudVolume
from cloudvolume.lib import Vec, Bbox, max2, min2, xyzrange, find_closest_divisor, yellow, jsonify
from cloudvolume.datasource.precomputed.sharding import ShardingSpecification
from cloudfiles import CloudFiles

from igneous.tasks import ( 
  SkeletonTask, UnshardedSkeletonMergeTask, 
  ShardedSkeletonMergeTask,
)

from .common import (
  operator_contact, FinelyDividedTaskIterator, 
  get_bounds, num_tasks, compute_shard_params_for_hashed
)

__all__ = [
  "create_skeletonizing_tasks",
  "create_unsharded_skeleton_merge_tasks",
  "create_sharded_skeleton_merge_tasks",
  "create_flat_graphene_skeleton_merge_tasks",
]

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
        'by': operator_contact(),
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
    shard_index_bytes=2**13,
    minishard_index_bytes=2**15,
    minishard_index_encoding='gzip', data_encoding='gzip',
    max_cable_length=None, spatial_index_db=None
  ): 
  cv = CloudVolume(layer_path, progress=True, spatial_index_db=spatial_index_db) 
  cv.mip = cv.skeleton.meta.mip

  # 17 sec to download for pinky100
  all_labels = cv.skeleton.spatial_index.query(cv.bounds * cv.resolution)
  
  (shard_bits, minishard_bits, preshift_bits) = \
    compute_shard_params_for_hashed(
      num_labels=len(all_labels),
      shard_index_bytes=int(shard_index_bytes),
      minishard_index_bytes=int(minishard_index_bytes),
    )

  spec = ShardingSpecification(
    type='neuroglancer_uint64_sharded_v1',
    preshift_bits=preshift_bits,
    hash='murmurhash3_x86_128',
    minishard_bits=minishard_bits,
    shard_bits=shard_bits,
    minishard_index_encoding=minishard_index_encoding,
    data_encoding=data_encoding,
  )
  cv.skeleton.meta.info['sharding'] = spec.to_dict()
  cv.skeleton.meta.commit_info()

  # rebuild b/c sharding changes the skeleton source
  cv = CloudVolume(layer_path, progress=True, spatial_index_db=spatial_index_db) 
  cv.mip = cv.skeleton.meta.mip

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
    'by': operator_contact(),
    'date': strftime('%Y-%m-%d %H:%M %Z'),
  }) 
  cv.commit_provenance()

  return [
    ShardedSkeletonMergeTask(
      layer_path, shard_no, 
      dust_threshold, tick_threshold,
      max_cable_length=max_cable_length
    )
    for shard_no in shard_labels.keys()
  ]

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
        'by': operator_contact(),
        'date': strftime('%Y-%m-%d %H:%M %Z'),
      }) 
      vol.commit_provenance()

  return UnshardedSkeletonMergeTaskIterator()
