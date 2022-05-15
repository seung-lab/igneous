import copy
from collections import defaultdict
from functools import reduce, partial
import itertools
import re
from typing import (
  Any, Dict, Optional, 
  Union, Tuple, cast,
  Iterator
)

from time import strftime

import numpy as np
from tqdm import tqdm

import cloudvolume
import cloudvolume.exceptions
from cloudvolume import CloudVolume
from cloudvolume.lib import Vec, Bbox, max2, min2, xyzrange, find_closest_divisor, yellow, jsonify
from cloudvolume.datasource.precomputed.sharding import ShardingSpecification
from cloudfiles import CloudFiles
import cloudfiles.paths

from igneous.tasks import (
  MeshTask, MeshManifestPrefixTask, 
  MeshManifestFilesystemTask, GrapheneMeshTask,
  SpatialIndexTask, MultiResShardedMeshMergeTask,
  MultiResUnshardedMeshMergeTask, 
  MultiResShardedFromUnshardedMeshMergeTask,
  TransferMeshFilesTask, DeleteMeshFilesTask
)
from .common import (
  operator_contact, FinelyDividedTaskIterator, 
  get_bounds, num_tasks, graphene_prefixes,
  compute_shard_params_for_hashed
)

__all__ = [
  "create_meshing_tasks",
  "create_mesh_manifest_tasks",
  "create_graphene_meshing_tasks",
  "create_graphene_hybrid_mesh_manifest_tasks",
  "create_spatial_index_mesh_tasks",
  "create_unsharded_multires_mesh_tasks",
  "create_sharded_multires_mesh_tasks",
  "create_sharded_multires_mesh_from_unsharded_tasks",
  "create_xfer_meshes_tasks",
  "create_mesh_deletion_tasks",
]

# split the work up into ~1000 tasks (magnitude 3)
def create_mesh_manifest_tasks(layer_path, magnitude=3, mesh_dir=None):
  assert int(magnitude) == magnitude
  assert magnitude >= 0

  protocol = cloudfiles.paths.get_protocol(layer_path)
  if protocol == "file":
    return [
      partial(MeshManifestFilesystemTask, 
        layer_path=layer_path,
        mesh_dir=mesh_dir,
      )
    ]

  start = 10 ** (magnitude - 1)
  end = 10 ** magnitude

  class MeshManifestTaskIterator(object):
    def __len__(self):
      return (10 ** magnitude) - 1
    def __iter__(self):
      for prefix in range(1, start):
        yield partial(MeshManifestPrefixTask, 
          layer_path=layer_path, 
          prefix=str(prefix) + ':', 
          mesh_dir=mesh_dir
        )

      # enumerate from e.g. 100 to 999
      for prefix in range(start, end):
        yield partial(MeshManifestPrefixTask, 
          layer_path=layer_path, 
          prefix=str(prefix),
          mesh_dir=mesh_dir
        )

  return MeshManifestTaskIterator()

def create_mesh_deletion_tasks(
  layer_path:str, 
  magnitude:int = 3, 
  mesh_dir:Optional[str] = None
):
  assert int(magnitude) == magnitude
  assert magnitude >= 0

  cv = CloudVolume(layer_path, mesh_dir=mesh_dir)

  cf = CloudFiles(cv.mesh.meta.layerpath)
  cf.delete('info')

  try:
    if cv.mesh.meta.is_sharded():
      return [ 
        partial(DeleteMeshFilesTask, 
          cloudpath=layer_path,
          prefix="",
          mesh_dir=mesh_dir,
        ) 
      ]

    start = 10 ** (magnitude - 1)
    end = 10 ** magnitude

    class MeshDeleteTaskIterator:
      def __len__(self):
        return (10 ** magnitude)
      def __iter__(self):
        # get spatial index files that start
        # with 0 too.
        yield partial(DeleteMeshFilesTask, 
          cloudpath=layer_path,
          prefix="0",
          mesh_dir=mesh_dir,
        )

        for prefix in range(1, start):
          yield partial(DeleteMeshFilesTask, 
            cloudpath=layer_path, 
            prefix=str(prefix) + ':', 
            mesh_dir=mesh_dir
          )

        # enumerate from e.g. 100 to 999
        for prefix in range(start, end):
          yield partial(DeleteMeshFilesTask, 
            cloudpath=layer_path, 
            prefix=str(prefix),
            mesh_dir=mesh_dir
          )

    return MeshDeleteTaskIterator()
  finally:
    cv.provenance.processing.append({
      'method': {
        'task': 'DeleteMeshFilesTask',
        'layer_path': layer_path,
        'mesh_dir': mesh_dir,
      },
      'by': operator_contact(),
      'date': strftime('%Y-%m-%d %H:%M %Z'),
    }) 
    cv.commit_provenance()


def create_meshing_tasks(
    layer_path, mip, shape=(448, 448, 448), 
    simplification=True, max_simplification_error=40,
    mesh_dir=None, cdn_cache=False, dust_threshold=None,
    object_ids=None, progress=False, fill_missing=False,
    encoding='precomputed', spatial_index=True, sharded=False,
    compress='gzip', closed_dataset_edges=True
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
        closed_dataset_edges=closed_dataset_edges,
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
          'closed_dataset_edges': closed_dataset_edges,
        },
        'by': operator_contact(),
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
        'by': operator_contact(),
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

def create_graphene_hybrid_mesh_manifest_tasks(
  cloudpath, mip, mip_bits, x_bits, y_bits, z_bits
):
  prefixes = graphene_prefixes(mip, mip_bits, (x_bits, y_bits, z_bits))

  class GrapheneHybridMeshManifestTaskIterator(object):
    def __len__(self):
      return len(prefixes)
    def __iter__(self):
      for prefix in prefixes:
        yield partial(MeshManifestPrefixTask, layer_path=cloudpath, prefix=str(prefix))

  return GrapheneHybridMeshManifestTaskIterator()

def create_spatial_index_mesh_tasks(
  cloudpath:str, 
  shape:Tuple[int,int,int] = (448,448,448), 
  mip:int = 0, 
  fill_missing:bool = False, 
  compress:Optional[Union[str,bool]] = 'gzip', 
  mesh_dir:Optional[str] = None
):
  """
  The main way to add a spatial index is to use the MeshTask,
  but old datasets or broken datasets may need it to be 
  reconstituted. An alternative use is create the spatial index
  over a different area size than the mesh task.
  """
  shape = Vec(*shape)

  vol = CloudVolume(cloudpath, mip=mip)

  if mesh_dir is None and not vol.info.get("mesh", None):
    mesh_dir = f"mesh_mip_{mip}_err_40"
  elif mesh_dir is None and vol.info.get("mesh", None):
    mesh_dir = vol.info["mesh"]

  if not "mesh" in vol.info:
    vol.info['mesh'] = mesh_dir
    vol.commit_info()

  cf = CloudFiles(cloudpath)
  info_filename = '{}/info'.format(mesh_dir)
  mesh_info = cf.get_json(info_filename) or {}
  new_mesh_info = copy.deepcopy(mesh_info)
  new_mesh_info['@type'] = new_mesh_info.get('@type', 'neuroglancer_legacy_mesh') 
  new_mesh_info['mip'] = new_mesh_info.get("mip", int(vol.mip))
  new_mesh_info['chunk_size'] = shape.tolist()
  new_mesh_info['spatial_index'] = {
    'resolution': vol.resolution.tolist(),
    'chunk_size': (shape * vol.resolution).tolist(),
  }
  if new_mesh_info != mesh_info:
    cf.put_json(info_filename, new_mesh_info)

  vol = CloudVolume(cloudpath, mip=mip) # reload spatial index

  class SpatialIndexMeshTaskIterator(FinelyDividedTaskIterator):
    def task(self, shape, offset):
      return partial(SpatialIndexTask, 
        cloudpath=cloudpath,
        shape=shape,
        offset=offset,
        subdir=mesh_dir,
        precision=vol.mesh.spatial_index.precision,
        mip=int(mip),
        fill_missing=bool(fill_missing),
        compress=compress,
      )

    def on_finish(self):
      vol.provenance.processing.append({
        'method': {
          'task': 'SpatialIndexTask',
          'cloudpath': vol.cloudpath,
          'shape': shape.tolist(),
          'mip': int(mip),
          'subdir': mesh_dir,
          'fill_missing': fill_missing,
          'compress': compress,
        },
        'by': operator_contact(),
        'date': strftime('%Y-%m-%d %H:%M %Z'),
      }) 
      vol.commit_provenance()

  return SpatialIndexMeshTaskIterator(vol.bounds, shape)

def configure_multires_info(
  cloudpath:str,
  vertex_quantization_bits:int, 
  mesh_dir:str
):
  """
  Computes properties and uploads a multires 
  mesh info file
  """
  assert vertex_quantization_bits in (10, 16), vertex_quantization_bits

  vol = CloudVolume(cloudpath)

  mesh_dir = mesh_dir or vol.info.get("mesh", None)

  if not "mesh" in vol.info:
    vol.info['mesh'] = mesh_dir
    vol.commit_info()

  res = vol.meta.resolution(vol.mesh.meta.mip)

  cf = CloudFiles(cloudpath)
  info_filename = f'{mesh_dir}/info'
  mesh_info = cf.get_json(info_filename) or {}
  new_mesh_info = copy.deepcopy(mesh_info)
  new_mesh_info['@type'] = "neuroglancer_multilod_draco"
  new_mesh_info['vertex_quantization_bits'] = vertex_quantization_bits
  new_mesh_info['transform'] = [ 
    res[0], 0,      0,      0,
    0,      res[1], 0,      0,
    0,      0,      res[2], 0,
  ]
  new_mesh_info['lod_scale_multiplier'] = 1.0

  if new_mesh_info != mesh_info:
    cf.put_json(
      info_filename, new_mesh_info, 
      cache_control="no-cache"
    )

def create_unsharded_multires_mesh_tasks(
  cloudpath:str, num_lod:int = 1, 
  magnitude:int = 3, mesh_dir:str = None,
  vertex_quantization_bits:int = 16,
  min_chunk_size:Tuple[int,int,int] = (512,512,512),
) -> Iterator:
  """
  vertex_quantization_bits: 10 or 16. Adjusts the precision
    of mesh vertices.
  """
  # split the work up into ~1000 tasks (magnitude 3)
  assert int(magnitude) == magnitude

  configure_multires_info(
    cloudpath, 
    vertex_quantization_bits, 
    mesh_dir
  )

  start = 10 ** (magnitude - 1)
  end = 10 ** magnitude

  class UnshardedMultiResTaskIterator:
    def __len__(self):
      return (10 ** magnitude) - 1
    def __iter__(self):
      for prefix in range(1, start):
        yield partial(MultiResUnshardedMeshMergeTask, 
          cloudpath=cloudpath, 
          prefix=str(prefix) + ':', 
          mesh_dir=mesh_dir,
          num_lod=num_lod,
          min_chunk_size=min_chunk_size,
        )

      # enumerate from e.g. 100 to 999
      for prefix in range(start, end):
        yield partial(MultiResUnshardedMeshMergeTask, 
          cloudpath=cloudpath, 
          prefix=prefix, 
          mesh_dir=mesh_dir,
          num_lod=num_lod,
          min_chunk_size=min_chunk_size,
        )

      self.update_provenance()

    def update_provenance(self):
      cv = CloudVolume(cloudpath)
      cv.provenance.processing.append({
        'method': {
          'task': 'MultiResUnshardedMeshMergeTask',
          'cloudpath': cloudpath,
          'magnitude': int(magnitude),
          'num_lod': int(num_lod),
          'vertex_quantization_bits': int(vertex_quantization_bits),
          'min_chunk_size': tuple(min_chunk_size),
          'mesh_dir': mesh_dir,
        },
        'by': operator_contact(),
        'date': strftime('%Y-%m-%d %H:%M %Z'),
      }) 
      cv.commit_provenance()

  return UnshardedMultiResTaskIterator()

def create_xfer_meshes_tasks(
  src:str,
  dest:str,
  mesh_dir:Optional[str] = None, 
  magnitude=2,
):
  cv_src = CloudVolume(src)
  cf_dest = CloudFiles(dest)

  if not mesh_dir:
    info = cf_dest.get_json("info")
    if info.get("mesh", None):
      mesh_dir = info.get("mesh")

  cf_dest.put_json(f"{mesh_dir}/info", cv_src.mesh.meta.info)

  alphabet = [ str(i) for i in range(10) ]
  if cv_src.mesh.meta.is_sharded():
    alphabet += [ 'a', 'b', 'c', 'd', 'e', 'f' ]

  prefixes = itertools.product(*([ alphabet ] * magnitude))
  prefixes = [ "".join(x) for x in prefixes ]

  # explicitly enumerate all prefixes smaller than the magnitude.
  for i in range(1, magnitude):
    explicit_prefix = itertools.product(*([ alphabet ] * i))
    explicit_prefix = [ "".join(x) for x in explicit_prefix ]
    if cv_src.mesh.meta.is_sharded():
      prefixes += [ f"{x}." for x in explicit_prefix ]
    else:
      prefixes += [ f"{x}:0" for x in explicit_prefix ]

  return [
    partial(TransferMeshFilesTask,
      src=src,
      dest=dest,
      prefix=prefix,
      mesh_dir=mesh_dir,
    )
    for prefix in prefixes
  ]

def create_sharded_multires_mesh_from_unsharded_tasks(
  src:str, 
  dest:str,
  shard_index_bytes=2**13, 
  minishard_index_bytes=2**15,
  min_shards:int = 1,
  num_lod:int = 1, 
  draco_compression_level:int = 1,
  vertex_quantization_bits:int = 16,
  minishard_index_encoding="gzip", 
  mesh_dir:Optional[str] = None, 
) -> Iterator[MultiResShardedMeshMergeTask]: 
  
  configure_multires_info(
    dest, 
    vertex_quantization_bits, 
    mesh_dir
  )

  cv_src = CloudVolume(src)
  cf = CloudFiles(cv_src.mesh.meta.layerpath)

  all_labels = []
  SEGID_RE = re.compile(r'(\d+)(?:(?::0(?:\.gz|\.br|\.zstd)?$)|\.index$)')
  for path in cf.list():
    match = SEGID_RE.search(path)
    if match is None:
      continue
    (segid,) = match.groups()
    all_labels.append(int(segid))

  (shard_bits, minishard_bits, preshift_bits) = \
    compute_shard_params_for_hashed(
      num_labels=len(all_labels),
      shard_index_bytes=int(shard_index_bytes),
      minishard_index_bytes=int(minishard_index_bytes),
      min_shards=int(min_shards),
    )

  cv_dest = CloudVolume(dest, mesh_dir=mesh_dir)
  cv_dest.mesh.meta.info["mip"] = cv_src.mesh.meta.mip
  cv_dest.commit_info()

  spec = ShardingSpecification(
    type='neuroglancer_uint64_sharded_v1',
    preshift_bits=preshift_bits,
    hash='murmurhash3_x86_128',
    minishard_bits=minishard_bits,
    shard_bits=shard_bits,
    minishard_index_encoding=minishard_index_encoding,
    data_encoding="raw", # draco encoded meshes
  )

  cv_dest.mesh.meta.info['sharding'] = spec.to_dict()
  cv_dest.mesh.meta.commit_info()

  cv_dest = CloudVolume(dest, mesh_dir=mesh_dir)

  # perf: ~66.5k hashes/sec on M1 ARM64
  shardfn = lambda lbl: cv_dest.mesh.reader.spec.compute_shard_location(lbl).shard_number

  shard_labels = defaultdict(list)
  for label in tqdm(all_labels, desc="Hashes"):
    shard_labels[shardfn(label)].append(label)
  del all_labels

  cf = CloudFiles(cv_dest.mesh.meta.layerpath, progress=True)
  files = ( 
    (str(shardno) + '.labels', labels) 
    for shardno, labels in shard_labels.items() 
  )
  cf.put_jsons(
    files, compress="gzip", 
    cache_control="no-cache", total=len(shard_labels)
  )

  cv_dest.provenance.processing.append({
    'method': {
      'task': 'MultiResShardedFromUnshardedMeshMergeTask',
      'src': src,
      'dest': dest,
      'num_lod': num_lod,
      'vertex_quantization_bits': vertex_quantization_bits,
      'preshift_bits': preshift_bits, 
      'minishard_bits': minishard_bits, 
      'shard_bits': shard_bits,
      'mesh_dir': mesh_dir,
      'draco_compression_level': draco_compression_level,
    },
    'by': operator_contact(),
    'date': strftime('%Y-%m-%d %H:%M %Z'),
  }) 
  cv_dest.commit_provenance()

  return [
    partial(MultiResShardedFromUnshardedMeshMergeTask,
      src=src, 
      dest=dest, 
      shard_no=shard_no, 
      num_lod=num_lod,
      mesh_dir=mesh_dir, 
      draco_compression_level=draco_compression_level,
    )
    for shard_no in shard_labels.keys()
  ]

def create_sharded_multires_mesh_tasks(
  cloudpath:str, 
  shard_index_bytes=2**13, 
  minishard_index_bytes=2**15,
  min_shards:int = 1,
  num_lod:int = 1, 
  draco_compression_level:int = 7,
  vertex_quantization_bits:int = 16,
  minishard_index_encoding="gzip", 
  mesh_dir:Optional[str] = None, 
  spatial_index_db:Optional[str] = None,
  min_chunk_size:Tuple[int,int,int] = (512,512,512)
) -> Iterator[MultiResShardedMeshMergeTask]: 

  configure_multires_info(
    cloudpath, 
    vertex_quantization_bits, 
    mesh_dir
  )

  # rebuild b/c sharding changes the mesh source class
  cv = CloudVolume(cloudpath, progress=True, spatial_index_db=spatial_index_db) 
  cv.mip = cv.mesh.meta.mip

  # 17 sec to download for pinky100
  all_labels = cv.mesh.spatial_index.query(cv.bounds * cv.resolution)
  
  (shard_bits, minishard_bits, preshift_bits) = \
    compute_shard_params_for_hashed(
      num_labels=len(all_labels),
      shard_index_bytes=int(shard_index_bytes),
      minishard_index_bytes=int(minishard_index_bytes),
      min_shards=min_shards,
    )

  spec = ShardingSpecification(
    type='neuroglancer_uint64_sharded_v1',
    preshift_bits=preshift_bits,
    hash='murmurhash3_x86_128',
    minishard_bits=minishard_bits,
    shard_bits=shard_bits,
    minishard_index_encoding=minishard_index_encoding,
    data_encoding="raw", # draco encoded meshes
  )

  cv.mesh.meta.info['sharding'] = spec.to_dict()
  cv.mesh.meta.commit_info()

  cv = CloudVolume(cloudpath)

  # perf: ~66.5k hashes/sec on M1 ARM64
  shardfn = lambda lbl: cv.mesh.reader.spec.compute_shard_location(lbl).shard_number

  shard_labels = defaultdict(list)
  for label in tqdm(all_labels, desc="Hashes"):
    shard_labels[shardfn(label)].append(label)
  del all_labels

  cf = CloudFiles(cv.mesh.meta.layerpath, progress=True)
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
      'task': 'MultiResShardedMeshMergeTask',
      'cloudpath': cloudpath,
      'mip': cv.mesh.meta.mip,
      'num_lod': num_lod,
      'vertex_quantization_bits': vertex_quantization_bits,
      'preshift_bits': preshift_bits, 
      'minishard_bits': minishard_bits, 
      'shard_bits': shard_bits,
      'mesh_dir': mesh_dir,
      'draco_compression_level': draco_compression_level,
      'min_chunk_size': min_chunk_size,
    },
    'by': operator_contact(),
    'date': strftime('%Y-%m-%d %H:%M %Z'),
  }) 
  cv.commit_provenance()

  return [
    partial(MultiResShardedMeshMergeTask,
      cloudpath, shard_no, 
      num_lod=num_lod,
      mesh_dir=mesh_dir, 
      spatial_index_db=spatial_index_db,
      draco_compression_level=draco_compression_level,
      min_chunk_size=min_chunk_size,
    )
    for shard_no in shard_labels.keys()
  ]
