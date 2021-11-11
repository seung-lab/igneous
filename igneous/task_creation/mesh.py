import copy
from functools import reduce, partial
from typing import (
  Any, Dict, Optional, 
  Union, Tuple, cast,
  Iterator
)

from time import strftime

import numpy as np

import cloudvolume
import cloudvolume.exceptions
from cloudvolume import CloudVolume
from cloudvolume.lib import Vec, Bbox, max2, min2, xyzrange, find_closest_divisor, yellow, jsonify
from cloudfiles import CloudFiles

from igneous.tasks import (
  MeshTask, MeshManifestTask, GrapheneMeshTask,
  MeshSpatialIndex, MultiResUnshardedMeshMergeTask
)
from .common import (
  operator_contact, FinelyDividedTaskIterator, 
  get_bounds, num_tasks, graphene_prefixes
)

__all__ = [
  "create_meshing_tasks",
  "create_mesh_manifest_tasks",
  "create_graphene_meshing_tasks",
  "create_graphene_hybrid_mesh_manifest_tasks",
  "create_spatial_index_mesh_tasks",
  "create_unsharded_multires_mesh_tasks",
]

# split the work up into ~1000 tasks (magnitude 3)
def create_mesh_manifest_tasks(layer_path, magnitude=3, mesh_dir=None):
  assert int(magnitude) == magnitude

  start = 10 ** (magnitude - 1)
  end = 10 ** magnitude

  class MeshManifestTaskIterator(object):
    def __len__(self):
      return (10 ** magnitude) - 1
    def __iter__(self):
      for prefix in range(1, start):
        yield MeshManifestTask(layer_path=layer_path, prefix=str(prefix) + ':', mesh_dir=mesh_dir)

      # enumerate from e.g. 100 to 999
      for prefix in range(start, end):
        yield MeshManifestTask(layer_path=layer_path, prefix=prefix, mesh_dir=mesh_dir)

  return MeshManifestTaskIterator()


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
        yield MeshManifestTask(layer_path=cloudpath, prefix=str(prefix))

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

  if mesh_dir is None:
    mesh_dir = f"mesh_mip_{mip}_err_40"

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

  class SpatialIndexMeshTaskIterator(FinelyDividedTaskIterator):
    def task(self, shape, offset):
      return partial(MeshSpatialIndex, 
        cloudpath=cloudpath,
        shape=shape,
        offset=offset,
        mip=int(mip),
        fill_missing=bool(fill_missing),
        compress=compress,
        mesh_dir=mesh_dir,
      )

    def on_finish(self):
      vol.provenance.processing.append({
        'method': {
          'task': 'MeshSpatialIndex',
          'cloudpath': vol.cloudpath,
          'shape': shape.tolist(),
          'mip': int(mip),
          'mesh_dir': mesh_dir,
          'fill_missing': fill_missing,
          'compress': compress,
        },
        'by': operator_contact(),
        'date': strftime('%Y-%m-%d %H:%M %Z'),
      }) 
      vol.commit_provenance()

  return SpatialIndexMeshTaskIterator(vol.bounds, shape)

def create_unsharded_multires_mesh_tasks(
  cloudpath:str, mip:int, num_lod:int = 1, 
  magnitude:int = 3, mesh_dir:str = None,
  vertex_quantization_bits:int = 16
) -> Iterator:
  """
  vertex_quantization_bits: 10 or 16. Adjusts the precision
    of mesh vertices.
  """
  # split the work up into ~1000 tasks (magnitude 3)
  assert int(magnitude) == magnitude
  assert vertex_quantization_bits in (10, 16)

  vol = CloudVolume(cloudpath, mip=mip)

  if mesh_dir is None:
    mesh_dir = f"mesh_mip_{mip}_err_40"

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

  start = 10 ** (magnitude - 1)
  end = 10 ** magnitude

  class UnshardedMultiResTaskIterator(object):
    def __len__(self):
      return (10 ** magnitude) - 1
    def __iter__(self):
      for prefix in range(1, start):
        yield partial(MultiResUnshardedMeshMergeTask, 
          cloudpath=cloudpath, 
          prefix=str(prefix) + ':', 
          mesh_dir=mesh_dir,
          num_lod=num_lod,
        )

      # enumerate from e.g. 100 to 999
      for prefix in range(start, end):
        yield partial(MultiResUnshardedMeshMergeTask, 
          cloudpath=cloudpath, 
          prefix=prefix, 
          mesh_dir=mesh_dir,
          num_lod=num_lod,
        )

  return UnshardedMultiResTaskIterator()
