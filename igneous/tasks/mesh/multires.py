from typing import Optional, List, Dict, Tuple

from collections import defaultdict
import functools
import itertools
import json
import math
import os
import pickle
import random
import re
import struct

import numpy as np
from tqdm import tqdm

from cloudfiles import CloudFiles

from cloudvolume import CloudVolume, Mesh, view
from cloudvolume.lib import Vec, Bbox, jsonify, sip, toiter, first
from cloudvolume.datasource.precomputed.mesh.multilod \
  import MultiLevelPrecomputedMeshManifest, to_stored_model_space
from cloudvolume.datasource.precomputed.sharding import synthesize_shard_files

import mapbuffer
from mapbuffer import MapBuffer
from taskqueue import queueable

import DracoPy
import fastremap
import pyfqmr
import trimesh
import zmesh

from .draco import draco_encoding_settings

__all__ = [
  "MultiResShardedMeshMergeTask",
  "MultiResUnshardedMeshMergeTask",
  "MultiResShardedFromUnshardedMeshMergeTask",
]

@queueable
def MultiResUnshardedMeshMergeTask(
  cloudpath:str, 
  prefix:str,
  cache_control:bool = False,
  draco_compression_level:int = 1,
  mesh_dir:Optional[str] = None,
  num_lod:int = 1,
  min_chunk_size:Tuple[int,int,int] = (512,512,512),
  progress:bool = False,
):
  cv = CloudVolume(cloudpath)
  
  if mesh_dir is None and 'mesh' in cv.info:
    mesh_dir = cv.info['mesh']

  files_per_label = get_mesh_filenames_subset(
    cloudpath, mesh_dir, prefix
  )

  cf = CloudFiles(cv.meta.join(cloudpath, mesh_dir))
  for label, filenames in tqdm(files_per_label.items(), disable=(not progress)):
    files = cf.get(filenames)
    # we should handle draco as well
    files = [ Mesh.from_precomputed(f["content"]) for f in files ]
    files = Mesh.concatenate(*files)

    (manifest, mesh) = process_mesh(
      cv, label, files, 
      num_lod, min_chunk_size, 
      draco_compression_level
    )

    if manifest is None:
      continue

    cf.put(f"{label}.index", manifest.to_binary(), cache_control="no-cache")
    cf.put(f"{label}", mesh, cache_control="no-cache")

def process_mesh(
  cv:CloudVolume,
  label:int,
  mesh:Mesh,
  num_lod:int,
  min_chunk_size:Tuple[int,int,int] = (512,512,512),
  draco_compression_level:int = 7,
) -> Tuple[MultiLevelPrecomputedMeshManifest, Mesh]:

  mesh.vertices /= cv.meta.resolution(cv.mesh.meta.mip)

  grid_origin = np.floor(np.min(mesh.vertices, axis=0))
  mesh_shape = (np.max(mesh.vertices, axis=0) - grid_origin).astype(int)

  if np.any(mesh_shape == 0):
    return (None, None)

  min_chunk_size = np.array(min_chunk_size, dtype=int)
  max_lod = int(max(np.min(np.log2(mesh_shape / min_chunk_size)), 0))
  max_lod = min(max_lod, num_lod)

  lods = generate_lods(label, mesh, max_lod)
  chunk_shape = np.ceil(mesh_shape / (2 ** (len(lods) - 1)))

  if np.any(chunk_shape == 0):
    return (None, None)

  lods = [
    create_octree_level_from_mesh(lods[lod], chunk_shape, lod, len(lods)) 
    for lod in range(len(lods)) 
  ]
  fragment_positions = [ nodes for submeshes, nodes in lods ]
  lods = [ submeshes for submeshes, nodes in lods ]

  manifest = MultiLevelPrecomputedMeshManifest(
    segment_id=label,
    chunk_shape=chunk_shape,
    grid_origin=grid_origin, 
    num_lods=len(lods), 
    lod_scales=[ 2**i for i in range(len(lods)) ],
    vertex_offsets=[[0,0,0]] * len(lods),
    num_fragments_per_lod=[ len(lods[lod]) for lod in range(len(lods)) ],
    fragment_positions=fragment_positions,
    fragment_offsets=[], # needs to be set when we have the final value
  )

  vqb = int(cv.mesh.meta.info["vertex_quantization_bits"])

  mesh_binaries = []
  for lod, submeshes in enumerate(lods):
    for frag_no, submesh in enumerate(submeshes):
      submesh.vertices = to_stored_model_space(
        submesh.vertices, manifest, 
        lod=lod,
        vertex_quantization_bits=vqb,
        frag=frag_no,
      )

      # test for totally degenerate meshes by checking if 
      # all of two axes match, meaning the mesh must be a 
      # point or a line.
      if np.sum(np.bitwise_and.reduce(submesh.vertices) == submesh.vertices[0,:]) >= 2:
        continue

      minpt = np.min(submesh.vertices, axis=0)
      quantization_range = np.max(submesh.vertices, axis=0) - minpt
      quantization_range = np.max(quantization_range)

      # mesh.vertices must be integer type or mesh will display
      # distorted in neuroglancer.
      submesh = DracoPy.encode(
        submesh.vertices, submesh.faces, 
        quantization_bits=vqb,
        compression_level=draco_compression_level,
        quantization_range=quantization_range,
        quantization_origin=minpt,
        create_metadata=True,
      )
      manifest.fragment_offsets.append(len(submesh))
      mesh_binaries.append(submesh)

  return (manifest, b''.join(mesh_binaries))

def get_mesh_filenames_subset(
  cloudpath:str, mesh_dir:str, prefix:str
):
  prefix = f'{mesh_dir}/{prefix}'
  segids = defaultdict(list)

  cf = CloudFiles(cloudpath)
  meshexpr = re.compile(r'(\d+):(\d+):')
  for filename in cf.list(prefix=prefix):
    filename = os.path.basename(filename)
    # `match` implies the beginning (^). `search` matches whole string
    matches = re.search(meshexpr, filename)

    if not matches:
      continue

    segid, lod = matches.groups()
    segid, lod = int(segid), int(lod)

    if lod != 0:
      continue

    segids[segid].append(filename)

  return segids

@queueable
def MultiResShardedMeshMergeTask(
  cloudpath:str,
  shard_no:str,
  draco_compression_level:int = 1,
  mesh_dir:Optional[str] = None,
  num_lod:int = 1,
  spatial_index_db:Optional[str] = None,
  min_chunk_size:Tuple[int,int,int] = (128,128,128),
  progress:bool = False
):
  cv = CloudVolume(cloudpath, spatial_index_db=spatial_index_db)
  cv.mip = cv.mesh.meta.mip
  if mesh_dir is None and 'mesh' in cv.info:
    mesh_dir = cv.info['mesh']

  # This looks messy because we are trying to avoid retaining
  # unnecessary memory. In the original skeleton iteration, this was 
  # using 50 GB+ memory on minnie65. So it makes sense to be just
  # as careful with a heavier type of object.
  locations = locations_for_labels(cv, labels_for_shard(cv, shard_no))
  filenames = set(itertools.chain(*locations.values()))
  labels = set(locations.keys())
  del locations
  meshes = collect_mesh_fragments(
    cv, labels, filenames, mesh_dir, progress
  )
  del filenames

  # important to iterate this way to avoid
  # creating a copy of meshes vs. { ... for in }
  for label in labels:
    meshes[label] = Mesh.concatenate(*meshes[label])
  del labels

  fname, shard = create_mesh_shard(
    cv, meshes, 
    num_lod, draco_compression_level,
    progress, shard_no, min_chunk_size
  )
  del meshes

  if shard is None:
    return

  cf = CloudFiles(cv.mesh.meta.layerpath)
  cf.put(
    fname, shard,
    compress=False,
    content_type='application/octet-stream',
    cache_control='no-cache',
  )

@queueable
def MultiResShardedFromUnshardedMeshMergeTask(
  src:str,
  dest:str,
  shard_no:str,
  cache_control:bool = False,
  draco_compression_level:int = 1,
  mesh_dir:Optional[str] = None,
  num_lod:int = 1,
  progress:bool = False,
  min_chunk_size:Tuple[int,int,int] = (512,512,512),
):
  cv_src = CloudVolume(src)

  if mesh_dir is None and 'mesh' in cv_src.info:
    mesh_dir = cv_src.info['mesh']

  cv_dest = CloudVolume(dest, mesh_dir=mesh_dir, progress=progress)

  labels = labels_for_shard(cv_dest, shard_no)

  if cv_src.mesh.meta.info['@type'] == "neuroglancer_multilod_draco":
    meshes = cv_src.mesh.get(labels)
  else:
    meshes = cv_src.mesh.get(labels, fuse=False)

  del labels
  
  fname, shard = create_mesh_shard(
    cv_dest, meshes, 
    num_lod, draco_compression_level,
    progress, shard_no, min_chunk_size
  )
  del meshes

  if shard is None:
    return

  cf = CloudFiles(cv_dest.mesh.meta.layerpath)
  cf.put(
    fname, shard, # fname, data
    compress=False,
    content_type='application/octet-stream',
    cache_control='no-cache',
  )

def generate_lods(
  label:int,
  mesh:Mesh,
  num_lods:int,
  decimation_factor:int = 2, 
  aggressiveness:float = 5.0,
  progress:bool = False,
):
  assert num_lods >= 0, num_lods

  lods = [ mesh ]

  # from pyfqmr documentation:
  # threshold = alpha * (iteration + K) ** agressiveness
  # 
  # Threshold is the total error that can be tolerated by
  # deleting a vertex.
  for i in range(1, num_lods+1):
    simplifier = pyfqmr.Simplify()
    simplifier.setMesh(mesh.vertices, mesh.faces)
    simplifier.simplify_mesh(
      target_count=max(int(len(mesh.faces) / (decimation_factor ** i)), 4),
      aggressiveness=aggressiveness,
      preserve_border=True,
      verbose=False,
      # Additional parameters to expose?
      # max_iterations=
      # K=
      # alpha=
      # update_rate=  # Number of iterations between each update.
      # lossless=
      # threshold_lossless=
    )

    lods.append(
      Mesh(*simplifier.getMesh())
    )

  return lods

def create_mesh_shard(
  cv:CloudVolume, meshes:Dict[int, Mesh],
  num_lod:int, draco_compression_level:int,
  progress:bool, shard_no:str, min_chunk_size:Tuple[int,int,int]
):
  meshes = {
    label: process_mesh(
      cv, label, mesh, 
      num_lod, min_chunk_size,
      draco_compression_level=draco_compression_level
    )
    for label, mesh in tqdm(meshes.items(), disable=(not progress))
  }
  data_offset = {
    label: len(manifest)
    for label, (manifest, mesh_binary) in meshes.items()
    if manifest is not None
  }
  meshes = {
    label: mesh_binary + manifest.to_binary()
    for label, (manifest, mesh_binary) in meshes.items()
    if manifest is not None
  }

  if len(meshes) == 0:
    return None, None

  shard_files = synthesize_shard_files(
    cv.mesh.reader.spec, meshes, data_offset
  )

  if len(shard_files) != 1:
    raise ValueError(
      "Only one shard file should be generated per task. "
      "Expected: {} Got: {} ".format(
        str(shard_no), ", ".join(shard_files.keys())
    ))

  filename = first(shard_files.keys())
  return filename, shard_files[filename]

def collect_mesh_fragments(
  cv:CloudVolume, 
  labels:List[int], 
  filenames:List[str], 
  mesh_dir:str, 
  progress:bool = False
) -> Dict[int, List[Mesh]]:
  dirfn = lambda loc: cv.meta.join(mesh_dir, loc)
  filenames = [ dirfn(loc) for loc in filenames ]

  block_size = 50

  if len(filenames) < block_size:
    blocks = [ filenames ]
    n_blocks = 1
  else:
    n_blocks = max(len(filenames) // block_size, 1)
    blocks = sip(filenames, block_size)

  all_meshes = defaultdict(list)
  for filenames_block in tqdm(blocks, desc="Filename Block", total=n_blocks, disable=(not progress)):
    if cv.meta.path.protocol == "file":
      all_files = {}
      prefix = cv.cloudpath.replace("file://", "")
      for filename in filenames_block:
        all_files[filename] = open(os.path.join(prefix, filename), "rb")
    else:
      all_files = cv.mesh.cache.download(filenames_block, progress=progress)
    
    for filename, content in tqdm(all_files.items(), desc="Scanning Fragments", disable=(not progress)):
      fragment = MapBuffer(content, frombytesfn=Mesh.from_precomputed)
      fragment.validate()

      for label in labels:
        try:
          mesh = fragment[label]
          mesh.id = label
          all_meshes[label].append((filename, mesh))
        except KeyError:
          continue

      if hasattr(content, "close"):
        content.close()

  # ensure consistent results across multiple runs
  # by sorting mesh fragments by filename
  for label in all_meshes:
    all_meshes[label].sort(key=lambda pair: pair[0])
    all_meshes[label] = [ pair[1] for pair in all_meshes[label] ]

  return all_meshes

def locations_for_labels(
  cv:CloudVolume, labels:List[int]
) -> Dict[int, List[str]]:

  SPATIAL_EXT = re.compile(r'\.spatial$')
  index_filenames = cv.mesh.spatial_index.file_locations_per_label(labels)
  for label, locations in index_filenames.items():
    for i, location in enumerate(locations):
      bbx = Bbox.from_filename(re.sub(SPATIAL_EXT, '', location))
      bbx /= cv.meta.resolution(cv.mesh.meta.mip)
      index_filenames[label][i] = bbx.to_filename() + '.frags'
  return index_filenames

def labels_for_shard(
  cv:CloudVolume, shard_no:str, progress:bool = False
) -> List[int]:
  """
  Try to fetch precalculated labels from `$shardno.labels` (faster) otherwise, 
  compute which labels are applicable to this shard from the shard index (much slower).
  """
  labels = CloudFiles(cv.mesh.meta.layerpath).get_json(shard_no + '.labels')
  if labels is not None:
    return labels

  labels = cv.mesh.spatial_index.query(cv.bounds * cv.resolution)
  spec = cv.mesh.reader.spec

  return [ 
    lbl for lbl in tqdm(labels, desc="Computing Shard Numbers", disable=(not progress))  \
    if spec.compute_shard_location(lbl).shard_number == shard_no 
  ]
  
## Below functons adapted from 
## https://github.com/google/neuroglancer/issues/272
## Thanks to Hythem Sidky (@hsidky) for sharing his 
## progress and code with the connectomics community.

def cmp_zorder(lhs, rhs) -> bool:
  def less_msb(x: int, y: int) -> bool:
    return x < y and x < (x ^ y)

  # Assume lhs and rhs array-like objects of indices.
  assert len(lhs) == len(rhs)
  # Will contain the most significant dimension.
  msd = 2
  # Loop over the other dimensions.
  for dim in [1, 0]:
    # Check if the current dimension is more significant
    # by comparing the most significant bits.
    if less_msb(lhs[msd] ^ rhs[msd], lhs[dim] ^ rhs[dim]):
      msd = dim
  return lhs[msd] - rhs[msd]

def create_octree_level_from_mesh(mesh, chunk_shape, lod, num_lods):
  """
  Create submeshes by slicing the orignal mesh to produce smaller chunks
  by slicing them from x,y,z dimensions.

  This creates (2^lod)^3 submeshes.
  """
  if lod == num_lods - 1:
    return ([ mesh ], ((0,0,0),) )

  mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)

  scale = Vec(*(np.array(chunk_shape) * (2 ** lod)))
  offset = Vec(*np.floor(mesh.vertices.min(axis=0)))
  grid_size = Vec(*np.ceil((mesh.vertices.max(axis=0) - offset) / scale), dtype=int)

  nx, ny, nz = np.eye(3)
  ox, oy, oz = offset * np.eye(3)

  submeshes = []
  nodes = []
  for x in range(0, grid_size.x):
    # list(...) required b/c it doesn't like Vec classes
    mesh_x = trimesh.intersections.slice_mesh_plane(mesh, plane_normal=nx, plane_origin=list(nx*x*scale.x+ox))
    mesh_x = trimesh.intersections.slice_mesh_plane(mesh_x, plane_normal=-nx, plane_origin=list(nx*(x+1)*scale.x+ox))
    for y in range(0, grid_size.y):
      mesh_y = trimesh.intersections.slice_mesh_plane(mesh_x, plane_normal=ny, plane_origin=list(ny*y*scale.y+oy))
      mesh_y = trimesh.intersections.slice_mesh_plane(mesh_y, plane_normal=-ny, plane_origin=list(ny*(y+1)*scale.y+oy))
      for z in range(0, grid_size.z):
        mesh_z = trimesh.intersections.slice_mesh_plane(mesh_y, plane_normal=nz, plane_origin=list(nz*z*scale.z+oz))
        mesh_z = trimesh.intersections.slice_mesh_plane(mesh_z, plane_normal=-nz, plane_origin=list(nz*(z+1)*scale.z+oz))

        if len(mesh_z.vertices) > 0:
          submeshes.append(mesh_z)
          nodes.append((x, y, z))

  # Sort in Z-curve order
  submeshes, nodes = zip(
    *sorted(zip(submeshes, nodes),
    key=functools.cmp_to_key(lambda x, y: cmp_zorder(x[1], y[1])))
  )
  # convert back from trimesh to CV Mesh class
  submeshes = [ Mesh(m.vertices, m.faces) for m in submeshes ]

  return (submeshes, nodes)
