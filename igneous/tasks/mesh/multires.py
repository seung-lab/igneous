from typing import Optional, List, Dict, Tuple

from collections import defaultdict
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

import cc3d
import DracoPy
import fastremap
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

    (manifest, mesh) = process_mesh(
      cv, label, files, 
      num_lod, draco_compression_level
    )

    cf.put(f"{label}.index", manifest.to_binary(), cache_control="no-cache")
    cf.put(f"{label}", mesh, cache_control="no-cache")

def process_mesh(
  cv:CloudVolume,
  label:int,
  mesh: Mesh,
  num_lod:int = 1,
  draco_compression_level:int = 1,
) -> Tuple[MultiLevelPrecomputedMeshManifest, Mesh]:

  manifest = MultiLevelPrecomputedMeshManifest(
    segment_id=label, 
    chunk_shape=cv.bounds.size3(),
    grid_origin=cv.bounds.minpt, 
    num_lods=1, 
    lod_scales=[ 1 ], 
    vertex_offsets=[[0,0,0]],
    num_fragments_per_lod=[1], 
    fragment_positions=[[[0,0,0]]], 
    fragment_offsets=[0], # needs to be set when we have the final value
  )

  vqb = int(cv.mesh.meta.info["vertex_quantization_bits"])

  mesh.vertices /= cv.meta.resolution(cv.mesh.meta.mip)
  mesh.vertices = to_stored_model_space(
    mesh.vertices, manifest, 
    lod=0, 
    vertex_quantization_bits=vqb,
    frag=0
  )

  quantization_range = np.max(mesh.vertices, axis=0) - np.min(mesh.vertices, axis=0)
  quantization_range = np.max(quantization_range)

  # mesh.vertices must be integer type or mesh will display
  # distored in neuroglancer.
  mesh = DracoPy.encode(
    mesh.vertices, mesh.faces, 
    quantization_bits=vqb,
    compression_level=draco_compression_level,
    quantization_range=quantization_range,
    quantization_origin=np.min(mesh.vertices, axis=0),
    create_metadata=True,
  )
  manifest.fragment_offsets = [ len(mesh) ]

  return (manifest, mesh)

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
    progress, shard_no
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
):
  cv_src = CloudVolume(src)

  if mesh_dir is None and 'mesh' in cv.info:
    mesh_dir = cv.info['mesh']

  cv_dest = CloudVolume(dest, mesh_dir=mesh_dir, progress=True)

  labels = labels_for_shard(cv_dest, shard_no)
  meshes = cv_src.mesh.get(labels, fuse=False)
  del labels
    
  fname, shard = create_mesh_shard(
    cv_dest, meshes, 
    num_lod, draco_compression_level,
    progress, shard_no
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

def create_mesh_shard(
  cv:CloudVolume, meshes:dict, 
  num_lod:int, draco_compression_level:int,
  progress:bool, shard_no:str
):
  meshes = {
    label: process_mesh(
      cv, label, mesh_frags,
      num_lod, draco_compression_level
    )
    for label, mesh_frags in tqdm(meshes.items(), disable=(not progress))
  }
  data_offset = { 
    label: len(manifest)
    for label, (manifest, mesh) in meshes.items() 
  }
  meshes = {
    label: mesh + manifest.to_binary()
    for label, (manifest, mesh) in meshes.items()
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
          all_meshes[label].append(mesh)
        except KeyError:
          continue

      if hasattr(content, "close"):
        content.close()

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
  
