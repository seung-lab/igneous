from typing import Optional

from collections import defaultdict

import json
import math
import os
import random
import re
import struct

import numpy as np
from tqdm import tqdm

from cloudfiles import CloudFiles

from cloudvolume import CloudVolume, Mesh, view
from cloudvolume.lib import Vec, Bbox, jsonify
from cloudvolume.datasource.precomputed.mesh.multilod \
  import MultiLevelPrecomputedMeshManifest, to_stored_model_space
import mapbuffer
from mapbuffer import MapBuffer
from taskqueue import queueable
import trimesh

import cc3d
import DracoPy
import fastremap
import zmesh

from .draco import draco_encoding_settings

__all__ = [
  "MultiResUnshardedMeshMergeTask",
]

@queueable
def MultiResUnshardedMeshMergeTask(
  cloudpath:str, 
  prefix:str,
  cache_control:bool = False,
  draco_compression_level:int = 1,
  mesh_dir:Optional[str] = None,
  num_lod:int = 1,
  # progress:bool = False,
  # sharded:bool = False,
):
  cv = CloudVolume(cloudpath)
  
  if mesh_dir is None and 'mesh' in cv.info:
    mesh_dir = cv.info['mesh']

  files_per_label = get_mesh_filenames_subset(
    cloudpath, mesh_dir, prefix
  )

  draco_settings = draco_encoding_settings(
    shape=cv.bounds.size3(),
    offset=cv.bounds.minpt,
    resolution=cv.resolution,
    compression_level=draco_compression_level,
    create_metadata=True,
  )

  cf = CloudFiles(cv.meta.join(cloudpath, mesh_dir))
  for label, filenames in files_per_label.items():
    files = cf.get(filenames)
    # we should handle draco as well
    files = [ Mesh.from_precomputed(f["content"]) for f in files ]
    mesh = Mesh.concatenate(*files)

    manifest = MultiLevelPrecomputedMeshManifest(
      segment_id=label, 
      chunk_shape=cv.bounds.size3(),
      grid_origin=cv.bounds.minpt, 
      num_lods=1, 
      lod_scales=2, 
      vertex_offsets=[[0,0,0]],
      num_fragments_per_lod=[1], 
      fragment_positions=[[[0,0,0]]], 
      fragment_offsets=[0],
    )

    mesh.vertices /= cv.meta.resolution(cv.mesh.meta.mip)
    mesh.vertices = to_stored_model_space(
      mesh.vertices, manifest, 
      lod=0, 
      vertex_quantization_bits=int(cv.mesh.meta.info["vertex_quantization_bits"]),
      frag=0
    )

    mesh = DracoPy.encode_mesh_to_buffer(
      mesh.vertices.flatten('C'), mesh.faces.flatten('C'), 
      **draco_settings
    )

    cf.put(f"{label}.index", manifest.to_binary(), cache_control="no-cache")
    cf.put(f"{label}", mesh, cache_control="no-cache")

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






