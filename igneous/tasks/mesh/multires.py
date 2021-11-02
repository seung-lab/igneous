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
  import MultiLevelPrecomputedMeshManifest
import mapbuffer
from mapbuffer import MapBuffer
from taskqueue import queueable
import trimesh

import cc3d
import DracoPy
import fastremap
import zmesh

from .draco import draco_encoding_settings

@queueable
def MultiResUnshardedMeshMergeTask(
  cloudpath:str, prefix:str,
  cache_control:bool = False,
  draco_compression_level:int = 1,
  draco_create_metadata:bool = False,
  dust_threshold:Optional[int] = None,
  encoding:str = 'precomputed',
  fill_missing:bool = False,
  max_simplification_error:int = 40,
  simplification_factor:int = 100,
  mesh_dir:Optional[str] = None,
  num_lod:int = 1,
  progress:bool = False,
  remap_table:Optional[dict] = None,
  spatial_index:bool = False,
  sharded:bool = False,
  timestamp:Optional[int] = None,
  agglomerate:Optional[bool] = True,
  stop_layer:Optional[int] = 2,
  compress:str = 'gzip',
):
  cv = CloudVolume(cloudpath)
  
  if mesh_dir is None and 'mesh' in cv.info:
    mesh_dir = cv.info['mesh']

  files_per_label = get_mesh_filenames_subset(
    cloudpath, mesh_dir, prefix
  )

  draco_settings = draco_encoding_settings(
    shape=cv.bounds.size3(),
    offset=offset,
    resolution=cv.resolution,
    create_metadata=True,
  )

  cf = CloudFiles(cloudpath)
  for label, filenames in files_per_label.items():
    # filenames = prepend mesh_dir
    files = cf.get(filenames)
    # we should handle draco as well
    files = [ Mesh.from_precomputed(f["content"]) for f in files ]
    mesh = Mesh.concatenate(files)
    
    mesh = DracoPy.encode_mesh_to_buffer(
      mesh.vertices.flatten('C'), mesh.faces.flatten('C'), 
      **draco_settings
    )

    manifest = MultiLevelPrecomputedMeshManifest(
      segment_id=label, 
      chunk_shape=cv.bounds.size3(),
      grid_origin=[0,0,0], 
      num_lods=1, 
      lod_scales=2, 
      vertex_offsets=[[0,0,0]],
      num_fragments_per_lod=[1], 
      fragment_positions=[[0,0,0]], 
      fragment_offsets=[0],
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






