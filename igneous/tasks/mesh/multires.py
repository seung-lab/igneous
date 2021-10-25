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

from cloudvolume import CloudVolume, view
from cloudvolume.lib import Vec, Bbox, jsonify
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
def MultiResMeshMergeTask(
  cloudpath:str, label:int,
  cache_control:bool = False,
  draco_compression_level:int = 1,
  draco_create_metadata:bool = False,
  dust_threshold:Optional[int] = None,
  encoding:str = 'precomputed',
  fill_missing:bool = False,
  max_simplification_error:int = 40,
  simplification_factor:int = 100,
  mesh_dir:Optional[str] = None,
  mip:int = 0,
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
  cv = CloudVolume(
    cloudpath, mip=mip, bounded=False,
    fill_missing=fill_missing
  )
  bounds = Bbox(offset, shape + offset)
  bounds = Bbox.clamp(bounds, cv.bounds)


  locations = cv.mesh.meta.spatial_index.file_locations_per_label(label)
  filenames = locations[label]

  draco_settings = draco_encoding_settings(
    shape=data_bounds.size3(),
    offset=offset,
    resolution=cv.resolution,
    create_metadata=True,
  )
  meshes = [ 
    DracoPy.encode_mesh_to_buffer(
      mesh.vertices.flatten('C'), mesh.faces.flatten('C'), 
      **draco_settings
    ) 
    for mesh in meshes 
  ]

  # create_multires_manifest(
  #   chunk_shape=shape, 
  #   grid_origin=cv.voxel_offset, 
  #   num_lods=1, lods_scales=[[1,1,1]],
  #   vertex_offsets=[[0,0,0]], 
  #   num_fragments_per_lod=[ len(meshes) ],
  #   fragment_positions=, 
  #   fragment_offsets=,
  # )


  







