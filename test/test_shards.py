import pytest
import operator
from functools import reduce

import numpy as np
from cloudvolume import CloudVolume, Bbox, Vec
import cloudvolume.lib as lib
from cloudvolume.datasource.precomputed.image.common import (
  gridpoints, compressed_morton_code
)
from cloudvolume.datasource.precomputed.sharding import ShardReader, ShardingSpecification

from igneous.task_creation.image import create_sharded_image_info
from igneous.shards import image_shard_shape_from_spec

def prod(x):
  return reduce(operator.mul, x, 1)

SCALES = [
  {
    'chunk_sizes': [[128, 128, 64]],
    'encoding': 'jpeg',
    'key': '48_48_30',
    'resolution': [48, 48, 30],
    'size': [1536, 1408, 2046],
    'voxel_offset': [0, 0, 0]
  },
  {
    'chunk_sizes': [[128, 128, 64]],
    'encoding': 'jpeg',
    'key': '24_24_30',
    'resolution': [24, 24, 30],
    'size': [3072, 2816, 2046],
    'voxel_offset': [0, 0, 0]
  },
  {
    'chunk_sizes': [[128, 128, 20]],
    'encoding': 'raw',
    'key': '4_4_40',
    'resolution': [4, 4, 40],
    'size': [40960, 40960, 990],
    'voxel_offset': [69632, 36864, 4855],
  },
]

@pytest.mark.parametrize("scale", SCALES)
def test_sharded_image_bits(scale):
  dataset_size = Vec(*scale["size"])
  chunk_size = Vec(*scale["chunk_sizes"][0])

  spec = create_sharded_image_info( 
    dataset_size=dataset_size,
    chunk_size=chunk_size,
    encoding="jpeg",
    dtype=np.uint8
  )

  shape = image_shard_shape_from_spec(
    spec, dataset_size, chunk_size
  )

  shape = lib.min2(shape, dataset_size)
  dataset_bbox = Bbox.from_vec(dataset_size)
  gpts = list(gridpoints(dataset_bbox, dataset_bbox, chunk_size))
  grid_size = np.ceil(dataset_size / chunk_size).astype(np.int64)

  spec = ShardingSpecification.from_dict(spec)
  reader = ShardReader(None, None, spec)

  morton_codes = compressed_morton_code(gpts, grid_size)
  min_num_shards = prod(dataset_size / shape)
  max_num_shards = prod(np.ceil(dataset_size / shape))
  
  assert 0 < min_num_shards <= 2 ** spec.shard_bits
  assert 0 < max_num_shards <= 2 ** spec.shard_bits

  real_num_shards = len(set(map(reader.get_filename, morton_codes)))

  assert min_num_shards <= real_num_shards <= max_num_shards




