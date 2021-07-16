import math

import numpy as np

from cloudvolume import Vec
from cloudvolume.datasource.precomputed.sharding import ShardingSpecification

from .types import ShapeType

def image_shard_shape_from_spec(
  spec: ShardingSpecification, 
  dataset_size: ShapeType, 
  chunk_size: ShapeType
) -> ShapeType:

  chunk_size = Vec(*chunk_size, dtype=np.uint64)
  dataset_size = Vec(*dataset_size, dtype=np.uint64)
  preshift_bits = np.uint64(spec["preshift_bits"])

  grid_size = np.ceil(dataset_size / chunk_size).astype(np.uint64)

  j = np.uint64(0)
  one = np.uint64(1)
  shape = Vec(0,0,0, dtype=np.uint64)

  if preshift_bits >= 64:
    raise ValueError(f"preshift_bits must be < 64. Got: {preshift_bits}")

  for i in range(preshift_bits):
    for dim in range(3):
      if 2 ** i < grid_size[dim]:
        shape[dim] += one
 
  shape = Vec(2 ** shape.x, 2 ** shape.y, 2 ** shape.z, dtype=np.uint64)
  return chunk_size * shape
