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
  minishard_bits = np.uint64(spec["minishard_bits"])
  shape_bits = preshift_bits + minishard_bits

  grid_size = np.ceil(dataset_size / chunk_size).astype(np.uint64)
  one = np.uint64(1)

  if shape_bits >= 64:
    raise ValueError(
      f"preshift_bits ({preshift_bits}) + minishard_bits ({minishard_bits}) must be < 64. Sum: {shape_bits}"
    )

  def compute_shape_bits():
    shape = Vec(0,0,0, dtype=np.uint64)

    i = 0
    over = [ False, False, False ]
    while i < shape_bits:
      changed = False
      for dim in range(3):
        if 2 ** (shape[dim] + 1) < grid_size[dim] * 2 and not over[dim]:
          if 2 ** (shape[dim] + 1) >= grid_size[dim]:
            over[dim] = True
          shape[dim] += one
          i += 1
          changed = True

        if i >= shape_bits:
          return shape

      if not changed:
        return shape

    return shape

  shape = compute_shape_bits()
  shape = Vec(2 ** shape.x, 2 ** shape.y, 2 ** shape.z, dtype=np.uint64)
  return chunk_size * shape
