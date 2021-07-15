import numpy as np

from cloudvolume import Vec
from cloudvolume.datasource.precomputed.sharding import ShardingSpecification

from .types import ShapeType

def image_shard_shape_from_spec(
  spec: ShardingSpecification, 
  dataset_size: ShapeType, 
  chunk_size: ShapeType
) -> ShapeType:
  # WARNING: Assumes data bounds are larger
  # than the shard size. The compressed
  # morton code logic changes if one of the
  # axes is truncated.
  chunk_size = Vec(*chunk_size, dtype=int)
  preshift_bits = spec["preshift_bits"]

  pot = preshift_bits // 3
  x = 2 ** pot
  y = 2 ** pot 
  z = 2 ** pot

  remainder = preshift_bits % 3
  if remainder == 1:
    x *= 2
  elif remainder == 2:
    x *= 2
    y *= 2

  shape = chunk_size * Vec(x,y,z)  
  dataset_size = Vec(*dataset_size)

  if np.any(shape > dataset_size):
    raise ValueError(f"shape {shape} > {dataset_size} datset size.")

  return shape