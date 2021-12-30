# @license
# Copyright 2016,2017 The Neuroglancer Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import numpy as np
from cloudvolume import CloudVolume, Vec
from cloudvolume.lib import min2, getprecision

DEFAULT_MAX_DOWNSAMPLING = 64 # maximum factor to downsample by
DEFAULT_MAX_DOWNSAMPLED_SIZE = 128 # minimum length of a side after downsampling
DEFAULT_MAX_DOWNSAMPLING_SCALES = float('inf')

def scale_series_to_downsample_factors(scales):
  fullscales = [ np.array(scale) for scale in scales ] 
  factors = []
  for i in range(1, len(fullscales)):
    factors.append( fullscales[i] / fullscales[i - 1]  )
  return [ factor.astype(int) for factor in factors ]


def compute_near_isotropic_downsampling_scales(size,
                                               voxel_size,
                                               dimensions_to_downsample,
                                               max_scales=DEFAULT_MAX_DOWNSAMPLING_SCALES,
                                               max_downsampling=DEFAULT_MAX_DOWNSAMPLING,
                                               max_downsampled_size=DEFAULT_MAX_DOWNSAMPLED_SIZE):
    """Compute a list of successive downsampling factors."""

    num_dims = len(voxel_size)
    cur_scale = np.ones((num_dims, ), dtype=int)
    scales = [tuple(cur_scale)]
    while (len(scales) < max_scales 
            and (np.prod(cur_scale) < max_downsampling) 
            and np.all(size / cur_scale > max_downsampled_size)
            and np.all(np.mod(size,cur_scale) == 0)): # the number of chunks should be integer
        # Find dimension with smallest voxelsize.
        cur_voxel_size = cur_scale * voxel_size
        smallest_cur_voxel_size_dim = dimensions_to_downsample[np.argmin(cur_voxel_size[
            dimensions_to_downsample])]
        cur_scale[smallest_cur_voxel_size_dim] *= 2
        target_voxel_size = cur_voxel_size[smallest_cur_voxel_size_dim] * 2
        for d in dimensions_to_downsample:
            if d == smallest_cur_voxel_size_dim:
                continue
            d_voxel_size = cur_voxel_size[d]
            if abs(d_voxel_size - target_voxel_size) > abs(d_voxel_size * 2 - target_voxel_size):
                cur_scale[d] *= 2
        scales.append(tuple(cur_scale))
    return scales

def compute_two_dimensional_near_isotropic_downsampling_scales(
        size,
        voxel_size,
        max_scales=float('inf'),
        max_downsampling=DEFAULT_MAX_DOWNSAMPLING,
        max_downsampled_size=DEFAULT_MAX_DOWNSAMPLED_SIZE):
    """Compute a list of successive downsampling factors for 2-d tiles."""

    max_scales = min(max_scales, 10)

    # First compute a set of 2-d downsamplings for XY, XZ, and YZ with a high
    # number of max_scales, and ignoring other criteria.
    scales_transpose = [
        compute_near_isotropic_downsampling_scales(
            size=size,
            voxel_size=voxel_size,
            dimensions_to_downsample=dimensions_to_downsample,
            max_scales=max_scales,
            max_downsampling=float('inf'),
            max_downsampled_size=0, ) for dimensions_to_downsample in [[0, 1], [0, 2], [1, 2]]
    ]

    # Truncate all list of scales to the same length, once the stopping criteria
    # is reached for all values of dimensions_to_downsample.
    scales = [((1, ) * 3, ) * 3]
    size = np.array(size)

    def scale_satisfies_criteria(scale):
        return np.prod(scale) < max_downsampling and (size / scale).max() > max_downsampled_size

    for i in range(1, max_scales):
        cur_scales = tuple(scales_transpose[d][i] for d in range(3))
        if all(not scale_satisfies_criteria(scale) for scale in cur_scales):
            break
        scales.append(cur_scales)
    return scales

def compute_plane_downsampling_scales(size, preserve_axis='z',
                                       max_scales=DEFAULT_MAX_DOWNSAMPLING_SCALES,
                                       max_downsampling=DEFAULT_MAX_DOWNSAMPLING,
                                       max_downsampled_size=DEFAULT_MAX_DOWNSAMPLED_SIZE):

    axis_map = { 'x': 0, 'y': 1, 'z': 2 }
    preserve_axis = axis_map[preserve_axis]

    size = np.array(size)

    if np.any(size <= 0):
        return [ (1,1,1) ]

    size[preserve_axis] = size[ (preserve_axis + 1) % 3 ]

    dimension = min(*size)
    num_downsamples = int(np.log2(dimension / max_downsampled_size))
    num_downsamples = min(num_downsamples, max_scales)

    factor = 2
    scales = [ (1,1,1) ]
    for i in range(num_downsamples):
        if factor > max_downsampling:
            break
        elif dimension / factor < max_downsampled_size:
            break

        scale = [ factor, factor, factor ]
        scale[preserve_axis] = 1

        scales.append(tuple(scale))
        factor *= 2

    return scales
    
def compute_factors(ds_shape, factor, chunk_size, volume_size):
  chunk_size = np.array(chunk_size)
  grid_size = Vec(*ds_shape, dtype=np.float32) / Vec(*chunk_size, dtype=np.float32)
  # find the dimension which will tolerate the smallest number of downsamples and 
  # return the number it will accept. + 0.0001 then truncate to compensate for FP errors
  # that would result in the log e.g. resulting in 1.9999999382 when it should be an
  # exact result.

  # This filtering is to avoid problems with dividing by log(1)
  grid_size = [ g for f, g in zip(factor, grid_size) if f != 1 ]
  grid_size = Vec(*grid_size, dtype=np.float32)

  factor_div = [ f for f in factor if f != 1 ]
  factor_div = Vec(*factor_div, dtype=np.float32)

  if len(factor_div) == 0:
    return []

  epsilon = 0.0001
  N = np.log(grid_size) / np.log(factor_div)
  N += epsilon
  N = min(N)

  if N < epsilon:
    return []

  dsvol = np.array(volume_size) / (np.array(factor) ** int(np.ceil(N)))
  dsvol = np.array([ dsvol[i] for i,f in enumerate(factor) if f != 1 ])
  chunk_size = np.array([ chunk_size[i] for i,f in enumerate(factor) if f != 1 ])

  N, fract = int(N), (N - float(int(N)))

  # incomplete downsamples are only legal when the
  # volume size is smaller than the chunk size.
  if all(dsvol < chunk_size) and fract > 0.05:
    N += 1

  return [ factor ] * N

def axis_to_factor(axis):
  if axis == 'x':
    return (1,2,2)
  elif axis == 'y':
    return (2,1,2)
  elif axis == 'z':
    return (2,2,1)
  else:
    raise ValueError("Axis not supported: " + str(axis))

def compute_scales(vol, mip, shape, axis, factor, chunk_size=None):
  shape = min2(vol.meta.volume_size(mip), shape)
  # sometimes we downsample a base layer of 512x512 
  # into underlying chunks of 64x64 which permits more scales
  underlying_mip = (mip + 1) if (mip + 1) in vol.available_mips else mip

  if chunk_size:
    scale_chunk_size = Vec(*chunk_size).astype(np.float32)
  else:
    scale_chunk_size = vol.meta.chunk_size(underlying_mip).astype(np.float32)

  if factor is None:
    factor = axis_to_factor(axis)

  factors = compute_factors(shape, factor, scale_chunk_size, vol.meta.volume_size(mip))
  scales = [ vol.meta.resolution(mip) ]

  precision = max(map(getprecision, vol.meta.resolution(mip)))

  def prec(x):
    if precision == 0:
      return int(x)
    return round(x, precision)

  for factor3 in factors:
    scales.append(
      list(map(prec, Vec(*scales[-1], dtype=np.float32) * Vec(*factor3)))
    )
  return scales[1:]

def create_downsample_scales(
    layer_path, mip, ds_shape, axis='z', 
    preserve_chunk_size=False, chunk_size=None,
    encoding=None, factor=None
  ):
  vol = CloudVolume(layer_path, mip)

  resolutions = compute_scales(vol, mip, ds_shape, axis, factor, chunk_size)

  if len(resolutions) == 0:
    print("WARNING: No scales generated.")

  for res in resolutions:
    vol.meta.add_resolution(res, encoding=encoding, chunk_size=chunk_size)

  if chunk_size is None:
    if preserve_chunk_size or len(resolutions) == 0:
      chunk_size = vol.scales[mip]['chunk_sizes']
    else:
      chunk_size = vol.scales[mip + 1]['chunk_sizes']
  else:
    chunk_size = [ chunk_size ]

  for i in range(mip + 1, mip + len(resolutions) + 1):
    vol.scales[i]['chunk_sizes'] = chunk_size

  vol.commit_info()
  return vol

def add_scale(
  layer_path, mip,
  preserve_chunk_size=True, chunk_size=None,
  encoding=None, factor=None
):
  vol = CloudVolume(layer_path, mip=mip)

  if factor is None:
    factor = (2,2,1)

  new_resolution = vol.resolution * Vec(*factor)
  vol.meta.add_resolution(
    new_resolution, encoding=encoding, chunk_size=chunk_size
  )

  if chunk_size is None:
    if preserve_chunk_size:
      chunk_size = vol.scales[mip]['chunk_sizes']
    else:
      chunk_size = vol.scales[mip + 1]['chunk_sizes']
  else:
    chunk_size = [ chunk_size ]

  if encoding is None:
    encoding = vol.scales[mip]['encoding']

  vol.scales[mip + 1]['chunk_sizes'] = chunk_size

  return vol

def downsample_shape_from_memory_target(
  data_width, cx, cy, cz, 
  factor, byte_target,
  max_mips=float('inf')
):
  """
  Compute the shape that will give the most downsamples for a given 
  memory target (e.g. 3e9 bytes aka 3 GB).

  data_width: byte size of dtype
  cx: chunk size x
  cy: chunk size y 
  cz: chunk size z
  factor: (2,2,1) or (2,2,2) are supported
  byte_target: memory used should be less than this

  Returns: Vec3 shape
  """
  # formulas come from solving the following optimization equations:
  #
  # factor (1,1,1)
  # find integers n and m such that
  # |n * cx - m * cy| is (approximately) minimized
  # treat cz as fixed to make thing easier.
  # We start with a guess that n = sqrt(byte_target / data_width / cx / cy / cz)
  #
  # factor (2,2,1)
  # 4/3 * data_width * cx^(2^n) * cy^(2^n) * cz < byte_target
  #
  # factor (2,2,2)
  # 8/7 * data_width * cx^(2^n) * cy^(2^n) * cz^(2^n) < byte_target
  #
  # it's possible to solve for an arbitrary factor, but more complicated
  # and we really only need those two as the blowup gets intense.
  if byte_target <= 0:
    raise ValueError(f"Unable to pick a shape for a byte budget <= 0. Got: {byte_target}")

  if cx * cy * cz <= 0:
    raise ValueError(f"Chunk size must have a positive integer volume. Got: <{cx},{cy},{cz}>")

  def n_shape(n, c_):
    num_downsamples = int(math.log2((c_ ** (2*n)) / c_))
    num_downsamples = int(min(num_downsamples, max_mips))
    return c_ * (2 ** num_downsamples)

  if factor == (1,1,1):
    n = int(math.sqrt(byte_target / data_width / cx / cy / cz))
    m = int(n * cx / cy)
    out = Vec(n * cx, m * cy, cz)
  elif factor == (2,2,1):
    if cx * cy == 1:
      size = 2 ** int(math.log2(math.sqrt(byte_target / cz)))
      out = Vec(size, size, cz)
    else:
      n = math.log(3/4 * byte_target / data_width / cz)
      n = n / 2 / math.log(cx * cy)
      shape = lambda c_: n_shape(n, c_) 
      out = Vec(shape(cx), shape(cy), cz)
  elif factor == (2,2,2):
    if cx * cy * cz == 1:
      size = 2 ** int(math.log2(round(byte_target ** (1/3), 5)))
      out = Vec(size, size, size)
    else:
      n = math.log(7/8 * byte_target / data_width)
      n = n / 2 / math.log(cx * cy * cz) 
      shape = lambda c_: n_shape(n, c_) 
      out = Vec(shape(cx), shape(cy), shape(cz))
  else:
    raise ValueError(f"This is now a harder optimization problem. Got: {factor}")

  out = out.astype(int)
  min_shape = Vec(cx,cy,cz)
  if any(out < min_shape):
    raise ValueError(
      f"Too little memory allocated to create a valid task."
      f" Got: {byte_target} Predicted Shape: {out} Minimum Shape: {min_shape}"
    )

  return out

