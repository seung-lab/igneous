from collections.abc import Sequence

import json
import math
import os
import random
from typing import Optional, Tuple, cast

import numpy as np
from tqdm import tqdm

from cloudfiles import CloudFiles
from taskqueue.registered_task import RegisteredTask

from cloudvolume import CloudVolume
from cloudvolume.exceptions import OutOfBoundsError
from cloudvolume.lib import min2, Vec, Bbox, mkdir, jsonify
from cloudvolume.datasource.precomputed.sharding import ShardingSpecification
from cloudvolume.frontends.precomputed import CloudVolumePrecomputed

from taskqueue import RegisteredTask, queueable
import tinybrain

import igneous.shards
from igneous import downsample_scales
from igneous.types import ShapeType

from .obsolete import (
  HyperSquareConsensusTask, WatershedRemapTask,
  MaskAffinitymapTask, InferenceTask
)

def downsample_and_upload(
    image, bounds, vol, ds_shape,
    mip=0, axis='z', skip_first=False,
    sparse=False, factor=None
  ):
    ds_shape = min2(vol.volume_size, ds_shape[:3])
    underlying_mip = (mip + 1) if (mip + 1) in vol.available_mips else mip
    chunk_size = vol.meta.chunk_size(underlying_mip).astype(np.float32)
    
    if factor is None:
      factor = downsample_scales.axis_to_factor(axis)
    factors = downsample_scales.compute_factors(ds_shape, factor, chunk_size, vol.volume_size)

    if len(factors) == 0:
      print("No factors generated. Image Shape: {}, Downsample Shape: {}, Volume Shape: {}, Bounds: {}".format(
          image.shape, ds_shape, vol.volume_size, bounds)
      )

    vol.mip = mip
    if not skip_first:
      vol[bounds.to_slices()] = image

    if len(factors) == 0:
      return

    num_mips = len(factors)

    mips = []
    if vol.layer_type == 'image':
      mips = tinybrain.downsample_with_averaging(
        image, factors[0], 
        num_mips=num_mips, sparse=sparse
      )
    elif vol.layer_type == 'segmentation':
      mips = tinybrain.downsample_segmentation(
        image, factors[0],
        num_mips=num_mips, sparse=sparse
      )
    else:
      mips = tinybrain.downsample_with_striding(image, factors[0], num_mips=num_mips)

    new_bounds = bounds.clone()

    for factor3 in factors:
      vol.mip += 1
      new_bounds //= factor3
      mipped = mips.pop(0)
      new_bounds.maxpt = new_bounds.minpt + Vec(*mipped.shape[:3])
      vol[new_bounds] = mipped

@queueable
def DeleteTask(layer_path:str, shape, offset, mip=0, num_mips=5):
  """Delete a block of images inside a layer on all mip levels."""
  shape = Vec(*shape)
  offset = Vec(*offset)
  vol = CloudVolume(layer_path, mip=mip, max_redirects=0)

  highres_bbox = Bbox( offset, offset + shape )

  top_mip = min(vol.available_mips[-1], mip + num_mips)

  for mip_i in range(mip, top_mip + 1):
    vol.mip = mip_i
    bbox = vol.bbox_to_mip(highres_bbox, mip, mip_i)
    bbox = bbox.round_to_chunk_size(vol.chunk_size, offset=vol.bounds.minpt)
    bbox = Bbox.clamp(bbox, vol.bounds)

    if bbox.volume() == 0:
      continue

    vol.delete(bbox)

@queueable
def BlackoutTask(
  cloudpath, mip, shape, offset,
  value=0, non_aligned_writes=False
):
  shape = Vec(*shape)
  offset = Vec(*offset)
  vol = CloudVolume(cloudpath, mip, non_aligned_writes=non_aligned_writes)
  bounds = Bbox(offset, shape + offset)
  bounds = Bbox.clamp(bounds, vol.bounds)
  img = np.zeros(bounds.size3(), dtype=vol.dtype) + value
  vol[bounds] = img

@queueable
def TouchTask(cloudpath, mip, shape, offset):
  # This could be made more sophisticated using exists
  vol = CloudVolume(cloudpath, mip, fill_missing=False)
  bounds = Bbox(offset, shape + offset)
  bounds = Bbox.clamp(bounds, vol.bounds)
  image = vol[bounds]

@queueable
def QuantizeTask(
  source_layer_path, dest_layer_path, 
  shape, offset, mip, fill_missing=False
):
  shape = Vec(*shape)
  offset = Vec(*offset)
  srcvol = CloudVolume(source_layer_path, mip=mip,
                       fill_missing=fill_missing)

  bounds = Bbox(offset, shape + offset)
  bounds = Bbox.clamp(bounds, srcvol.bounds)

  image = srcvol[bounds.to_slices()][:, :, :, :1]  # only use x affinity
  image = (image * 255.0).astype(np.uint8)

  destvol = CloudVolume(dest_layer_path, mip=mip)
  downsample_and_upload(image, bounds, destvol, shape, mip=mip, axis='z')

class ContrastNormalizationTask(RegisteredTask):
  """TransferTask + Contrast Correction based on LuminanceLevelsTask output."""
  # translate = change of origin

  def __init__(
    self, src_path, dest_path, levels_path, shape,
    offset, mip, clip_fraction, fill_missing,
    translate, minval, maxval
  ):

    super(ContrastNormalizationTask, self).__init__(
      src_path, dest_path, levels_path, shape, offset,
      mip, clip_fraction, fill_missing, translate,
      minval, maxval
    )
    self.src_path = src_path
    self.dest_path = dest_path
    self.shape = Vec(*shape)
    self.offset = Vec(*offset)
    self.fill_missing = fill_missing
    self.translate = Vec(*translate)
    self.mip = int(mip)
    if isinstance(clip_fraction, Sequence):
      assert len(clip_fraction) == 2
      self.lower_clip_fraction = float(clip_fraction[0])
      self.upper_clip_fraction = float(clip_fraction[1])
    else:
      self.lower_clip_fraction = self.upper_clip_fraction = float(clip_fraction)

    self.minval = minval
    self.maxval = maxval

    self.levels_path = levels_path if levels_path else self.src_path

    assert 0 <= self.lower_clip_fraction <= 1
    assert 0 <= self.upper_clip_fraction <= 1
    assert self.lower_clip_fraction + self.upper_clip_fraction <= 1

  def execute(self):
    srccv = CloudVolume(
        self.src_path, fill_missing=self.fill_missing, mip=self.mip)
    destcv = CloudVolume(
        self.dest_path, fill_missing=self.fill_missing, mip=self.mip)

    bounds = Bbox(self.offset, self.shape[:3] + self.offset)
    bounds = Bbox.clamp(bounds, srccv.bounds)
    image = srccv[bounds.to_slices()].astype(np.float32)

    zlevels = self.fetch_z_levels(bounds)

    nbits = np.dtype(srccv.dtype).itemsize * 8
    maxval = float(2 ** nbits - 1)

    for z in range(bounds.minpt.z, bounds.maxpt.z):
      imagez = z - bounds.minpt.z
      zlevel = zlevels[imagez]
      (lower, upper) = self.find_section_clamping_values(
          zlevel, self.lower_clip_fraction, 1 - self.upper_clip_fraction)
      if lower == upper:
        continue
      img = image[:, :, imagez]
      img = (img - float(lower)) * (maxval / (float(upper) - float(lower)))
      image[:, :, imagez] = img

    image = np.round(image)

    minval = self.minval if self.minval is not None else 0.0
    maxval = self.maxval if self.maxval is not None else maxval

    image = np.clip(image, minval, maxval).astype(destcv.dtype)

    bounds += self.translate
    downsample_and_upload(image, bounds, destcv, self.shape, mip=self.mip)

  def find_section_clamping_values(self, zlevel, lowerfract, upperfract):
    filtered = np.copy(zlevel)

    # remove pure black from frequency counts as
    # it has no information in our images
    filtered[0] = 0

    cdf = np.zeros(shape=(len(filtered),), dtype=np.uint64)
    cdf[0] = filtered[0]
    for i in range(1, len(filtered)):
      cdf[i] = cdf[i - 1] + filtered[i]

    total = cdf[-1]

    if total == 0:
      return (0, 0)

    lower = 0
    for i, val in enumerate(cdf):
      if float(val) / float(total) > lowerfract:
        break
      lower = i

    upper = 0
    for i, val in enumerate(cdf):
      if float(val) / float(total) > upperfract:
        break
      upper = i

    return (lower, upper)

  def fetch_z_levels(self, bounds):
    cf = CloudFiles(self.levels_path)

    levelfilenames = [
      cf.join('levels', f"{self.mip}", f"{z}")
      for z in range(bounds.minpt.z, bounds.maxpt.z)
    ]

    levels = cf.get(levelfilenames)

    errors = [
      level['path'] \
      for level in levels if level['content'] == None
    ]

    if len(errors):
      raise Exception(", ".join(
          errors) + " were not defined. Did you run a LuminanceLevelsTask for these slices?")

    levels = [(
      int(os.path.basename(item['path'])),
      json.loads(item['content'].decode('utf-8'))
    ) for item in levels ]

    levels.sort(key=lambda x: x[0])
    levels = [x[1] for x in levels]
    return [ np.array(x['levels'], dtype=np.uint64) for x in levels ]


class LuminanceLevelsTask(RegisteredTask):
  """Generate a frequency count of luminance values by random sampling. Output to $PATH/levels/$MIP/$Z"""

  def __init__(self, src_path, levels_path, shape, offset, coverage_factor, mip):
    super(LuminanceLevelsTask, self).__init__(
      src_path, levels_path, shape,
      offset, coverage_factor, mip
    )
    self.src_path = src_path
    self.shape = Vec(*shape)
    self.offset = Vec(*offset)
    self.coverage_factor = coverage_factor
    self.mip = int(mip)
    self.levels_path = levels_path

    assert 0 < coverage_factor <= 1, "Coverage Factor must be between 0 and 1"

  def execute(self):
    srccv = CloudVolume(self.src_path, mip=self.mip, fill_missing=True)

    # Accumulate a histogram of the luminance levels
    nbits = np.dtype(srccv.dtype).itemsize * 8
    levels = np.zeros(shape=(2 ** nbits,), dtype=np.uint64)

    bounds = Bbox(self.offset, self.shape[:3] + self.offset)
    bounds = Bbox.clamp(bounds, srccv.bounds)

    bboxes = self.select_bounding_boxes(bounds)
    for bbox in bboxes:
      img2d = srccv[bbox.to_slices()].reshape((bbox.volume()))
      cts = np.bincount(img2d)
      levels[0:len(cts)] += cts.astype(np.uint64)

    covered_area = sum([bbx.volume() for bbx in bboxes])

    bboxes = [(bbox.volume(), bbox.size3()) for bbox in bboxes]
    bboxes.sort(key=lambda x: x[0])
    biggest = bboxes[-1][1]

    output = {
      "levels": levels.tolist(),
      "patch_size": biggest.tolist(),
      "num_patches": len(bboxes),
      "coverage_ratio": covered_area / self.shape.rectVolume(),
    }

    path = self.levels_path if self.levels_path else self.src_path
    path = os.path.join(path, 'levels')

    cf = CloudFiles(path)
    cf.put_json(
      path="{}/{}".format(self.mip, self.offset.z),
      content=output,
      cache_control='no-cache'
    )

  def select_bounding_boxes(self, dataset_bounds):
    # Sample patches until coverage factor is satisfied.
    # Ensure the patches are non-overlapping and random.
    sample_shape = Bbox((0, 0, 0), (2048, 2048, 1))
    area = self.shape.rectVolume()

    total_patches = int(math.ceil(area / sample_shape.volume()))
    N = int(math.ceil(float(total_patches) * self.coverage_factor))

    # Simplification: We are making patch selection against a discrete
    # grid instead of a continuous space. This removes the influence of
    # overlap in a less complex fashion.
    patch_indicies = set()
    while len(patch_indicies) < N:
      ith_patch = random.randint(0, (total_patches - 1))
      patch_indicies.add(ith_patch)

    gridx = int(math.ceil(self.shape.x / sample_shape.size3().x))

    bboxes = []
    for i in patch_indicies:
      patch_start = Vec(i % gridx, i // gridx, 0)
      patch_start *= sample_shape.size3()
      patch_start += self.offset
      bbox = Bbox(patch_start, patch_start + sample_shape.size3())
      bbox = Bbox.clamp(bbox, dataset_bounds)
      bboxes.append(bbox)
    return bboxes

@queueable
def TransferTask(
  src_path, dest_path,
  mip, shape, offset,
  translate=(0,0,0),
  fill_missing=False,
  skip_first=False,
  skip_downsamples=False,
  delete_black_uploads=False,
  background_color=0,
  sparse=False,
  axis='z',
  agglomerate=False,
  timestamp=None,
  compress='gzip',
  factor=None
):
  """
  Transfer an image to a new location while enabling
  rechunking, translation, reencoding, recompressing,
  and downsampling. For graphene, we can also generate
  proofread segmentation using the agglomerate flag.
  """
  shape = Vec(*shape)
  offset = Vec(*offset)
  fill_missing = bool(fill_missing)
  translate = Vec(*translate)
  delete_black_uploads = bool(delete_black_uploads)
  sparse = bool(sparse)
  skip_first = bool(skip_first)
  skip_downsamples = bool(skip_downsamples)

  src_cv = CloudVolume(
    src_path, fill_missing=fill_missing,
    mip=mip, bounded=False
  )
  dest_cv = CloudVolume(
    dest_path, fill_missing=fill_missing,
    mip=mip, delete_black_uploads=delete_black_uploads,
    background_color=background_color, compress=compress
  )

  dst_bbox = Bbox(offset, shape + offset)
  dst_bbox = Bbox.clamp(dst_bbox, dest_cv.bounds)

  if (
    skip_downsamples
    and agglomerate == False
    and src_cv.scale == dest_cv.scale
    and src_cv.dtype == dest_cv.dtype
    and np.all(translate == (0,0,0))
  ):
    # most efficient transfer type, just copy
    # files possibly without even decompressing
    src_cv.image.transfer_to(
      dest_path, dst_bbox, mip,
      compress=compress
    )
    return

  src_bbox = dst_bbox - translate
  image = src_cv.download(
    src_bbox, agglomerate=agglomerate, timestamp=timestamp
  )

  if skip_downsamples:
    dest_cv[dst_bbox] = image
  else:
    downsample_and_upload(
      image, dst_bbox, dest_cv,
      shape, mip=mip,
      skip_first=skip_first,
      sparse=sparse, axis=axis,
      factor=factor
    )

@queueable
def DownsampleTask(
  layer_path, mip, shape, offset,
  fill_missing=False, axis='z', sparse=False,
  delete_black_uploads=False, background_color=0,
  dest_path=None, compress="gzip", factor=None
):
  """
  Downsamples a cutout of the volume. By default it performs
  2x2x1 downsamples along the specified axis. The factor argument
  overrrides this functionality.
  """
  if dest_path is None:
    dest_path = layer_path

  return TransferTask(
    layer_path, dest_path,
    mip, shape, offset,
    translate=(0,0,0),
    fill_missing=fill_missing,
    skip_first=True,
    skip_downsamples=False,
    delete_black_uploads=delete_black_uploads,
    background_color=background_color,
    sparse=sparse,
    axis=axis,
    compress=compress,
    factor=factor,
  )

@queueable
def ImageShardTransferTask(
  src_path: str,
  dst_path: str,
  shape: ShapeType,
  offset: ShapeType,
  mip: int = 0,
  fill_missing: bool = False,
  translate: ShapeType = (0, 0, 0),
  agglomerate: bool = False,
  timestamp: Optional[int] = None,
):
  """
  Generates a sharded image volume from
  a preexisting CloudVolume readable data 
  source. Downsamples are not generated.

  The sharded specification can be read here:
  Shard Container: 
  https://github.com/google/neuroglancer/blob/056a3548abffc3c76c93c7a906f1603ce02b5fa3/src/neuroglancer/datasource/precomputed/sharded.md
  Sharded Images:    
  https://github.com/google/neuroglancer/blob/056a3548abffc3c76c93c7a906f1603ce02b5fa3/src/neuroglancer/datasource/precomputed/volume.md#unsharded-chunk-storage
  """
  shape = Vec(*shape)
  offset = Vec(*offset)
  mip = int(mip)
  fill_missing = bool(fill_missing)
  translate = Vec(*translate)

  src_vol = CloudVolume(
    src_path, fill_missing=fill_missing, 
    mip=mip, bounded=False
  )
  dst_vol = CloudVolume(
    dst_path,
    fill_missing=fill_missing,
    mip=mip,
    compress=None
  )

  dst_bbox = Bbox(offset, offset + shape)
  dst_bbox = Bbox.clamp(dst_bbox, dst_vol.meta.bounds(mip))
  dst_bbox = dst_bbox.expand_to_chunk_size(
    dst_vol.meta.chunk_size(mip), 
    offset=dst_vol.meta.voxel_offset(mip)
  )
  src_bbox = dst_bbox - translate

  img = src_vol.download(
    src_bbox, agglomerate=agglomerate, timestamp=timestamp
  )
  (filename, shard) = dst_vol.image.make_shard(
    img, dst_bbox, mip, progress=False
  )
  del img

  basepath = dst_vol.meta.join(
    dst_vol.cloudpath, dst_vol.meta.key(mip)
  )

  CloudFiles(basepath).put(filename, shard)

@queueable
def ImageShardDownsampleTask(
  src_path: str,
  shape: ShapeType,
  offset: ShapeType,
  mip: int = 0,
  fill_missing: bool = False,
  sparse: bool = False,
  agglomerate: bool = False,
  timestamp: Optional[int] = None,
  factor: ShapeType = (2,2,1)
):
  """
  Generate a single downsample level for a shard.
  Shards are usually hundreds of megabytes to several
  gigabyte of data, so it is usually unrealistic from a
  memory perspective to make more than one mip at a time.
  """
  shape = Vec(*shape)
  offset = Vec(*offset)
  mip = int(mip)
  fill_missing = bool(fill_missing)

  src_vol = CloudVolume(
    src_path, fill_missing=fill_missing, 
    mip=mip, bounded=False, progress=False
  )
  chunk_size = src_vol.meta.chunk_size(mip)

  bbox = Bbox(offset, offset + shape)
  bbox = Bbox.clamp(bbox, src_vol.meta.bounds(mip))
  bbox = bbox.expand_to_chunk_size(
    chunk_size, offset=src_vol.meta.voxel_offset(mip)
  )

  shard_shape = igneous.shards.image_shard_shape_from_spec(
    src_vol.scales[mip + 1]["sharding"], 
    src_vol.meta.volume_size(mip + 1), 
    src_vol.meta.chunk_size(mip + 1)
  )
  upper_offset = offset // Vec(*factor)
  shape_bbox = Bbox(upper_offset, upper_offset + shard_shape)
  shape_bbox = shape_bbox.astype(np.int64)
  shape_bbox = Bbox.clamp(shape_bbox, src_vol.meta.bounds(mip + 1))
  shape_bbox = shape_bbox.expand_to_chunk_size(src_vol.meta.chunk_size(mip + 1))

  if shape_bbox.subvoxel():
    return

  shard_shape = list(shape_bbox.size3()) + [ 1 ]

  output_img = np.zeros(shard_shape, dtype=src_vol.dtype)
  nz = int(math.ceil(bbox.dz / chunk_size.z))

  dsfn = tinybrain.downsample_with_averaging
  if src_vol.layer_type == "segmentation":
    dsfn = tinybrain.downsample_segmentation

  zbox = bbox.clone()
  zbox.maxpt.z = zbox.minpt.z + chunk_size.z
  for z in range(nz):
    img = src_vol.download(
      zbox, agglomerate=agglomerate, timestamp=timestamp
    )
    (ds_img,) = dsfn(img, factor, num_mips=1, sparse=sparse)
    # ds_img[slc] b/c sometimes the size round up in tinybrain
    # makes this too large by one voxel on an axis
    output_img[:,:,(z*chunk_size.z):(z+1)*chunk_size.z] = ds_img

    del img
    del ds_img
    zbox.minpt.z += chunk_size.z
    zbox.maxpt.z += chunk_size.z

  (filename, shard) = src_vol.image.make_shard(
    output_img, shape_bbox, (mip + 1), progress=False
  )
  basepath = src_vol.meta.join(
    src_vol.cloudpath, src_vol.meta.key(mip + 1)
  )
  CloudFiles(basepath).put(filename, shard)
