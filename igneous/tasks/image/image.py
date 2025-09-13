from collections.abc import Sequence
from collections import defaultdict

from functools import partial
import json
import math
import os
import random
from typing import Optional, Tuple, cast, Iterable, Union

import numpy as np
from tqdm import tqdm

import fastremap
from mapbuffer import MapBuffer

from cloudfiles import CloudFiles, CloudFile
from taskqueue.registered_task import RegisteredTask

from cloudvolume import CloudVolume
from cloudvolume.exceptions import OutOfBoundsError
from cloudvolume.lib import min2, Vec, Bbox, mkdir, jsonify
from cloudvolume.datasource.precomputed.sharding import ShardingSpecification

from taskqueue import RegisteredTask, queueable
import tinybrain

import igneous.shards
from igneous import downsample_scales
from igneous.types import ShapeType, DownsampleMethods

from .obsolete import (
  HyperSquareConsensusTask, WatershedRemapTask,
  MaskAffinitymapTask, InferenceTask
)

def downsample_method_to_fn(method, sparse, vol):
  if method == DownsampleMethods.AUTO:
    if vol.layer_type == 'image':
      method = DownsampleMethods.AVERAGE_POOLING
    elif vol.layer_type == 'segmentation':
      method = DownsampleMethods.MODE_POOLING
    else:
      method = DownsampleMethods.STRIDING

  if method == DownsampleMethods.MIN_POOLING:
    return tinybrain.downsample_with_min_pooling
  elif method == DownsampleMethods.MAX_POOLING:
    return tinybrain.downsample_with_max_pooling
  elif method == DownsampleMethods.AVERAGE_POOLING:
    return partial(tinybrain.downsample_with_averaging, sparse=sparse)
  elif method == DownsampleMethods.MODE_POOLING:
    return partial(tinybrain.downsample_segmentation, sparse=sparse)
  else:
    return tinybrain.downsample_with_striding

def downsample_and_upload(
  image, bounds, vol, ds_shape,
  mip=0, axis='z', skip_first=False,
  sparse=False, factor=None, max_mips=None,
  method=DownsampleMethods.AUTO,
):
    ds_shape = min2(vol.volume_size, ds_shape[:3])
    underlying_mip = (mip + 1) if (mip + 1) in vol.available_mips else mip
    chunk_size = vol.meta.chunk_size(underlying_mip).astype(np.float32)
    
    if factor is None:
      factor = downsample_scales.axis_to_factor(axis)
    factors = downsample_scales.compute_factors(ds_shape, factor, chunk_size, vol.volume_size)

    if max_mips is not None:
      factors = factors[:max_mips]

    if len(factors) == 0 and max_mips:
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

    fn = downsample_method_to_fn(method, sparse, vol)
    mips = fn(image, factors[0], num_mips=num_mips)
        
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

@queueable
def CLAHETask(
  src:str, 
  dest:str, 
  mip:int, 
  fill_missing:bool,
  shape:tuple[int,int,int], 
  offset:tuple[int,int,int],
  clip_limit:float = 40.0, 
  tile_grid_size:tuple[int,int] = (8,8),
):
  import cv2

  # OpenCV will "try" to use the specified number of threads. 0 means run sequentially.
  # Igneous uses a multiprocessing model, so threads just fight with each other.
  # https://docs.opencv.org/3.4/db/de0/group__core__utils.html#gae78625c3c2aa9e0b83ed31b73c6549c0
  cv2.setNumThreads(0)

  shape = Vec(*shape)
  offset = Vec(*offset)

  clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

  src_cv = CloudVolume(src, mip=mip, fill_missing=fill_missing)

  bounds = Bbox(offset, shape + offset)
  bounds = Bbox.clamp(bounds, src_cv.bounds)

  overlapped_bbx = bounds.clone()
  overlapped_bbx.minpt.x -= tile_grid_size[0]
  overlapped_bbx.maxpt.x += tile_grid_size[0]
  overlapped_bbx.minpt.y -= tile_grid_size[1]
  overlapped_bbx.maxpt.y += tile_grid_size[1]
  overlapped_bbx = Bbox.clamp(overlapped_bbx, src_cv.bounds)

  image_stack = src_cv[overlapped_bbx][...,0]

  for z in range(image_stack.shape[2]):
    image_stack[:,:,z] = clahe.apply(image_stack[:,:,z])

  crop_bbx = bounds.clone()
  crop_bbx.minpt -= overlapped_bbx.minpt
  crop_bbx.maxpt -= overlapped_bbx.minpt

  dest_cv = CloudVolume(dest, mip=mip)
  dest_cv[bounds] = image_stack[crop_bbx.to_slices()]

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

    if len(bboxes) == 0:
      return

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
      if not bbox.subvoxel():
        bboxes.append(bbox)
    return bboxes

@queueable
def TransferTask(
  src_path:str, dest_path:str,
  mip:int, shape, offset,
  translate=(0,0,0),
  fill_missing:bool = False,
  skip_first:bool = False,
  skip_downsamples:bool = False,
  delete_black_uploads:bool = False,
  background_color:int = 0,
  sparse:bool = False,
  axis:chr = 'z',
  agglomerate:bool = False,
  timestamp:Optional[int] = None,
  compress='gzip',
  factor=None,
  max_mips:Optional[int] = None,
  stop_layer:Optional[int] = None,
  downsample_method:str = DownsampleMethods.AUTO,
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
    background_color=background_color, compress=compress,
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
    src_bbox, 
    agglomerate=agglomerate, 
    timestamp=timestamp,
    stop_layer=stop_layer,
  )

  if skip_downsamples:
    dest_cv[dst_bbox] = image
  else:
    downsample_and_upload(
      image, dst_bbox, dest_cv,
      shape, mip=mip,
      skip_first=skip_first,
      sparse=sparse, axis=axis,
      factor=factor, max_mips=max_mips,
      method=downsample_method,
    )

@queueable
def DownsampleTask(
  layer_path, mip, shape, offset,
  fill_missing=False, axis='z', sparse=False,
  delete_black_uploads=False, background_color=0,
  dest_path=None, compress="gzip", factor=None,
  max_mips=None, method=DownsampleMethods.AUTO,
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
    max_mips=max_mips,
    downsample_method=DownsampleMethods.AUTO,
  )

@queueable
def ReorderTask(
  src:str, dest:str,
  mip:int, 
  mapping:dict,
  shape:Iterable[int], 
  offset:Iterable[int],
  fill_missing:bool = False,
  delete_black_uploads:bool = False,
  background_color:int = 0,
  compress:Union[str,bool] = 'br',
):
  shape = Vec(*shape)
  offset = Vec(*offset)
  mapping = { int(k):int(v) for k,v, in mapping.items() }

  src_cv = CloudVolume(
    src, mip=mip,
    fill_missing=fill_missing,
  )
  dest_cv = CloudVolume(
    dest, fill_missing=bool(fill_missing),
    mip=mip, 
    delete_black_uploads=delete_black_uploads,
    background_color=int(background_color), 
    compress=compress,
  )

  bbox = Bbox(offset, shape + offset)
  bbox = Bbox.clamp(bbox, src_cv.bounds)

  img = src_cv[bbox]
  zbox = bbox.clone()
  if len(mapping) == 0:
    dest_cv[zbox] = img
    return

  for z in range(img.shape[2]):
    img_slc = img[:,:,z]
    new_z = mapping.get(z, z)
    zbox.minpt.z = bbox.minpt.z + z
    zbox.maxpt.z = zbox.minpt.z + 1
    dest_cv[zbox] = img_slc

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
  stop_layer: Optional[int] = None,
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
    compress=None,
  )

  dst_bbox = Bbox(offset, offset + shape)
  dst_bbox = Bbox.clamp(dst_bbox, dst_vol.meta.bounds(mip))
  dst_bbox = dst_bbox.expand_to_chunk_size(
    dst_vol.meta.chunk_size(mip), 
    offset=dst_vol.meta.voxel_offset(mip)
  )
  src_bbox = dst_bbox - translate

  fullpathfn = lambda vol, fname: vol.meta.join(vol.cloudpath, vol.meta.key(mip), fname)

  if (
    src_vol.scale == dst_vol.scale 
    and src_bbox == dst_bbox 
    and agglomerate == False
  ):
    filename = dst_vol.image.shard_filename(dst_bbox, mip=mip)
    dst_fullpath = fullpathfn(dst_vol, filename)
    src_fullpath = fullpathfn(src_vol, filename)
    CloudFile(dst_fullpath).transfer_from(src_fullpath)
  else:
    img = src_vol.download(
      src_bbox, 
      agglomerate=agglomerate, 
      timestamp=timestamp,
      stop_layer=stop_layer,
    )
    (filename, shard) = dst_vol.image.make_shard(
      img, dst_bbox, mip, progress=False
    )
    del img

    dst_fullpath = fullpathfn(dst_vol, filename)
    CloudFile(dst_fullpath).put(shard)

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
  factor: ShapeType = (2,2,1),
  method: int = DownsampleMethods.AUTO,
  num_mips: int = 1,
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

  def shard_shapefn(mip):
    return igneous.shards.image_shard_shape_from_spec(
      src_vol.scales[mip]["sharding"], 
      src_vol.meta.volume_size(mip), 
      src_vol.meta.chunk_size(mip)
    )    

  shard_shape = shard_shapefn(mip + 1)
  upper_offset = offset // Vec(*factor)
  shape_bbox = Bbox(upper_offset, upper_offset + shard_shape)
  shape_bbox = shape_bbox.astype(np.int64)
  shape_bbox = Bbox.clamp(shape_bbox, src_vol.meta.bounds(mip + 1))

  if shape_bbox.subvoxel():
    return

  output_shards_by_mip = []
  for mip_i in range(mip + 1, mip + num_mips + 1):
    output_shards_by_mip.append(defaultdict(dict))

  cz = chunk_size.z * (factor[2] ** num_mips)
  nz = int(math.ceil(bbox.dz / cz))

  dsfn = downsample_method_to_fn(method, sparse, src_vol)

  zbox = bbox.clone()
  zbox.maxpt.z = zbox.minpt.z + cz

  # Need to save memory for segmentation... it's big.
  renumber = src_vol.layer_type == "segmentation"

  for z in range(nz):
    if renumber:
      img, mapping = src_vol.download(
        zbox, 
        agglomerate=agglomerate, 
        timestamp=timestamp,
        renumber=True,
      )
      mapping = { v:k for k,v in mapping.items() }
    else:
      img = src_vol.download(
        zbox, 
        agglomerate=agglomerate, 
        timestamp=timestamp,
      )

    ds_imgs = dsfn(img, factor, num_mips=num_mips, sparse=sparse)
    del img

    factor = np.array(factor, dtype=int)

    for i in range(num_mips):
      shard_shape = shard_shapefn(mip + i + 1)

      num_x_shards = int(factor[0] ** (num_mips - i))
      num_y_shards = int(factor[1] ** (num_mips - i))
      
      num_x_shards = int(min(num_x_shards, np.ceil(zbox.size().x / shard_shape[0])) // (2**i))
      num_y_shards = int(min(num_y_shards, np.ceil(zbox.size().y / shard_shape[1])) // (2**i))

      num_x_shards = int(min(num_x_shards, np.ceil(src_vol.meta.volume_size(mip+i).x / shard_shape[0])))
      num_y_shards = int(min(num_y_shards, np.ceil(src_vol.meta.volume_size(mip+i).y / shard_shape[1])))
      
      num_x_shards = max(num_x_shards, 1)
      num_y_shards = max(num_y_shards, 1)

      shard_z = int(z / shard_shape[2])

      for shard_y in range(num_y_shards):
        for shard_x in range(num_x_shards):
          xoff = int(shard_shape[0] * shard_x)
          yoff = int(shard_shape[1] * shard_y)

          shard_cutout = ds_imgs[i][
            xoff:int(xoff+shard_shape[0]), 
            yoff:int(yoff+shard_shape[1]),
            :
          ]

          if shard_cutout.size == 0:
            continue

          if renumber:
            shard_cutout = fastremap.remap(shard_cutout, mapping)

          shard_bbox = zbox.clone()
          shard_bbox.minpt //= (factor ** (i+1))
          shard_bbox.maxpt //= (factor ** (i+1))

          shard_bbox.minpt += np.array([ xoff, yoff, 0 ])
          shard_bbox.maxpt = shard_bbox.minpt + np.array([
            shard_cutout.shape[0],
            shard_cutout.shape[1],
            shard_cutout.shape[2],
          ])

          if shard_bbox.minpt.z >= src_vol.meta.bounds(i+1).maxpt.z:
            continue

          if renumber:
            shard_cutout = shard_cutout.astype(src_vol.dtype, copy=False)

          chunk_dict = src_vol.image.make_shard_chunks(shard_cutout, shard_bbox, mip + i + 1)
          output_shards_by_mip[i][(shard_x, shard_y, shard_z)].update(chunk_dict)

    del ds_imgs
    zbox.minpt.z += cz
    zbox.maxpt.z += cz

  for i in range(num_mips):
    shard_shape = shard_shapefn(mip + i + 1)

    for (shard_x, shard_y, shard_z), chunk_dict in output_shards_by_mip[i].items():
      minpt = np.array([ shard_x * shard_shape[0], shard_y * shard_shape[1], shard_z * shard_shape[2] ])
      shard_bbox = Bbox(minpt, minpt + shard_shape)
      mip_offset = src_vol.meta.voxel_offset(mip + i + 1)
      shard_bbox.minpt += mip_offset
      shard_bbox.maxpt += mip_offset

      (filename, shard) = src_vol.image.make_shard(
        chunk_dict, shard_bbox, (mip + i + 1), progress=False
      )
      basepath = src_vol.meta.join(
        src_vol.cloudpath, src_vol.meta.key(mip + i + 1)
      )
      CloudFiles(basepath).put(filename, shard)
    output_shards_by_mip[i] = None # Free RAM

@queueable
def CountVoxelsTask(
  cloudpath:str,
  shape:ShapeType,
  offset:ShapeType,
  mip:int = 0,
  fill_missing:bool = False,
  agglomerate:bool = False,
  timestamp:Optional[int] = None
):
  shape = Vec(*shape)
  offset = Vec(*offset)
  mip = int(mip)
  fill_missing = bool(fill_missing)

  cv = CloudVolume(
    cloudpath, fill_missing=fill_missing, 
    mip=mip, bounded=False, progress=False
  )
  bbox = Bbox(offset, offset + shape)
  bbox = Bbox.clamp(bbox, cv.meta.bounds(mip))
  
  labels = cv.download(
    bbox, 
    agglomerate=agglomerate, 
    timestamp=timestamp
  )
  uniq, cts = fastremap.unique(labels, return_counts=True)
  voxel_counts = { str(segid): ct for segid, ct in zip(uniq, cts) }

  cf = CloudFiles(cloudpath)

  cf.put_json(
    cf.join(f'{cv.key}', 'stats', 'voxel_counts', f'{bbox.to_filename()}.json'),
    voxel_counts
  )

