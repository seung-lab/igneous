from functools import reduce, partial
import operator
from typing import Any, Dict, Union, Tuple, cast, Optional, Iterator

import copy
import json
import math
from time import strftime

import fastremap
import numpy as np
from tqdm import tqdm

from cloudfiles import CloudFiles

import cloudvolume
import cloudvolume.exceptions
from cloudvolume import CloudVolume
from cloudvolume.lib import Vec, Bbox, max2, min2, xyzrange, find_closest_divisor, yellow, jsonify

from igneous import downsample_scales
from igneous.tasks import (
  BlackoutTask, QuantizeTask, DeleteTask, 
  ContrastNormalizationTask, DownsampleTask,
  TransferTask, TouchTask, LuminanceLevelsTask,
  HyperSquareConsensusTask, # HyperSquareTask,
  ImageShardTransferTask, ImageShardDownsampleTask,
  CCLFacesTask, CCLEquivalancesTask, RelabelCCLTask
)

from igneous.shards import image_shard_shape_from_spec
from igneous.types import ShapeType

from .common import (
  operator_contact, FinelyDividedTaskIterator, 
  get_bounds, num_tasks, prod
)

__all__  = [
  "create_blackout_tasks",
  "create_touch_tasks",
  "create_downsampling_tasks", 
  "create_image_shard_downsample_tasks",
  "create_deletion_tasks",
  "create_transfer_tasks",
  "create_image_shard_transfer_tasks",
  "create_quantized_affinity_info",
  "create_quantize_tasks",
  "create_hypersquare_ingest_tasks",
  "create_hypersquare_consensus_tasks",
  "create_luminance_levels_tasks",
  "create_contrast_normalization_tasks",
  "create_ccl_face_tasks",
  "create_ccl_equivalence_tasks",
  "create_ccl_relabel_tasks",
]

# A reasonable size for processing large
# image chunks.
MEMORY_TARGET = int(3.5e9)

def create_blackout_tasks(
  cloudpath:str, bounds:Bbox,
  mip:int = 0, shape:ShapeType = (2048, 2048, 64), 
  value:int = 0, non_aligned_writes:bool = False
):

  vol = CloudVolume(cloudpath, mip=mip)

  shape = Vec(*shape)
  bounds = Bbox.create(bounds)
  bounds = vol.bbox_to_mip(bounds, mip=0, to_mip=mip)

  if not non_aligned_writes:
    bounds = bounds.expand_to_chunk_size(vol.chunk_size, vol.voxel_offset)
    
  bounds = Bbox.clamp(bounds, vol.mip_bounds(mip))

  class BlackoutTaskIterator(FinelyDividedTaskIterator):
    def task(self, shape, offset):
      bounded_shape = min2(shape, vol.bounds.maxpt - offset)
      return partial(igneous.tasks.BlackoutTask, 
        cloudpath=cloudpath, 
        mip=mip, 
        shape=shape.clone(), 
        offset=offset.clone(),
        value=value, 
        non_aligned_writes=non_aligned_writes,
      )

    def on_finish(self):
      vol.provenance.processing.append({
        'method': {
          'task': 'BlackoutTask',
          'cloudpath': cloudpath,
          'mip': mip,
          'non_aligned_writes': non_aligned_writes,
          'value': value,
          'shape': shape.tolist(),
          'bounds': [
            bounds.minpt.tolist(),
            bounds.maxpt.tolist(),
          ],
        },
        'by': operator_contact(),
        'date': strftime('%Y-%m-%d %H:%M %Z'),
      })

  return BlackoutTaskIterator(bounds, shape)
  
def create_touch_tasks(
    cloudpath, 
    mip=0, shape=(2048, 2048, 64),
    bounds=None
  ):

  vol = CloudVolume(cloudpath, mip=mip)

  shape = Vec(*shape)

  if bounds is None:
    bounds = vol.bounds.clone()

  bounds = Bbox.create(bounds)
  bounds = vol.bbox_to_mip(bounds, mip=0, to_mip=mip)
  bounds = Bbox.clamp(bounds, vol.mip_bounds(mip))

  class TouchTaskIterator(FinelyDividedTaskIterator):
    def task(self, shape, offset):
      bounded_shape = min2(shape, vol.bounds.maxpt - offset)
      return partial(igneous.tasks.TouchTask,
        cloudpath=cloudpath,
        shape=bounded_shape.clone(),
        offset=offset.clone(),
        mip=mip,
      )

    def on_finish(self):
      vol.provenance.processing.append({
        'method': {
            'task': 'TouchTask',
            'mip': mip,
            'shape': shape.tolist(),
            'bounds': [
              bounds.minpt.tolist(),
              bounds.maxpt.tolist(),
            ],
          },
        'by': operator_contact(),
        'date': strftime('%Y-%m-%d %H:%M %Z'),
      })
      vol.commit_provenance()

  return TouchTaskIterator(bounds, shape)

def create_downsampling_tasks(
    layer_path, mip=0, fill_missing=False, 
    axis='z', num_mips=5, preserve_chunk_size=True,
    sparse=False, bounds=None, chunk_size=None,
    encoding=None, delete_black_uploads=False, 
    background_color=0, dest_path=None, compress=None,
    factor=None, bounds_mip=0
  ):
    """
    mip: Download this mip level, writes to mip levels greater than this one.
    fill_missing: interpret missing chunks as black instead of issuing an EmptyVolumeException
    axis: technically 'x' and 'y' are supported, but no one uses them.
    num_mips: download a block chunk * 2**num_mips size and generate num_mips mips. If you have
      memory problems, try reducing this number.
    preserve_chunk_size: if true, maintain chunk size of starting mip, else, find the closest
      evenly divisible chunk size to 64,64,64 for this shape and use that. The latter can be
      useful when mip 0 uses huge chunks and you want to simply visualize the upper mips.
    chunk_size: (overrides preserve_chunk_size) force chunk size for new layers to be this.
    sparse: When downsampling segmentation, if true, don't count black pixels when computing
      the mode. Useful for e.g. synapses and point labels.
    bounds: By default, downsample everything, but you can specify restricted bounding boxes
      instead. The bounding box will be expanded to the nearest chunk. Bbox is specifed in mip 0
      coordinates.
    delete_black_uploads: issue delete commands instead of upload chunks
      that are all background.
    background_color: Designates which color should be considered background.
    dest_path: (optional) instead of writing downsamples to the existing 
      volume, write them somewhere else. This can be useful e.g. if someone 
      doesn't want you to touch the existing info file.
    compress: None, 'gzip', or 'br' Determines which compression algorithm to use 
      for new uploaded files.
    factor: (overrides axis) can manually specify what each downsampling round is
      supposed to do: e.g. (2,2,1), (2,2,2), etc
    """
    def ds_shape(mip, chunk_size=None, factor=None):
      if chunk_size:
        shape = Vec(*chunk_size)
      else:
        shape = vol.meta.chunk_size(mip)[:3]

      if factor is None:
        factor = downsample_scales.axis_to_factor(axis)

      shape.x *= factor[0] ** num_mips
      shape.y *= factor[1] ** num_mips
      shape.z *= factor[2] ** num_mips
      return shape

    vol = CloudVolume(layer_path, mip=mip)
    shape = ds_shape(mip, chunk_size, factor)

    vol = downsample_scales.create_downsample_scales(
      layer_path, mip, shape, 
      preserve_chunk_size=preserve_chunk_size, chunk_size=chunk_size,
      encoding=encoding, factor=factor
    )

    if not preserve_chunk_size or chunk_size:
      shape = ds_shape(mip + 1, chunk_size, factor)

    bounds = get_bounds(
      vol, bounds, mip,
      bounds_mip=bounds_mip,
      chunk_size=vol.chunk_size,
    )
    
    class DownsampleTaskIterator(FinelyDividedTaskIterator):
      def task(self, shape, offset):
        return partial(DownsampleTask, 
          layer_path=layer_path,
          mip=vol.mip,
          shape=shape.clone(),
          offset=offset.clone(),
          axis=axis,
          fill_missing=fill_missing,
          sparse=sparse,
          delete_black_uploads=delete_black_uploads,
          background_color=background_color,
          dest_path=dest_path,
          compress=compress,
          factor=factor,
        )

      def on_finish(self):
        vol.provenance.processing.append({
          'method': {
            'task': 'DownsampleTask',
            'mip': mip,
            'num_mips': num_mips,
            'shape': shape.tolist(),
            'axis': axis,
            'method': 'downsample_with_averaging' if vol.layer_type == 'image' else 'downsample_segmentation',
            'sparse': sparse,
            'bounds': str(bounds),
            'chunk_size': (list(chunk_size) if chunk_size else None),
            'preserve_chunk_size': preserve_chunk_size,
            'encoding': encoding,
            'fill_missing': bool(fill_missing),
            'delete_black_uploads': bool(delete_black_uploads),
            'background_color': background_color,
            'dest_path': dest_path,
            'compress': compress,
            'factor': (tuple(factor) if factor else None),
          },
          'by': operator_contact(),
          'date': strftime('%Y-%m-%d %H:%M %Z'),
        })
        vol.commit_provenance()

    return DownsampleTaskIterator(bounds, shape)

def create_sharded_image_info(
  dataset_size: ShapeType,
  chunk_size: ShapeType,
  encoding: str,
  dtype: Any,
  uncompressed_shard_bytesize: int = MEMORY_TARGET, 
  max_shard_index_bytes: int = 8192, # 2^13
  max_minishard_index_bytes: int = 40000,
  max_labels_per_minishard: int = 4000
) -> Dict[str, Any]:
  """
  Create a recommended sharding scheme. These recommendations are based
  on the following principles:

  1. Compressed shard sizes should be smaller than 2 GB
  2. Uncompressed shard sizes should be smaller than about 3.5 GB
  3. The number of shard files should be minimized.
  4. The size of the shard index should be small (< ~8 KiB)
  5. The size of the minishard index should be small (< ~32 KiB)
    and each index should contain between hundreds to thousands
    of labels.

  Rationale:

  1. Large file transfers are more difficult to parallelize. Large
    files > 4 GB, or > 5 GB may run into various limits (
    can't be stored on FAT32, needs chunked upload to GCS/S3 which 
    is not supported by every tool.)
  2. Shard construction should fit in a reasonable amount of memory.
  3. Easier to organize a transfer of shards. Shard
     indices are cached efficiently.
  4. Shard indices should not take up significant memory in cache
    and should download quickly on 10 Mbps connections.
  5. Minishard indices should download quickly, but should not be too small
    else the cache becomes useless. The more minishards there are, the larger
    the shard index becomes as well.

  Achieving these goals requires approximate knowledge of the compression 
  ratio and the number of labels per a unit volume.

  Parameters:
    chunk_info: A precomputed info for a SINGLE MIP (e.g. `cv.scales[0]`)
    byte_width: number of bytes in the dtype e.g. uint8: 1, uint64: 8
    labels_per_voxel: A quantity derived experimentally from your dataset.
      Take a representative sample, count the number of labels and divide
      by the number of voxels in the sample. It only needs to be in the 
      ballpark.
  
  Returns: sharding recommendation (if OK, add as `cv.scales[0]['sharding']`)
  """
  if isinstance(dtype, int):
    byte_width = dtype
  elif isinstance(dtype, str) or np.issubdtype(dtype, np.integer):
    byte_width = np.dtype(dtype).itemsize
  else:
    raise ValueError(f"{dtype} must be int, str, or np.integer.")

  voxels = prod(dataset_size)
  chunk_voxels = prod(chunk_size)
  num_chunks = Bbox([0,0,0], dataset_size).num_chunks(chunk_size)

  # maximum amount of information in the morton codes
  grid_size = np.ceil(Vec(*dataset_size) / Vec(*chunk_size)).astype(np.int64)
  max_bits = sum([ math.ceil(math.log2(size)) for size in grid_size ])
  if max_bits > 64:
    raise ValueError(
      f"{max_bits}, more than a 64-bit integer, "
      "would be required to describe the chunk positions "
      "in this dataset. Try increasing the chunk size or "
      "increasing dataset bounds."
      f"Dataset Size: {dataset_size} Chunk Size: {chunk_size}"
    )

  chunks_per_shard = math.ceil(uncompressed_shard_bytesize / (chunk_voxels * byte_width))
  chunks_per_shard = 2 ** int(math.log2(chunks_per_shard))

  if num_chunks < chunks_per_shard:
    chunks_per_shard = 2 ** int(math.ceil(math.log2(num_chunks)))

  # approximate, would need to account for rounding effects to be exact
  # rounding is corrected for via max_bits - pre - mini below.
  num_shards = num_chunks / chunks_per_shard 
  
  def update_bits():
    shard_bits = int(math.ceil(math.log2(num_shards)))
    preshift_bits = int(math.ceil(math.log2(chunks_per_shard)))
    preshift_bits = min(preshift_bits, max_bits - shard_bits)
    return (shard_bits, preshift_bits)
  
  shard_bits, preshift_bits = update_bits()

  # each chunk is one morton code, and so # chunks = # labels
  num_labels_per_minishard = chunks_per_shard
  minishard_bits = 0
  while num_labels_per_minishard > max_labels_per_minishard:
    num_labels_per_minishard /= 2
    minishard_bits += 1

    # 3 fields, each a uint64 with # of labels rows
    minishard_size = 3 * 8 * num_labels_per_minishard
    # two fields, each uint64 for each row w/ 2^minishard bits rows
    shard_index_size = 2 * 8 * (2 ** minishard_bits)

    minishard_index_too_big = (
      minishard_size > max_minishard_index_bytes 
      and minishard_bits > preshift_bits
    )

    if (
      minishard_index_too_big
      or (shard_index_size > max_shard_index_bytes)
    ):
      minishard_bits -= 1
      num_shards *= 2
      shard_bits, preshift_bits = update_bits()

  # preshift_bits + minishard_bits = number of indexable chunks
  # Since we try to hold the number of indexable chunks fixed, we steal
  # from preshift_bits to get space for the minishard bits.
  # We need to make use of the maximum amount of information available
  # in the morton codes, so if there's any slack from rounding, the
  # remainder goes into shard bits.
  preshift_bits = preshift_bits - minishard_bits
  shard_bits = max_bits - preshift_bits - minishard_bits

  if preshift_bits < 0:
    raise ValueError(f"Preshift bits cannot be negative. ({shard_bits}, {minishard_bits}, {preshift_bits}), total info: {max_bits} bits")

  if preshift_bits + shard_bits + minishard_bits > max_bits:
    raise ValueError(f"{preshift_bits} preshift_bits {shard_bits} shard_bits + {minishard_bits} minishard_bits must be <= {max_bits}. Try reducing the number of minishards.")

  data_encoding = "gzip"
  if encoding in ("jpeg", "kempressed", "fpzip"):
    data_encoding = "raw"

  return {
    "@type": "neuroglancer_uint64_sharded_v1",
    "data_encoding": data_encoding,
    "hash": "identity",
    "minishard_bits": minishard_bits,
    "minishard_index_encoding": "gzip",
    "preshift_bits": preshift_bits, 
    "shard_bits": shard_bits,
  }

def create_image_shard_transfer_tasks(
  src_layer_path: str,
  dst_layer_path: str,
  mip: int = 0,
  chunk_size: Optional[ShapeType] = None,
  encoding: bool = None,
  bounds: Optional[Bbox] = None,
  bounds_mip : int = 0,
  fill_missing: bool = False,
  translate: ShapeType = (0, 0, 0),
  dest_voxel_offset: Optional[ShapeType] = None,
  agglomerate: bool = False, 
  timestamp: int = None,
  memory_target: int = MEMORY_TARGET,
  clean_info: bool = False
):
  src_vol = CloudVolume(src_layer_path, mip=mip)

  if dest_voxel_offset:
    dest_voxel_offset = Vec(*dest_voxel_offset, dtype=int)
  else:
    dest_voxel_offset = src_vol.voxel_offset.clone()

  if not chunk_size:
    chunk_size = src_vol.info['scales'][mip]['chunk_sizes'][0]
  chunk_size = Vec(*chunk_size)

  try:
    dest_vol = CloudVolume(dst_layer_path, mip=mip)
  except cloudvolume.exceptions.InfoUnavailableError:
    info = copy.deepcopy(src_vol.info)
    dest_vol = CloudVolume(dst_layer_path, info=info, mip=mip)
    dest_vol.commit_info()

  if dest_voxel_offset is not None:
    dest_vol.scale["voxel_offset"] = dest_voxel_offset

  # If translate is not set, but dest_voxel_offset is then it should naturally be
  # only be the difference between datasets.
  if translate is None:
    translate = dest_vol.voxel_offset - src_vol.voxel_offset # vector pointing from src to dest
  else:
    translate = Vec(*translate) // src_vol.downsample_ratio

  if encoding is not None:
    dest_vol.info['scales'][mip]['encoding'] = encoding
    if encoding == 'compressed_segmentation' and 'compressed_segmentation_block_size' not in dest_vol.info['scales'][mip]:
      dest_vol.info['scales'][mip]['compressed_segmentation_block_size'] = (8,8,8)
  dest_vol.info['scales'] = dest_vol.info['scales'][:mip+1]
  dest_vol.info['scales'][mip]['chunk_sizes'] = [ chunk_size.tolist() ]

  spec = create_sharded_image_info(
    dataset_size=dest_vol.scale["size"], 
    chunk_size=dest_vol.scale["chunk_sizes"][0], 
    encoding=dest_vol.scale["encoding"], 
    dtype=dest_vol.dtype,
    uncompressed_shard_bytesize=memory_target,
  )
  dest_vol.scale["sharding"] = spec
  if clean_info:
    dest_vol.info = clean_xfer_info(dest_vol.info)
  dest_vol.commit_info()

  shape = image_shard_shape_from_spec(spec, dest_vol.scale["size"], chunk_size)

  bounds = get_bounds(
    dest_vol, bounds, mip, 
    bounds_mip=bounds_mip, 
    chunk_size=chunk_size,
  )

  class ImageShardTransferTaskIterator(FinelyDividedTaskIterator):
    def task(self, shape, offset):
      return partial(ImageShardTransferTask,
        src_layer_path,
        dst_layer_path,
        shape=shape,
        offset=offset,
        fill_missing=fill_missing,
        translate=translate,
        mip=mip,
        agglomerate=agglomerate,
        timestamp=timestamp,
      )

    def on_finish(self):
      job_details = {
        "method": {
          "task": "ImageShardTransferTask",
          "src": src_layer_path,
          "dest": dst_layer_path,
          "shape": list(map(int, shape)),
          "fill_missing": fill_missing,
          "translate": list(map(int, translate)),
          "bounds": [bounds.minpt.tolist(), bounds.maxpt.tolist()],
          "mip": mip,
        },
        "by": operator_contact(),
        "date": strftime("%Y-%m-%d %H:%M %Z"),
      }

      dvol = CloudVolume(dst_layer_path)
      dvol.provenance.sources = [src_layer_path]
      dvol.provenance.processing.append(job_details)
      dvol.commit_provenance()

  return ImageShardTransferTaskIterator(bounds, shape)

def create_image_shard_downsample_tasks(
  cloudpath, mip=0, fill_missing=False, 
  sparse=False, chunk_size=None,
  encoding=None, memory_target=MEMORY_TARGET,
  agglomerate=False, timestamp=None,
  factor=(2,2,1), bounds=None, bounds_mip=0
):
  """
  Downsamples an existing image layer that may be
  sharded or unsharded to create a sharded layer.
  
  Only 2x2x1 downsamples are supported for now.
  """
  cv = downsample_scales.add_scale(
    cloudpath, mip, 
    preserve_chunk_size=True, chunk_size=chunk_size,
    encoding=encoding, factor=factor
  )
  cv.mip = mip + 1
  cv.scale["sharding"] = create_sharded_image_info(
    dataset_size=cv.scale["size"], 
    chunk_size=cv.scale["chunk_sizes"][0], 
    encoding=cv.scale["encoding"], 
    dtype=cv.dtype,
    uncompressed_shard_bytesize=int(memory_target),
  )
  cv.commit_info()

  shape = image_shard_shape_from_spec(
    cv.scale["sharding"], cv.volume_size, cv.chunk_size
  )
  shape = Vec(*shape) * factor

  cv.mip = mip
  bounds = get_bounds(
    cv, bounds, mip, 
    bounds_mip=bounds_mip, 
    chunk_size=cv.meta.chunk_size(mip + 1)
  )

  class ImageShardDownsampleTaskIterator(FinelyDividedTaskIterator):
    def task(self, shape, offset):
      return partial(ImageShardDownsampleTask,
        cloudpath,
        shape=tuple(shape),
        offset=tuple(offset),
        mip=int(mip),
        fill_missing=bool(fill_missing),
        sparse=bool(sparse),
        agglomerate=bool(agglomerate),
        timestamp=timestamp,
        factor=tuple(factor),
      )

    def on_finish(self):
      job_details = {
        "method": {
          "task": "ImageShardDownsampleTask",
          "cloudpath": cloudpath,
          "shape": list(map(int, shape)),
          "fill_missing": fill_missing,
          "sparse": bool(sparse),
          "bounds": [bounds.minpt.tolist(), bounds.maxpt.tolist()],
          "mip": mip,
          "agglomerate": agglomerate,
          "timestamp": timestamp,
        },
        "by": operator_contact(),
        "date": strftime("%Y-%m-%d %H:%M %Z"),
      }

      cv.provenance.sources = [cloudpath]
      cv.provenance.processing.append(job_details)
      cv.commit_provenance()

  return ImageShardDownsampleTaskIterator(bounds, shape)

def create_deletion_tasks(
    layer_path, mip=0, num_mips=5, 
    shape=None, bounds=None
  ):
  vol = CloudVolume(layer_path, max_redirects=0)
  
  if shape is None:
    shape = vol.mip_underlying(mip)[:3]
    shape.x *= 2 ** num_mips
    shape.y *= 2 ** num_mips
  else:
    shape = Vec(*shape)

  if not bounds:
    bounds = vol.mip_bounds(mip).clone()

  class DeleteTaskIterator(FinelyDividedTaskIterator):
    def task(self, shape, offset):
      bounded_shape = min2(shape, bounds.maxpt - offset)
      return partial(DeleteTask,
        layer_path=layer_path,
        shape=bounded_shape.clone(),
        offset=offset.clone(),
        mip=mip,
        num_mips=num_mips,
      )

    def on_finish(self):
      vol = CloudVolume(layer_path, max_redirects=0)
      vol.provenance.processing.append({
        'method': {
          'task': 'DeleteTask',
          'mip': mip,
          'num_mips': num_mips,
          'shape': shape.tolist(),
        },
        'by': operator_contact(),
        'date': strftime('%Y-%m-%d %H:%M %Z'),
      })
      vol.commit_provenance()

  return DeleteTaskIterator(bounds, shape)

def clean_xfer_info(info):
  """Removes fields that could interfere with additional processing."""
  info.pop("mesh", None)
  info.pop("meshing", None)
  info.pop("skeletons", None)
  return info

def create_transfer_tasks(
  src_layer_path:str, dest_layer_path:str, 
  chunk_size:ShapeType = None, 
  shape:ShapeType = None, 
  fill_missing:bool = False, 
  translate:ShapeType = None,
  bounds:Optional[Bbox] = None, 
  mip:int = 0, 
  preserve_chunk_size:bool = True,
  encoding=None, 
  skip_downsamples:bool = False,
  delete_black_uploads:bool = False, 
  background_color:int = 0,
  agglomerate:bool = False, 
  timestamp:Optional[int] = None, 
  compress:Union[str,bool] = 'gzip',
  factor:ShapeType = None, 
  sparse:bool = False, 
  dest_voxel_offset:ShapeType = None,
  memory_target:int = MEMORY_TARGET, 
  max_mips:int = 5, 
  clean_info:bool = False, 
  no_src_update:bool = False, 
  bounds_mip:int = 0,
) -> Iterator:
  """
  Transfer data to a new data layer. You can use this operation
  to make changes to the dataset representation as well. For 
  example, you can change the chunk size, compression, bounds,
  and offset.

  Downsamples will be automatically generated while transferring
  unless skip_downsamples is set. The number of downsamples will
  be determined by the chunk size and the task shape.

  bounds: Bbox specified in terms of the destination image and its
    highest resolution.
  translate: Vec3 pointing from source bounds to dest bounds
    and is in terms of the highest resolution of the source image.
    This allows you to compensate for differing voxel offsets
    or enables you to move part of the image to a new location.
  dest_voxel_offset: When creating a new image, move the 
    global coordinate origin to this point. This is commonly
    used to "zero" a newly aligned image (e.g. (0,0,0)) 

  background_color: Designates which color should be considered background.
  chunk_size: (overrides preserve_chunk_size) force chunk size for new layers to be this.
  clean_info: scrub additional fields from the info file that might interfere
    with later processing (e.g. mesh and skeleton related info).
  compress: None, 'gzip', or 'br' Determines which compression algorithm to use 
    for new uploaded files.
  delete_black_uploads: issue delete commands instead of upload chunks
    that are all background.
  encoding: "raw", "jpeg", "compressed_segmentation", "compresso", "fpzip", or "kempressed"
    depending on which kind of data you're dealing with. raw works for everything (no compression) 
    but you might get better compression with another encoding. You can think of encoding as the
    image type-specific first stage of compression and the "compress" flag as the data
    agnostic second stage compressor. For example, compressed_segmentation and gzip work
    well together, but not jpeg and gzip.
  factor: (overrides axis) can manually specify what each downsampling round is
    supposed to do: e.g. (2,2,1), (2,2,2), etc
  fill_missing: Treat missing image tiles as zeroed for both src and dest.
  max_mips: (pairs with memory_target) maximum number of downsamples to generate even
    if the memory budget is large enough for more.
  memory_target: given a task size in bytes, pick the task shape that will produce the 
    maximum number of downsamples. Only works for (2,2,1) or (2,2,2).
  no_src_update: don't update the source's provenance file
  preserve_chunk_size: if true, maintain chunk size of starting mip, else, find the closest
    evenly divisible chunk size to 64,64,64 for this shape and use that. The latter can be
    useful when mip 0 uses huge chunks and you want to simply visualize the upper mips.
  shape: (overrides memory_target) The 3d size of each task. Choose a shape that meets 
    the following criteria unless you're doing something out of the ordinary.
    (a) 2^n multiple of destination chunk size (b) doesn't consume too much memory
    (c) n is related to the downsample factor for each axis, so for a factor of (2,2,1) (default)
      z only needs to be a single chunk, but x and y should be 2, 4, 8,or 16 times the chunk size.
    Remember to multiply 4/3 * shape.x * shape.y * shape.z * data_type to estimate how much memory 
    each task will require. If downsamples are off, you can skip the 4/3. In the future, if chunk
    sizes match we might be able to do a simple file transfer. The problem can be formulated as 
    producing the largest number of downsamples within a given memory target.

    EXAMPLE: destination is uint64 with chunk size (128, 128, 64) with a memory target of
      at most 3GB per task and a downsample factor of (2,2,1).

      The largest number of downsamples is 4 using 2048 * 2048 * 64 sized tasks which will
      use 2.9 GB of memory. The next size up would use 11.5GB and is too big. 

  sparse: When downsampling segmentation, if true, don't count black pixels when computing
    the mode. Useful for e.g. synapses and point labels.

  agglomerate: (graphene only) remap the watershed layer to a proofread segmentation.
  timestamp: (graphene only) integer UNIX timestamp indicating the proofreading state
    to represent.
  """
  src_vol = CloudVolume(src_layer_path, mip=mip)

  if dest_voxel_offset:
    dest_voxel_offset = Vec(*dest_voxel_offset, dtype=int)
  else:
    dest_voxel_offset = src_vol.voxel_offset.clone()

  if factor is None:
    factor = (2,2,1)

  if skip_downsamples:
    factor = (1,1,1)

  if not chunk_size:
    chunk_size = src_vol.info['scales'][mip]['chunk_sizes'][0]
  chunk_size = Vec(*chunk_size)

  try:
    dest_vol = CloudVolume(dest_layer_path, mip=mip)
  except cloudvolume.exceptions.InfoUnavailableError:
    info = copy.deepcopy(src_vol.info)
    dest_vol = CloudVolume(dest_layer_path, info=info, mip=mip)
    dest_vol.commit_info()

  if dest_voxel_offset is not None:
    dest_vol.scale["voxel_offset"] = dest_voxel_offset

  # If translate is not set, but dest_voxel_offset is then it should naturally be
  # only be the difference between datasets.
  if translate is None:
    translate = dest_vol.voxel_offset - src_vol.voxel_offset # vector pointing from src to dest
  else:
    translate = Vec(*translate) // src_vol.downsample_ratio

  if encoding is not None:
    dest_vol.info['scales'][mip]['encoding'] = encoding
    if encoding == 'compressed_segmentation' and 'compressed_segmentation_block_size' not in dest_vol.info['scales'][mip]:
      dest_vol.info['scales'][mip]['compressed_segmentation_block_size'] = (8,8,8)
  dest_vol.info['scales'] = dest_vol.info['scales'][:mip+1]
  dest_vol.info['scales'][mip]['chunk_sizes'] = [ chunk_size.tolist() ]

  if clean_info:
    dest_vol.info = clean_xfer_info(dest_vol.info)

  dest_vol.commit_info()

  if shape is None:
    if memory_target is not None:
      shape = downsample_scales.downsample_shape_from_memory_target(
        np.dtype(src_vol.dtype).itemsize, 
        dest_vol.chunk_size.x, dest_vol.chunk_size.y, dest_vol.chunk_size.z, 
        factor, memory_target, max_mips
      )
    else:
      raise ValueError("Either shape or memory_target must be specified.")

  shape = Vec(*shape)

  if factor[2] == 1:
    shape.z = int(dest_vol.chunk_size.z * round(shape.z / dest_vol.chunk_size.z))

  if not skip_downsamples:
    downsample_scales.create_downsample_scales(dest_layer_path, 
      mip=mip, ds_shape=shape, factor=factor,
      preserve_chunk_size=preserve_chunk_size,
      encoding=encoding
    )

  dest_bounds = get_bounds(
    dest_vol, bounds, mip, 
    bounds_mip=bounds_mip,
    chunk_size=chunk_size
  )

  class TransferTaskIterator(FinelyDividedTaskIterator):
    def task(self, shape, offset):  
      return partial(TransferTask,
        src_path=src_layer_path,
        dest_path=dest_layer_path,
        shape=shape.clone(),
        offset=offset.clone(),
        fill_missing=fill_missing,
        translate=translate,
        mip=mip,
        skip_downsamples=skip_downsamples,
        delete_black_uploads=bool(delete_black_uploads),
        background_color=background_color,
        agglomerate=agglomerate,
        timestamp=timestamp,
        compress=compress,
        factor=factor,
        sparse=sparse,
      )

    def on_finish(self):
      job_details = {
        'method': {
          'task': 'TransferTask',
          'src': src_layer_path,
          'dest': dest_layer_path,
          'shape': list(map(int, shape)),
          'fill_missing': fill_missing,
          'translate': list(map(int, translate)),
          'skip_downsamples': skip_downsamples,
          'delete_black_uploads': bool(delete_black_uploads),
          'background_color': background_color,
          'bounds': [
            dest_bounds.minpt.tolist(),
            dest_bounds.maxpt.tolist()
          ],
          'mip': mip,
          'agglomerate': bool(agglomerate),
          'timestamp': timestamp,
          'compress': compress,
          'encoding': encoding,
          'memory_target': memory_target,
          'factor': (tuple(factor) if factor else None),
          'sparse': bool(sparse),
        },
        'by': operator_contact(),
        'date': strftime('%Y-%m-%d %H:%M %Z'),
      }

      dest_vol = CloudVolume(dest_layer_path)
      dest_vol.provenance.sources = [ src_layer_path ]
      dest_vol.provenance.processing.append(job_details) 
      dest_vol.commit_provenance()

      if not no_src_update and src_vol.meta.path.protocol in ('gs', 's3', 'file'):
        src_vol.provenance.processing.append(job_details)
        src_vol.commit_provenance()

  return TransferTaskIterator(dest_bounds, shape)

def create_contrast_normalization_tasks(
    src_path, dest_path, levels_path=None,
    shape=None, mip=0, clip_fraction=0.01, 
    fill_missing=False, translate=(0,0,0),
    minval=None, maxval=None, bounds=None,
    bounds_mip=0
  ):
  """
  Use the output of luminence levels to contrast
  correct the image by stretching the histogram
  to cover the full range of the data type.
  """
  srcvol = CloudVolume(src_path, mip=mip)
  
  try:
    dvol = CloudVolume(dest_path, mip=mip)
  except Exception: # no info file
    info = copy.deepcopy(srcvol.info)
    dvol = CloudVolume(dest_path, mip=mip, info=info)
    dvol.info['scales'] = dvol.info['scales'][:mip+1]
    dvol.commit_info()

  if bounds is None:
    bounds = srcvol.bounds.clone()

  if shape is None:
    shape = Bbox( (0,0,0), (2048, 2048, 64) )
    shape = shape.shrink_to_chunk_size(dvol.underlying).size3()
    shape = Vec.clamp(shape, (1,1,1), bounds.size3() )
  
  shape = Vec(*shape)

  downsample_scales.create_downsample_scales(dest_path, mip=mip, ds_shape=shape, preserve_chunk_size=True)
  dvol.refresh_info()

  bounds = get_bounds(srcvol, bounds, mip, bounds_mip=bounds_mip)

  class ContrastNormalizationTaskIterator(FinelyDividedTaskIterator):
    def task(self, shape, offset):
      return ContrastNormalizationTask( 
        src_path=src_path, 
        dest_path=dest_path,
        levels_path=levels_path,
        shape=shape.clone(), 
        offset=offset.clone(), 
        clip_fraction=clip_fraction,
        mip=mip,
        fill_missing=fill_missing,
        translate=translate,
        minval=minval,
        maxval=maxval,
      )
    
    def on_finish(self):
      dvol.provenance.processing.append({
        'method': {
          'task': 'ContrastNormalizationTask',
          'src_path': src_path,
          'dest_path': dest_path,
          'shape': Vec(*shape).tolist(),
          'clip_fraction': clip_fraction,
          'mip': mip,
          'translate': Vec(*translate).tolist(),
          'minval': minval,
          'maxval': maxval,
          'bounds': [
            bounds.minpt.tolist(),
            bounds.maxpt.tolist()
          ],
        },
        'by': operator_contact(),
        'date': strftime('%Y-%m-%d %H:%M %Z'),
      }) 
      dvol.commit_provenance()

  return ContrastNormalizationTaskIterator(bounds, shape)

def create_luminance_levels_tasks(
  layer_path, levels_path=None, coverage_factor=0.01, 
  shape=None, offset=None, mip=0, 
  bounds_mip=0, bounds=None
):
  """
  Compute per slice luminance level histogram and write them as
  $layer_path/levels/$z. Each z file looks like:

  {
    "levels": [ 0, 35122, 12, ... ], # 256 indices, index = luminance i.e. 0 is black, 255 is white 
    "patch_size": [ sx, sy, sz ], # metadata on how large the patches were
    "num_patches": 20, # metadata on
    "coverage_ratio": 0.011, # actual sampled area on this slice normalized by ROI size.
  }

  Args:
    layer_path: source image to sample from
    levels_path: which path to write ./levels/ to (default: $layer_path)
    coverage_factor: what fraction of the image to sample

    offset & shape: Allows you to specify an ROI if much of
      the edges are black. Defaults to entire image.
    mip (int): which mip to work with, default maximum resolution
    bounds_mip (int): mip of the input bounds
    bounds (Bbox-like)
  """
  if shape or offset:
    print(yellow(
      "Create Luminance Levels Tasks: Deprecation Notice: "
      "shape and offset parameters are deprecated in favor of the bounds argument."
    ))

  vol = CloudVolume(layer_path, mip=mip)

  if bounds is None:
    bounds = vol.bounds.clone()

  bounds = get_bounds(vol, bounds, mip, bounds_mip=bounds_mip)

  if shape is None and bounds is None:
    shape = Vec(*vol.shape)
    shape.z = 1
  elif bounds is not None:
    shape = Vec(*bounds.size3())
    shape.z = 1

  if not offset:
    offset = bounds.minpt

  offset = Vec(*offset)
  zoffset = offset.clone()

  protocol = vol.meta.path.protocol

  class LuminanceLevelsTaskIterator(object):
    def __len__(self):
      return bounds.maxpt.z - bounds.minpt.z
    def __iter__(self):
      for z in range(bounds.minpt.z, bounds.maxpt.z + 1):
        zoffset.z = z
        yield LuminanceLevelsTask( 
          src_path=layer_path, 
          levels_path=levels_path,
          shape=shape, 
          offset=zoffset, 
          coverage_factor=coverage_factor,
          mip=mip,
        )

      if protocol == 'boss':
        raise StopIteration()

      if levels_path:
        try:
          vol = CloudVolume(levels_path)
        except cloudvolume.exceptions.InfoUnavailableError:
          vol = CloudVolume(levels_path, info=vol.info)
      else:
        vol = CloudVolume(layer_path, mip=mip)
      
      vol.provenance.processing.append({
        'method': {
          'task': 'LuminanceLevelsTask',
          'src': layer_path,
          'levels_path': levels_path,
          'shape': Vec(*shape).tolist(),
          'offset': Vec(*offset).tolist(),
          'bounds': [
            bounds.minpt.tolist(),
            bounds.maxpt.tolist()
          ],
          'coverage_factor': coverage_factor,
          'mip': mip,
        },
        'by': operator_contact(),
        'date': strftime('%Y-%m-%d %H:%M %Z'),
      }) 
      vol.commit_provenance()

  return LuminanceLevelsTaskIterator()

def compute_fixup_offsets(vol, points, shape):
  pts = map(np.array, points)

  # points are specified in high res coordinates 
  # because that's what people read off the screen.
  def nearest_offset(pt):
    mip0offset = (np.floor((pt - vol.mip_voxel_offset(0)) / shape) * shape) + vol.mip_voxel_offset(0)
    return mip0offset / vol.downsample_ratio

  return map(nearest_offset, pts)

def create_fixup_downsample_tasks(
    layer_path, points, 
    shape=Vec(2048, 2048, 64), mip=0, axis='z'
  ):
  """you can use this to fix black spots from when downsample tasks fail
  by specifying a point inside each black spot.
  """
  vol = CloudVolume(layer_path, mip)
  offsets = compute_fixup_offsets(vol, points, shape)

  class FixupDownsampleTasksIterator():
    def __len__(self):
      return len(offsets)
    def __iter__(self):
      for offset in offsets:
        yield DownsampleTask(
          layer_path=layer_path,
          mip=mip,
          shape=shape,
          offset=offset,
          axis=axis,
        )

  return FixupDownsampleTasksIterator()

def create_quantized_affinity_info(
    src_layer, dest_layer, shape, mip, 
    chunk_size, encoding
  ):
  srcvol = CloudVolume(src_layer)
  
  info = copy.deepcopy(srcvol.info)
  info['num_channels'] = 1
  info['data_type'] = 'uint8'
  info['type'] = 'image'
  info['scales'] = info['scales'][:mip+1]
  for i in range(mip+1):
    info['scales'][i]['encoding'] = encoding
    info['scales'][i]['chunk_sizes'] = [ chunk_size ]
  return info

def create_quantize_tasks(
    src_layer, dest_layer, shape, 
    mip=0, fill_missing=False, 
    chunk_size=(128, 128, 64), 
    encoding='raw', bounds=None
  ):

  shape = Vec(*shape)

  info = create_quantized_affinity_info(
    src_layer, dest_layer, shape, 
    mip, chunk_size, encoding
  )
  destvol = CloudVolume(dest_layer, info=info, mip=mip)
  destvol.commit_info()

  downsample_scales.create_downsample_scales(
    dest_layer, mip=mip, ds_shape=shape, 
    chunk_size=chunk_size, encoding=encoding
  )

  if bounds is None:
    bounds = destvol.mip_bounds(mip)
  else:
    bounds = destvol.bbox_to_mip(bounds, mip=0, to_mip=mip)
    bounds = bounds.expand_to_chunk_size(
      destvol.mip_chunk_size(mip), destvol.mip_voxel_offset(mip)
    )

  class QuantizeTasksIterator(FinelyDividedTaskIterator):
    def task(self, shape, offset):
      return partial(QuantizeTask,
        source_layer_path=src_layer,
        dest_layer_path=dest_layer,
        shape=shape.tolist(),
        offset=offset.tolist(),
        fill_missing=fill_missing,
        mip=mip,
      )

    def on_finish(self):
      destvol.provenance.sources = [ src_layer ]
      destvol.provenance.processing.append({
        'method': {
          'task': 'QuantizeTask',
          'source_layer_path': src_layer,
          'dest_layer_path': dest_layer,
          'shape': shape.tolist(),
          'fill_missing': fill_missing,
          'mip': mip,
        },
        'by': operator_contact(),
        'date': strftime('%Y-%m-%d %H:%M %Z'),
      }) 
      destvol.commit_provenance()

  return QuantizeTasksIterator(bounds, shape)

def create_hypersquare_ingest_tasks(
    hypersquare_bucket_name, dataset_name, 
    hypersquare_chunk_size, resolution, 
    voxel_offset, volume_size, overlap
  ):
  
  def crtinfo(layer_type, dtype, encoding):
    return CloudVolume.create_new_info(
      num_channels=1,
      layer_type=layer_type,
      data_type=dtype,
      encoding=encoding,
      resolution=resolution,
      voxel_offset=voxel_offset,
      volume_size=volume_size,
      chunk_size=[ 56, 56, 56 ],
    )

  imginfo = crtinfo('image', 'uint8', 'jpeg')
  seginfo = crtinfo('segmentation', 'uint16', 'raw')

  scales = downsample_scales.compute_plane_downsampling_scales(hypersquare_chunk_size)[1:] # omit (1,1,1)

  IMG_LAYER_NAME = 'image'
  SEG_LAYER_NAME = 'segmentation'

  imgvol = CloudVolume(dataset_name, IMG_LAYER_NAME, 0, info=imginfo)
  segvol = CloudVolume(dataset_name, SEG_LAYER_NAME, 0, info=seginfo)

  print("Creating info files for image and segmentation...")
  imgvol.commit_info()
  segvol.commit_info()

  def crttask(volname, tasktype, layer_name):
    return HyperSquareTask(
      bucket_name=hypersquare_bucket_name,
      dataset_name=dataset_name,
      layer_name=layer_name,
      volume_dir=volname,
      layer_type=tasktype,
      overlap=overlap,
      resolution=resolution,
    )

  print("Listing hypersquare bucket...")
  volumes_listing = lib.gcloud_ls('gs://{}/'.format(hypersquare_bucket_name))

  # download this from: 
  # with open('e2198_volumes.json', 'r') as f:
  #   volumes_listing = json.loads(f.read())

  volumes_listing = [ x.split('/')[-2] for x in volumes_listing ]

  class CreateHypersquareIngestTaskIterator(object):
    def __len__(self):
      return len(volumes_listing)
    def __iter__(self):
      for cloudpath in volumes_listing:
        # img_task = crttask(cloudpath, 'image', IMG_LAYER_NAME)
        yield crttask(cloudpath, 'segmentation', SEG_LAYER_NAME)
        # seg_task.execute()

  return CreateHypersquareIngestTaskIterator()

def create_hypersquare_consensus_tasks(
    src_path, dest_path, 
    volume_map_file, consensus_map_path
  ):
  """
  Transfer an Eyewire consensus into neuroglancer. This first requires
  importing the raw segmentation via a hypersquare ingest task. However,
  this can probably be streamlined at some point.

  The volume map file should be JSON encoded and 
  look like { "X-X_Y-Y_Z-Z": EW_VOLUME_ID }

  The consensus map file should look like:
  { VOLUMEID: { CELLID: [segids] } }
  """

  with open(volume_map_file, 'r') as f:
    volume_map = json.loads(f.read())

  vol = CloudVolume(dest_path)

  class HyperSquareConsensusTaskIterator(object):
    def __len__(self):
      return len(volume_map)
    def __iter__(self):
      for boundstr, volume_id in volume_map.items():
        bbox = Bbox.from_filename(boundstr)
        bbox.minpt = Vec.clamp(bbox.minpt, vol.bounds.minpt, vol.bounds.maxpt)
        bbox.maxpt = Vec.clamp(bbox.maxpt, vol.bounds.minpt, vol.bounds.maxpt)

        yield HyperSquareConsensusTask(
          src_path=src_path,
          dest_path=dest_path,
          ew_volume_id=int(volume_id),
          consensus_map_path=consensus_map_path,
          shape=bbox.size3(),
          offset=bbox.minpt.clone(),
        )

  return HyperSquareConsensusTaskIterator()


def create_ccl_face_tasks(
  cloudpath, mip, shape=(512,512,512),
  threshold_gte:Optional[Union[float,int]] = None,
  threshold_lte:Optional[Union[float,int]] = None,
  fill_missing:bool = False,
):
  """pass 1"""
  vol = CloudVolume(cloudpath, mip=mip)

  shape = Vec(*shape)
  bounds = vol.bounds.clone()    

  class CCLFaceTaskIterator(FinelyDividedTaskIterator):
    def task(self, shape, offset):
      return partial(CCLFacesTask, 
        cloudpath=cloudpath, 
        mip=mip, 
        shape=shape.clone(), 
        offset=offset.clone(),
        threshold_gte=threshold_gte,
        threshold_lte=threshold_lte,
        fill_missing=fill_missing,
      )

    def on_finish(self):
      vol.provenance.processing.append({
        'method': {
          'task': 'CCLFacesTask',
          'cloudpath': cloudpath,
          'mip': mip,
          'shape': shape.tolist(),
          'threshold_gte': threshold_gte,
          'threshold_lte': threshold_lte,
          'fill_missing': bool(fill_missing),
        },
        'by': operator_contact(),
        'date': strftime('%Y-%m-%d %H:%M %Z'),
      })
      vol.commit_provenance()

  return CCLFaceTaskIterator(bounds, shape)

def create_ccl_equivalence_tasks(
  cloudpath, mip, shape=(512,512,512),
  threshold_gte:Optional[Union[float,int]] = None,
  threshold_lte:Optional[Union[float,int]] = None,
  fill_missing:bool = False,
):
  """pass 2. Note: shape MUST match pass 1."""
  vol = CloudVolume(cloudpath, mip=mip)

  shape = Vec(*shape)
  bounds = vol.bounds.clone()    

  class CCLEquivalencesTaskIterator(FinelyDividedTaskIterator):
    def task(self, shape, offset):
      return partial(CCLEquivalancesTask, 
        cloudpath=cloudpath, 
        mip=mip, 
        shape=shape.clone(), 
        offset=offset.clone(),
        threshold_gte=threshold_gte,
        threshold_lte=threshold_lte,
        fill_missing=fill_missing,
      )

    def on_finish(self):
      vol.provenance.processing.append({
        'method': {
          'task': 'CCLEquivalancesTask',
          'cloudpath': cloudpath,
          'mip': mip,
          'shape': shape.tolist(),
          'threshold_gte': threshold_gte,
          'threshold_lte': threshold_lte,
          'fill_missing': bool(fill_missing),
        },
        'by': operator_contact(),
        'date': strftime('%Y-%m-%d %H:%M %Z'),
      })
      vol.commit_provenance()

  return CCLEquivalencesTaskIterator(bounds, shape)

def create_ccl_relabel_tasks(
  src_path, dest_path, 
  mip, shape=(512,512,512),
  chunk_size=None, encoding=None,
  threshold_gte:Optional[Union[float,int]] = None,
  threshold_lte:Optional[Union[float,int]] = None,
  fill_missing:bool = False,
):
  """pass 3"""

  src_vol = CloudVolume(src_path, mip=mip)

  cf = CloudFiles(src_path)
  max_label = int(cf.get_json(cf.join(src_vol.key, "ccl", "max_label.json"))[0])

  smallest_dtype = fastremap.fit_dtype(np.uint64, max_label)
  smallest_dtype = np.dtype(smallest_dtype).name

  try:
    dest_vol = CloudVolume(dest_path, mip=mip)
  except cloudvolume.exceptions.InfoUnavailableError:
    info = copy.deepcopy(src_vol.info)
    info["data_type"] = smallest_dtype
    scale = info["scales"][mip]
    if chunk_size:
      scale["chunk_sizes"] = [chunk_size]
    if encoding:
      scale["encoding"] = encoding
      if encoding == "compressed_segmentation":
        scale['compressed_segmentation_block_size'] = (8,8,8)
    if "sharding" in scale:
      del scale["sharding"]
    dest_vol = CloudVolume(dest_path, info=info, mip=mip)
    dest_vol.reset_scales()
    dest_vol.commit_info()

  shape = Vec(*shape)
  bounds = src_vol.bounds.clone()  

  class RelabelCCLTaskIterator(FinelyDividedTaskIterator):
    def task(self, shape, offset):
      return partial(RelabelCCLTask, 
        src_path=src_path, 
        dest_path=dest_path,
        mip=mip, 
        shape=shape.clone(), 
        offset=offset.clone(),
        threshold_gte=threshold_gte,
        threshold_lte=threshold_lte,
        fill_missing=fill_missing,
      )

    def on_finish(self):
      dest_vol.provenance.processing.append({
        'method': {
          'task': 'RelabelCCLTask',
          'src_path': src_path, 
          'dest_path': dest_path,
          'mip': mip,
          'shape': shape.tolist(),
          'threshold_gte': threshold_gte,
          'threshold_lte': threshold_lte,
          'fill_missing': bool(fill_missing),
        },
        'by': operator_contact(),
        'date': strftime('%Y-%m-%d %H:%M %Z'),
      })
      dest_vol.commit_provenance()

  return RelabelCCLTaskIterator(bounds, shape)

