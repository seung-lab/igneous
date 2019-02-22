from __future__ import print_function

from six.moves import range
from itertools import product
from functools import reduce
import operator

import copy
import json
import math
import os
import re
import subprocess
import time
from time import strftime

import numpy as np
from tqdm import tqdm
import cloudvolume
from cloudvolume import CloudVolume, Storage
from cloudvolume.lib import Vec, Bbox, max2, min2, xyzrange, find_closest_divisor, yellow
from taskqueue import TaskQueue, MockTaskQueue 

from igneous import downsample_scales, chunks
from igneous.tasks import (
  IngestTask, HyperSquareConsensusTask, 
  MeshTask, MeshManifestTask, DownsampleTask, QuantizeTask, 
  TransferTask, WatershedRemapTask, DeleteTask, 
  LuminanceLevelsTask, ContrastNormalizationTask, MaskAffinitymapTask, 
  InferenceTask
)
# from igneous.tasks import BigArrayTask

# for provenance files

if __name__ == '__main__':
  try:
    OPERATOR_CONTACT = subprocess.check_output("git config user.email", shell=True)
    OPERATOR_CONTACT = str(OPERATOR_CONTACT.rstrip())
  except:
    try:
      print(yellow('Unable to determine provenance contact email. Set "git config user.email". Using unix $USER instead.'))
      OPERATOR_CONTACT = os.environ['USER']
    except:
      print(yellow('$USER was not set. The "owner" field of the provenance file will be blank.'))
      OPERATOR_CONTACT = ''

def get_bounds(vol, bounds, shape, mip, chunk_size=None):
  if bounds is None:
    bounds = vol.bounds.clone()
  else:
    bounds = Bbox.create(bounds)
    bounds = vol.bbox_to_mip(bounds, mip=0, to_mip=mip)
    if chunk_size is not None:
      bounds = bounds.expand_to_chunk_size(chunk_size, vol.mip_voxel_offset(mip))
    bounds = Bbox.clamp(bounds, vol.mip_bounds(mip))

  print("Volume Bounds: ", vol.mip_bounds(mip))
  print("Selected ROI:  ", bounds)

  return bounds

def create_ingest_task(cloudpath):
  """
  Creates one task for each ingest chunk present in the build folder.
  It is required that the info file is already placed in order for this task
  to run succesfully.
  """
  class IngestTaskIterator():
    def __iter__(self):
      with Storage(cloudpath) as storage:
        for filename in storage.list_files(prefix='build/'):
          yield IngestTask(
            chunk_path=storage.get_path_to_file('build/'+filename),
            chunk_encoding='npz',
            layer_path=cloudpath,
          )
  return IngestTaskIterator()

# def create_bigarray_task(cloudpath):
#   """
#   Creates one task for each bigarray chunk present in the bigarray folder.
#   These tasks will convert the bigarray chunks into chunks that ingest tasks are able to understand.
#   """
#   class BigArrayTaskIterator():
#     def __iter__(self):    
#       with Storage(cloudpath) as storage:
#         for filename in storage.list_blobs(prefix='bigarray/'):
#           yield BigArrayTask(
#             chunk_path=storage.get_path_to_file('bigarray/'+filename),
#             chunk_encoding='npz', # npz_uint8 to convert affinites float32 affinties to uint8
#             version='{}/{}'.format(storage._path.dataset_name, storage._path.layer_name)
#           )
#   return BigArrayTaskIterator()

def compute_build_bounding_box(storage, prefix='build/'):
    bboxes = []
    for filename in tqdm(storage.list_files(prefix=prefix), desc='Computing Bounds'):
        bbox = Bbox.from_filename(filename) 
        bboxes.append(bbox)

    bounds = Bbox.expand(*bboxes)
    chunk_size = reduce(max2, map(lambda bbox: bbox.size3(), bboxes))

    print('bounds={} (size: {}); chunk_size={}'.format(bounds, bounds.size3(), chunk_size))
  
    return bounds, chunk_size

def get_build_data_type_and_shape(storage):
    for filename in storage.list_files(prefix='build/'):
        arr = chunks.decode_npz(storage.get_file(filename))
        return arr.dtype.name, arr.shape[3] #num_channels

def create_info_file_from_build(layer_path, layer_type, resolution, encoding):
  assert layer_type in ('image', 'segmentation', 'affinities')

  with Storage(layer_path) as storage:
    bounds, build_chunk_size = compute_build_bounding_box(storage)
    data_type, num_channels = get_build_data_type_and_shape(storage)

  neuroglancer_chunk_size = find_closest_divisor(build_chunk_size, closest_to=[64,64,64])

  info = CloudVolume.create_new_info(
    num_channels=num_channels, 
    layer_type=layer_type, 
    data_type=data_type, 
    encoding=encoding, 
    resolution=resolution, 
    voxel_offset=bounds.minpt.tolist(), 
    volume_size=bounds.size3(),
    mesh=(layer_type == 'segmentation'), 
    chunk_size=list(map(int, neuroglancer_chunk_size)),
  )

  vol = CloudVolume(layer_path, mip=0, info=info).commit_info()
  vol = create_downsample_scales(layer_path, mip=0, ds_shape=build_chunk_size, axis='z')
  
  return vol.info

def create_downsample_scales(
    layer_path, mip, ds_shape, axis='z', 
    preserve_chunk_size=False, chunk_size=None,
    encoding=None
  ):
  vol = CloudVolume(layer_path, mip)
  shape = min2(vol.volume_size, ds_shape)

  # sometimes we downsample a base layer of 512x512 
  # into underlying chunks of 64x64 which permits more scales
  underlying_mip = (mip + 1) if (mip + 1) in vol.available_mips else mip
  underlying_shape = vol.mip_underlying(underlying_mip).astype(np.float32)

  if chunk_size:
    underlying_shape = Vec(*chunk_size).astype(np.float32)

  toidx = { 'x': 0, 'y': 1, 'z': 2 }
  preserved_idx = toidx[axis]
  underlying_shape[preserved_idx] = float('inf')

  scales = downsample_scales.compute_plane_downsampling_scales(
    size=shape, 
    preserve_axis=axis, 
    max_downsampled_size=int(min(*underlying_shape)),
  ) 
  scales = scales[1:] # omit (1,1,1)
  scales = [ list(map(int, vol.downsample_ratio * Vec(*factor3))) for factor3 in scales ]

  if len(scales) == 0:
    print("WARNING: No scales generated.")

  for scale in scales:
    vol.add_scale(scale, encoding=encoding, chunk_size=chunk_size)

  if chunk_size is None:
    if preserve_chunk_size or len(scales) == 0:
      chunk_size = vol.scales[mip]['chunk_sizes']
    else:
      chunk_size = vol.scales[mip + 1]['chunk_sizes']
  else:
    chunk_size = [ chunk_size ]

  if encoding is None:
    encoding = vol.scales[mip]['encoding']

  for i in range(mip + 1, mip + len(scales) + 1):
    vol.scales[i]['chunk_sizes'] = chunk_size

  return vol.commit_info()

def create_downsampling_tasks(
    layer_path, mip=0, fill_missing=False, 
    axis='z', num_mips=5, preserve_chunk_size=True,
    sparse=False, bounds=None, chunk_size=None,
    encoding=None
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
    """
    def ds_shape(mip):
      shape = vol.mip_underlying(mip)[:3]
      shape.x *= 2 ** num_mips
      shape.y *= 2 ** num_mips
      return shape

    vol = CloudVolume(layer_path, mip=mip)
    shape = ds_shape(vol.mip)
    vol = create_downsample_scales(
      layer_path, mip, shape, 
      preserve_chunk_size=preserve_chunk_size, chunk_size=chunk_size,
      encoding=encoding
    )

    if not preserve_chunk_size or chunk_size:
      shape = ds_shape(vol.mip + 1)

    bounds = get_bounds(vol, bounds, shape, mip, vol.chunk_size)
    
    class DownsampleTaskIterator():
      def __len__(self):
        return reduce(operator.mul, np.ceil(bounds.size3() / shape))
      def __iter__(self):
        for startpt in xyzrange( bounds.minpt, bounds.maxpt, shape ):
          yield DownsampleTask(
            layer_path=layer_path,
            mip=vol.mip,
            shape=shape.clone(),
            offset=startpt.clone(),
            axis=axis,
            fill_missing=fill_missing,
            sparse=sparse,
          )

        vol.provenance.processing.append({
          'method': {
            'task': 'DownsampleTask',
            'mip': mip,
            'shape': shape.tolist(),
            'axis': axis,
            'method': 'downsample_with_averaging' if vol.layer_type == 'image' else 'downsample_segmentation',
            'sparse': sparse,
            'bounds': str(bounds),
            'chunk_size': (list(chunk_size) if chunk_size else None),
            'preserve_chunk_size': preserve_chunk_size,
          },
          'by': OPERATOR_CONTACT,
          'date': strftime('%Y-%m-%d %H:%M %Z'),
        })
        vol.commit_provenance()

    return DownsampleTaskIterator()

def create_deletion_tasks(layer_path, mip=0, num_mips=5):
  vol = CloudVolume(layer_path)
  
  shape = vol.mip_underlying(mip)[:3]
  shape.x *= 2 ** num_mips
  shape.y *= 2 ** num_mips

  class DeleteTaskIterator():
    def __len__(self):
      return reduce(operator.mul, np.ceil(vol.bounds.size3() / shape))
    def __iter__(self):
      for startpt in xyzrange( vol.bounds.minpt, vol.bounds.maxpt, shape ):
        bounded_shape = min2(shape, vol.bounds.maxpt - startpt)
        yield DeleteTask(
          layer_path=layer_path,
          shape=bounded_shape.clone(),
          offset=startpt.clone(),
          mip=mip,
          num_mips=num_mips,
        )

      vol = CloudVolume(layer_path)
      vol.provenance.processing.append({
        'method': {
          'task': 'DeleteTask',
          'mip': mip,
          'num_mips': num_mips,
          'shape': shape.tolist(),
        },
        'by': OPERATOR_CONTACT,
        'date': strftime('%Y-%m-%d %H:%M %Z'),
      })
      vol.commit_provenance()

  return DeleteTaskIterator()


def create_meshing_tasks(
    layer_path, mip, 
    shape=Vec(512, 512, 64), max_simplification_error=40,
    mesh_dir=None, cdn_cache=False 
  ):
  shape = Vec(*shape)

  vol = CloudVolume(layer_path, mip)

  if mesh_dir is None:
    mesh_dir = 'mesh_mip_{}_err_{}'.format(mip, max_simplification_error)

  if not 'mesh' in vol.info:
    vol.info['mesh'] = mesh_dir
    vol.commit_info()

  class MeshTaskIterator():
    def __len__(self):
      return reduce(operator.mul, np.ceil(vol.bounds.size3() / shape))
    def __iter__(self):
      for startpt in tqdm(xyzrange( vol.bounds.minpt, vol.bounds.maxpt, shape ), desc="Inserting Mesh Tasks"):
        yield MeshTask(
          shape.clone(),
          startpt.clone(),
          layer_path,
          mip=vol.mip,
          max_simplification_error=max_simplification_error,
          mesh_dir=mesh_dir, 
          cache_control=('' if cdn_cache else 'no-cache'),
        )

      vol.provenance.processing.append({
        'method': {
          'task': 'MeshTask',
          'layer_path': layer_path,
          'mip': vol.mip,
          'shape': shape.tolist(),
          'max_simplification_error': max_simplification_error,
          'mesh_dir': mesh_dir,
          'cdn_cache': cdn_cache,
        },
        'by': OPERATOR_CONTACT,
        'date': strftime('%Y-%m-%d %H:%M %Z'),
      }) 
      vol.commit_provenance()

  return MeshTaskIterator()

def create_transfer_tasks(
    src_layer_path, dest_layer_path, 
    chunk_size=None, shape=Vec(2048, 2048, 64), 
    fill_missing=False, translate=(0,0,0), 
    bounds=None, mip=0, preserve_chunk_size=True
  ):
  """
  Transfer data from one data layer to another. It's possible
  to transfer from a lower resolution mip level within a given
  bounding box. The bounding box should be specified in terms of
  the highest resolution.
  """
  shape = Vec(*shape)
  vol = CloudVolume(src_layer_path, mip=mip)
  translate = Vec(*translate) // vol.downsample_ratio
 
  if not chunk_size:
    chunk_size = vol.info['scales'][mip]['chunk_sizes'][0]
  chunk_size = Vec(*chunk_size)

  try:
    dvol = CloudVolume(dest_layer_path, mip=mip)
  except Exception: # no info file
    info = copy.deepcopy(vol.info)
    dvol = CloudVolume(dest_layer_path, info=info)
    dvol.commit_info()

  dvol.info['scales'] = dvol.info['scales'][:mip+1]
  dvol.info['scales'][mip]['chunk_sizes'] = [ chunk_size.tolist() ]
  dvol.commit_info()

  create_downsample_scales(dest_layer_path, 
    mip=mip, ds_shape=shape, preserve_chunk_size=preserve_chunk_size)
  
  if bounds is None:
    bounds = vol.bounds.clone()
  else:
    bounds = vol.bbox_to_mip(bounds, mip=0, to_mip=mip)
    bounds = Bbox.clamp(bounds, vol.bounds)

  class TransferTaskIterator(object):
    def __len__(self):
      return int(reduce(operator.mul, np.ceil(bounds.size3() / shape)))
    def __iter__(self):
      for startpt in xyzrange( bounds.minpt, bounds.maxpt, shape ):
        yield TransferTask(
          src_path=src_layer_path,
          dest_path=dest_layer_path,
          shape=shape.clone(),
          offset=startpt.clone(),
          fill_missing=fill_missing,
          translate=translate,
          mip=mip,
        )

      job_details = {
        'method': {
          'task': 'TransferTask',
          'src': src_layer_path,
          'dest': dest_layer_path,
          'shape': list(map(int, shape)),
          'fill_missing': fill_missing,
          'translate': list(map(int, translate)),
          'bounds': [
            bounds.minpt.tolist(),
            bounds.maxpt.tolist()
          ],
          'mip': mip,
        },
        'by': OPERATOR_CONTACT,
        'date': strftime('%Y-%m-%d %H:%M %Z'),
      }

      dvol = CloudVolume(dest_layer_path)
      dvol.provenance.sources = [ src_layer_path ]
      dvol.provenance.processing.append(job_details) 
      dvol.commit_provenance()

      if vol.path.protocol != 'boss':
        vol.provenance.processing.append(job_details)
        vol.commit_provenance()

  return TransferTaskIterator()

def create_contrast_normalization_tasks(
    src_path, dest_path, levels_path=None,
    shape=None, mip=0, clip_fraction=0.01, 
    fill_missing=False, translate=(0,0,0),
    minval=None, maxval=None, bounds=None
  ):

  srcvol = CloudVolume(src_path, mip=mip)
  
  try:
    dvol = CloudVolume(dest_path, mip=mip)
  except Exception: # no info file
    info = copy.deepcopy(srcvol.info)
    dvol = CloudVolume(dest_path, mip=mip, info=info)
    dvol.info['scales'] = dvol.info['scales'][:mip+1]
    dvol.commit_info()

  if shape == None:
    shape = Bbox( (0,0,0), (2048, 2048, 64) )
    shape = shape.shrink_to_chunk_size(dvol.underlying).size3()

  shape = Vec(*shape)

  create_downsample_scales(dest_path, mip=mip, ds_shape=shape, preserve_chunk_size=True)
  dvol.refresh_info()

  bounds = get_bounds(srcvol, bounds, shape, mip)

  class ContrastNormalizationTaskIterator(object):
    def __len__(self):
      return int(reduce(operator.mul, np.ceil(bounds.size3() / shape)))
    def __iter__(self):
      for startpt in xyzrange( bounds.minpt, bounds.maxpt, shape ):
        task_shape = min2(shape.clone(), srcvol.bounds.maxpt - startpt)
        yield ContrastNormalizationTask( 
          src_path=src_path, 
          dest_path=dest_path,
          levels_path=levels_path,
          shape=task_shape, 
          offset=startpt.clone(), 
          clip_fraction=clip_fraction,
          mip=mip,
          fill_missing=fill_missing,
          translate=translate,
          minval=minval,
          maxval=maxval,
        )
        
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
        'by': OPERATOR_CONTACT,
        'date': strftime('%Y-%m-%d %H:%M %Z'),
      }) 
      dvol.commit_provenance()

  return ContrastNormalizationTaskIterator()

def create_luminance_levels_tasks(
    layer_path, levels_path=None, coverage_factor=0.01, 
    shape=None, offset=(0,0,0), mip=0, bounds=None
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

  layer_path: source image to sample from
  levels_path: which path to write ./levels/ to (default: $layer_path)
  coverage_factor: what fraction of the image to sample

  offset & shape: Allows you to specify an ROI if much of
    the edges are black. Defaults to entire image.
  mip: int, which mip to work with, default maximum resolution
  """
  vol = CloudVolume(layer_path, mip=mip)

  if shape == None:
    shape = vol.shape.clone()
    shape.z = 1

  offset = Vec(*offset)
  zoffset = offset.clone()

  bounds = get_bounds(vol, bounds, shape, mip)
  protocol = vol.path.protocol

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
        'by': OPERATOR_CONTACT,
        'date': strftime('%Y-%m-%d %H:%M %Z'),
      }) 
      vol.commit_provenance()

  return LuminanceLevelsTaskIterator()

def create_watershed_remap_tasks(
    map_path, src_layer_path, dest_layer_path, 
    shape=Vec(2048, 2048, 64)
  ):
  shape = Vec(*shape)
  vol = CloudVolume(src_layer_path)

  create_downsample_scales(dest_layer_path, mip=0, ds_shape=shape)

  class WatershedRemapTaskIterator():
    def __len__(self):
      return int(reduce(operator.mul, np.ceil(bounds.size3() / shape)))
    def __iter__(self):
      for startpt in xyzrange( vol.bounds.minpt, vol.bounds.maxpt, shape ):
        yield WatershedRemapTask(
          map_path=map_path,
          src_path=src_layer_path,
          dest_path=dest_layer_path,
          shape=shape.clone(),
          offset=startpt.clone(),
        )
    
      dvol = CloudVolume(dest_layer_path)
      dvol.provenance.processing.append({
        'method': {
          'task': 'WatershedRemapTask',
          'src': src_layer_path,
          'dest': dest_layer_path,
          'remap_file': map_path,
          'shape': list(shape),
        },
        'by': OPERATOR_CONTACT,
        'date': strftime('%Y-%m-%d %H:%M %Z'),
      }) 
      dvol.commit_provenance()

  return WatershedRemapTaskIterator()

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

def create_quantized_affinity_info(src_layer, dest_layer, shape, mip, chunk_size):
  srcvol = CloudVolume(src_layer)
  
  info = copy.deepcopy(srcvol.info)
  info['num_channels'] = 1
  info['data_type'] = 'uint8'
  info['type'] = 'image'
  info['scales'] = info['scales'][:mip+1]
  for i in range(mip+1):
    info['scales'][i]['chunk_sizes'] = [ chunk_size ]
  return info

def create_quantize_tasks(
    src_layer, dest_layer, shape, 
    mip=0, fill_missing=False, chunk_size=[64,64,64]
  ):

  shape = Vec(*shape)

  info = create_quantized_affinity_info(
    src_layer, dest_layer, shape, mip, chunk_size
  )
  destvol = CloudVolume(dest_layer, info=info, mip=mip)
  destvol.commit_info()

  create_downsample_scales(dest_layer, mip=mip, ds_shape=shape)

  class QuantizeTasksIterator():
    def __len__(self):
      return  int(reduce(operator.mul, np.ceil(destvol.bounds.size3() / shape)))
    def __iter__(self):
      for startpt in xyzrange( destvol.bounds.minpt, destvol.bounds.maxpt, shape ):
        yield QuantizeTask(
          source_layer_path=src_layer,
          dest_layer_path=dest_layer,
          shape=shape.tolist(),
          offset=startpt.tolist(),
          fill_missing=fill_missing,
          mip=mip,
        )

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
        'by': OPERATOR_CONTACT,
        'date': strftime('%Y-%m-%d %H:%M %Z'),
      }) 
      destvol.commit_provenance()

  return QuantizeTasksIterator()

# split the work up into ~1000 tasks (magnitude 3)
def create_mesh_manifest_tasks(layer_path, magnitude=3):
  assert int(magnitude) == magnitude

  start = 10 ** (magnitude - 1)
  end = 10 ** magnitude

  # For a prefix like 100, tasks 1-99 will be missed. Account for them by
  # enumerating them individually with a suffixed ':' to limit matches to
  # only those small numbers

  class MeshManifestTaskIterator(object):
    def __len__(self):
      return 10 ** magnitude
    def __iter__(self):
      for prefix in range(1, start):
        yield MeshManifestTask(layer_path=layer_path, prefix=str(prefix) + ':')

      # enumerate from e.g. 100 to 999
      for prefix in range(start, end):
        yield MeshManifestTask(layer_path=layer_path, prefix=prefix)

  return MeshManifestTaskIterator()

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

def create_mask_affinity_map_tasks(
    aff_input_layer_path, aff_output_layer_path, 
    aff_mip, mask_layer_path, mask_mip, output_block_start, 
    output_block_size, grid_size 
  ):
    """
    affinity map masking block by block. The block coordinates should be aligned with 
    cloud storage. 
    """

    class MaskAffinityMapTaskIterator():
      def __len__(self):
        return int(reduce(operator.mul, grid_size))
      def __iter__(self):
        for x, y, z in xyzrange(grid_size):
          output_bounds = Bbox.from_slices(tuple(slice(s+x*b, s+x*b+b)
                  for (s, x, b) in zip(output_block_start, (z, y, x), output_block_size)))
          yield MaskAffinitymapTask(
              aff_input_layer_path=aff_input_layer_path,
              aff_output_layer_path=aff_output_layer_path,
              aff_mip=aff_mip, 
              mask_layer_path=mask_layer_path,
              mask_mip=mask_mip,
              output_bounds=output_bounds,
          )

        vol = CloudVolume(output_layer_path, mip=aff_mip)
        vol.provenance.processing.append({
            'method': {
                'task': 'InferenceTask',
                'aff_input_layer_path': aff_input_layer_path,
                'aff_output_layer_path': aff_output_layer_path,
                'aff_mip': aff_mip,
                'mask_layer_path': mask_layer_path,
                'mask_mip': mask_mip,
                'output_block_start': output_block_start,
                'output_block_size': output_block_size, 
                'grid_size': grid_size,
            },
            'by': OPERATOR_CONTACT,
            'date': strftime('%Y-%m-%d %H:%M %Z'),
        })
        vol.commit_provenance()

  return MaskAffinityMapTaskIterator()


def create_inference_tasks(
    image_layer_path, convnet_path, 
    mask_layer_path, output_layer_path, output_block_start, output_block_size, 
    grid_size, patch_size, patch_overlap, cropping_margin_size,
    output_key='output', num_output_channels=3, 
    image_mip=1, output_mip=1, mask_mip=3
  ):
    """
    convnet inference block by block. The block coordinates should be aligned with 
    cloud storage. 
    """
    class InferenceTaskIterator():
      def __len__(self):
        return int(reduce(operator.mul, grid_size))
      def __iter__(self):
        for x, y, z in xyzrange(grid_size):
          output_offset = tuple(s+x*b for (s, x, b) in 
                                zip(output_block_start, (z, y, x), 
                                    output_block_size))
          yield InferenceTask(
              image_layer_path=image_layer_path,
              convnet_path=convnet_path,
              mask_layer_path=mask_layer_path,
              output_layer_path=output_layer_path,
              output_offset=output_offset,
              output_shape=output_block_size,
              patch_size=patch_size, 
              patch_overlap=patch_overlap,
              cropping_margin_size=cropping_margin_size,
              output_key=output_key,
              num_output_channels=num_output_channels,
              image_mip=image_mip,
              output_mip=output_mip,
              mask_mip=mask_mip
          )


        vol = CloudVolume(output_layer_path, mip=output_mip)
        vol.provenance.processing.append({
            'method': {
                'task': 'InferenceTask',
                'image_layer_path': image_layer_path,
                'convnet_path': convnet_path,
                'mask_layer_path': mask_layer_path,
                'output_layer_path': output_layer_path,
                'output_offset': output_offset,
                'output_shape': output_block_size,
                'patch_size': patch_size,
                'patch_overlap': patch_overlap,
                'cropping_margin_size': cropping_margin_size,
                'output_key': output_key,
                'num_output_channels': num_output_channels,
                'image_mip': image_mip,
                'output_mip': output_mip,
                'mask_mip': mask_mip,
            },
            'by': OPERATOR_CONTACT,
            'date': strftime('%Y-%m-%d %H:%M %Z'),
        })
        vol.commit_provenance()

  return InferenceTaskIterator()


def upload_build_chunks(storage, volume, offset=[0, 0, 0], build_chunk_size=[1024,1024,128]):
  offset = Vec(*offset)
  shape = Vec(*volume.shape[:3])
  build_chunk_size = Vec(*build_chunk_size)

  for spt in xyzrange( (0,0,0), shape, build_chunk_size):
    ept = min2(spt + build_chunk_size, shape)
    bbox = Bbox(spt, ept)
    chunk = volume[ bbox.to_slices() ]
    bbox += offset
    filename = 'build/{}'.format(bbox.to_filename())
    storage.put_file(filename, chunks.encode_npz(chunk))
  storage.wait()


def cascade(tq, fnlist):
    for fn in fnlist:
      fn(tq)
      N = tq.enqueued
      while N > 0:
        N = tq.enqueued
        print('\r {} remaining'.format(N), end='')
        time.sleep(2)

