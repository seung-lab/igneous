from __future__ import print_function

from six.moves import range
from itertools import product
from functools import reduce

import copy
import json
import math
import os
import re
import time
from time import strftime

import numpy as np
from tqdm import tqdm
from cloudvolume import CloudVolume, Storage
from cloudvolume.lib import Vec, Bbox, max2, min2, xyzrange, find_closest_divisor
from taskqueue import TaskQueue, MockTaskQueue, LocalTaskQueue

from igneous import downsample_scales, chunks
from igneous.tasks import (
  IngestTask, HyperSquareConsensusTask, 
  MeshTask, MeshManifestTask, DownsampleTask, QuantizeAffinitiesTask, 
  TransferTask, WatershedRemapTask, DeleteTask, 
  LuminanceLevelsTask, ContrastNormalizationTask
)
# from igneous.tasks import BigArrayTask

USER_EMAIL = 'ws9@princeton.edu' # for provenance files

def create_ingest_task(storage, task_queue):
    """
    Creates one task for each ingest chunk present in the build folder.
    It is required that the info file is already placed in order for this task
    to run succesfully.
    """
    for filename in storage.list_files(prefix='build/'):
        t = IngestTask(
          chunk_path=storage.get_path_to_file('build/'+filename),
          chunk_encoding='npz',
          layer_path=storage.layer_path,
        )
        task_queue.insert(t)

# def create_bigarray_task(storage, task_queue):
#     """
#     Creates one task for each bigarray chunk present in the bigarray folder.
#     These tasks will convert the bigarray chunks into chunks that ingest tasks are able to understand.
#     """
#     for filename in tqdm(storage.list_blobs(prefix='bigarray/')):   
#         t = BigArrayTask(
#             chunk_path=storage.get_path_to_file('bigarray/'+filename),
#             chunk_encoding='npz', #npz_uint8 to convert affinites float32 affinties to uint8
#             version='{}/{}'.format(storage._path.dataset_name, storage._path.layer_name))
#         task_queue.insert(t)

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

def create_downsample_scales(layer_path, mip, ds_shape, axis='z', preserve_chunk_size=False):
  vol = CloudVolume(layer_path, mip)
  shape = min2(vol.volume_size, ds_shape)

  # sometimes we downsample a base layer of 512x512 
  # into underlying chunks of 64x64 which permits more scales
  underlying_mip = (mip + 1) if (mip + 1) in vol.available_mips else mip
  underlying_shape = vol.mip_underlying(underlying_mip).astype(np.float32)

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

  for scale in scales:
    vol.add_scale(scale)

  if preserve_chunk_size:
    for i in range(mip + 1, mip + len(scales) + 1):
      vol.scales[i]['chunk_sizes'] = vol.scales[mip]['chunk_sizes']

  return vol.commit_info()

def create_downsampling_tasks(
    task_queue, layer_path, 
    mip=-1, fill_missing=False, axis='z', 
    num_mips=5, preserve_chunk_size=True,
    sparse=False
  ):
  
    def ds_shape(mip):
      shape = vol.mip_underlying(mip)[:3]
      shape.x *= 2 ** num_mips
      shape.y *= 2 ** num_mips
      return shape

    vol = CloudVolume(layer_path, mip=mip)
    shape = ds_shape(vol.mip)
    vol = create_downsample_scales(layer_path, mip, shape, preserve_chunk_size=preserve_chunk_size)

    if not preserve_chunk_size:
      shape = ds_shape(vol.mip + 1)

    bounds = vol.bounds.clone()
    for startpt in tqdm(xyzrange( bounds.minpt, bounds.maxpt, shape ), desc="Inserting Downsample Tasks"):
      task = DownsampleTask(
        layer_path=layer_path,
        mip=vol.mip,
        shape=shape.clone(),
        offset=startpt.clone(),
        axis=axis,
        fill_missing=fill_missing,
        sparse=sparse,
      )
      task_queue.insert(task)
    task_queue.wait('Uploading')
    vol.provenance.processing.append({
      'method': {
        'task': 'DownsampleTask',
        'mip': mip,
        'shape': shape.tolist(),
        'axis': axis,
        'method': 'downsample_with_averaging' if vol.layer_type == 'image' else 'downsample_segmentation',
        'sparse': sparse,
      },
      'by': USER_EMAIL,
      'date': strftime('%Y-%m-%d %H:%M %Z'),
    })
    vol.commit_provenance()

def create_deletion_tasks(task_queue, layer_path):
  vol = CloudVolume(layer_path)
  shape = vol.underlying * 10

  for startpt in tqdm(xyzrange( vol.bounds.minpt, vol.bounds.maxpt, shape ), desc="Inserting Deletion Tasks"):
    bounded_shape = min2(shape, vol.bounds.maxpt - startpt)
    task = DeleteTask(
      layer_path=layer_path,
      shape=bounded_shape.clone(),
      offset=startpt.clone(),
    )
    task_queue.insert(task)
  task_queue.wait('Uploading DeleteTasks')

def create_meshing_tasks(task_queue, layer_path, mip, shape=Vec(512, 512, 512)):
  shape = Vec(*shape)
  max_simplification_error = 40

  vol = CloudVolume(layer_path, mip)

  if not 'mesh' in vol.info:
    vol.info['mesh'] = 'mesh_mip_{}_err_{}'.format(mip, max_simplification_error)
    vol.commit_info()

  for startpt in tqdm(xyzrange( vol.bounds.minpt, vol.bounds.maxpt, shape ), desc="Inserting Mesh Tasks"):
    task = MeshTask(
      layer_path=layer_path,
      mip=vol.mip,
      shape=shape.clone(),
      offset=startpt.clone(),
      max_simplification_error=max_simplification_error,
    )
    task_queue.insert(task)
  task_queue.wait('Uploading MeshTasks')

  vol.provenance.processing.append({
    'method': {
      'task': 'MeshTask',
      'layer_path': layer_path,
      'mip': vol.mip,
      'shape': shape.tolist(),      
    },
    'by': USER_EMAIL,
    'date': strftime('%Y-%m-%d %H:%M %Z'),
  }) 
  vol.commit_provenance()

def create_transfer_tasks(task_queue, src_layer_path, dest_layer_path, chunk_size=None, shape=Vec(2048, 2048, 64), fill_missing=False, translate=(0,0,0)):
  shape = Vec(*shape)
  translate = Vec(*translate)
  vol = CloudVolume(src_layer_path)
 
  if not chunk_size:
    chunk_size = vol.info['scales'][0]['chunk_sizes'][0]
  chunk_size = Vec(*chunk_size)

  try:
    dvol = CloudVolume(dest_layer_path)
  except Exception: # no info file
    info = copy.deepcopy(vol.info)
    dvol = CloudVolume(dest_layer_path, info=info)
    dvol.commit_info()

  if chunk_size is not None:
    dvol.info['scales'] = dvol.info['scales'][:1]
    dvol.info['scales'][0]['chunk_sizes'] = [ chunk_size.tolist() ]
    dvol.commit_info()

  create_downsample_scales(dest_layer_path, mip=0, ds_shape=shape, preserve_chunk_size=True)
  
  bounds = vol.bounds.clone()
  for startpt in tqdm(xyzrange( bounds.minpt, bounds.maxpt, shape ), desc="Inserting Transfer Tasks"):
    task = TransferTask(
      src_path=src_layer_path,
      dest_path=dest_layer_path,
      shape=shape.clone(),
      offset=startpt.clone(),
      fill_missing=fill_missing,
      translate=translate,
    )
    task_queue.insert(task)
  task_queue.wait('Uploading Transfer Tasks')

  dvol = CloudVolume(dest_layer_path)
  dvol.provenance.processing.append({
    'method': {
      'task': 'TransferTask',
      'src': src_layer_path,
      'dest': dest_layer_path,
      'shape': list(map(int, shape)),
      'fill_missing': fill_missing,
      'translate': list(map(int, translate)),
    },
    'by': USER_EMAIL,
    'date': strftime('%Y-%m-%d %H:%M %Z'),
  }) 
  dvol.commit_provenance()

def create_contrast_normalization_tasks(task_queue, src_path, dest_path, 
  shape=None, mip=0, clip_fraction=0.01, fill_missing=False, translate=(0,0,0)):

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

  bounds = srcvol.bounds.clone()
  for startpt in tqdm(xyzrange( bounds.minpt, bounds.maxpt, shape ), desc="Inserting Contrast Normalization Tasks"):
    task_shape = min2(shape.clone(), srcvol.bounds.maxpt - startpt)
    task = ContrastNormalizationTask( 
      src_path=src_path, 
      dest_path=dest_path,
      shape=task_shape, 
      offset=startpt.clone(), 
      clip_fraction=clip_fraction,
      mip=mip,
      fill_missing=fill_missing,
      translate=translate,
    )
    task_queue.insert(task)
  task_queue.wait('Uploading Contrast Normalization Tasks')

  dvol.provenance.processing.append({
    'method': {
      'task': 'ContrastNormalizationTask',
      'src_path': src_path,
      'dest_path': dest_path,
      'shape': Vec(*shape).tolist(),
      'clip_fraction': clip_fraction,
      'mip': mip,
      'translate': Vec(*translate).tolist(),
    },
    'by': USER_EMAIL,
    'date': strftime('%Y-%m-%d %H:%M %Z'),
  }) 
  dvol.commit_provenance()

def create_luminance_levels_tasks(task_queue, layer_path, coverage_factor=0.01, shape=None, offset=(0,0,0), mip=0):
  vol = CloudVolume(layer_path)

  if shape == None:
    shape = vol.shape.clone()
    shape.z = 1

  offset = Vec(*offset)

  for z in range(vol.bounds.minpt.z, vol.bounds.maxpt.z + 1):
    offset.z = z
    task = LuminanceLevelsTask( 
      src_path=layer_path, 
      shape=shape, 
      offset=offset, 
      coverage_factor=coverage_factor,
      mip=mip,
    )
    task_queue.insert(task)
  task_queue.wait('Uploading Luminance Levels Tasks')

  vol.provenance.processing.append({
    'method': {
      'task': 'LuminanceLevelsTask',
      'src': layer_path,
      'shape': Vec(*shape).tolist(),
      'offset': Vec(*offset).tolist(),
      'coverage_factor': coverage_factor,
      'mip': mip,
    },
    'by': USER_EMAIL,
    'date': strftime('%Y-%m-%d %H:%M %Z'),
  }) 
  vol.commit_provenance()

def create_boss_transfer_tasks(task_queue, src_layer_path, dest_layer_path, shape=Vec(1024, 1024, 64)):
  # Note: Weird errors with datatype changing to float64 when requesting 2048,2048,64
  # 1024,1024,64 worked nicely though.
  shape = Vec(*shape)
  vol = CloudVolume(dest_layer_path)

  create_downsample_scales(dest_layer_path, mip=0, ds_shape=shape)

  for startpt in tqdm(xyzrange( vol.bounds.minpt, vol.bounds.maxpt, shape ), desc="Inserting Boss Transfer Tasks"):
    task = BossTransferTask(
      src_path=src_layer_path,
      dest_path=dest_layer_path,
      shape=shape.clone(),
      offset=startpt.clone(),
    )
    task_queue.insert(task)
  task_queue.wait('Uploading Boss Transfer Tasks')

def create_watershed_remap_tasks(task_queue, map_path, src_layer_path, dest_layer_path, shape=Vec(2048, 2048, 64)):
  shape = Vec(*shape)
  vol = CloudVolume(src_layer_path)

  create_downsample_scales(dest_layer_path, mip=0, ds_shape=shape)

  for startpt in tqdm(xyzrange( vol.bounds.minpt, vol.bounds.maxpt, shape ), desc="Inserting Remap Tasks"):
    task = WatershedRemapTask(
      map_path=map_path,
      src_path=src_layer_path,
      dest_path=dest_layer_path,
      shape=shape.clone(),
      offset=startpt.clone(),
    )
    task_queue.insert(task)
  task_queue.wait('Uploading Remap Tasks')
  dvol = CloudVolume(dest_layer_path)
  dvol.provenance.processing.append({
    'method': {
      'task': 'WatershedRemapTask',
      'src': src_layer_path,
      'dest': dest_layer_path,
      'remap_file': map_path,
      'shape': list(shape),
    },
    'by': USER_EMAIL,
    'date': strftime('%Y-%m-%d %H:%M %Z'),
  }) 
  dvol.commit_provenance()

def compute_fixup_offsets(vol, points, shape):
  pts = map(np.array, points)

  # points are specified in high res coordinates 
  # because that's what people read off the screen.
  def nearest_offset(pt):
    mip0offset = (np.floor((pt - vol.mip_voxel_offset(0)) / shape) * shape) + vol.mip_voxel_offset(0)
    return mip0offset / vol.downsample_ratio

  return map(nearest_offset, pts)

def create_fixup_downsample_tasks(task_queue, layer_path, points, shape=Vec(2048, 2048, 64), mip=0, axis='z'):
  """you can use this to fix black spots from when downsample tasks fail
  by specifying a point inside each black spot.
  """
  vol = CloudVolume(layer_path, mip)
  offsets = compute_fixup_offsets(vol, points, shape)

  for offset in tqdm(offsets, desc="Inserting Corrective Downsample Tasks"):
    task = DownsampleTask(
      layer_path=layer_path,
      mip=mip,
      shape=shape,
      offset=offset,
      axis=axis,
    )
    task_queue.insert(task)
  task_queue.wait('Uploading')

def create_quantized_affinity_info(src_layer, dest_layer, shape):
  srcvol = CloudVolume(src_layer)
  
  info = copy.deepcopy(srcvol.info)
  info['num_channels'] = 1
  info['data_type'] = 'uint8'
  info['type'] = 'segmentation'
  info['scales'] = info['scales'][:1]
  info['scales'][0]['chunk_sizes'] = [[ 64, 64, 64 ]]
  return info

def create_quantized_affinity_tasks(taskqueue, src_layer, dest_layer, shape, fill_missing=False):
  shape = Vec(*shape)

  info = create_quantized_affinity_info(src_layer, dest_layer, shape)
  destvol = CloudVolume(dest_layer, info=info)
  destvol.commit_info()

  create_downsample_scales(dest_layer, mip=0, ds_shape=shape)

  for startpt in tqdm(xyzrange( destvol.bounds.minpt, destvol.bounds.maxpt, shape ), desc="Inserting QuantizeAffinities Tasks"):
    task = QuantizeAffinitiesTask(
      source_layer_path=src_layer,
      dest_layer_path=dest_layer,
      shape=list(shape.clone()),
      offset=list(startpt.clone()),
      fill_missing=fill_missing,
    )
    task_queue.insert(task)
  task_queue.wait('Uploading')

def create_fixup_quantize_tasks(task_queue, src_layer, dest_layer, shape, points):
  shape = Vec(*shape)
  vol = CloudVolume(src_layer, 0)
  offsets = compute_fixup_offsets(vol, points, shape)

  for offset in tqdm(offsets, desc="Inserting Corrective Quantization Tasks"):
    task = QuantizeAffinitiesTask(
      source_layer_path=src_layer,
      dest_layer_path=dest_layer,
      shape=list(shape.clone()),
      offset=list(offset.clone()),
    )
    # task.execute()
    task_queue.insert(task)
  task_queue.wait('Uploading')

# split the work up into ~1000 tasks (magnitude 3)
def create_mesh_manifest_tasks(task_queue, layer_path, magnitude=3):
  assert int(magnitude) == magnitude

  start = 10 ** (magnitude - 1)
  end = 10 ** magnitude

  # For a prefix like 100, tasks 1-99 will be missed. Account for them by
  # enumerating them individually with a suffixed ':' to limit matches to
  # only those small numbers
  for prefix in range(1, start):
    task = MeshManifestTask(layer_path=layer_path, prefix=str(prefix) + ':')
    task_queue.insert(task)

  # enumerate from e.g. 100 to 999
  for prefix in range(start, end):
    task = MeshManifestTask(layer_path=layer_path, prefix=prefix)
    task_queue.insert(task)

  task_queue.wait('Uploading Manifest Tasks')

def create_hypersquare_ingest_tasks(task_queue, hypersquare_bucket_name, dataset_name, hypersquare_chunk_size, resolution, voxel_offset, volume_size, overlap):
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

  for cloudpath in tqdm(volumes_listing, desc="Creating Ingest Tasks"):
    # print(cloudpath)
    # img_task = crttask(cloudpath, 'image', IMG_LAYER_NAME)
    seg_task = crttask(cloudpath, 'segmentation', SEG_LAYER_NAME)
    # seg_task.execute()
    tq.insert(seg_task)

def create_hypersquare_consensus_tasks(task_queue, src_path, dest_path, volume_map_file, consensus_map_path):
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

  for boundstr, volume_id in tqdm(volume_map.items(), desc="Inserting HyperSquare Consensus Remap Tasks"):
    bbox = Bbox.from_filename(boundstr)
    bbox.minpt = Vec.clamp(bbox.minpt, vol.bounds.minpt, vol.bounds.maxpt)
    bbox.maxpt = Vec.clamp(bbox.maxpt, vol.bounds.minpt, vol.bounds.maxpt)

    task = HyperSquareConsensusTask(
      src_path=src_path,
      dest_path=dest_path,
      ew_volume_id=int(volume_id),
      consensus_map_path=consensus_map_path,
      shape=bbox.size3(),
      offset=bbox.minpt.clone(),
    )
    task_queue.insert(task)
  task_queue.wait()

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

