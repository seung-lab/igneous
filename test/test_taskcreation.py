import pytest

import os
import shutil

from cloudvolume import Storage
from cloudvolume.lib import Bbox
import numpy as np

import igneous.task_creation as task_creation
from layer_harness import layer_path, delete_layer, create_storage

def test_upload_build_chunks():
  delete_layer()
  storage = create_storage()

  # Easiest test, power of two sizes, no offset
  random_data = np.random.randint(255, size=(1024, 1024, 256, 1), dtype=np.uint8)
  task_creation.upload_build_chunks(storage, random_data, offset=(0,0,0), build_chunk_size=(512,512,64))

  build_path = os.path.join(layer_path, 'layer/build')

  files = os.listdir(build_path)
  assert len(files) == (2 * 2 * 4)
  assert '0-512_0-512_0-64' in files 
  assert '512-1024_512-1024_192-256' in files
  assert '0-512_512-1024_0-64' in files

  # Prime numbered offsets

  delete_layer()
  task_creation.upload_build_chunks(storage, random_data, offset=(3,23,5), build_chunk_size=(512,512,64))
  files = os.listdir(build_path)
  assert len(files) == (2 * 2 * 4)
  assert '3-515_23-535_5-69' in files 
  assert '515-1027_535-1047_197-261' in files
  assert '3-515_535-1047_5-69' in files


  # Edges are not easily divided
  random_data2 = random_data[:1000, :1000, :156, :]
  delete_layer()
  task_creation.upload_build_chunks(storage, random_data2, offset=(0,0,0), build_chunk_size=(512,512,64))
  files = os.listdir(build_path)
  assert len(files) == (2 * 2 * 3)
  assert '0-512_0-512_0-64' in files 
  assert '512-1000_512-1000_128-156' in files
  assert '0-512_512-1000_0-64' in files

  # Edges not easily divided + offset
  delete_layer()
  task_creation.upload_build_chunks(storage, random_data2, offset=(3,23,5), build_chunk_size=(512,512,64))
  files = os.listdir(build_path)
  assert len(files) == (2 * 2 * 3)
  assert '3-515_23-535_5-69' in files 
  assert '515-1003_535-1023_133-161' in files
  assert '3-515_535-1023_5-69' in files

def test_compute_build_bounding_box():
  delete_layer()
  storage = create_storage()
  random_data = np.random.randint(255, size=(1024, 1024, 256), dtype=np.uint8)

  # Easy, power of two, 0 offsets
  task_creation.upload_build_chunks(storage, random_data, offset=(0,0,0), build_chunk_size=(512,512,64))
  bounds, chunk_size = task_creation.compute_build_bounding_box(storage)

  assert np.all(chunk_size == (512, 512, 64))
  assert bounds == Bbox( (0,0,0), (1024, 1024, 256) )

  # Prime offsets
  delete_layer()
  task_creation.upload_build_chunks(storage, random_data, offset=(3,23,5), build_chunk_size=(512,512,64))
  bounds, chunk_size = task_creation.compute_build_bounding_box(storage)

  assert np.all(chunk_size == (512, 512, 64))
  assert bounds == Bbox( (3,23,5), (1027, 1047, 261) )

  # Non-power of two edges, 0 offsets
  random_data2 = random_data[:1000, :1000, :156]
  delete_layer()
  task_creation.upload_build_chunks(storage, random_data2, offset=(0,0,0), build_chunk_size=(512,512,64))
  bounds, chunk_size = task_creation.compute_build_bounding_box(storage)

  assert np.all(chunk_size == (512, 512, 64))
  assert bounds == Bbox( (0,0,0), (1000, 1000, 156) )

  # Non-power of two edges, offsets
  delete_layer()
  task_creation.upload_build_chunks(storage, random_data2, offset=(3,23,5), build_chunk_size=(512,512,64))
  bounds, chunk_size = task_creation.compute_build_bounding_box(storage)

  assert np.all(chunk_size == (512, 512, 64))
  assert bounds == Bbox( (3,23,5), (1003, 1023, 161) )

def test_create_info_file_from_build():
  delete_layer()
  random_data = np.random.randint(255, size=(1024, 1024, 128, 3), dtype=np.uint8)
  
  storage = create_storage()
  task_creation.upload_build_chunks(storage, random_data, offset=(3,23,5), build_chunk_size=(512,512,64))
  info = task_creation.create_info_file_from_build(storage.layer_path, 'image', resolution=[7,5,17], encoding='raw')

  assert info['num_channels'] == 3
  assert info['type'] == 'image'
  assert info['data_type'] == 'uint8'
  assert info['scales'][0]['encoding'] == 'raw'
  assert len(info['scales']) == 4

  assert np.all(tuple(info['scales'][0]['resolution']) == (7,5,17))
  assert np.all(tuple(info['scales'][0]['size']) == (1024,1024,128))
  assert np.all(tuple(info['scales'][0]['voxel_offset']) == (3,23,5))
  assert np.all(tuple(info['scales'][0]['chunk_sizes'][0]) == (64,64,64))

  assert np.all(tuple(info['scales'][1]['resolution']) == (14,10,17))
  assert np.all(tuple(info['scales'][1]['size']) == (512,512,128))
  assert np.all(tuple(info['scales'][1]['voxel_offset']) == (1,11,5))
  assert np.all(tuple(info['scales'][1]['chunk_sizes'][0]) == (64,64,64))

  assert np.all(tuple(info['scales'][2]['resolution']) == (28,20,17))
  assert np.all(tuple(info['scales'][2]['size']) == (256,256,128))
  assert np.all(tuple(info['scales'][2]['voxel_offset']) == (0,5,5))
  assert np.all(tuple(info['scales'][2]['chunk_sizes'][0]) == (64,64,64))

  assert np.all(tuple(info['scales'][3]['resolution']) == (56,40,17))
  assert np.all(tuple(info['scales'][3]['size']) == (128,128,128))
  assert np.all(tuple(info['scales'][3]['voxel_offset']) == (0,2,5))
  assert np.all(tuple(info['scales'][3]['chunk_sizes'][0]) == (64,64,64))







