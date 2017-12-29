import pytest

import os
import shutil

from cloudvolume import Storage
from cloudvolume.lib import Bbox
import numpy as np

import igneous.task_creation as task_creation
from .layer_harness import layer_path, delete_layer, create_storage

def test_upload_build_chunks():
  delete_layer()
  storage = create_storage()

  # Easiest test, power of two sizes, no offset
  random_data = np.random.randint(255, size=(256, 256, 128, 1), dtype=np.uint8)
  task_creation.upload_build_chunks(storage, random_data, offset=(0,0,0), build_chunk_size=(64,64,32))

  build_path = os.path.join(layer_path, 'layer/build')

  files = os.listdir(build_path)
  assert len(files) == (4 * 4 * 4)
  assert '0-64_0-64_0-32' in files 
  assert '64-128_64-128_64-96' in files
  assert '0-64_128-192_0-32' in files

  # Prime numbered offsets

  delete_layer()
  task_creation.upload_build_chunks(storage, random_data, offset=(3,23,5), build_chunk_size=(64,64,64))
  files = os.listdir(build_path)
  assert len(files) == (4 * 4 * 2)
  assert '3-67_23-87_5-69' in files 
  assert '67-131_87-151_69-133' in files
  assert '3-67_87-151_5-69' in files


  # Edges are not easily divided
  random_data2 = random_data[:100, :100, :77, :]
  delete_layer()
  task_creation.upload_build_chunks(storage, random_data2, offset=(0,0,0), build_chunk_size=(32,32,32))
  files = os.listdir(build_path)
  assert len(files) == (4 * 4 * 3)
  assert '0-32_0-32_0-32' in files 
  assert '96-100_96-100_64-77' in files
  assert '0-32_96-100_0-32' in files

  # Edges not easily divided + offset
  delete_layer()
  task_creation.upload_build_chunks(storage, random_data2, offset=(3,23,5), build_chunk_size=(32,32,32))
  files = os.listdir(build_path)
  assert len(files) == (4 * 4 * 3)
  assert '3-35_23-55_5-37' in files 
  assert '99-103_119-123_69-82' in files
  assert '3-35_87-119_5-37' in files

def test_compute_build_bounding_box():
  delete_layer()
  storage = create_storage()
  random_data = np.random.randint(255, size=(256, 256, 128), dtype=np.uint8)

  # Easy, power of two, 0 offsets
  task_creation.upload_build_chunks(storage, random_data, offset=(0,0,0), build_chunk_size=(128,128,32))
  bounds, chunk_size = task_creation.compute_build_bounding_box(storage)

  assert np.all(chunk_size == (128, 128, 32))
  assert bounds == Bbox( (0,0,0), (256, 256, 128) )

  # Prime offsets
  delete_layer()
  task_creation.upload_build_chunks(storage, random_data, offset=(3,23,5), build_chunk_size=(128,128,32))
  bounds, chunk_size = task_creation.compute_build_bounding_box(storage)

  assert np.all(chunk_size == (128, 128, 32))
  assert bounds == Bbox( (3,23,5), (256+3, 256+23, 128+5) )

  # Non-power of two edges, 0 offsets
  random_data2 = random_data[:100, :100, :106]
  delete_layer()
  task_creation.upload_build_chunks(storage, random_data2, offset=(0,0,0), build_chunk_size=(32,32,32))
  bounds, chunk_size = task_creation.compute_build_bounding_box(storage)

  assert np.all(chunk_size == (32, 32, 32))
  assert bounds == Bbox( (0,0,0), (100, 100, 106) )

  # Non-power of two edges, offsets
  delete_layer()
  task_creation.upload_build_chunks(storage, random_data2, offset=(3,23,5), build_chunk_size=(64,64,32))
  bounds, chunk_size = task_creation.compute_build_bounding_box(storage)

  assert np.all(chunk_size == (64, 64, 32))
  assert bounds == Bbox( (3,23,5), (100+3, 100+23, 106+5) )

def test_create_info_file_from_build():
  delete_layer()
  random_data = np.random.randint(255, size=(256, 256, 128, 3), dtype=np.uint8)
  
  storage = create_storage()
  task_creation.upload_build_chunks(storage, random_data, offset=(3,23,5), build_chunk_size=(256,256,64))
  info = task_creation.create_info_file_from_build(storage.layer_path, 'image', resolution=[7,5,17], encoding='raw')

  print(info)

  assert info['num_channels'] == 3
  assert info['type'] == 'image'
  assert info['data_type'] == 'uint8'
  assert info['scales'][0]['encoding'] == 'raw'
  assert len(info['scales']) == 3

  assert np.all(tuple(info['scales'][0]['resolution']) == (7,5,17))
  assert np.all(tuple(info['scales'][0]['size']) == (256,256,128))
  assert np.all(tuple(info['scales'][0]['voxel_offset']) == (3,23,5))
  assert np.all(tuple(info['scales'][0]['chunk_sizes'][0]) == (64,64,64))

  assert np.all(tuple(info['scales'][1]['resolution']) == (14,10,17))
  assert np.all(tuple(info['scales'][1]['size']) == (128,128,128))
  assert np.all(tuple(info['scales'][1]['voxel_offset']) == (1,11,5))
  assert np.all(tuple(info['scales'][1]['chunk_sizes'][0]) == (64,64,64))

  assert np.all(tuple(info['scales'][2]['resolution']) == (28,20,17))
  assert np.all(tuple(info['scales'][2]['size']) == (64,64,128))
  assert np.all(tuple(info['scales'][2]['voxel_offset']) == (0,5,5))
  assert np.all(tuple(info['scales'][2]['chunk_sizes'][0]) == (64,64,64))







