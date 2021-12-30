import pytest

import copy
from functools import partial
from collections import defaultdict
import os.path
import shutil

import numpy as np
from cloudvolume import CloudVolume, EmptyVolumeException, view
import cloudvolume.lib as lib
from cloudfiles import CloudFiles
from taskqueue import MockTaskQueue, TaskQueue
import tinybrain

import igneous
import igneous.task_creation as tc
from igneous.downsample_scales import create_downsample_scales

CHECKER = False

@pytest.fixture(scope="module", params=[ (512,512,128), (600, 600, 200) ])
def transfer_data(request):
  if not CHECKER:
    data = np.random.randint(0xFF, size=request.param, dtype=np.uint8)
  else:
    data = np.zeros((512,512,128), dtype=np.uint8)
    i = 1
    for x in range(8):
      for y in range(8):
        for z in range(2):
          data[64*x:64*(x+1), 64*y:64*(y+1), 64*z:64*(z+1)] = i
          i += 1 

  ds_data = tinybrain.downsample_with_averaging(data, factor=[2, 2, 1, 1], num_mips=5)
  ds_data.insert(0, data)

  for i in range(len(ds_data)):
    while ds_data[i].ndim < 4:
      ds_data[i] = ds_data[i][..., np.newaxis]

  return ds_data  

path_root = "/tmp/removeme/transfertask"
srcpath = f"file://{path_root}/src"
destpath = f"file://{path_root}/dest"

def rmdest():
  directory = f"{path_root}/dest"
  if os.path.exists(directory):
    shutil.rmtree(directory)
def rmsrc():
  directory = f"{path_root}/src"
  if os.path.exists(directory):
    shutil.rmtree(directory)

@pytest.fixture(scope="module")
def tq():
  return MockTaskQueue()

@pytest.fixture(scope="function")
def src_cv(transfer_data):
  rmsrc()
  rmdest()
  return CloudVolume.from_numpy(
    transfer_data[0], 
    vol_path=srcpath,
    resolution=(1,1,1), 
    voxel_offset=(0,0,0), 
    chunk_size=(64,64,64), 
    layer_type="image", 
    max_mip=0,
  )

def test_transfer_task_vanilla(tq, src_cv, transfer_data):
  tasks = tc.create_transfer_tasks(src_cv.cloudpath, destpath)
  tq.insert_all(tasks)

  dest_cv = CloudVolume(destpath)
  assert np.all(src_cv[:] == dest_cv[:])
  rmsrc()
  rmdest()

def test_transfer_task_rechunk(tq, src_cv, transfer_data):
  tasks = tc.create_transfer_tasks(
    src_cv.cloudpath, destpath, chunk_size=(50,50,50)
  )
  tq.insert_all(tasks)
  dest_cv = CloudVolume(destpath)
  assert len(dest_cv.scales) == 5
  assert np.all(src_cv[:] == dest_cv[:])
  for mip in range(1, 5):
    dest_cv.mip = mip
    assert np.all(dest_cv[:] == transfer_data[mip])
  rmsrc()
  rmdest()

# chunk size (64,64,64) should test transfer_to pathway
@pytest.mark.parametrize("chunk_size", [ (50,50,50), (64,64,64) ])
def test_transfer_task_skip_downsample(tq, src_cv, chunk_size):
  tasks = tc.create_transfer_tasks(
    src_cv.cloudpath, destpath, 
    chunk_size=chunk_size, skip_downsamples=True
  )
  tq.insert_all(tasks)
  dest_cv = CloudVolume(destpath)
  assert len(dest_cv.scales) == 1
  assert np.all(src_cv[:] == dest_cv[:])
  rmsrc()
  rmdest()

def test_transfer_task_dest_offset(tq, src_cv, transfer_data):
  tasks = tc.create_transfer_tasks(
    src_cv.cloudpath, destpath, 
    chunk_size=(50,50,50), dest_voxel_offset=(100, 100, 100)
  )
  tq.insert_all(tasks)
  dest_cv = CloudVolume(destpath)
  assert len(dest_cv.scales) == 5
  assert tuple(dest_cv.voxel_offset) == (100, 100, 100)
  assert tuple(src_cv.voxel_offset) == (0, 0, 0)
  assert np.all(src_cv[:] == dest_cv[:])
  for mip in range(1, 5):
    dest_cv.mip = mip
    assert np.all(dest_cv[:] == transfer_data[mip])
  rmsrc()
  rmdest()

def test_transfer_task_subset(tq, src_cv, transfer_data):
  dest_cv = CloudVolume(destpath, info=copy.deepcopy(src_cv.info))
  dest_cv.scale["size"] = [256, 256, 64]
  dest_cv.commit_info()

  tasks = tc.create_transfer_tasks(
    src_cv.cloudpath, destpath, 
    chunk_size=(64,64,64), 
    translate=(-128, -128, -64),
  )
  tq.insert_all(tasks)
  dest_cv.refresh_info()

  assert len(dest_cv.scales) == 3
  assert np.all(src_cv[128:128+256, 128:128+256, 64:64+64] == dest_cv[:])

  rmsrc()
  rmdest()

