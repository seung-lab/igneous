import pytest

import copy
from functools import partial
from collections import defaultdict
import os.path
import shutil

import numpy as np
from cloudvolume import CloudVolume, EmptyVolumeException
import cloudvolume.lib as lib
from cloudfiles import CloudFiles
from taskqueue import LocalTaskQueue, TaskQueue
import tinybrain

import igneous
import igneous.task_creation as tc
from igneous.downsample_scales import create_downsample_scales

CHECKER = False # for debugging

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

@pytest.fixture(scope="module", params=[ (512,512,128) ])
def smooth_transfer_data(request):
  def make_smooth_test_image(h, w, d):
    img3d = np.zeros([h,w,d], dtype=np.uint8, order="F")

    for z in range(d):
      x = np.linspace(0, 255, w)
      y = np.linspace(0, 255, h)
      xx, yy = np.meshgrid(x, y)
      # mix a few low frequencies to stress more DCT coefficients
      img2d = (0.5 * xx + 0.3 * yy + 0.2 * np.sin(xx / 20 + z) * 50)
      img3d[:,:,z] = np.clip(img2d, 0, 255).astype(np.uint8)

    return img3d

  chunk_size = request.param
  data = make_smooth_test_image(*chunk_size)

  ds_data = tinybrain.downsample_with_averaging(data, factor=[2, 2, 1, 1], num_mips=5)
  ds_data.insert(0, data)

  for i in range(len(ds_data)):
    while ds_data[i].ndim < 4:
      ds_data[i] = ds_data[i][..., np.newaxis]

  return ds_data

path_root = "/tmp/removeme/transfertask"
srcpath = f"file://{path_root}/src"
destpath = f"file://{path_root}/dest"

def rmdest(suffix=''):
  directory = f"{path_root}/dest{suffix}"
  if os.path.exists(directory):
    shutil.rmtree(directory)
def rmsrc():
  directory = f"{path_root}/src"
  if os.path.exists(directory):
    shutil.rmtree(directory)

@pytest.fixture(scope="module")
def tq():
  return LocalTaskQueue()

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

@pytest.fixture(scope="function")
def smooth_src_cv(smooth_transfer_data):
  rmsrc()
  rmdest()
  return CloudVolume.from_numpy(
    smooth_transfer_data[0], 
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

def test_transfer_task_sharded(tq, src_cv, transfer_data):
  tasks = tc.create_image_shard_transfer_tasks(
    src_cv.cloudpath, destpath
  )
  tq.insert_all(tasks)

  dest_cv = CloudVolume(destpath)

  assert np.all(src_cv[:] == dest_cv[:])

  destpath2 = destpath + "2"

  tasks = tc.create_image_shard_transfer_tasks(
    destpath, destpath2
  )

  tq.insert_all(tasks)

  dest_cv2 = CloudVolume(destpath2)
  assert np.all(dest_cv[:] == dest_cv2[:])

  rmsrc()
  rmdest()
  rmdest('2')

def test_transfer_task_sharded_jpeg_jxl_lossless(tq, smooth_src_cv, smooth_transfer_data):
  sharded_path = f"{srcpath}/sharded/"

  smooth_src_cv.transfer_to(sharded_path, smooth_src_cv.bounds, sharded=True)

  tasks = tc.create_image_shard_transfer_tasks(
    sharded_path, destpath, 
    encoding="jpeg"
  )
  tq.insert_all(tasks)

  destpath2 = f"{destpath}2"
  destpath3 = f"{destpath}3"

  tasks = tc.create_image_shard_transfer_tasks(
    destpath, destpath2, 
    encoding="jxl",
    encoding_level=100,
    encoding_effort=1,
  )
  tq.insert_all(tasks)

  tasks = tc.create_image_shard_transfer_tasks(
    destpath2, destpath3, 
    encoding="jpeg",
  )
  tq.insert_all(tasks)

  jpg_cv = CloudVolume(destpath)
  jxl_cv = CloudVolume(destpath2)
  jpg_cv2 = CloudVolume(destpath3)

  jpg_img = jpg_cv[:][...,0]
  jxl_img = jxl_cv[:][...,0]
  jpg_img2 = jpg_cv2[:][...,0]

  assert np.all(jpg_img == jpg_img2)

  # jxl<->jpg lossless transcoding allows for 
  # the perfect recovery of jpg bitstreams, but
  # in jxl format, the jxl decoder is used and
  # the result can be +/- 1

  # A smooth or natural image is required to avoid stressing 
  # IDCT unnecessarily
  # The random image we were using prior produced large deltas
  # but that was pathological behavior.
  assert np.allclose(jpg_img, jxl_img, atol=1)

  rmsrc()
  rmdest()
  rmdest('2')
  rmdest('3')







