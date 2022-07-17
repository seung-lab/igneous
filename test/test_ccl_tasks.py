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

@pytest.fixture(scope="module", params=[ (512,512,128) ])
def transfer_data(request):
  data = np.zeros((512,512,128), dtype=np.uint8)
  i = 1
  for x in range(data.shape[0] // 64):
    for y in range(data.shape[1] // 64):
      for z in range(data.shape[2] // 64):
        data[64*x:64*(x+1), 64*y:64*(y+1), 64*z:64*(z+1)] = i
        i += 1 

  return data  

path_root = "/tmp/removeme/igneous/ccltask"
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
    transfer_data, 
    vol_path=srcpath,
    resolution=(1,1,1), 
    voxel_offset=(0,0,0), 
    chunk_size=(64,64,64), 
    layer_type="image", 
    max_mip=0,
  )

@pytest.mark.parametrize("lower", (None,255,0))
@pytest.mark.parametrize("fill_missing", [True,False])
def test_ccl_tasks(tq, src_cv, transfer_data, lower, fill_missing):
  shape = (128,128,128)
  tasks = tc.create_ccl_face_tasks(
    src_cv.cloudpath, mip=0, shape=shape, 
    threshold_lte=lower, fill_missing=fill_missing,
  )
  tq.insert_all(tasks)

  cf = CloudFiles(src_cv.cloudpath)
  faces = list(cf.list(cf.join(src_cv.key, "ccl", "faces")))

  answer = [
    '1_1_1/ccl/faces/0-0-0-xy.cpso', '1_1_1/ccl/faces/0-0-0-xz.cpso', '1_1_1/ccl/faces/0-0-0-yz.cpso', 
    '1_1_1/ccl/faces/0-1-0-xy.cpso', '1_1_1/ccl/faces/0-1-0-xz.cpso', '1_1_1/ccl/faces/0-1-0-yz.cpso', 
    '1_1_1/ccl/faces/0-2-0-xy.cpso', '1_1_1/ccl/faces/0-2-0-xz.cpso', '1_1_1/ccl/faces/0-2-0-yz.cpso', 
    '1_1_1/ccl/faces/0-3-0-xy.cpso', '1_1_1/ccl/faces/0-3-0-xz.cpso', '1_1_1/ccl/faces/0-3-0-yz.cpso', 
    '1_1_1/ccl/faces/1-0-0-xy.cpso', '1_1_1/ccl/faces/1-0-0-xz.cpso', '1_1_1/ccl/faces/1-0-0-yz.cpso', 
    '1_1_1/ccl/faces/1-1-0-xy.cpso', '1_1_1/ccl/faces/1-1-0-xz.cpso', '1_1_1/ccl/faces/1-1-0-yz.cpso', 
    '1_1_1/ccl/faces/1-2-0-xy.cpso', '1_1_1/ccl/faces/1-2-0-xz.cpso', '1_1_1/ccl/faces/1-2-0-yz.cpso', 
    '1_1_1/ccl/faces/1-3-0-xy.cpso', '1_1_1/ccl/faces/1-3-0-xz.cpso', '1_1_1/ccl/faces/1-3-0-yz.cpso', 
    '1_1_1/ccl/faces/2-0-0-xy.cpso', '1_1_1/ccl/faces/2-0-0-xz.cpso', '1_1_1/ccl/faces/2-0-0-yz.cpso', 
    '1_1_1/ccl/faces/2-1-0-xy.cpso', '1_1_1/ccl/faces/2-1-0-xz.cpso', '1_1_1/ccl/faces/2-1-0-yz.cpso', 
    '1_1_1/ccl/faces/2-2-0-xy.cpso', '1_1_1/ccl/faces/2-2-0-xz.cpso', '1_1_1/ccl/faces/2-2-0-yz.cpso', 
    '1_1_1/ccl/faces/2-3-0-xy.cpso', '1_1_1/ccl/faces/2-3-0-xz.cpso', '1_1_1/ccl/faces/2-3-0-yz.cpso', 
    '1_1_1/ccl/faces/3-0-0-xy.cpso', '1_1_1/ccl/faces/3-0-0-xz.cpso', '1_1_1/ccl/faces/3-0-0-yz.cpso', 
    '1_1_1/ccl/faces/3-1-0-xy.cpso', '1_1_1/ccl/faces/3-1-0-xz.cpso', '1_1_1/ccl/faces/3-1-0-yz.cpso', 
    '1_1_1/ccl/faces/3-2-0-xy.cpso', '1_1_1/ccl/faces/3-2-0-xz.cpso', '1_1_1/ccl/faces/3-2-0-yz.cpso', 
    '1_1_1/ccl/faces/3-3-0-xy.cpso', '1_1_1/ccl/faces/3-3-0-xz.cpso', '1_1_1/ccl/faces/3-3-0-yz.cpso'
  ]
  faces.sort()
  answer.sort()

  assert len(faces) == 16 * 3
  assert faces == answer

  tasks = tc.create_ccl_equivalence_tasks(
    src_cv.cloudpath, mip=0, shape=shape, 
    threshold_lte=lower, fill_missing=fill_missing,
  )
  tq.insert_all(tasks)

  cf = CloudFiles(src_cv.cloudpath)
  equivs = list(cf.list(cf.join(src_cv.key, "ccl", "equivalences")))

  answer = [
    '1_1_1/ccl/equivalences/0-0-0.json', '1_1_1/ccl/equivalences/0-1-0.json', '1_1_1/ccl/equivalences/0-2-0.json', 
    '1_1_1/ccl/equivalences/0-3-0.json', '1_1_1/ccl/equivalences/1-0-0.json', '1_1_1/ccl/equivalences/1-1-0.json', 
    '1_1_1/ccl/equivalences/1-2-0.json', '1_1_1/ccl/equivalences/1-3-0.json', '1_1_1/ccl/equivalences/2-0-0.json', 
    '1_1_1/ccl/equivalences/2-1-0.json', '1_1_1/ccl/equivalences/2-2-0.json', '1_1_1/ccl/equivalences/2-3-0.json', 
    '1_1_1/ccl/equivalences/3-0-0.json', '1_1_1/ccl/equivalences/3-1-0.json', '1_1_1/ccl/equivalences/3-2-0.json', 
    '1_1_1/ccl/equivalences/3-3-0.json'
  ]
  answer.sort()
  equivs.sort()
  assert equivs == answer

  import igneous.tasks.image.ccl
  igneous.tasks.image.ccl.create_relabeling(src_cv.cloudpath, mip=0, shape=shape)

  tasks = tc.create_ccl_relabel_tasks(
    src_cv.cloudpath, destpath, 
    mip=0, shape=shape,
    threshold_lte=lower,
    fill_missing=fill_missing,
  )
  tq.insert_all(tasks)

  cv_dest = CloudVolume(destpath)
  cc_labels = cv_dest[:][:,:,:,0]

  uniq = np.unique(cc_labels)
  if lower is None:
    assert len(uniq) == 128
    assert np.all(uniq == np.arange(1,129))
  elif lower == 255:
    assert len(uniq) == 1
    assert uniq[0] == 1
  elif lower == 0:
    assert len(uniq) == 1
    assert uniq[0] == 0

  rmsrc()
  rmdest()

