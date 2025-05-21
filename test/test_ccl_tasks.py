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
from taskqueue import MockTaskQueue, TaskQueue
import tinybrain

import igneous
import igneous.task_creation as tc
from igneous.downsample_scales import create_downsample_scales

@pytest.fixture(scope="module")
def checker_data(request):
  data = np.zeros((512,512,128), dtype=np.uint8)
  i = 1
  for x in range(data.shape[0] // 64):
    for y in range(data.shape[1] // 64):
      for z in range(data.shape[2] // 64):
        data[64*x:64*(x+1), 64*y:64*(y+1), 64*z:64*(z+1)] = i
        i += 1 

  return data

@pytest.fixture(scope="module")
def connectomics_data(request):
  import crackle
  return crackle.load("./test/connectomics.npy.ckl.gz")

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
def src_cv(checker_data):
  rmsrc()
  rmdest()
  return CloudVolume.from_numpy(
    checker_data, 
    vol_path=srcpath,
    resolution=(1,1,1), 
    voxel_offset=(0,0,0), 
    chunk_size=(128,128,64), 
    layer_type="image", 
    max_mip=0,
  )

@pytest.fixture(scope="function")
def connectomics_cv(connectomics_data):
  rmsrc()
  rmdest()
  return CloudVolume.from_numpy(
    connectomics_data, 
    vol_path=srcpath,
    resolution=(1,1,1), 
    voxel_offset=(0,0,0), 
    chunk_size=(128,128,64), 
    layer_type="segmentation", 
    max_mip=0,
  )

@pytest.mark.parametrize("dtype", [ np.uint32, np.float32 ])
def test_threshold_image(dtype):
  import igneous.tasks.image.ccl as ccl
  sz = 100
  image = np.arange(0,sz**3).reshape((sz,sz,sz), order="F")
  image = image.astype(dtype)

  res = ccl.threshold_image(image, None, None)  
  assert np.all(image == res)

  res = ccl.threshold_image(image, sz**3+1, None)  
  assert np.all(res == 1)

  res = ccl.threshold_image(image, None, 0)  
  assert np.all(res == 1)

  res = ccl.threshold_image(image, sz**3+1, 0)
  assert np.all(res == 1)

  res = ccl.threshold_image(image, None, 1)
  assert np.all(res[0,0,0] == 0)
  res[0,0,0] = 1
  assert np.all(res == 1)

  res = ccl.threshold_image(image, sz**3+1, 1)
  assert np.all(res[0,0,0] == 0)
  res[0,0,0] = 1
  assert np.all(res == 1)

@pytest.mark.parametrize("upper", (None,255,0))
@pytest.mark.parametrize("fill_missing", [True,False])
@pytest.mark.parametrize("dust_threshold", [0, (64**3) + 1])
def test_ccl_tasks(
  tq, src_cv, checker_data, 
  upper, fill_missing,
  dust_threshold
):
  shape = (128,128,128)
  tasks = tc.create_ccl_face_tasks(
    src_cv.cloudpath, mip=0, shape=shape, 
    threshold_lte=upper, fill_missing=fill_missing,
    dust_threshold=dust_threshold,
  )
  tq.insert_all(tasks)

  cf = CloudFiles(src_cv.cloudpath)
  faces = list(cf.list(cf.join(src_cv.key, "ccl", "faces")))

  answer = [
    '1_1_1/ccl/faces/0-0-0-xy.ckl', '1_1_1/ccl/faces/0-0-0-xz.ckl', '1_1_1/ccl/faces/0-0-0-yz.ckl', 
    '1_1_1/ccl/faces/0-1-0-xy.ckl', '1_1_1/ccl/faces/0-1-0-xz.ckl', '1_1_1/ccl/faces/0-1-0-yz.ckl', 
    '1_1_1/ccl/faces/0-2-0-xy.ckl', '1_1_1/ccl/faces/0-2-0-xz.ckl', '1_1_1/ccl/faces/0-2-0-yz.ckl', 
    '1_1_1/ccl/faces/0-3-0-xy.ckl', '1_1_1/ccl/faces/0-3-0-xz.ckl', '1_1_1/ccl/faces/0-3-0-yz.ckl', 
    '1_1_1/ccl/faces/1-0-0-xy.ckl', '1_1_1/ccl/faces/1-0-0-xz.ckl', '1_1_1/ccl/faces/1-0-0-yz.ckl', 
    '1_1_1/ccl/faces/1-1-0-xy.ckl', '1_1_1/ccl/faces/1-1-0-xz.ckl', '1_1_1/ccl/faces/1-1-0-yz.ckl', 
    '1_1_1/ccl/faces/1-2-0-xy.ckl', '1_1_1/ccl/faces/1-2-0-xz.ckl', '1_1_1/ccl/faces/1-2-0-yz.ckl', 
    '1_1_1/ccl/faces/1-3-0-xy.ckl', '1_1_1/ccl/faces/1-3-0-xz.ckl', '1_1_1/ccl/faces/1-3-0-yz.ckl', 
    '1_1_1/ccl/faces/2-0-0-xy.ckl', '1_1_1/ccl/faces/2-0-0-xz.ckl', '1_1_1/ccl/faces/2-0-0-yz.ckl', 
    '1_1_1/ccl/faces/2-1-0-xy.ckl', '1_1_1/ccl/faces/2-1-0-xz.ckl', '1_1_1/ccl/faces/2-1-0-yz.ckl', 
    '1_1_1/ccl/faces/2-2-0-xy.ckl', '1_1_1/ccl/faces/2-2-0-xz.ckl', '1_1_1/ccl/faces/2-2-0-yz.ckl', 
    '1_1_1/ccl/faces/2-3-0-xy.ckl', '1_1_1/ccl/faces/2-3-0-xz.ckl', '1_1_1/ccl/faces/2-3-0-yz.ckl', 
    '1_1_1/ccl/faces/3-0-0-xy.ckl', '1_1_1/ccl/faces/3-0-0-xz.ckl', '1_1_1/ccl/faces/3-0-0-yz.ckl', 
    '1_1_1/ccl/faces/3-1-0-xy.ckl', '1_1_1/ccl/faces/3-1-0-xz.ckl', '1_1_1/ccl/faces/3-1-0-yz.ckl', 
    '1_1_1/ccl/faces/3-2-0-xy.ckl', '1_1_1/ccl/faces/3-2-0-xz.ckl', '1_1_1/ccl/faces/3-2-0-yz.ckl', 
    '1_1_1/ccl/faces/3-3-0-xy.ckl', '1_1_1/ccl/faces/3-3-0-xz.ckl', '1_1_1/ccl/faces/3-3-0-yz.ckl'
  ]
  faces.sort()
  answer.sort()

  assert len(faces) == 16 * 3
  assert faces == answer

  tasks = tc.create_ccl_equivalence_tasks(
    src_cv.cloudpath, mip=0, shape=shape, 
    threshold_lte=upper, fill_missing=fill_missing,
    dust_threshold=dust_threshold,
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
    threshold_lte=upper,
    fill_missing=fill_missing,
    dust_threshold=dust_threshold,
  )
  tq.insert_all(tasks)

  cv_dest = CloudVolume(destpath)
  cc_labels = cv_dest[:][:,:,:,0]

  uniq = np.unique(cc_labels)
  if dust_threshold > 0:
    if upper in (None, 0):
      assert len(uniq) == 1
      assert uniq[0] == 0
    else:
      assert len(uniq) == 1
      assert uniq[0] == 1
  else:
    if upper is None:
      assert len(uniq) == 128
      assert np.all(uniq == np.arange(1,129))
    elif upper == 255:
      assert len(uniq) == 1
      assert uniq[0] == 1
    elif upper == 0:
      assert len(uniq) == 1
      assert uniq[0] == 0

  rmsrc()
  rmdest()

def test_ccl_tasks_connectomics(
  tq, connectomics_cv, connectomics_data,
):
  shape = (128,128,128)
  tasks = tc.create_ccl_face_tasks(
    connectomics_cv.cloudpath, mip=0, shape=shape,
  )
  tq.insert_all(tasks)

  tasks = tc.create_ccl_equivalence_tasks(
    connectomics_cv.cloudpath, mip=0, shape=shape, 
  )
  tq.insert_all(tasks)

  import igneous.tasks.image.ccl
  igneous.tasks.image.ccl.create_relabeling(connectomics_cv.cloudpath, mip=0, shape=shape)

  tasks = tc.create_ccl_relabel_tasks(
    connectomics_cv.cloudpath, destpath, 
    mip=0, shape=shape,
  )
  tq.insert_all(tasks)

  cv_dest = CloudVolume(destpath)
  cc_labels = cv_dest[:][:,:,:,0]

  import fastremap
  import cc3d

  cc_orig = cc3d.connected_components(
    connectomics_data, connectivity=6
  )

  cc_labels, _ = fastremap.renumber(cc_labels)
  cc_orig, _ = fastremap.renumber(cc_orig)

  assert np.all(cc_labels == cc_orig)

  rmsrc()
  rmdest()

def test_numberify():
  import igneous_cli

  assert igneous_cli.numberify('.111') == float(0.111)
  assert igneous_cli.numberify('33') == int(33)
  assert igneous_cli.numberify(33) == int(33)
  assert igneous_cli.numberify(12.2342) == 12.2342
  assert igneous_cli.numberify('1.23') == float('1.23')
  assert igneous_cli.numberify(1) == 1
  assert igneous_cli.numberify(0) == 0

