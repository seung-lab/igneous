import pytest

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

@pytest.fixture(scope="module")
def transfer_data():
    random_data = np.random.randint(0xFF, size=(512,512,128), dtype=np.uint8)
    ds_data = tinybrain.downsample_with_averaging(random_data, factor=[2, 2, 1, 1], num_mips=5)
    ds_data.insert(0, random_data)

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

@pytest.fixture(scope="module")
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
    tasks = tc.create_transfer_tasks(src_cv.cloudpath, destpath, chunk_size=(50,50,50))
    tq.insert_all(tasks)
    dest_cv = CloudVolume(destpath)
    assert len(dest_cv.scales) == 4
    assert np.all(src_cv[:] == dest_cv[:])
    for mip in range(1, 4):
        dest_cv.mip = mip
        assert np.all(dest_cv[:] == transfer_data[mip])
    rmsrc()
    rmdest()

def test_transfer_task_skip_downsample(tq, src_cv):
    tasks = tc.create_transfer_tasks(
        src_cv.cloudpath, destpath, 
        chunk_size=(50,50,50), skip_downsamples=True
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
    assert len(dest_cv.scales) == 4
    assert tuple(dest_cv.voxel_offset) == (100, 100, 100)
    assert tuple(src_cv.voxel_offset) == (0, 0, 0)
    assert np.all(src_cv[:] == dest_cv[:])
    for mip in range(1, 4):
        dest_cv.mip = mip
        assert np.all(dest_cv[:] == transfer_data[mip])
    rmsrc()
    rmdest()