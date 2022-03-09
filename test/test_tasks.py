import pytest

from functools import partial
from builtins import range
from collections import defaultdict
import json
import os.path
import shutil

import numpy as np
from cloudvolume import CloudVolume, EmptyVolumeException, view
import cloudvolume.lib as lib
from cloudfiles import CloudFiles
from taskqueue import MockTaskQueue, TaskQueue
import tinybrain

from igneous import (
    DownsampleTask, MeshTask, 
    MeshManifestPrefixTask, MeshManifestFilesystemTask,
    QuantizeTask, HyperSquareConsensusTask,
    DeleteTask, BlackoutTask, ContrastNormalizationTask
)
import igneous.task_creation as tc
from igneous.task_creation import create_downsampling_tasks, create_quantized_affinity_info
from igneous.downsample_scales import create_downsample_scales
from .layer_harness import delete_layer, create_layer

@pytest.mark.parametrize("compression_method", ( None, 'gzip', 'br',))
def test_downsample_no_offset(compression_method):
    delete_layer()
    cf, data = create_layer(size=(1024,1024,128,1), offset=(0,0,0))
    cv = CloudVolume(cf.cloudpath)
    assert len(cv.scales) == 1
    assert len(cv.available_mips) == 1

    cv.commit_info()

    tq = MockTaskQueue()
    tasks = create_downsampling_tasks(cf.cloudpath, mip=0, num_mips=4, compress=compression_method)
    tq.insert_all(tasks)

    cv.refresh_info()

    assert len(cv.available_mips) == 5
    assert np.array_equal(cv.mip_volume_size(0), [ 1024, 1024, 128 ])
    assert np.array_equal(cv.mip_volume_size(1), [  512,  512, 128 ])
    assert np.array_equal(cv.mip_volume_size(2), [  256,  256, 128 ])
    assert np.array_equal(cv.mip_volume_size(3), [  128,  128, 128 ])
    assert np.array_equal(cv.mip_volume_size(4), [   64,   64, 128 ])
    
    slice64 = np.s_[0:64, 0:64, 0:64]

    cv.mip = 0
    assert np.all(cv[slice64] == data[slice64])

    data_ds1, = tinybrain.downsample_with_averaging(data, factor=[2, 2, 1, 1])
    cv.mip = 1
    assert np.all(cv[slice64] == data_ds1[slice64])

    data_ds2, = tinybrain.downsample_with_averaging(data, factor=[4, 4, 1, 1])
    cv.mip = 2
    assert np.all(cv[slice64] == data_ds2[slice64])

    data_ds3, = tinybrain.downsample_with_averaging(data, factor=[8, 8, 1, 1])
    cv.mip = 3
    assert np.all(cv[slice64] == data_ds3[slice64])

    data_ds4, = tinybrain.downsample_with_averaging(data, factor=[16, 16, 1, 1])
    cv.mip = 4
    assert np.all(cv[slice64] == data_ds4[slice64])

def test_downsample_no_offset_2x2x2():
    delete_layer()
    cf, data = create_layer(size=(512,512,512,1), offset=(0,0,0))
    cv = CloudVolume(cf.cloudpath)
    assert len(cv.scales) == 1
    assert len(cv.available_mips) == 1

    cv.commit_info()

    tq = MockTaskQueue()
    tasks = create_downsampling_tasks(
        cf.cloudpath, mip=0, num_mips=3, 
        compress=None, factor=(2,2,2)
    )
    tq.insert_all(tasks)

    cv.refresh_info()

    assert len(cv.available_mips) == 4
    assert np.array_equal(cv.mip_volume_size(0), [ 512, 512, 512 ])
    assert np.array_equal(cv.mip_volume_size(1), [ 256, 256, 256 ])
    assert np.array_equal(cv.mip_volume_size(2), [ 128, 128, 128 ])
    assert np.array_equal(cv.mip_volume_size(3), [  64,  64,  64 ])
    
    slice64 = np.s_[0:64, 0:64, 0:64]

    cv.mip = 0
    assert np.all(cv[slice64] == data[slice64])

    data_ds1, = tinybrain.downsample_with_averaging(data, factor=[2, 2, 2, 1])
    cv.mip = 1
    assert np.all(cv[slice64] == data_ds1[slice64])

    data_ds2, = tinybrain.downsample_with_averaging(data, factor=[4, 4, 4, 1])
    cv.mip = 2
    assert np.all(cv[slice64] == data_ds2[slice64])

    data_ds3, = tinybrain.downsample_with_averaging(data, factor=[8, 8, 8, 1])
    cv.mip = 3
    assert np.all(cv[slice64] == data_ds3[slice64])

def test_downsample_with_offset():
    delete_layer()
    cf, data = create_layer(size=(512,512,128,1), offset=(3,7,11))
    cv = CloudVolume(cf.cloudpath)
    assert len(cv.scales) == 1
    assert len(cv.available_mips) == 1

    cv.commit_info()

    tq = MockTaskQueue()
    tasks = create_downsampling_tasks(cf.cloudpath, mip=0, num_mips=3)
    tq.insert_all(tasks)

    cv.refresh_info()

    assert len(cv.available_mips) == 4
    assert np.array_equal(cv.mip_volume_size(0), [ 512, 512, 128 ])
    assert np.array_equal(cv.mip_volume_size(1), [ 256, 256, 128 ])
    assert np.array_equal(cv.mip_volume_size(2), [ 128, 128, 128 ])
    assert np.array_equal(cv.mip_volume_size(3), [  64,  64, 128 ])

    assert np.all(cv.mip_voxel_offset(3) == (0,0,11))
    
    cv.mip = 0
    assert np.all(cv[3:67, 7:71, 11:75] == data[0:64, 0:64, 0:64])

    data_ds1, = tinybrain.downsample_with_averaging(data, factor=[2, 2, 1, 1])
    cv.mip = 1
    assert np.all(cv[1:33, 3:35, 11:75] == data_ds1[0:32, 0:32, 0:64])

    data_ds2, = tinybrain.downsample_with_averaging(data, factor=[4, 4, 1, 1])
    cv.mip = 2
    assert np.all(cv[0:16, 1:17, 11:75] == data_ds2[0:16, 0:16, 0:64])

    data_ds3, = tinybrain.downsample_with_averaging(data, factor=[8, 8, 1, 1])
    cv.mip = 3
    assert np.all(cv[0:8, 0:8, 11:75] == data_ds3[0:8,0:8,0:64])

def test_downsample_w_missing():
    delete_layer()
    cf, data = create_layer(size=(512,512,128,1), offset=(3,7,11))
    cv = CloudVolume(cf.cloudpath)
    assert len(cv.scales) == 1
    assert len(cv.available_mips) == 1
    delete_layer()

    cv.commit_info()

    tq = MockTaskQueue()

    try:
        tasks = create_downsampling_tasks(cf.cloudpath, mip=0, num_mips=3, fill_missing=False)
        tq.insert_all(tasks)
    except EmptyVolumeException:
        pass

    tq = MockTaskQueue()

    tasks = create_downsampling_tasks(cf.cloudpath, mip=0, num_mips=3, fill_missing=True)
    tq.insert_all(tasks)

    cv.refresh_info()

    assert len(cv.available_mips) == 4
    assert np.array_equal(cv.mip_volume_size(0), [ 512, 512, 128 ])
    assert np.array_equal(cv.mip_volume_size(1), [ 256, 256, 128 ])
    assert np.array_equal(cv.mip_volume_size(2), [ 128, 128, 128 ])
    assert np.array_equal(cv.mip_volume_size(3), [  64,  64, 128 ])

    assert np.all(cv.mip_voxel_offset(3) == (0,0,11))
    
    cv.mip = 0
    cv.fill_missing = True
    assert np.count_nonzero(cv[3:67, 7:71, 11:75]) == 0

def test_downsample_higher_mip():
    delete_layer()
    cf, data = create_layer(size=(512,512,64,1), offset=(3,7,11))
    cv = CloudVolume(cf.cloudpath)
    cv.info['scales'] = cv.info['scales'][:1]
    
    tq = MockTaskQueue()

    cv.commit_info()
    tasks = create_downsampling_tasks(cf.cloudpath, mip=0, num_mips=2)
    tq.insert_all(tasks)
    cv.refresh_info()
    assert len(cv.available_mips) == 3

    tasks = create_downsampling_tasks(cf.cloudpath, mip=1, num_mips=2)
    tq.insert_all(tasks)
    cv.refresh_info()
    assert len(cv.available_mips) == 4

    cv.mip = 3
    assert cv[:,:,:].shape == (64,64,64,1)

def test_delete():
    delete_layer()
    cf, _ = create_layer(size=(128,64,64,1), offset=(0,0,0), layer_type="segmentation")
    cv = CloudVolume(cf.cloudpath)

    DeleteTask(
        layer_path=cf.cloudpath,
        offset=(0,0,0),
        shape=(128, 64, 64),
    )

    fnames = list(cf.list())
    
    assert '1_1_1/0-64_0-64_0-64' not in fnames
    assert '1_1_1/64-128_0-64_0-64' not in fnames

def test_blackout_tasks():
    delete_layer()
    cf, _ = create_layer(size=(128,64,64,1), offset=(0,0,0), layer_type="image")
    cv = CloudVolume(cf.cloudpath)

    tq = TaskQueue("fq:///tmp/removeme/blackout/")

    tq.insert(
        partial(BlackoutTask, 
            cloudpath=cf.cloudpath,
            mip=0,
            offset=(0,0,0),
            shape=(128, 64, 64),
            value=11,
            non_aligned_writes=False
        )
    )
    tq.lease().execute()

    img = cv[:,:,:]
    assert np.all(img == 11)

    BlackoutTask(
        cloudpath=cf.cloudpath,
        mip=0,
        offset=(0,0,0),
        shape=(37, 64, 64),
        value=23,
        non_aligned_writes=True
    )

    img = cv[:37,:,:]
    assert np.all(img == 23)

    img = cv[:]
    items, counts = np.unique(img, return_counts=True)
    counts = {
        items[0]: counts[0],
        items[1]: counts[1]
    }

    twenty_threes = 37 * 64 * 64
    assert counts[23] == twenty_threes
    assert counts[11] == (128 * 64 * 64) - twenty_threes
    
@pytest.mark.parametrize('compress', ('gzip', 'br'))
def test_mesh(compress):
    delete_layer()
    cf, _ = create_layer(size=(64,64,64,1), offset=(0,0,0), layer_type="segmentation")
    cv = CloudVolume(cf.cloudpath)
    # create a box of ones surrounded by zeroes
    data = np.zeros(shape=(64,64,64,1), dtype=np.uint32)
    data[1:-1,1:-1,1:-1,:] = 1
    cv[0:64,0:64,0:64] = data
    cv.info['mesh'] = 'mesh'
    cv.commit_info()

    t = MeshTask(
        shape=(64,64,64),
        offset=(0,0,0),
        layer_path=cf.cloudpath,
        mip=0,
        remap_table={"1": "10"},
        low_padding=0,
        high_padding=1,
        compress=compress
    )
    t.execute()
    assert cf.get('mesh/10:0:0-64_0-64_0-64') is not None 
    assert list(cf.list('mesh/')) == ['mesh/10:0:0-64_0-64_0-64']


def test_quantize():
    qpath = 'file:///tmp/removeme/quantized/'

    delete_layer()
    delete_layer(qpath)

    cf, _ = create_layer(size=(256,256,128,3), offset=(0,0,0), layer_type="affinities")
    cv = CloudVolume(cf.cloudpath)

    shape = (128, 128, 64)
    slices = np.s_[ :shape[0], :shape[1], :shape[2], :1 ]

    data = cv[slices]
    data *= 255.0
    data = data.astype(np.uint8)

    task = partial(QuantizeTask,
        source_layer_path=cf.cloudpath,
        dest_layer_path=qpath,
        shape=shape,
        offset=(0,0,0),
        mip=0,
    )

    info = create_quantized_affinity_info(
        cf.cloudpath, qpath, shape, 
        mip=0, chunk_size=[64,64,64], encoding='raw'
    )
    qcv = CloudVolume(qpath, info=info)
    qcv.commit_info()

    create_downsample_scales(qpath, mip=0, ds_shape=shape)

    task()

    qcv.mip = 0

    qdata = qcv[slices]

    assert np.all(data.shape == qdata.shape)
    assert np.all(data == qdata)
    assert data.dtype == np.uint8

def test_mesh_manifests_filesystem():
    directory = '/tmp/removeme/mesh_manifests_fs/'
    layer_path = 'file://' + directory
    mesh_dir = 'mesh_mip_3_error_40'

    delete_layer(layer_path)

    to_path = lambda filename: os.path.join(directory, mesh_dir, filename)

    n_segids = 100
    n_lods = 2
    n_fragids = 5

    CloudFiles(layer_path).put_json('info', {"mesh":"mesh_mip_3_error_40"})

    for segid in range(n_segids):
        for lod in range(n_lods):
            for fragid in range(n_fragids):
                filename = '{}:{}:{}'.format(segid, lod, fragid)
                lib.touch(to_path(filename))

    MeshManifestFilesystemTask(layer_path=layer_path, lod=0)

    for segid in range(n_segids):
        for fragid in range(n_fragids):
            filename = '{}:0'.format(segid)
            assert os.path.exists(to_path(filename)), filename
            filename = '{}:1'.format(segid)
            assert not os.path.exists(to_path(filename)), filename

    MeshManifestFilesystemTask(layer_path=layer_path, lod=1)

    for segid in range(n_segids):
        for fragid in range(n_fragids):
            filename = '{}:0'.format(segid)
        assert os.path.exists(to_path(filename)), filename
        filename = '{}:1'.format(segid)
        assert os.path.exists(to_path(filename)), filename

    with open(to_path('50:0'), 'r') as f:
        content = json.loads(f.read())
        content["fragments"].sort()
        assert content == {"fragments": [ "50:0:0","50:0:1","50:0:2","50:0:3","50:0:4" ]}

    if os.path.exists(directory):
        shutil.rmtree(directory)

def test_mesh_manifests_prefixes():
    directory = '/tmp/removeme/mesh_manifests_prefix/'
    layer_path = 'file://' + directory
    mesh_dir = 'mesh_mip_3_error_40'

    delete_layer(layer_path)

    to_path = lambda filename: os.path.join(directory, mesh_dir, filename)

    n_segids = 100
    n_lods = 2
    n_fragids = 5

    CloudFiles(layer_path).put_json('info', {"mesh":"mesh_mip_3_error_40"})

    for segid in range(n_segids):
        for lod in range(n_lods):
            for fragid in range(n_fragids):
                filename = '{}:{}:{}'.format(segid, lod, fragid)
                lib.touch(to_path(filename))

    for i in range(10):
        MeshManifestPrefixTask(layer_path=layer_path, prefix=str(i), lod=0)

    for segid in range(n_segids):
        for fragid in range(n_fragids):
            filename = '{}:0'.format(segid)
            assert os.path.exists(to_path(filename))
            filename = '{}:1'.format(segid)
            assert not os.path.exists(to_path(filename))

    for i in range(10):
        MeshManifestPrefixTask(layer_path=layer_path, prefix=str(i), lod=1)

    for segid in range(n_segids):
        for fragid in range(n_fragids):
            filename = '{}:0'.format(segid)
        assert os.path.exists(to_path(filename))
        filename = '{}:1'.format(segid)
        assert os.path.exists(to_path(filename))

    with open(to_path('50:0'), 'r') as f:
        content = json.loads(f.read())
        assert content == {"fragments": [ "50:0:0","50:0:1","50:0:2","50:0:3","50:0:4" ]}

    if os.path.exists(directory):
        shutil.rmtree(directory)

def test_luminance_levels_task():
    directory = '/tmp/removeme/luminance_levels/'
    layer_path = 'file://' + directory

    delete_layer(layer_path)

    cf, imgd = create_layer(
        size=(256,256,128,1), offset=(0,0,0), 
        layer_type="image", layer_name='luminance_levels'
    )

    tq = MockTaskQueue()
    tasks = tc.create_luminance_levels_tasks( 
        layer_path=layer_path,
        coverage_factor=0.01, 
        shape=None, 
        offset=(0,0,0), 
        mip=0
    )
    tq.insert_all(tasks)

    gt = [ 0 ] * 256
    for x,y,z in lib.xyzrange( (0,0,0), list(imgd.shape[:2]) + [1] ):
        gt[ imgd[x,y,0,0] ] += 1

    with open('/tmp/removeme/luminance_levels/levels/0/0', 'rt') as f:
        levels = f.read()

    levels = json.loads(levels)
    assert levels['coverage_ratio'] == 1.0
    assert levels['levels'] == gt

def test_contrast_normalization_task():
    directory = '/tmp/removeme/contrast_normalization/'
    src_path = 'file://' + directory
    dest_path = src_path[:-1] + '2'

    delete_layer(src_path)
    delete_layer(dest_path)

    cf, imgd = create_layer(
        size=(300,300,129,1), offset=(0,0,0), 
        layer_type="image", layer_name='contrast_normalization'
    )
    tq = MockTaskQueue()
    tasks = tc.create_luminance_levels_tasks( 
        layer_path=src_path,
        coverage_factor=0.01, 
        shape=None, 
        offset=(0,0,0), 
        mip=0
    )
    tq.insert_all(tasks)

    tasks = tc.create_contrast_normalization_tasks( 
        src_path=src_path, 
        dest_path=dest_path, 
        levels_path=None,
        shape=None, 
        mip=0, 
        clip_fraction=0.01, 
        fill_missing=False, 
        translate=(0,0,0),
        minval=None, 
        maxval=None, 
        bounds=None,
        bounds_mip=0,
    )
    tq.insert_all(tasks)


def test_skeletonization_task():
    directory = '/tmp/removeme/skeleton/'
    layer_path = 'file://' + directory
    delete_layer(layer_path)

    img = np.ones((256,256,256), dtype=np.uint64)
    img[:,:,:] = 2
    cv = CloudVolume.from_numpy(
        img,
        layer_type='segmentation',
        vol_path=layer_path, 
    )

    tq = MockTaskQueue()
    tasks = tc.create_skeletonizing_tasks(layer_path, mip=0, teasar_params={
        'scale': 10,
        'const': 10,
    })
    tq.insert_all(tasks)

