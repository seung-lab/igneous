from builtins import range
import json
import os.path
import shutil

import numpy as np
from cloudvolume import Storage, CloudVolume, EmptyVolumeException
import cloudvolume.lib as lib
from taskqueue import MockTaskQueue

from igneous import (
    DownsampleTask, MeshTask, MeshManifestTask, 
    QuantizeAffinitiesTask, HyperSquareConsensusTask,
    DeleteTask
)
from igneous import downsample
from igneous.task_creation import create_downsample_scales, create_downsampling_tasks, create_quantized_affinity_info
from .layer_harness import delete_layer, create_layer

def test_ingest_image():
    delete_layer()
    storage, data = create_layer(size=(256,256,128,1), offset=(0,0,0), layer_type='image')
    cv = CloudVolume(storage.layer_path)
    assert len(cv.scales) == 3
    assert len(cv.available_mips) == 3

    slice64 = np.s_[0:64, 0:64, 0:64]

    cv.mip = 0
    assert np.all(cv[slice64] == data[slice64])

    assert len(cv.available_mips) == 3
    assert np.array_equal(cv.mip_volume_size(0), [ 256, 256, 128 ])
    assert np.array_equal(cv.mip_volume_size(1), [ 128, 128, 128 ])
    assert np.array_equal(cv.mip_volume_size(2), [ 64, 64, 128 ])
    
    slice64 = np.s_[0:64, 0:64, 0:64]

    cv.mip = 0
    assert np.all(cv[slice64] == data[slice64])

    data_ds1 = downsample.downsample_with_averaging(data, factor=[2, 2, 1, 1])
    cv.mip = 1
    assert np.all(cv[slice64] == data_ds1[slice64])

    data_ds2 = downsample.downsample_with_averaging(data_ds1, factor=[2, 2, 1, 1])
    cv.mip = 2
    assert np.all(cv[slice64] == data_ds2[slice64])


def test_ingest_segmentation():
    delete_layer()
    storage, data = create_layer(size=(256,256,128,1), offset=(0,0,0), layer_type='segmentation')
    cv = CloudVolume(storage.layer_path)
    assert len(cv.scales) == 3
    assert len(cv.available_mips) == 3

    slice64 = np.s_[0:64, 0:64, 0:64]

    cv.mip = 0
    assert np.all(cv[slice64] == data[slice64])

    assert len(cv.available_mips) == 3
    assert np.array_equal(cv.mip_volume_size(0), [ 256, 256, 128 ])
    assert np.array_equal(cv.mip_volume_size(1), [ 128, 128, 128 ])
    assert np.array_equal(cv.mip_volume_size(2), [  64,  64, 128 ])
    
    slice64 = np.s_[0:64, 0:64, 0:64]

    cv.mip = 0
    assert np.all(cv[slice64] == data[slice64])

    data_ds1 = downsample.downsample_segmentation(data, factor=[2, 2, 1, 1])
    cv.mip = 1
    assert np.all(cv[slice64] == data_ds1[slice64])

    data_ds2 = downsample.downsample_segmentation(data_ds1, factor=[2, 2, 1, 1])
    cv.mip = 2
    assert np.all(cv[slice64] == data_ds2[slice64])

def test_downsample_no_offset():
    delete_layer()
    storage, data = create_layer(size=(1024,1024,128,1), offset=(0,0,0))
    cv = CloudVolume(storage.layer_path)
    assert len(cv.scales) == 5
    assert len(cv.available_mips) == 5

    cv.commit_info()

    create_downsampling_tasks(MockTaskQueue(), storage.layer_path, mip=0, num_mips=4)

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

    data_ds1 = downsample.downsample_with_averaging(data, factor=[2, 2, 1, 1])
    cv.mip = 1
    assert np.all(cv[slice64] == data_ds1[slice64])

    data_ds2 = downsample.downsample_with_averaging(data_ds1, factor=[2, 2, 1, 1])
    cv.mip = 2
    assert np.all(cv[slice64] == data_ds2[slice64])

    data_ds3 = downsample.downsample_with_averaging(data_ds2, factor=[2, 2, 1, 1])
    cv.mip = 3
    assert np.all(cv[slice64] == data_ds3[slice64])

    data_ds4 = downsample.downsample_with_averaging(data_ds3, factor=[2, 2, 1, 1])
    cv.mip = 4
    assert np.all(cv[slice64] == data_ds4[slice64])

def test_downsample_with_offset():
    delete_layer()
    storage, data = create_layer(size=(512,512,128,1), offset=(3,7,11))
    cv = CloudVolume(storage.layer_path)
    assert len(cv.scales) == 4
    assert len(cv.available_mips) == 4

    cv.commit_info()

    create_downsampling_tasks(MockTaskQueue(), storage.layer_path, mip=0, num_mips=3)

    cv.refresh_info()

    assert len(cv.available_mips) == 4
    assert np.array_equal(cv.mip_volume_size(0), [ 512, 512, 128 ])
    assert np.array_equal(cv.mip_volume_size(1), [ 256, 256, 128 ])
    assert np.array_equal(cv.mip_volume_size(2), [ 128, 128, 128 ])
    assert np.array_equal(cv.mip_volume_size(3), [  64,  64, 128 ])

    assert np.all(cv.mip_voxel_offset(3) == (0,0,11))
    
    cv.mip = 0
    assert np.all(cv[3:67, 7:71, 11:75] == data[0:64, 0:64, 0:64])

    data_ds1 = downsample.downsample_with_averaging(data, factor=[2, 2, 1, 1])
    cv.mip = 1
    assert np.all(cv[1:33, 3:35, 11:75] == data_ds1[0:32, 0:32, 0:64])

    data_ds2 = downsample.downsample_with_averaging(data_ds1, factor=[2, 2, 1, 1])
    cv.mip = 2
    assert np.all(cv[0:16, 1:17, 11:75] == data_ds2[0:16, 0:16, 0:64])

    data_ds3 = downsample.downsample_with_averaging(data_ds2, factor=[2, 2, 1, 1])
    cv.mip = 3
    assert np.all(cv[0:8, 0:8, 11:75] == data_ds3[0:8,0:8,0:64])

def test_downsample_w_missing():
    delete_layer()
    storage, data = create_layer(size=(512,512,128,1), offset=(3,7,11))
    cv = CloudVolume(storage.layer_path)
    assert len(cv.scales) == 4
    assert len(cv.available_mips) == 4
    delete_layer()

    cv.commit_info()

    try:
        create_downsampling_tasks(MockTaskQueue(), storage.layer_path, mip=0, num_mips=3, fill_missing=False)
    except EmptyVolumeException:
        pass

    create_downsampling_tasks(MockTaskQueue(), storage.layer_path, mip=0, num_mips=3, fill_missing=True)

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
    storage, data = create_layer(size=(512,512,64,1), offset=(3,7,11))
    cv = CloudVolume(storage.layer_path)
    cv.info['scales'] = cv.info['scales'][:1]
    
    cv.commit_info()
    create_downsampling_tasks(MockTaskQueue(), storage.layer_path, mip=0, num_mips=2)
    cv.refresh_info()
    assert len(cv.available_mips) == 3

    create_downsampling_tasks(MockTaskQueue(), storage.layer_path, mip=1, num_mips=2)
    cv.refresh_info()
    assert len(cv.available_mips) == 4

    cv.mip = 3
    assert cv[:,:,:].shape == (64,64,64,1)

def test_delete():
    delete_layer()
    storage, _ = create_layer(size=(128,64,64,1), offset=(0,0,0), layer_type="segmentation")
    cv = CloudVolume(storage.layer_path)

    DeleteTask(
        layer_path=storage.layer_path,
        offset=(0,0,0),
        shape=(128, 64, 64),
    ).execute()

    fnames = [ _ for _ in storage.list_files() ]
    
    assert '1_1_1/0-64_0-64_0-64' not in fnames
    assert '1_1_1/64-128_0-64_0-64' not in fnames


def test_mesh():
    delete_layer()
    storage, _ = create_layer(size=(64,64,64,1), offset=(0,0,0), layer_type="segmentation")
    cv = CloudVolume(storage.layer_path)
    # create a box of ones surrounded by zeroes
    data = np.zeros(shape=(64,64,64,1), dtype=np.uint32)
    data[1:-1,1:-1,1:-1,:] = 1
    cv[0:64,0:64,0:64] = data

    t = MeshTask(
        shape=(64,64,64),
        offset=(0,0,0),
        layer_path=storage.layer_path,
        mip=0,
        remap_table={"1": "10"},
        low_padding=0,
        high_padding=1
    )
    t.execute()
    assert storage.get_file('mesh/10:0:0-64_0-64_0-64') is not None 
    assert list(storage.list_files('mesh/')) == ['mesh/10:0:0-64_0-64_0-64']


def test_quantize_affinities():
    qpath = 'file:///tmp/removeme/quantized/'

    delete_layer()
    delete_layer(qpath)

    storage, _ = create_layer(size=(256,256,128,3), offset=(0,0,0), layer_type="affinities")
    cv = CloudVolume(storage.layer_path)

    shape = (128, 128, 64)
    slices = np.s_[ :shape[0], :shape[1], :shape[2], :1 ]

    data = cv[slices]
    data *= 255.0
    data = data.astype(np.uint8)

    task = QuantizeAffinitiesTask(
        source_layer_path=storage.layer_path,
        dest_layer_path=qpath,
        shape=shape,
        offset=(0,0,0),
    )

    info = create_quantized_affinity_info(storage.layer_path, qpath, shape)
    qcv = CloudVolume(qpath, info=info)
    qcv.commit_info()

    create_downsample_scales(qpath, mip=0, ds_shape=shape)

    task.execute()

    qcv.mip = 0

    qdata = qcv[slices]

    assert np.all(data.shape == qdata.shape)
    assert np.all(data == qdata)
    assert data.dtype == np.uint8


def test_mesh_manifests():
    directory = '/tmp/removeme/mesh_manifests/'
    layer_path = 'file://' + directory
    mesh_dir = 'mesh_mip_3_error_40'

    delete_layer(layer_path)

    to_path = lambda filename: os.path.join(directory, mesh_dir, filename)

    n_segids = 100
    n_lods = 2
    n_fragids = 5

    with Storage(layer_path) as stor:
        stor.put_file('info', '{"mesh":"mesh_mip_3_error_40"}'.encode('utf8'))

    for segid in range(n_segids):
        for lod in range(n_lods):
            for fragid in range(n_fragids):
                filename = '{}:{}:{}'.format(segid, lod, fragid)
                lib.touch(to_path(filename))

    for i in range(10):
        MeshManifestTask(layer_path=layer_path, prefix=i, lod=0).execute()

    for segid in range(n_segids):
        for fragid in range(n_fragids):
            filename = '{}:0'.format(segid)
            assert os.path.exists(to_path(filename))
            filename = '{}:1'.format(segid)
            assert not os.path.exists(to_path(filename))

    for i in range(10):
        MeshManifestTask(layer_path=layer_path, prefix=i, lod=1).execute()

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
        
        
# def test_watershed():
#     return # needs expensive julia stuff enabled in dockerfile
#     from igneous import WatershedTask
#     delete_layer('affinities')
#     storage, data = create_layer(size=(64,64,64,3), offset=(0,0,0), layer_type='affinities', layer_name='affinities')

#     delete_layer('segmentation')
#     storage, data = create_layer(size=(64,64,64,1), offset=(0,0,0), layer_type='segmentation', layer_name='segmentation')

#     WatershedTask(chunk_position='0-64_0-64_0-64',
#                   crop_position='0-64_0-64_0-64',
#                   layer_path_affinities='file:///tmp/removeme/affinities',
#                   layer_path_segmentation='file:///tmp/removeme/segmentation',
#                   high_threshold=0.999987, low_threshold=0.003, merge_threshold=0.3, 
#                   merge_size=800, dust_size=800).execute()


# def test_real_data():
#     return # this is to expensive to be test by travis
#     from igneous import WatershedTask
#     from tqdm import tqdm
#     from itertools import product
#     vol = CloudVolume('s3://neuroglancer/pinky40_v11/affinitymap-jnet')
#     scale = vol.info['scales'][0]
#     for x_min in range(0, scale['size'][0], 512):
#         for y_min in range(0, scale['size'][1], 512):
#             for z_min in range(0, scale['size'][2], 1024):
#                 x_max = min(scale['size'][0], x_min + 768)
#                 y_max = min(scale['size'][1], y_min + 768)
#                 z_max = min(scale['size'][2], z_min + 1024)

#                 #adds offsets
#                 x_min += scale['voxel_offset'][0]; x_max += scale['voxel_offset'][0]
#                 y_min += scale['voxel_offset'][1]; y_max += scale['voxel_offset'][1]
#                 z_min += scale['voxel_offset'][2]; z_max += scale['voxel_offset'][2]
#                 WatershedTask(chunk_position='{}-{}_{}-{}_{}-{}'.format(x_min, x_max, y_min, y_max, z_min, z_max),
#                   crop_position='128-640_128-640_0-1024',
#                   layer_path_affinities='s3://neuroglancer/pinky40_v11/affinitymap-jnet',
#                   layer_path_segmentation='s3://neuroglancer/pinky40_v11/chunked_watershed',
#                   high_threshold=0.999987, low_threshold=0.003, merge_threshold=0.3, 
#                   merge_size=800, dust_size=800).execute()
   