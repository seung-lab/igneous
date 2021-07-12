import pytest

import shutil
import numpy as np

import os

from cloudvolume import CloudVolume
from cloudfiles import CloudFiles

layer_path = '/tmp/removeme/'

def create_storage(layer_name='layer'):
    stor_path = os.path.join(layer_path, layer_name)
    return CloudFiles('file://' + stor_path)

def create_layer(size, offset, layer_type="image", layer_name='layer', dtype=None):

    default = lambda dt: dtype or dt

    if layer_type == "image":
        random_data = np.random.randint(255, size=size, dtype=default(np.uint8))
    elif layer_type == 'affinities':
        random_data = np.random.uniform(low=0, high=1, size=size).astype(default(np.float32))
    elif layer_type == "segmentation":
        random_data = np.random.randint(0xFFFFFF, size=size, dtype=np.uint32)
    else:
        high = np.array([0], dtype=default(np.uint32)) - 1
        random_data = np.random.randint(high[0], size=size, dtype=default(np.uint32))
    
    storage = create_storage(layer_name)

    CloudVolume.from_numpy(
        random_data, 
        vol_path='file://' + layer_path + '/' + layer_name,
        resolution=(1,1,1), voxel_offset=offset, 
        chunk_size=(64,64,64), layer_type=layer_type, 
        max_mip=0,
    )
        
    return storage, random_data

def delete_layer(path=layer_path):
    if os.path.exists(path):
        shutil.rmtree(path)  

    
    
    