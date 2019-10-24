"""
This script is used for collecting all sharded
skeleton framents onto a single machine and 
processing them into fully realized shards.
"""
from collections import defaultdict
import os 
import numpy as np 
import pickle
import sys 

from tqdm import tqdm

from cloudvolume import CloudVolume, Storage, PrecomputedSkeleton
from cloudvolume.datasource.precomputed.sharding import ShardingSpecification, synthesize_shard_files
import kimimaro

cloudpath = sys.argv[1]

cv = CloudVolume(cloudpath)

# setup sharded

spec = ShardingSpecification(
  'neuroglancer_uint64_sharded_v1', 
  preshift_bits=9, 
  hash='murmurhash3_x86_128', 
  minishard_bits=6, 
  shard_bits=6, 
  minishard_index_encoding='raw', 
  data_encoding='gzip',
)

cv.skeleton.meta.info['sharding'] = spec.to_dict()
cv.skeleton.meta.commit_info()

# actually get data

print("Downloading list of files...")
print(cv.skeleton.meta.layerpath)
with Storage(cv.skeleton.meta.layerpath, progress=True) as stor:
  all_files = list(stor.list_files())

all_files = [ 
  fname for fname in all_files if os.path.splitext(fname)[1] == '.frags' 
]

print("Downloading files...")
with Storage(cv.skeleton.meta.layerpath, progress=True) as stor:
  all_files = stor.get_files(all_files)

# CHECKPOINT?

all_files = [ pickle.loads(res['content']) for res in tqdm(all_files, desc='Unpickling') ]

# group by segid

skeletons = defaultdict(list)

for fragment in tqdm(all_files, desc='Aggregating Fragments'):
  for label, skel_frag in tqdm(fragment.items()):
    skeletons[label].append(skel_frag)

# CHECKPOINT? 

for label, skels in tqdm(skeletons.items(), desc='Merging'):
  skeleton = PrecomputedSkeleton.simple_merge(skels).consolidate()
  skeleton = kimimaro.postprocess(
    skeleton, 
    dust_threshold=1000, # voxels 
    tick_threshold=1300, # nm
  )
  skeleton.id = label
  skeletons[label] = skeleton.to_precomputed()

shard_files = synthesize_shard_files(spec, skeletons, progress=True)
print(shard_files.keys())







