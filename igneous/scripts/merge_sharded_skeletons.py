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
cv.skeleton.meta.info['vertex_attributes'] = [ 
  attr for attr in cv.skeleton.meta.info['vertex_attributes'] \
  if attr['data_type'] == 'float32'
] 
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

all_files = [ 
  pickle.loads(res['content']) for res in tqdm(all_files, desc='Unpickling') 
]

# group by segid

skeletons = defaultdict(list)

for fragment in tqdm(all_files, desc='Aggregating Fragments'):
  for label, skel_frag in fragment.items():
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
  skeleton.extra_attributes = [ 
    attr for attr in skeleton.extra_attributes \
    if attr['data_type'] == 'float32' 
  ] 
  skeletons[label] = skeleton.to_precomputed()

shard_files = synthesize_shard_files(spec, skeletons, progress=False)
# for fname, data in shard_files.items():
#   print(fname, ":", len(data) / 1e6)

uploadable = [ (fname, data) for fname, data in shard_files.items() ]
with Storage(cv.skeleton.meta.layerpath) as stor:
  stor.put_files(
    files=uploadable, 
    compress=False,
    content_type='application/octet-stream',
    cache_control='no-cache',
  )








