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

import pathos.pools
from tqdm import tqdm

from cloudvolume import CloudVolume, Storage, PrecomputedSkeleton
from cloudvolume.datasource.precomputed.sharding import ShardingSpecification, synthesize_shard_files
import kimimaro

cloudpath = sys.argv[1]

if len(sys.argv):
  parallel = int(sys.argv[2])
else:
  parallel = 1

cv = CloudVolume(cloudpath)

# setup sharded

spec = ShardingSpecification(
  'neuroglancer_uint64_sharded_v1', 
  preshift_bits=9, 
  hash='murmurhash3_x86_128', 
  minishard_bits=6, 
  shard_bits=6, 
  minishard_index_encoding='gzip', 
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

for i, res in enumerate(tqdm(all_files, desc='Unpickling')):
  all_files[i] = pickle.loads(res['content'])

# group by segid

skeletons = defaultdict(list)
for fragment in all_files:
  for label, skel_frag in fragment.items():
    skeletons[label].append(skel_frag)

del all_files

# CHECKPOINT? 

for label, skels in tqdm(skeletons.items(), desc='Simple Merging'):
  skeleton = PrecomputedSkeleton.simple_merge(skels)
  skeleton.id = label
  skeleton.extra_attributes = [ 
    attr for attr in skeleton.extra_attributes \
    if attr['data_type'] == 'float32' 
  ] 
  skeletons[label] = skeleton 

def complex_merge(label):
  return kimimaro.postprocess(
    skeletons[label], 
    dust_threshold=1000, # voxels 
    tick_threshold=1300, # nm
  )

merged_skeletons = {}

with tqdm(total=len(skeletons), disable=False, desc="Final Merging") as pbar:
  if parallel == 1:
    for label in skeletons.keys():
      skel = complex_merge(label)
      merged_skeletons[skel.id] = skel.to_precomputed()
      pbar.update(1)
  else:
    pool = pathos.pools.ProcessPool(parallel)
    for skel in pool.uimap(complex_merge, skeletons.keys()):
      merged_skeletons[skel.id] = skel.to_precomputed()
      pbar.update(1)
    pool.close()
    pool.join()
    pool.clear()

del skeletons

shard_files = synthesize_shard_files(spec, merged_skeletons, progress=True)

uploadable = [ (fname, data) for fname, data in shard_files.items() ]
with Storage(cv.skeleton.meta.layerpath) as stor:
  stor.put_files(
    files=uploadable, 
    compress=False,
    content_type='application/octet-stream',
    cache_control='no-cache',
  )








