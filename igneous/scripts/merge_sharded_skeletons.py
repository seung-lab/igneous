"""
This script is used for collecting all sharded
skeleton framents onto a single machine and 
processing them into fully realized shards.
"""
from collections import defaultdict
import os 
import numpy as np 
import pickle
import time
import sys 

from multiprocessing import Manager
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

manager = Manager()
crt_dict = lambda: manager.dict() if parallel > 1 else {}
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

def load_raw_skeletons():
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

  unfused_skeletons = defaultdict(list)
  while all_files:
    fragment = all_files.pop()
    for label, skel_frag in fragment.items():
      unfused_skeletons[label].append(skel_frag)  

  # CHECKPOINT? 

  skeletons = crt_dict()
  labels = list(unfused_skeletons.keys())
  for label in tqdm(labels, desc='Simple Merging'):
    skels = unfused_skeletons[label]
    skeleton = PrecomputedSkeleton.simple_merge(skels)
    skeleton.id = label
    skeleton.extra_attributes = [ 
      attr for attr in skeleton.extra_attributes \
      if attr['data_type'] == 'float32' 
    ] 
    skeletons[label] = skeleton 
    del unfused_skeletons[label]

  return skeletons

if os.path.exists('checkpoint.pkl'):
  print("Loading checkpoint...")
  s = time.time()
  with open("checkpoint.pkl", 'rb') as f:
    skeletons = pickle.load(f)
  print("Checkpoint loaded in " + str(time.time() - s) + " sec.")
else:
  skeletons = load_raw_skeletons()
  print("Saving checkpoint...")
  s = time.time()
  with open("checkpoint.pkl", 'wb') as f:
    pickle.dump(skeletons, f)
  print("Checkpoint saved in " + str(time.time() - s) + " sec.")

def complex_merge(skel):
  return kimimaro.postprocess(
    skel, 
    dust_threshold=1000, # voxels 
    tick_threshold=1300, # nm
  )

merged_skeletons = crt_dict()
labels = list(skeletons.keys())

with tqdm(total=len(skeletons), disable=False, desc="Final Merging") as pbar:
  if parallel == 1:
    for label in labels:
      skel = complex_merge(skeletons[label])
      merged_skeletons[skel.id] = skel.to_precomputed()
      del skeletons[label]
      pbar.update(1)
  else:
    pool = pathos.pools.ProcessPool(parallel)
    for skel in pool.uimap(complex_merge, skeletons.values()):
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

