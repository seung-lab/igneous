"""
Perform connected components labeling on the image.

Result will be a 6-connected labeling of the input
image. All steps must use the same task shape. 

As each task uses a label offset to differentiate its
CCL labels from adjacent tasks, the largest possible
image that can be handled has 2^64 voxels which is
order of magnitude about size of a whole mouse brain.

Each of the steps are labeled with their sequence number.
Their order is:
  (1) Generate 3 back faces for each task with 
    1 voxel overlap (so they can be referenced by 
    adjacent tasks). [ CCLFacesTask ]
  (2) Compute linkages between CCL tasks and save the results 
    in a database. [ CCLEquivalancesTask ]
  (3) Compute a global union find from the linkage 
    data and from that a global relabeling scheme which 
    is saved in the database. [ create_relabeling ]
  (4) Apply the relabeling scheme to the image. [ RelabelCCLTask ]
"""
from typing import Optional, Union
from collections import defaultdict

import cc3d
import fastremap
import numpy as np
import compresso

from tqdm import tqdm

from cloudfiles import CloudFiles
from taskqueue import queueable

from cloudvolume import CloudVolume, Bbox, Vec
from cloudvolume.lib import sip

from ...types import ShapeType

__all__ = [
  "CCLFacesTask",
  "CCLEquivalancesTask",
  "RelabelCCLTask",
]

class DisjointSet:
  def __init__(self):
    self.data = {} 
  def makeset(self, x):
    self.data[x] = x
    return x
  def find(self, x):
    if not x in self.data:
      return None
    i = self.data[x]
    while i != self.data[i]:
      self.data[i] = self.data[self.data[i]]
      i = self.data[i]
    return i
  def union(self, x, y):
    i = self.find(x)
    j = self.find(y)
    if i is None:
      i = self.makeset(x)
    if j is None:
      j = self.makeset(y)

    if i < j:
      self.data[j] = i
    else:
      self.data[i] = j

def compute_task_number(grid_size, gridpoint) -> int:
  return int(
    gridpoint.x + grid_size.x * (
      gridpoint.y + grid_size.y * gridpoint.z
    )
  )

def compute_label_offset(shape, grid_size, gridpoint) -> int:
  # a task sequence number counting from 0 according to
  # where the task is located in space (so we can recompute 
  # it in the second pass easily)
  task_num = compute_task_number(grid_size, gridpoint)
  return task_num * shape.x * shape.y * shape.z

def threshold_image(image, threshold_lte, threshold_gte) -> np.ndarray:
  if threshold_gte is None and threshold_lte is None:
    return image

  thresholded = np.zeros(image.shape, dtype=np.uint8, order="F")
  if threshold_gte is not None:
    thresholded += image >= threshold_gte
  if threshold_lte is not None:
    thresholded += image <= threshold_lte
  return thresholded

def blackout_non_face_rails(
  labels:np.ndarray, shape:ShapeType
) -> np.ndarray:
  """
  For 6-connectivity, we need to black out the
  "rails" that would represent a higher connectivity
  to ensure that the labels are actually present
  when computing the relabeling later.
  """
  slcs = [
    np.s_[shape[0],shape[1],:],
    np.s_[shape[0],:,shape[2]],
    np.s_[:,shape[1],shape[2]]
  ]

  for slc in slcs:
    try:
      labels[slc] = 0
    except IndexError:
      pass

  return labels

@queueable
def CCLFacesTask(
  cloudpath:str, mip:int, 
  shape:ShapeType, offset:ShapeType,
  threshold_gte:Optional[Union[float,int]] = None,
  threshold_lte:Optional[Union[float,int]] = None,
  fill_missing:bool = False,
):
  """
  (1) Generate x,y,z back faces of each 1vx overlap task
  as compresso encoded 2D images.

  For continuous data, greater than or equal to (gte) 
  or less than or equal to (lte) thresholds 
  can be applied either individually or together. If neither
  is specified, no modification of the input will occur.

  These images are stored in e.g. 32_32_40/ccl/faces/ and have
  the following scheme where the numbers are the gridpoint
  location and the letters indicate which face plane.
    1-2-0-xy.cpso
    1-2-0-xz.cpso
    1-2-0-yz.cpso
  """
  shape = Vec(*shape)
  offset = Vec(*offset)
  bounds = Bbox(offset, offset + shape + 1) # 1 vx overlap

  if bounds.subvoxel():
    return

  cv = CloudVolume(cloudpath, mip=mip, fill_missing=fill_missing)
  bounds = Bbox.clamp(bounds, cv.meta.bounds(mip))

  grid_size = np.ceil((cv.bounds / shape).size3())
  gridpoint = np.floor(bounds.center() / shape).astype(int)
  label_offset = compute_label_offset(shape + 1, grid_size, gridpoint)
  
  labels = threshold_image(cv[bounds][...,0], threshold_lte, threshold_gte)
  labels = blackout_non_face_rails(labels, shape)
  cc_labels = cc3d.connected_components(labels, connectivity=6, out_dtype=np.uint64)
  cc_labels += label_offset
  cc_labels[labels == 0] = 0

  # Uploads leading faces for adjacent tasks to examine
  slices = [
    cc_labels[:,:,-1],
    cc_labels[:,-1,:],
    cc_labels[-1,:,:],
  ]
  slices = [ compresso.compress(slc) for slc in slices ]
  filenames = [
    f'{gridpoint.x}-{gridpoint.y}-{gridpoint.z}-xy.cpso',
    f'{gridpoint.x}-{gridpoint.y}-{gridpoint.z}-xz.cpso',
    f'{gridpoint.x}-{gridpoint.y}-{gridpoint.z}-yz.cpso'
  ]

  cf = CloudFiles(cloudpath)
  filenames = [
    cf.join(cv.key, 'ccl', 'faces', fname) for fname in filenames
  ]
  cf.puts(zip(filenames, slices), compress='br')

@queueable
def CCLEquivalancesTask(
  cloudpath:str, mip:int,
  shape:ShapeType, offset:ShapeType,
  threshold_gte:Optional[Union[float,int]] = None,
  threshold_lte:Optional[Union[float,int]] = None,
  fill_missing:bool = False,
):
  """
  (2) Generate linkages between tasks by comparing the 
  front face of the task with the back faces of the 
  three adjacent tasks saved from the first step.

  Writes output to e.g. 32_32_40/ccl/equivalences/
  """
  shape = Vec(*shape)
  offset = Vec(*offset)
  bounds = Bbox(offset, offset + shape + 1) # 1 vx overlap

  if bounds.subvoxel():
    return

  cv = CloudVolume(cloudpath, mip=mip, fill_missing=fill_missing)
  bounds = Bbox.clamp(bounds, cv.meta.bounds(mip))

  grid_size = np.ceil((cv.bounds / shape).size3())
  gridpoint = np.floor(bounds.center() / shape).astype(int)
  label_offset = compute_label_offset(shape + 1, grid_size, gridpoint)
  
  equivalences = DisjointSet()

  labels = threshold_image(cv[bounds][...,0], threshold_lte, threshold_gte)
  labels = blackout_non_face_rails(labels, shape)
  cc_labels, N = cc3d.connected_components(
    labels, connectivity=6, 
    out_dtype=np.uint64, return_N=True
  )
  cc_labels += label_offset
  cc_labels[labels == 0] = 0

  for i in range(1, N+1):
    equivalences.makeset(i + label_offset)

  cf = CloudFiles(cloudpath)
  filenames = [
    f'{gridpoint.x}-{gridpoint.y}-{gridpoint.z-1}-xy.cpso',
    f'{gridpoint.x}-{gridpoint.y-1}-{gridpoint.z}-xz.cpso',
    f'{gridpoint.x-1}-{gridpoint.y}-{gridpoint.z}-yz.cpso'
  ]
  filenames = [
    cf.join(cv.key, 'ccl', 'faces', fname) for fname in filenames
  ]

  slices = cf.get(filenames, return_dict=True)
  for key in slices:
    if slices[key] is None:
      continue
    face = compresso.decompress(slices[key])
    if '-xy' in key:
      face = face[:shape.x,:shape.y,0]
    elif '-xz' in key:
      face = face[:shape.x,:shape.z,0]
    else:
      face = face[:shape.y,:shape.z,0]
    slices[key] = face

  prev = [ slices[filenames[i]] for i in (0,1,2) ]

  cur = [
    cc_labels[:shape.x,:shape.y,0],
    cc_labels[:shape.x,0,:shape.z],
    cc_labels[0,:shape.y,:shape.z],
  ]

  for prev_i, cur_i in zip(prev, cur):
    if prev_i is None:
      continue

    mapping = fastremap.inverse_component_map(cur_i, prev_i)
    for task_label, adj_labels in mapping.items():
      if task_label == 0:
        continue
      for adj_label in fastremap.unique(adj_labels):
        if adj_label != 0:
          equivalences.union(int(task_label), int(adj_label))

  cf = CloudFiles(cloudpath)
  out_name = cf.join(cv.key, 'ccl', 'equivalences', f'{gridpoint.x}-{gridpoint.y}-{gridpoint.z}.json')
  cf.put_json(
    out_name, 
    { str(k): int(v) for k,v in equivalences.data.items() }, 
    compress='br'
  )

@queueable
def RelabelCCLTask(
  src_path:str, dest_path:str, mip:int,
  shape:ShapeType, offset:ShapeType,
  threshold_gte:Optional[Union[float,int]] = None,
  threshold_lte:Optional[Union[float,int]] = None,
  fill_missing:bool = False,
):
  """
  (4) Retrieves the relabeling for this task from the
  database, applies it, and saves the resulting image
  to the destination path. Upon saving, the 1 voxel
  overlap is omitted.
  """
  shape = Vec(*shape)
  offset = Vec(*offset)
  bounds = Bbox(offset, offset + shape + 1) # 1 vx overlap

  if bounds.subvoxel():
    return

  cv = CloudVolume(src_path, mip=mip, fill_missing=fill_missing)
  bounds = Bbox.clamp(bounds, cv.meta.bounds(mip))

  grid_size = np.ceil((cv.bounds / shape).size3())
  gridpoint = np.floor(bounds.center() / shape).astype(int)
  label_offset = compute_label_offset(shape + 1, grid_size, gridpoint)
  task_num = compute_task_number(grid_size, gridpoint)
  
  cf = CloudFiles(src_path)
  mapping_path = cf.join(cv.key, "ccl", "relabel", f"{task_num}.json")
  mapping = cf.get_json(mapping_path) or {}
  mapping =  { int(k):int(v) for k,v in mapping.items() }
  mapping[0] = 0

  labels = threshold_image(cv[bounds][...,0], threshold_lte, threshold_gte)
  labels = blackout_non_face_rails(labels, shape)
  cc_labels, N = cc3d.connected_components(
    labels, connectivity=6, 
    out_dtype=np.uint64, return_N=True
  )
  cc_labels += label_offset
  cc_labels[labels == 0] = 0

  task_voxels = shape.x * shape.y * shape.z
  fastremap.remap(cc_labels, mapping, in_place=True)

  # Final upload without overlap
  dest_cv = CloudVolume(dest_path, mip=mip)
  bounds = Bbox(offset, offset + shape)
  bounds = Bbox.clamp(bounds, dest_cv.meta.bounds(mip))
  cc_labels = cc_labels[:shape.x,:shape.y,:shape.z].astype(dest_cv.dtype)
  cc_labels = cc_labels[:,:,:,np.newaxis]
  dest_cv[bounds] = cc_labels

def create_relabeling(cloudpath, mip, shape):
  """
  (3) Computes a relabeling from the linkages saved 
  from (2) and then saves them.

  Writes output to e.g. 32_32_40/ccl/relabel/
  and also .../ccl/max_label.json which contains
  the largest label.
  """
  cv = CloudVolume(cloudpath, mip=mip)
  cf = CloudFiles(cloudpath)
  all_eqpaths = cf.list(cf.join(cv.key, "ccl", "equivalences"))

  equivalences = DisjointSet()

  with tqdm(desc="Creating Union-Find", total=0) as pbar:
    for eqpaths in sip(all_eqpaths, 5000):
      eqdicts = cf.get_json(eqpaths)
      pbar.total += sum(( len(datum) for datum in eqdicts ))
      pbar.refresh()
      
      for data in eqdicts:
        for val1, val2 in data.items():
          equivalences.union(int(val1), int(val2))
          pbar.update()

      del eqdicts

  relabel = {}
  next_label = 1
  for key in tqdm(equivalences.data.keys(), desc="Renumbering"):
    lbl = equivalences.find(key)
    if lbl not in relabel:
      relabel[key] = next_label
      relabel[lbl] = next_label
      next_label += 1
    else:
      relabel[key] = relabel[lbl]

  del equivalences

  max_label_fname = cf.join(cv.key, "ccl", "max_label.json")
  cf.put_json(max_label_fname, [ next_label - 1 ])

  task_size = Vec(*shape) + 1
  task_voxels = task_size.x * task_size.y * task_size.z

  buckets = defaultdict(dict)
  for before_val, after_val in relabel.items():
    task_num = int(before_val // task_voxels)
    buckets[task_num][before_val] = after_val

  del relabel

  cf.put_jsons(
    (
      (cf.join(cv.key, "ccl", "relabel", f"{task_num}.json"), relabeling) 
      for task_num, relabeling in buckets.items()
    ),
    total=len(buckets),
    compress="br",
    progress=True
  )

def clean_intermediate_files(src, mip):
  cv = CloudVolume(src, mip)
  cf = CloudFiles(src)
  cf.delete(cf.list(cf.join(cv.key, "ccl")))





