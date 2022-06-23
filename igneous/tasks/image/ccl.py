import cc3d
import fastremap
import numpy as np
import compresso

from tqdm import tqdm

from cloudfiles import CloudFiles
from taskqueue import queueable

from cloudvolume import CloudVolume, Bbox, Vec

from . import sql
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

def compute_label_offset(shape, grid_size, gridpoint):
# a task sequence number counting from 0 according to
  # where the task is located in space (so we can recompute 
  # it in the second pass easily)
  task_num = int(
    gridpoint.x + grid_size.x * (
      gridpoint.y + grid_size.y * gridpoint.z
    )
  )
  return task_num * shape.x * shape.y * shape.z

@queueable
def CCLFacesTask(
  cloudpath:str, mip:int, 
  shape:ShapeType, offset:ShapeType
):
  shape = Vec(*shape)
  offset = Vec(*offset)
  bounds = Bbox(offset, offset + shape)

  if bounds.subvoxel():
    return

  cv = CloudVolume(cloudpath, mip=mip)
  grid_size = np.ceil((cv.bounds / shape).size3())
  gridpoint = np.floor(bounds.center() / shape).astype(int)
  label_offset = compute_label_offset(shape, grid_size, gridpoint)
  
  labels = cv[bounds][...,0]
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
    cf.join(cv.key, 'ccl', fname) for fname in filenames
  ]
  cf.puts(zip(filenames, slices), compress='br')

@queueable
def CCLEquivalancesTask(
  cloudpath:str, mip:int, db_path:str,
  shape:ShapeType, offset:ShapeType,
):
  shape = Vec(*shape)
  offset = Vec(*offset)
  bounds = Bbox(offset, offset + shape)

  if bounds.subvoxel():
    return

  cv = CloudVolume(cloudpath, mip=mip)
  grid_size = np.ceil((cv.bounds / shape).size3())
  gridpoint = np.floor(bounds.center() / shape).astype(int)
  label_offset = compute_label_offset(shape, grid_size, gridpoint)
  
  equivalences = DisjointSet()

  labels = cv[bounds][...,0]
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
    f'{gridpoint.x-1}-{gridpoint.y}-{gridpoint.z}-xy.cpso',
    f'{gridpoint.x}-{gridpoint.y-1}-{gridpoint.z}-xz.cpso',
    f'{gridpoint.x}-{gridpoint.y}-{gridpoint.z-1}-yz.cpso'
  ]
  filenames = [
    cf.join(cv.key, 'ccl', fname) for fname in filenames
  ]

  slices = cf.get(filenames, return_dict=True)
  for key in slices:
    if slices[key] is None:
      continue
    slices[key] = compresso.decompress(slices[key])[:,:,0]

  prev = [ slices[filenames[i]] for i in (2,1,0) ]

  cur = [
    cc_labels[0,:,:],
    cc_labels[:,0,:],
    cc_labels[:,:,0],
  ]

  for prev_i, cur_i in zip(prev, cur):
    if prev_i is None:
      continue

    mapping = fastremap.inverse_component_map(cur_i, prev_i)
    for task_label, adj_labels in mapping.items():
      for adj_label in adj_labels:
        if task_label != 0 and adj_label != 0:
          equivalences.union(task_label, adj_label)

  sql.insert_equivalences(db_path, equivalences.data)

@queueable
def RelabelCCLTask(
  src_path:str, dest_path:str, mip:int, 
  db_path:str,
  shape:ShapeType, offset:ShapeType,
):
  shape = Vec(*shape)
  offset = Vec(*offset)
  bounds = Bbox(offset, offset + shape)

  if bounds.subvoxel():
    return

  cv = CloudVolume(src_path, mip=mip)
  grid_size = np.ceil((cv.bounds / shape).size3())
  gridpoint = np.floor(bounds.center() / shape).astype(int)
  label_offset = compute_label_offset(shape, grid_size, gridpoint)

  labels = cv[bounds][...,0]
  cc_labels, N = cc3d.connected_components(
    labels, connectivity=6, 
    out_dtype=np.uint64, return_N=True
  )
  cc_labels += label_offset
  cc_labels[labels == 0] = 0

  task_voxels = shape.x * shape.y * shape.z
  mapping = sql.get_relabeling(db_path, label_offset, task_voxels)
  mapping[0] = 0
  fastremap.remap(cc_labels, mapping, in_place=True)

  dest_cv = CloudVolume(dest_path, mip=mip)
  dest_cv[bounds] = cc_labels

def create_relabeling(db_path):
  rows = sql.retrieve_equivalences(db_path)
  equivalences = DisjointSet()

  for val1, val2 in tqdm(rows, desc="Creating Union-Find"):
    equivalences.union(val1, val2)

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

  sql.insert_relabeling(db_path, relabel)





