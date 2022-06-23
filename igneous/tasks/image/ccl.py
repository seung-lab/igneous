import cc3d
import fastremap
import numpy as np
import compresso

from cloudfiles import CloudFiles
from taskqueue import queueable

from cloudvolume import CloudVolume, Bbox, Vec

from .sql import insert_equivalences, get_relabeling
from ....types import ShapeType

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

    # We are not choosing the minimum representative
    # to ensure that there are no overlapping rows
    # inserted into the database.
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
def ComputeCCLFaces(
  cloudpath:str, mip:int, 
  shape:ShapeType, offset:ShapeType
):
  shape = Vec(*shape)
  offset = Vec(*offset)
  bounds = Bbox(offset, offset + shape)

  if bounds.subvoxel():
    return

  cv = CloudVolume(cloudpath, mip=mip)
  grid_size = np.ceil(cv.bounds / shape)
  gridpoint = np.floor(bounds.center() / grid_size).astype(int)
  label_offset = compute_label_offset(shape, grid_size, gridpoint)
  
  labels = cv[bounds]
  cc_labels = cc3d.connected_components(labels, connectivity=6, out_dtype=np.uint64)
  cc_labels += label_offset

  # Uploads leading faces for adjacent tasks to examine
  slices = [
    cc_labels[:,:,-1],
    cc_labels[:,-1,:],
    cc_labels[-1,:,:],
  ]
  slices = [ compresso.compress(slc) for slc in slices ]
  filenames = [
    f'{gridpoint.x}-{gridpoint.y}-{gridpoint.z}-xy'
    f'{gridpoint.x}-{gridpoint.y}-{gridpoint.z}-xz'
    f'{gridpoint.x}-{gridpoint.y}-{gridpoint.z}-yz'
  ]

  cf = CloudFiles(cloudpath)
  filenames = [
    cf.join(cv.key, 'ccl', fname) for fname in filenames
  ]
  cf.puts(zip(filenames, slices), compression='br')

@queueable
def ComputeCCLEquivalances(
  cloudpath:str, mip:int, db_path:str,
  shape:ShapeType, offset:ShapeType,
):
  shape = Vec(*shape)
  offset = Vec(*offset)
  bounds = Bbox(offset, offset + shape)

  if bounds.subvoxel():
    return

  cv = CloudVolume(cloudpath, mip=mip)
  grid_size = np.ceil(cv.bounds / shape)
  gridpoint = np.floor(bounds.center() / grid_size).astype(int)
  label_offset = compute_label_offset(shape, grid_size, gridpoint)
  
  equivalences = DisjointSet()

  labels = cv[bounds]
  cc_labels, N = cc3d.connected_components(
    labels, connectivity=6, 
    out_dtype=np.uint64, return_N=True
  )
  cc_labels += label_offset

  for i in range(1, N+1):
    equivalences.makeset(i + label_offset)

  cf = CloudFiles(cloudpath)
  filenames = [
    f'{gridpoint.x-1}-{gridpoint.y}-{gridpoint.z}-xy'
    f'{gridpoint.x}-{gridpoint.y-1}-{gridpoint.z}-xz'
    f'{gridpoint.x}-{gridpoint.y}-{gridpoint.z-1}-yz'
  ]
  filenames = [
    cf.join(cv.key, 'ccl', fname) for fname in filenames
  ]

  slices = cf.get(full_filenames, return_dict=True)
  for key in slices:
    if slices[key] is None:
      continue
    slices[key] = compresso.decompress(slices[key])

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

  insert_equivalences(db_path, equivalances.data)

@queueable
def RelabelCCL(
  src_path:str, dest_path:src, mip:int, 
  db_path:str,
  shape:ShapeType, offset:ShapeType,
):
  shape = Vec(*shape)
  offset = Vec(*offset)
  bounds = Bbox(offset, offset + shape)

  if bounds.subvoxel():
    return

  cv = CloudVolume(src_path, mip=mip)
  grid_size = np.ceil(cv.bounds / shape)
  gridpoint = np.floor(bounds.center() / grid_size).astype(int)
  label_offset = compute_label_offset(shape, grid_size, gridpoint)

  labels = cv[bounds]
  cc_labels, N = cc3d.connected_components(
    labels, connectivity=6, 
    out_dtype=np.uint64, return_N=True
  )
  cc_labels += label_offset

  task_voxels = shape.x * shape.y * shape.z
  mapping = get_relabeling(db_path, label_offset, task_voxels)

  fastremap.remap(cc_labels, mapping, in_place=True)

  dest_cv = CloudVolume(dest_path, mip=mip)
  dest_cv[bounds] = cc_labels
