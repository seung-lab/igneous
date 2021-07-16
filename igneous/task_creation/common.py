import copy
from functools import reduce, partial
import operator
import os
import subprocess

import numpy as np

from cloudvolume.lib import Vec, Bbox, max2, min2, xyzrange, find_closest_divisor, yellow, jsonify

def operator_contact():
  contact = ''
  try:
    contact = subprocess.check_output("git config user.email", shell=True)
    contact = str(contact.rstrip())
  except:
    try:
      print(yellow('Unable to determine provenance contact email. Set "git config user.email". Using unix $USER instead.'))
      contact = os.environ['USER']
    except:
      print(yellow('$USER was not set. The "owner" field of the provenance file will be blank.'))
      contact = ''

  return contact

def prod(x):
  return reduce(operator.mul, x, 1)

def get_bounds(vol, bounds, mip, chunk_size=None):
  if bounds is None:
    bounds = vol.bounds.clone()
  else:
    bounds = Bbox.create(bounds)
    bounds = vol.bbox_to_mip(bounds, mip=0, to_mip=mip)
    if chunk_size is not None:
      bounds = bounds.expand_to_chunk_size(chunk_size, vol.meta.voxel_offset(mip))
    bounds = Bbox.clamp(bounds, vol.meta.bounds(mip))
  

  print("Volume Bounds: ", vol.meta.bounds(mip))
  print("Selected ROI:  ", bounds)

  return bounds

def num_tasks(bounds, shape):
  return int(reduce(operator.mul, np.ceil(bounds.size3() / shape)))

class FinelyDividedTaskIterator():
  """
  Parallelizes tasks that do not have overlap.

  Evenly splits tasks between processes without 
  regards to whether the dividing line lands in
  the middle of a slice. 
  """
  def __init__(self, bounds, shape):
    self.bounds = bounds 
    self.shape = Vec(*shape)
    self.start = 0
    self.end = num_tasks(bounds, shape)

  def __len__(self):
    return self.end - self.start
  
  def __getitem__(self, slc):
    itr = copy.deepcopy(self)
    itr.start = max(self.start + slc.start, self.start)
    itr.end = min(self.start + slc.stop, self.end)
    return itr

  def __iter__(self):
    for i in range(self.start, self.end):
      pt = self.to_coord(i)
      offset = pt * self.shape + self.bounds.minpt
      yield self.task(self.shape.clone(), offset.clone())

    self.on_finish()

  def to_coord(self, index):
    """Convert an index into a grid coordinate defined by the task shape."""
    sx, sy, sz = np.ceil(self.bounds.size3() / self.shape).astype(int)
    sxy = sx * sy
    z = index // sxy
    y = (index - (z * sxy)) // sx
    x = index - sx * (y + z * sy)
    return Vec(x,y,z)

  def task(self, shape, offset):
    raise NotImplementedError()

  def on_finish(self):
    pass

def graphene_prefixes(
    mip=1, mip_bits=8, 
    coord_bits=(10, 10, 10), 
    prefix_length=6
  ):
  """
  Graphene structures segids as decimal numbers following
  the below format:

  mip x y z segid

  Typical parameter values are 
  mip_bits=4 or 8, x_bits=8 or 10, y_bits=8 or 10
  """
  coord_bits = Vec(*coord_bits)

  mip_shift = 64 - mip_bits
  x_shift = mip_shift - coord_bits.x
  y_shift = x_shift - coord_bits.y
  z_shift = y_shift - coord_bits.z

  x_range = 2 ** coord_bits.x 
  y_range = 2 ** coord_bits.y
  z_range = 2 ** coord_bits.z

  prefixes = set()
  for x in range(x_range):
    for y in range(y_range):
      num = (mip << mip_shift) + (x << x_shift) + (y << y_shift)
      num = str(num)[:prefix_length]
      prefixes.add(num)

  return prefixes