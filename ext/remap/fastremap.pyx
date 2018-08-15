"""
Functions related to remapping volumes into smaller data types.
For example, a uint64 volume can contain values as high as 2^64,
however, if the volume is only 512x512x512 voxels, the maximum
spread of values would be 134,217,728 (ratio: 7.2e-12). 

For some operations, we can save memory and improve performance
by performing operations on a remapped volume and remembering the
mapping back to the original value.

Author: William Silversmith
Affiliation: Seung Lab, Princeton Neuroscience Institute
Date: August 2018
"""
cimport cython
from libc.stdint cimport *
from libcpp.unordered_map cimport unordered_map

import numpy as np
cimport numpy as cnp

ctypedef fused ALLINT:
  uint8_t
  uint16_t
  uint32_t
  uint64_t
  int8_t
  int16_t
  int32_t
  int64_t

ctypedef fused UINT:
  uint8_t
  uint16_t
  uint32_t
  uint64_t

@cython.boundscheck(False)
def renumber(arr, immutable=None, uint32_t start=1):
  """
  Given an array of integers, renumber all the unique values starting
  from 1. This can allow us to reduce the size of the data width required
  to represent it.

  arr: A 1d numpy array
  immutable: None or int, an optional value that shouldn't change
  start: Start renumbering from this value

  Return: a renumbered array
  """
  shape = arr.shape

  cdef uint32_t[:] arrview = arr.flatten()
  remap_dict = {}
  
  cdef uint32_t remap_id = start
  cdef int i = 0

  cdef uint32_t elem

  cdef uint32_t immut = 0
  cdef int size = arr.size

  if immutable is None:
    for i in range(size):
      elem = arrview[i]
      if elem in remap_dict:
        arrview[i] = remap_dict[elem]
      else:
        arrview[i] = remap_id
        remap_dict[elem] = remap_id
        remap_id += 1
  else:
    for i in range(size):
      elem = arrview[i]
      if elem == immut:
        continue
      if elem in remap_dict:
        arrview[i] = remap_dict[elem]
      else:
        arrview[i] = remap_id
        remap_dict[elem] = remap_id
        remap_id += 1
        
  if start < 0:
    types = [ np.int8, np.int16, np.int32, np.int64 ]
  else:
    types = [ np.uint8, np.uint16, np.uint32, np.uint64 ]
  
  factor = max(abs(start), abs(remap_id))

  if factor < 2 ** 8:
    return arr.astype(types[0]).reshape( shape ), remap_dict
  elif factor < 2 ** 16:
    return arr.astype(types[1]).reshape( shape ), remap_dict
  elif factor < 2 ** 32:
    return arr.astype(types[2]).reshape( shape ), remap_dict
  else:
    return arr.astype(types[3]).reshape( shape ), remap_dict

@cython.boundscheck(False)
def remap_from_array(cnp.ndarray[UINT] arr, cnp.ndarray[UINT] vals):
  cdef UINT[:] valview = vals
  cdef UINT[:] arrview = arr

  cdef size_t i = 0
  cdef size_t size = arr.size 
  cdef size_t maxkey = vals.size - 1
  cdef UINT elem

  with nogil:
    for i in range(size):
      elem = arr[i]
      if elem < 0 or elem > maxkey:
        continue
      arrview[i] = vals[elem]

  return arr

@cython.boundscheck(False)
def remap(cnp.ndarray[ALLINT] arr, cnp.ndarray[ALLINT] keys, cnp.ndarray[ALLINT] vals):
  cdef ALLINT[:] keyview = keys
  cdef ALLINT[:] valview = vals
  cdef ALLINT[:] arrview = arr
  cdef unordered_map[ALLINT, ALLINT] remap_dict

  assert keys.size == vals.size

  cdef size_t i = 0
  cdef size_t size = keys.size 
  cdef ALLINT elem

  with nogil:
    for i in range(size):
      remap_dict[keys[i]] = vals[i]

  i = 0
  size = arr.size 

  with nogil:
    for i in range(size):
      elem = arr[i]
      if remap_dict.find(elem) == remap_dict.end():
        continue
      else:
          arrview[i] = remap_dict[elem]

  return arr
          
  

