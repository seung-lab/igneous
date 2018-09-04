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
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def renumber(arr, uint64_t start=1):
  """
  Given an array of integers, renumber all the unique values starting
  from 1. This can allow us to reduce the size of the data width required
  to represent it.

  arr: A numpy array
  start (default: 1): Start renumbering from this value

  Return: a renumbered array, dict with remapping of oldval => newval
  """
  shape = arr.shape

  cdef uint64_t[:] arrview64 
  cdef uint32_t[:] arrview32

  sixyfourbit = np.dtype(arr.dtype).itemsize > 4
  order = 'F' if arr.flags['F_CONTIGUOUS'] else 'C'

  if sixyfourbit:
    arrview64 = arr.astype(np.uint64).flatten(order)
  else:
    arrview32 = arr.astype(np.uint32).flatten(order)

  remap_dict = { 0: 0 }
  
  cdef uint64_t remap_id = start
  cdef int i = 0

  cdef uint64_t elem
  cdef int size = arr.size
  if sixyfourbit:
    for i in range(size):
      elem = arrview64[i]
      if elem in remap_dict:
        arrview64[i] = remap_dict[elem]
      else:
        arrview64[i] = remap_id
        remap_dict[elem] = remap_id
        remap_id += 1
  else:
    for i in range(size):
      elem = arrview32[i]
      if elem in remap_dict:
        arrview32[i] = remap_dict[elem]
      else:
        arrview32[i] = remap_id
        remap_dict[elem] = remap_id
        remap_id += 1

  if start < 0:
    types = [ np.int8, np.int16, np.int32, np.int64 ]
  else:
    types = [ np.uint8, np.uint16, np.uint32, np.uint64 ]
  
  factor = max(abs(start), abs(remap_id))

  if factor < 2 ** 8:
    final_type = types[0]
  elif factor < 2 ** 16:
    final_type = types[1]
  elif factor < 2 ** 32:
    final_type = types[2]
  else:
    final_type = types[3]

  if sixyfourbit:
    output = bytearray(arrview64)
    intermediate_dtype = np.uint64
  else:
    output = bytearray(arrview32)
    intermediate_dtype = np.uint32

  output = np.frombuffer(output, dtype=intermediate_dtype).astype(final_type)
  output = output.reshape( arr.shape, order=order)
  return output, remap_dict

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
          
  

