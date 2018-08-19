"""
Certain operations have to be fast for the skeletonization
procedure. The ones that didn't fit elsewhere (e.g. dijkstra
and the euclidean distance transform) have a home here.

Author: William Silversmith
Affiliation: Seung Lab, Princeton Neuroscience Institute
Date: August 2018
"""
cimport cython
from libc.stdlib cimport calloc, free
from libc.stdint cimport (
  uint32_t, uint8_t, int8_t, int16_t, int32_t, int64_t
)
from libcpp cimport bool
from cpython cimport array 
import array
import sys

from libcpp.vector cimport vector
cimport numpy as cnp
import numpy as np

cdef extern from "math.h":
  float INFINITY

@cython.boundscheck(False)
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def first_label(cnp.ndarray[uint8_t, cast=True, ndim=3] labels):
  cdef int sx, sy, sz 
  cdef int  x,  y,  z

  sx = labels.shape[0]
  sy = labels.shape[1]
  sz = labels.shape[2]

  for z in range(0, sz):
    for y in range(0, sy):
      for x in range(0, sx):
        if labels[x,y,z]:
          return (x,y,z)

  return None

@cython.boundscheck(False)
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def find_target(
    cnp.ndarray[uint8_t, cast=True, ndim=3] labels, 
    cnp.ndarray[float, ndim=3] PDRF
  ):
  cdef int x,y,z
  cdef sx, sy, sz

  sx = labels.shape[0]
  sy = labels.shape[1]
  sz = labels.shape[2]

  cdef int mx, my, mz

  mx = -1
  my = -1
  mz = -1

  cdef float maxpdrf = -INFINITY
  for x in range(0, sx):
    for y in range(0, sy):
      for z in range(0, sz):
        if labels[x,y,z] and PDRF[x,y,z] > maxpdrf:
          maxpdrf = PDRF[x,y,z]
          mx = x
          my = y
          mz = z

  return (mx, my, mz)

@cython.boundscheck(False)
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def roll_invalidation_ball(
    cnp.ndarray[uint8_t, cast=True, ndim=3] labels, 
    cnp.ndarray[uint32_t, ndim=2] path, 
    float scale, float const
  ):
  
  cdef int sx, sy, sz 

  sx = labels.shape[0]
  sy = labels.shape[1]
  sz = labels.shape[2]
  
  cdef float radius, dist
  cdef int ceil_r
  cdef int minx, maxx, miny, maxy, minz, maxz

  cdef int x,y,z
  cdef int x0, y0, z0

  cdef int invalidated = 0

  for coord in path:
    x0, y0, z0 = coord[0], coord[1], coord[2]
    radius = labels[x0,y0,z0] * scale + const
    ceil_r = <int>(radius + 0.5)

    minx = max(0, x0 - ceil_r)
    maxx = min(sx, x0 + ceil_r)
    miny = max(0, y0 - ceil_r)
    maxy = min(sy, y0 + ceil_r)
    minz = max(0, z0 - ceil_r)
    maxz = min(sz, z0 + ceil_r)

    radius *= radius 

    for x in range(minx, maxx):
      for y in range(miny, maxy):
        for z in range(minz, maxz):
          dist = (x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2
          if dist <= radius and labels[x,y,z]:
            invalidated += 1
            labels[x,y,z] = 0

  return invalidated, labels





