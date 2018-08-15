"""
Certain operations have to be fast for the skeletonization
procedure. The ones that didn't fit elsewhere (e.g. dijkstra
and the euclidean distance transform) have a home here.

Author: William Silversmith
Affiliation: Seung Lab, Princeton Neuroscience Institute
Date: August 2018
"""

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

def first_label(cnp.ndarray[uint8_t, cast=True, ndim=3] labels):
  cdef int sx, sy, sz 
  cdef int  x,  y,  z

  sx = labels.shape[0]
  sy = labels.shape[1]
  sz = labels.shape[2]

  for x in range(0, sx):
    for y in range(0, sy):
      for z in range(0, sz):
        if labels[x,y,z]:
          return (x,y,z)

  return None

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
  cdef int minx, maxx, miny, maxy, minz, maxz

  cdef int x,y,z

  cdef int invalidated = 0

  for coord in path:
    radius = labels[coord[0], coord[1], coord[2]] * scale + const

    minx = max(0, coord[0] - radius)
    maxx = min(sx - 1, coord[0] + radius)
    miny = max(0, coord[1] - radius)
    maxy = min(sy - 1, coord[1] + radius)
    minz = max(0, coord[2] - radius)
    maxz = min(sz - 1, coord[2] + radius)

    radius *= radius 

    for x in (minx, maxx):
      for y in (miny, maxy):
        for z in (minz, maxz):
          dist = (x - coord[0]) ** 2 + (y - coord[1]) ** 2 + (z - coord[2]) ** 2
          if dist <= radius and labels[x,y,z]:
            invalidated += 1
            labels[x,y,z] = 0

  return invalidated





