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
  uint32_t, int8_t, int16_t, int32_t, int64_t
)
from cpython cimport array 
import array
import sys

from libcpp.vector cimport vector
cimport numpy as cnp
import numpy as np

cdef extern from "math.h":
  float INFINITY

def roll_invalidation_ball(dbf, path, float scale, float const):
  cdef int sx, sy, sz 
  cdef float[:,:,:] dbfv = dbf

  sx = dbf.shape[0]
  sy = dbf.shape[1]
  sz = dbf.shape[2]
  
  cdef float radius, dist
  cdef int minx, maxx, miny, maxy, minz, maxz

  cdef int x,y,z

  for coord in path:
    radius = dbfv[coord[0], coord[1], coord[2]] * scale + const

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
          if (dist <= radius):
            dbfv[x,y,z] = INFINITY

  return dbf





