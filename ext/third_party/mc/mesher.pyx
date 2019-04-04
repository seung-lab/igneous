# distutils: language = c++
# distutils: sources = ext/third_party/mc/cMesher.cpp

# Cython interface file for wrapping the object
#
#

from libc.stdint cimport uint64_t, uint32_t
from libcpp.vector cimport vector
from libcpp cimport bool
from libcpp.string cimport string

import numpy as np

# c++ interface to cython
cdef extern from "cMesher.h":
  cdef struct MeshObject:
    vector[float] points
    vector[float] normals
    vector[unsigned int] faces

  cdef cppclass CMesher[T]:
    CMesher(vector[uint32_t] voxel_res) except +
    void mesh(vector[T], unsigned int, unsigned int, unsigned int)
    vector[T] ids()
    MeshObject get_mesh(T id, bool normals, int simplification_factor, int max_simplification_error)

# creating a cython wrapper class
cdef class Mesher32:
  cdef CMesher[uint32_t] *thisptr      # hold a C++ instance which we're wrapping
  def __cinit__(self, voxel_res):
    self.thisptr = new CMesher[uint32_t](voxel_res.astype(np.uint32))
  def __dealloc__(self):
    del self.thisptr
  def mesh(self, data):    
    self.thisptr.mesh(
      data.flatten().astype(np.uint32), 
      data.shape[0], data.shape[1], data.shape[2]
    )
  def ids(self):
    return self.thisptr.ids()
  def get_mesh(self, mesh_id, normals=False, simplification_factor=0, max_simplification_error=8):
    return self.thisptr.get_mesh(mesh_id, normals, simplification_factor, max_simplification_error)

cdef class Mesher64:
  cdef CMesher[uint64_t] *thisptr      # hold a C++ instance which we're wrapping
  def __cinit__(self, voxel_res):
    self.thisptr = new CMesher[uint64_t](voxel_res.astype(np.uint32))
  def __dealloc__(self):
    del self.thisptr
  def mesh(self, data):    
    self.thisptr.mesh(
      data.flatten().astype(np.uint64), 
      data.shape[0], data.shape[1], data.shape[2]
    )
  def ids(self):
    return self.thisptr.ids()
  def get_mesh(self, mesh_id, normals=False, simplification_factor=0, max_simplification_error=8):
    return self.thisptr.get_mesh(mesh_id, normals, simplification_factor, max_simplification_error)
