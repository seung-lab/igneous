# distutils: language = c++
# distutils: sources = ext/third_party/mc/cMesher.cpp

# Cython interface file for wrapping the object
#
#

from libc.stdint cimport uint64_t
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

    cdef cppclass CMesher:
        CMesher() except +
        void mesh(vector[uint64_t], unsigned int, unsigned int, unsigned int)
        vector[uint64_t] ids()
        MeshObject get_mesh(uint64_t, bool normals, int simplification_factor, int max_simplification_error)

# creating a cython wrapper class
cdef class Mesher:
    cdef CMesher *thisptr      # hold a C++ instance which we're wrapping
    def __cinit__(self):
        self.thisptr = new CMesher()
    def __dealloc__(self):
        del self.thisptr
    def mesh(self, data):    
        self.thisptr.mesh(
            data.astype(np.uint64).flatten(), 
            data.shape[0], data.shape[1], data.shape[2]
        )
    def ids(self):
        return self.thisptr.ids()
    def get_mesh(self, mesh_id, normals=False, simplification_factor=0, max_simplification_error=8):
        return self.thisptr.get_mesh(mesh_id, normals, simplification_factor, max_simplification_error)
