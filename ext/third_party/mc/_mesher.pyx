# distutils: language = c++
# distutils: sources = ext/third_party/mc/cMesher.cpp

# Cython interface file for wrapping the object
#
#

from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t
from libcpp.vector cimport vector
from libcpp cimport bool
from libcpp.string cimport string

import numpy as np

# c++ interface to cython
cdef extern from "cMesher.h":
    cdef struct meshobj:
        vector[float] points
        vector[float] normals
        vector[unsigned int] faces

    cdef cppclass cMesher[T]:
        cMesher() except +
        void mesh(vector[T], unsigned int, unsigned int, unsigned int)
        vector[T] ids()
        meshobj get_mesh(T, bool normals, int simplification_factor, int max_simplification_error)
        bool write_obj(T id, string filename)

cdef class Mesher8:
    cdef cMesher[uint8_t] *thisptr      # hold a C++ instance which we're wrapping
    def __cinit__(self):
        self.thisptr = new cMesher[uint8_t]()
    def __dealloc__(self):
        del self.thisptr
    def mesh(self, data, sx, sy, sz):
        self.thisptr.mesh(data, sx, sy, sz)
    def ids(self):
        return self.thisptr.ids()
    def get_mesh(self, mesh_id, normals=False, simplification_factor=0, max_simplification_error=8):
        return self.thisptr.get_mesh(mesh_id, normals, simplification_factor, max_simplification_error)
    def write_obj(self, mesh_id, filename):
        return self.thisptr.write_obj(mesh_id, filename)

cdef class Mesher16:
    cdef cMesher[uint16_t] *thisptr      # hold a C++ instance which we're wrapping
    def __cinit__(self):
        self.thisptr = new cMesher[uint16_t]()
    def __dealloc__(self):
        del self.thisptr  
    def mesh(self, data, sx, sy, sz):
        self.thisptr.mesh(data, sx, sy, sz)
    def ids(self):
        return self.thisptr.ids()
    def get_mesh(self, mesh_id, normals=False, simplification_factor=0, max_simplification_error=8):
        return self.thisptr.get_mesh(mesh_id, normals, simplification_factor, max_simplification_error)
    def write_obj(self, mesh_id, filename):
        return self.thisptr.write_obj(mesh_id, filename)

cdef class Mesher32:
    cdef cMesher[uint32_t] *thisptr      # hold a C++ instance which we're wrapping
    def __cinit__(self):
        self.thisptr = new cMesher[uint32_t]()    
    def __dealloc__(self):
        del self.thisptr
    def mesh(self, data, sx, sy, sz):
        self.thisptr.mesh(data, sx, sy, sz)
    def ids(self):
        return self.thisptr.ids()
    def get_mesh(self, mesh_id, normals=False, simplification_factor=0, max_simplification_error=8):
        return self.thisptr.get_mesh(mesh_id, normals, simplification_factor, max_simplification_error)
    def write_obj(self, mesh_id, filename):
        return self.thisptr.write_obj(mesh_id, filename)

cdef class Mesher64:
    cdef cMesher[uint64_t] *thisptr      # hold a C++ instance which we're wrapping
    def __cinit__(self):
        self.thisptr = new cMesher[uint64_t]()
    def __dealloc__(self):
        del self.thisptr
    def mesh(self, data, sx, sy, sz):
        self.thisptr.mesh(data, sx, sy, sz)
    def ids(self):
        return self.thisptr.ids()
    def get_mesh(self, mesh_id, normals=False, simplification_factor=0, max_simplification_error=8):
        return self.thisptr.get_mesh(mesh_id, normals, simplification_factor, max_simplification_error)
    def write_obj(self, mesh_id, filename):
        return self.thisptr.write_obj(mesh_id, filename)
