# distutils: language = c++
# distutils: sources = ext/third_party/mc/cMesher.cpp

# Cython interface file for wrapping the object
#
#

from libc.stdint cimport uint64_t, uint32_t, uint16_t, uint8_t
from libcpp.vector cimport vector
from libcpp cimport bool
from libcpp.string cimport string

import numpy as np

# c++ interface to cython
cdef extern from "cMesher.hpp":
    cdef struct MeshObject:
        vector[float] points
        vector[float] normals
        vector[unsigned int] faces

    cdef cppclass CMesher[P,L]:
        CMesher(vector[uint32_t] voxel_res) except +
        void mesh(vector[L], unsigned int, unsigned int, unsigned int)
        vector[L] ids()
        MeshObject get_mesh(L, bool normals, int simplification_factor, int max_simplification_error)
        bool erase(L id)
        void clear()

# creating a cython wrapper class
# cdef class Mesher:
#     cdef CMesher[uint64_t, uint64_t] *ptr      # hold a C++ instance which we're wrapping

#     def __cinit__(self, voxel_res):
#         self.ptr = new CMesher[uint64_t, uint64_t](voxel_res.astype(np.uint32))

#     def __dealloc__(self):
#         del self.ptr

#     def mesh(self, data):
#         self.ptr.mesh(
#             data.astype(np.uint64).flatten(), 
#             data.shape[0], data.shape[1], data.shape[2]
#         )

#     def ids(self):
#         return self.ptr.ids()
    
#     def get_mesh(self, mesh_id, normals=False, simplification_factor=0, max_simplification_error=8):
#         return self.ptr.get_mesh(mesh_id, normals, simplification_factor, max_simplification_error)
    
#     def clear(self):
#         self.ptr.clear()

#     def erase(self, mesh_id):
#         return self.ptr.erase(mesh_id)

cdef class Mesher:
    cdef CMesher[uint32_t, uint64_t] *ptr      # hold a C++ instance which we're wrapping

    def __cinit__(self, voxel_res):
        self.ptr = new CMesher[uint32_t, uint64_t](voxel_res.astype(np.uint32))

    def __dealloc__(self):
        del self.ptr

    def mesh(self, data):
        self.ptr.mesh(
            data.astype(np.uint64).flatten(), 
            data.shape[0], data.shape[1], data.shape[2]
        )

    def ids(self):
        return self.ptr.ids()
    
    def get_mesh(self, mesh_id, normals=False, simplification_factor=0, max_simplification_error=8):
        return self.ptr.get_mesh(mesh_id, normals, simplification_factor, max_simplification_error)
    
    def clear(self):
        self.ptr.clear()

    def erase(self, mesh_id):
        return self.ptr.erase(mesh_id)








