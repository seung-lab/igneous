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
cdef extern from "cMesher.h":
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
cdef class Mesher:
    cdef CMesher[uint64_t, uint64_t] *ptr88      # hold a C++ instance which we're wrapping
    cdef CMesher[uint32_t, uint64_t] *ptr48
    cdef CMesher[uint64_t, uint32_t] *ptr84
    cdef CMesher[uint32_t, uint32_t] *ptr44
    cdef CMesher[uint64_t, uint16_t] *ptr82
    cdef CMesher[uint32_t, uint16_t] *ptr42
    cdef CMesher[uint64_t, uint8_t]  *ptr81
    cdef CMesher[uint32_t, uint8_t]  *ptr41

    def __cinit__(self, voxel_res):
        self.voxel_res = voxel_res.astype(np.uint32)
        self.ptrs = ( 
            self.ptr88, self.ptr48, self.ptr84, 
            self.ptr44, self.ptr82, self.ptr42, 
            self.ptr81, self.ptr41 
        )

    def __dealloc__(self):
        self._clear_ptrs()

    def ptr(self):
        return [ _ for _ in self.ptrs ][0]

    def _clear_ptrs(self):
        del self.ptr88
        del self.ptr48
        del self.ptr84
        del self.ptr44
        del self.ptr82
        del self.ptr42
        del self.ptr81
        del self.ptr41

    def mesh(self, data):
        self._clear_ptrs()

        cdef unsigned int limit32 = 1024 # 10 bit fields
        cdef bool sixtyfourbit = np.any(data.shape[:3] > limit32) 

        label_bytes = np.dtype(data.dtype).itemsize

        if label_bytes == 8:
            data = data.view(np.uint64)
            if sixtyfourbit:            
                self.ptr88 = new CMesher[uint64_t, uint64_t](self.voxel_res) 
            else:
                self.ptr48 = new CMesher[uint32_t, uint64_t](self.voxel_res)
        elif label_bytes == 4:
            data = data.view(np.uint32)
            if sixtyfourbit:            
                self.ptr84 = new CMesher[uint64_t, uint32_t](self.voxel_res) 
            else:
                self.ptr44 = new CMesher[uint32_t, uint32_t](self.voxel_res)
        elif label_bytes == 2:
            data = data.view(np.uint16)
            if sixtyfourbit:            
                self.ptr82 = new CMesher[uint64_t, uint16_t](self.voxel_res) 
            else:
                self.ptr42 = new CMesher[uint32_t, uint16_t](self.voxel_res)
        elif label_bytes == 1:
            data = data.view(np.uint8)
            if sixtyfourbit:            
                self.ptr81 = new CMesher[uint64_t, uint8_t](self.voxel_res) 
            else:
                self.ptr41 = new CMesher[uint32_t, uint8_t](self.voxel_res)

        self.ptr().mesh(
            data.flatten(), 
            data.shape[0], data.shape[1], data.shape[2]
        )

    def ids(self):
        return self.ptr().ids()
    def get_mesh(self, mesh_id, normals=False, simplification_factor=0, max_simplification_error=8):
        return self.ptr().get_mesh(mesh_id, normals, simplification_factor, max_simplification_error)
    def clear(self):
        self.ptr().clear()
    def erase(self, mesh_id):
        return self.ptr().erase(mesh_id)











