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
        MeshObject get_mesh(L segid, bool normals, int simplification_factor, int max_simplification_error)
        bool erase(L segid)
        void clear()

class Mesher:
    def __init__(self, voxel_res):
        self._mesher = Mesher6464(voxel_res)
        self.voxel_res = voxel_res

    def mesh(self, data):
        del self._mesher

        shape = np.array(data.shape)
        nbytes = np.dtype(data.dtype).itemsize

        # 1024 = 10 bits each allocated to X, Y, Z fields
        if np.any(shape >= 1024):
            MesherClass = Mesher6432 if nbytes <= 4 else Mesher6464
        else:
            MesherClass = Mesher3232 if nbytes <= 4 else Mesher3264

        self._mesher = MesherClass(self.voxel_res)

        return self._mesher.mesh(data)

    def ids(self):
        return self._mesher.ids()

    def get_mesh(self, mesh_id, normals=False, simplification_factor=0, max_simplification_error=8):
        return self._mesher.get_mesh(
            mesh_id, normals, simplification_factor, max_simplification_error
        )
    
    def clear(self):
        self._mesher.clear()
   
    def erase(self, segid):
        return self._mesher.erase(segid)


# creating a cython wrapper class
cdef class Mesher6464:
    cdef CMesher[uint64_t, uint64_t] *ptr      # hold a C++ instance which we're wrapping

    def __cinit__(self, voxel_res):
        self.ptr = new CMesher[uint64_t, uint64_t](voxel_res.astype(np.uint32))

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

cdef class Mesher6432:
    cdef CMesher[uint64_t, uint32_t] *ptr      # hold a C++ instance which we're wrapping

    def __cinit__(self, voxel_res):
        self.ptr = new CMesher[uint64_t, uint32_t](voxel_res.astype(np.uint32))

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

cdef class Mesher3264:
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

cdef class Mesher3232:
    cdef CMesher[uint32_t, uint32_t] *ptr      # hold a C++ instance which we're wrapping

    def __cinit__(self, voxel_res):
        self.ptr = new CMesher[uint32_t, uint32_t](voxel_res.astype(np.uint32))

    def __dealloc__(self):
        del self.ptr

    def mesh(self, data):
        self.ptr.mesh(
            data.astype(np.uint32).flatten(), 
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











