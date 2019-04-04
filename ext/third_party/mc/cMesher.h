/*
Passing variables / arrays between cython and cpp
Example from
http://docs.cython.org/src/userguide/wrapping_CPlusPlus.html

Adapted to include passing of multidimensional arrays

*/

#include <vector>
#include <zi/mesh/marching_cubes.hpp>

struct MeshObject {
  std::vector<float> points;
  std::vector<float> normals;
  std::vector<unsigned int> faces;
};

template <typename T>
class CMesher {
 private:
  zi::mesh::marching_cubes<T> marchingcubes_;
  zi::mesh::simplifier<double> simplifier_;
  std::vector<uint32_t> voxelresolution_;

 public:
  CMesher(const std::vector<uint32_t> &voxelresolution);
  ~CMesher();
  void mesh(
    const std::vector<T> &data, 
    const std::size_t sx, const std::size_t sy, const std::size_t sz
  );
  std::vector<T> ids();
  MeshObject get_mesh(
    T id, bool generate_normals, 
    int simplification_factor, int max_error
  );
};
