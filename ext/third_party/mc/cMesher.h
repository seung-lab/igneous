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

class CMesher {
 private:
  zi::mesh::marching_cubes<uint64_t> marchingcubes_;
  zi::mesh::simplifier<double> simplifier_;

 public:
  CMesher();
  ~CMesher();
  void mesh(const std::vector<uint64_t> &data, unsigned int sx, unsigned int sy,
            unsigned int sz);
  std::vector<uint64_t> ids();
  MeshObject get_mesh(uint64_t id, bool generate_normals, int simplification_factor, int max_error);
};
