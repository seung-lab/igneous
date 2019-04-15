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

template <typename PositionType, typename LabelType>
class CMesher {
 private:
  zi::mesh::marching_cubes<PositionType, LabelType> marchingcubes_;
  zi::mesh::simplifier<double> simplifier_;
  std::vector<uint32_t> voxelresolution_;

 public:
  CMesher(const std::vector<uint32_t> &voxelresolution);
  ~CMesher();
  void mesh(
    const std::vector<LabelType> &data, 
    unsigned int sx, unsigned int sy, unsigned int sz
  );
  std::vector<LabelType> ids();
  MeshObject get_mesh(LabelType id, bool generate_normals, int simplification_factor, int max_error);
  void clear();
  bool erase(LabelType id);
};
