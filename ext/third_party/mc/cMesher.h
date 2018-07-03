/*
Passing variables / arrays between cython and cpp
Example from 
http://docs.cython.org/src/userguide/wrapping_CPlusPlus.html

Adapted to include passing of multidimensional arrays

*/

#include <vector>
#include <iostream>
#include <zi/mesh/marching_cubes.hpp>

struct meshobj {
  std::vector<float> points;
  std::vector<float> normals;
  std::vector<unsigned int> faces;
};

class cMesher {
  private:
    zi::mesh::marching_cubes<uint64_t> mc;
    zi::mesh::simplifier<double> s;
  public:
    cMesher();
    ~cMesher();
    void mesh(const std::vector<uint64_t> &data,
              unsigned int sx, unsigned int sy, unsigned int sz);
    std::vector<uint64_t> ids();
    meshobj get_mesh(const uint64_t id, const bool generate_normals, const int simplification_factor, const int max_error);
    bool write_obj(const uint64_t id, const std::string &filename);
};
