/*
Passing variables / arrays between cython and cpp
Example from
http://docs.cython.org/src/userguide/wrapping_CPlusPlus.html

Adapted to include passing of multidimensional arrays

*/

#include <vector>
#include <zi/mesh/int_mesh.hpp>
#include <zi/mesh/quadratic_simplifier.hpp>
#include <zi/vl/vec.hpp>

#include "cMesher.h"

//////////////////////////////////
CMesher::CMesher(const std::vector<uint32_t> &voxelresolution) {
  voxelresolution_ = voxelresolution;
}

CMesher::~CMesher() {}

void CMesher::mesh(const std::vector<uint64_t> &data, unsigned int sx,
                  unsigned int sy, unsigned int sz) {
  // Create Marching Cubes class for type T volume

  const uint64_t *a = &data[0];
  // Run global marching cubes, a mesh is generated for each segment ID group
  marchingcubes_.marche(a, sx, sy, sz);
}

std::vector<uint64_t> CMesher::ids() {
  std::vector<uint64_t> keys;
  for (auto it = marchingcubes_.meshes().begin();
       it != marchingcubes_.meshes().end(); ++it) {
    keys.push_back(it->first);
  }

  return keys;
}

MeshObject CMesher::get_mesh(uint64_t id, bool generate_normals,
                            int simplification_factor,
                            int max_simplification_error) {
  MeshObject obj;

  if (marchingcubes_.count(id) == 0) {  // MC produces no triangles if either
                                        // none or all voxels were labeled!
    return obj;
  }

  zi::mesh::int_mesh im;
  im.add(marchingcubes_.get_triangles(id));
  im.fill_simplifier<double>(simplifier_, 0, 0, 0, voxelresolution_[2],
      voxelresolution_[1], voxelresolution_[0]);
  simplifier_.prepare(generate_normals);

  if (simplification_factor > 0) {
    // This is the most cpu intensive line
    // max_error expects an error quadric, also MC scales the mesh by a factor
    // of 2. Therefore we have to square the max_distance, and account for the
    // factor of 2 from MC.
    simplifier_.optimize(
        simplifier_.face_count() / simplification_factor,
        (2.0 * max_simplification_error) * (2.0 * max_simplification_error));
  }

  std::vector<zi::vl::vec3d> points;
  std::vector<zi::vl::vec3d> normals;
  std::vector<zi::vl::vec<unsigned, 3> > faces;

  simplifier_.get_faces(points, normals, faces);
  obj.points.reserve(3 * points.size());
  obj.faces.reserve(3 * faces.size());

  if (generate_normals) {
    obj.normals.reserve(3 * points.size());
  }

  for (auto v = points.begin(); v != points.end(); ++v) {
    obj.points.push_back((*v)[2]);
    obj.points.push_back((*v)[1]);
    obj.points.push_back((*v)[0]);
  }

  if (generate_normals) {
    for (auto vn = normals.begin(); vn != normals.end(); ++vn) {
      obj.normals.push_back((*vn)[2]);
      obj.normals.push_back((*vn)[1]);
      obj.normals.push_back((*vn)[0]);
    }
  }

  for (auto f = faces.begin(); f != faces.end(); ++f) {
    obj.faces.push_back((*f)[0]);
    obj.faces.push_back((*f)[2]);
    obj.faces.push_back((*f)[1]);
  }

  return obj;
}
