/*
Passing variables / arrays between cython and cpp
Example from
http://docs.cython.org/src/userguide/wrapping_CPlusPlus.html

Adapted to include passing of multidimensional arrays

*/

#include <vector>
#include <zi/mesh/int_mesh.hpp>
#include <zi/mesh/face_mesh.hpp>
#include <zi/mesh/quadratic_simplifier.hpp>
#include <zi/vl/vec.hpp>

#include "cMesher.h"

//////////////////////////////////
CMesher::CMesher(const std::vector<uint32_t> &voxelresolution) {
  voxelresolution_ = voxelresolution;
}

CMesher::~CMesher() {}

MeshObject CMesher::collect_simplified_mesh(bool generate_normals) {
  MeshObject obj;
  std::vector<zi::vl::vec3d> points;
  std::vector<zi::vl::vec3d> normals;
  std::vector<zi::vl::vec<unsigned, 3>> faces;

  simplifier_.get_faces(points, normals, faces);
  obj.points.reserve(3 * points.size());
  obj.faces.reserve(3 * faces.size());

  if (generate_normals) {
    obj.normals.reserve(3 * normals.size());
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
    simplifier_.optimize(
        simplifier_.face_count() / simplification_factor,
        max_simplification_error);
  }

  return collect_simplified_mesh(generate_normals);
}

MeshObject CMesher::simplify(const MeshObject &pymesh,
                             int simplification_factor,
                             int max_simplification_error,
                             bool generate_normals = false) {
  zi::mesh::face_mesh<float> zimesh;

  size_t v_cnt = pymesh.points.size();
  size_t f_cnt = pymesh.faces.size();

  const zi::vl::vec<float, 3> * const p =
      (const zi::vl::vec<float, 3> * const)pymesh.points.data();
  const zi::vl::vec<uint32_t, 3> * const f =
      (const zi::vl::vec<uint32_t, 3> * const)pymesh.faces.data();

  bool has_normals = true;
  const zi::vl::vec<float, 3> * n;

  if (pymesh.normals.size() == v_cnt) {
    n = (const zi::vl::vec<float, 3> *)pymesh.normals.data();
  } else {
    // No vertex normals found, but face_mesh.add() expects a vector to copy
    // from - just set it to points as a placeholder.
    // This is unrelated to the `generate_normals` flag. The simplifier
    // will take care of real vertex normal calculation, if desired.
    n = (const zi::vl::vec<float, 3> *)pymesh.points.data();
    has_normals = false;
  }

  zimesh.add(p, n, v_cnt/3, f, f_cnt/3);

  zimesh.fill_simplifier<double>(simplifier_);
  simplifier_.prepare(generate_normals && !has_normals);

  if (simplification_factor > 0) {
    simplifier_.optimize(simplifier_.face_count() / simplification_factor,
                         max_simplification_error);
  }

  return collect_simplified_mesh(generate_normals);
}
