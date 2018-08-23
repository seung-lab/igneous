#include <cmath>
#include <cstdint>
#include <stack>

#ifndef FIND_TARGET_HPP
#define FIND_TARGET_HPP

namespace skeletontricks {

inline void compute_neighborhood(
  int *neighborhood, const size_t loc, 
  const int x, const int y, const int z,
  const size_t sx, const size_t sy, const size_t sz) {

  for (int i = 0; i < 6; i++) {
    neighborhood[i] = 0;
  }

  const int sxy = sx * sy;

  // 6-hood

  if (x > 0) {
    neighborhood[0] = -1;
  }
  if (x < (int)sx - 1) {
    neighborhood[1] = 1;
  }
  if (y > 0) {
    neighborhood[2] = -(int)sx;
  }
  if (y < (int)sy - 1) {
    neighborhood[3] = (int)sx;
  }
  if (z > 0) {
    neighborhood[4] = -sxy;
  }
  if (z < (int)sz - 1) {
    neighborhood[5] = sxy;
  }
}

int _find_target_in_shape(
    uint8_t* labels, uint8_t* eroded_labels, float* field,
    const size_t sx, const size_t sy, const size_t sz, 
    const size_t source
  ) {

  const size_t voxels = sx * sy * sz;
  const size_t sxy = sx * sy;

  const bool power_of_two = !((sx & (sx - 1)) || (sy & (sy - 1))); 
  const int xshift = std::log2(sx); // must use log2 here, not lg/lg2 to avoid fp errors
  const int yshift = std::log2(sy);

  bool *visited = new bool[voxels]();

  int neighborhood[6];
  size_t loc;

  int x, y, z;

  std::stack<int> tovisit;
  tovisit.push(source);

  float maximum = -1;
  int max_location = -1;

  while (!tovisit.empty()) {
    loc = tovisit.top();
    tovisit.pop();

    if (visited[loc] || !labels[loc]) {
      continue;
    }

    visited[loc] = true;

    if (eroded_labels[loc] && field[loc] > maximum) {
      maximum = field[loc];
      max_location = loc;
    }

    if (power_of_two) {
      z = loc >> (xshift + yshift);
      y = (loc - (z << (xshift + yshift))) >> xshift;
      x = loc - ((y + (z << yshift)) << xshift);
    }
    else {
      z = loc / sxy;
      y = (loc - (z * sxy)) / sx;
      x = loc - sx * (y + z * sy);
    }

    compute_neighborhood(neighborhood, loc, x, y, z, sx, sy, sz);

    for (int i = 0; i < 6; i++) {
      if (neighborhood[i]) {
        tovisit.push(loc + neighborhood[i]);
      }
    }
  }

  delete []visited;

  return max_location;
}

};

#endif
