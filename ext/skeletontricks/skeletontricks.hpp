/* Includes such classic hits as roll_invalidation_cube.
 * 
 * Author: William Silversmith
 * Affiliation: Seung Lab, Princeton University
 * Date: September 2018
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <queue>
#include <vector>

#ifndef SKELETONTRICKS_HPP
#define SKELETONTRICKS_HPP

namespace skeletontricks {

int _roll_invalidation_cube(
  uint8_t* labels, float* DBF,
  const int sx, const int sy, const int sz,
  const float wx, const float wy, const float wz,
  int* path, const int path_size,
  const float scale, const float constant) {

  if (path_size == 0) {
    return 0;
  }

  const int sxy = sx * sy;
  const int voxels = sxy * sz;

  int minx, maxx, miny, maxy, minz, maxz;
  int x, y, z;

  int global_minx = sx;
  int global_maxx = 0;

  int16_t* topology = new int16_t[voxels]();
  
  const bool power_of_two = !((sx & (sx - 1)) || (sy & (sy - 1))); 
  const int xshift = std::log2(sx); // must use log2 here, not lg/lg2 to avoid fp errors
  const int yshift = std::log2(sy);

  int loc;
  float radius;

  // First pass: compute toplology
  for (int i = 0; i < path_size; i++) {
    loc = path[i];
    radius = scale * DBF[loc] + constant;

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

    minx = std::max(0,     (int)(x - (radius / wx)));
    maxx = std::min(sx-1,  (int)(0.5 + (x + (radius / wx))));
    miny = std::max(0,     (int)(y - (radius / wy)));
    maxy = std::min(sy-1,  (int)(0.5 + (y + (radius / wy))));
    minz = std::max(0,     (int)(z - (radius / wz)));
    maxz = std::min(sz-1,  (int)(0.5 + (z + (radius / wz))));

    global_minx = std::min(global_minx, minx);
    global_maxx = std::max(global_maxx, maxx);

    for (y = miny; y <= maxy; y++) {
      for (z = minz; z <= maxz; z++) {
        topology[minx + sx * y + sxy * z] += 1;
        topology[maxx + sx * y + sxy * z] -= 1;
      }
    }
  }

  // Second pass: invalidate labels
  int coloring;
  int idx = 0;
  int invalidated = 0;
  while (idx < voxels) {
    coloring = 0;
    idx += global_minx;
    for (int i = global_minx; i <= global_maxx; i++, idx++) {
      coloring += topology[idx];
      if (coloring > 0 || topology[idx]) {
        invalidated += labels[idx];
        labels[idx] = 0;
      }
    }
    idx += (sx - global_maxx - 1);
  }

  free(topology);

  return invalidated;
}

};

#endif