/* Includes such classic hits as roll_invalidation_ball.
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

  const int sxy = sx * sy;
  const int voxels = sxy * sz;

  int minx, maxx, miny, maxy, minz, maxz;
  int x, y, z;

  int16_t* topology = new int16_t[voxels];
  
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

    // perf: could do * iwx
    minx = std::max(0,  (int)(0.5 + (x - (radius / wx))));
    maxx = std::min(sx, (int)(0.5 + (x + (radius / wx))));
    miny = std::max(0,  (int)(0.5 + (y - (radius / wy))));
    maxy = std::min(sy, (int)(0.5 + (y + (radius / wy))));
    minz = std::max(0,  (int)(0.5 + (z - (radius / wz))));
    maxz = std::min(sz, (int)(0.5 + (z + (radius / wz))));

    // perf: could do min_offset, max_offset
    for (int y = miny; y < maxy; y++) {
      for (int z = minz; z < maxz; z++) {
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
    for (int i = 0; i < sx; i++, idx++) {
      coloring += topology[idx];
      // perf: could try invalidated += (coloring > 0) * labels[idx] etc
      if (coloring > 0) {
        invalidated += labels[idx];
        labels[idx] = 0;
      }
    }
  }

  free(topology);

  return invalidated;
}

};

#endif