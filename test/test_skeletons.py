import pytest

import numpy as np

from cloudvolume import PrecomputedSkeleton

from igneous.tasks.skeletonization import postprocess

def test_consolidate():

  skel = PrecomputedSkeleton(
    vertices=np.array([
      (0, 0, 0),
      (1, 0, 0),
      (2, 0, 0),
      (0, 0, 0),
      (2, 1, 0),
      (2, 2, 0),
      (2, 2, 1),
      (2, 2, 2),
    ], dtype=np.float32),

    edges=np.array([
      [0, 1],
      [1, 2],
      [2, 3],
      [3, 4],
      [4, 5],
      [5, 6],
      [6, 7],
    ], dtype=np.uint32),

    radii=np.array([
      0, 1, 2, 3, 4, 5, 6, 7
    ], dtype=np.float32),

    vertex_types=np.array([
      0, 1, 2, 3, 4, 5, 6, 7
    ], dtype=np.uint8),
  )

  correct_skel = PrecomputedSkeleton(
    vertices=np.array([
      (0, 0, 0),
      (1, 0, 0),
      (2, 0, 0),
      (2, 1, 0),
      (2, 2, 0),
      (2, 2, 1),
      (2, 2, 2),
    ], dtype=np.float32),

    edges=np.array([
      [0, 1],
      [1, 2],
      [2, 0],
      [0, 3],
      [3, 4],
      [4, 5],
      [5, 6],
    ], dtype=np.uint32),

    radii=np.array([
      0, 1, 2, 4, 5, 6, 7
    ], dtype=np.float32),

    vertex_types=np.array([
      0, 1, 2, 4, 5, 6, 7
    ], dtype=np.uint8),
  )

  consolidated_skel = postprocess.consolidate_skeleton(skel)

  assert np.all(consolidated_skel.vertices == correct_skel.vertices)
  assert np.all(np.sort(consolidated_skel.edges, axis=0) == np.sort(correct_skel.edges, axis=0))
  assert np.all(consolidated_skel.radii == correct_skel.radii)
  assert np.all(consolidated_skel.vertex_types == correct_skel.vertex_types)

  
