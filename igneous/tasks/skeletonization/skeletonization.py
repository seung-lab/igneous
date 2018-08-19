"""
Skeletonization algorithm based on TEASAR (Sato et al. 2000).

Authors: Alex Bae and Will Silversmith
Affiliation: Seung Lab, Princeton Neuroscience Institue
Date: June-August 2018
"""
from collections import defaultdict

import numpy as np
from scipy import ndimage

import igneous.dijkstra 
import igneous.skeletontricks

from .definitions import Skeleton, path2edge

from cloudvolume.lib import save_images

def TEASAR(DBF, parameters):
  """
  Given a point cloud, convert it into a skeleton.
  
  Based on the algorithm by:

  Sato, et al. "TEASAR: tree-structure extraction algorithm for accurate and robust skeletons"  
    Proc. the Eighth Pacific Conference on Computer Graphics and Applications. Oct. 2000.
    doi:10.1109/PCCGA.2000.883951 (https://ieeexplore.ieee.org/document/883951/)

  object_points: n x 3 point cloud format of the object
  parameters: list of "scale" and "constant" parameters (first: "scale", second: "constant")
               larger values mean less senstive to detecting small branches

  Returns: Skeleton object
  """
  # Wasteful, can reduce to finding the first non-zero pixel
  
  labels = (DBF != 0).astype(np.bool)  
  any_voxel = igneous.skeletontricks.first_label(labels)   

  if any_voxel is None:
    return Skeleton()

  M = 1 / (np.max(DBF) ** 1.01)

  print("M=", M, "any_voxel=", any_voxel)

  # "4.4 DAF:  Compute distance from any voxel field"
  # Compute DAF, but we immediately convert to the PDRF
  # The extremal point of the PDRF is a valid root node
  # even if the DAF is computed from an arbitrary pixel.
  DBF[ DBF == 0 ] = np.inf
  DAF = igneous.dijkstra.distance_field(DBF, any_voxel)
  root = igneous.skeletontricks.find_target(labels, DAF)
  DAF = igneous.dijkstra.distance_field(DBF, root)

  print("root=", root)

  # Add p(v) to the DAF (pp. 4, section 4.5)
  # "4.5 PDRF: Compute penalized distance from root voxel field"
  # Let M > max(DBF)
  # p(v) = 5000 * (1 - DBF(v) / M)^16
  # 5000 is chosen to allow skeleton segments to be up to 3000 voxels
  # long without exceeding floating point precision.
  del DAF  
  PDRF = (20 * 5000) * ((1 - (DBF * M)) ** 16) # 20x is a variation on TEASAR
  PDRF = PDRF.astype(np.float32)

  def xyskelprojection(path):
    black = np.zeros((512,512,1), dtype=np.uint8)
    outline = labels.any(axis=-1).astype(np.uint8) * 77
    outline = outline.reshape( (512,512,1) )
    black += outline
    for coord in path:
      black[coord[0], coord[1]] = 255
    save_images(black, directory="./saved_images/projections/")

  paths = []
  valid_labels = np.count_nonzero(labels)

  while valid_labels > 0:
    target = igneous.skeletontricks.find_target(labels, PDRF)
    path = igneous.dijkstra.dijkstra(PDRF, root, target)
    invalidated, labels = igneous.skeletontricks.roll_invalidation_ball(
      labels, path, parameters[0], parameters[1]
    )
    valid_labels -= invalidated
    paths.append(path)

  skel_verts, skel_edges = path_union(paths)
  skel_radii = DBF[skel_nodes[::3], skel_nodes[1::3], skel_nodes[2::3]]

  return Skeleton(skel_nodes, skel_edges, skel_radii)

def path_union(paths):
  tree = defaultdict(set)
  tree_id = {}

  ct = 0
  for path in paths:
    for i in range(path.shape[0] - 1):
      parent = tuple(path[i, :].tolist())
      child = tuple(path[i + 1, :].tolist())
      tree[parent].update(child)
      if not parent in tree_id:
        tree_id[parent] = ct
        ct += 1

  root = tuple(paths[0][0,:].tolist())
  vertices = []
  edges = []

  def dfs(node):
    vertices.append(node)
    for child in tree[node]:
      edges.append([ tree_id[parent], tree_id[child] ])
      dfs(child)

  dfs(root)

  npv = np.zeros((len(vertices) * 3,), dtype=np.float32)
  for i, vert in enumerate(vertices):
    npv[ 3 * i + 0 ] = vertices[i][0]
    npv[ 3 * i + 1 ] = vertices[i][1]
    npv[ 3 * i + 2 ] = vertices[i][2]

  npe = np.zeros((len(edges) * 2,), dtype=np.uint32)
  for i, edge in enumerate(edges):
    npe[ 2 * i + 0 ] = edges[i][0]
    npe[ 2 * i + 1 ] = edges[i][1]

  return npv, npe

