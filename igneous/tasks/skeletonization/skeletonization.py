"""
Skeletonization algorithm based on TEASAR (Sato et al. 2000).

Authors: Alex Bae and Will Silversmith
Affiliation: Seung Lab, Princeton Neuroscience Institue
Date: June-August 2018
"""
from collections import defaultdict

import numpy as np
from scipy import ndimage
from PIL import Image

import igneous.dijkstra 
import igneous.skeletontricks

from .definitions import Skeleton, path2edge
from math import log
from cloudvolume.lib import save_images, mkdir

def check_kth_power(n, k):
    return log(n, k) % 1 == 0

def TEASAR(labels, DBF, scale=10, const=10, anisotropy=(1,1,1), max_boundary_distance=5000, pdrf_scale = 5000, pdrf_exponent=16 ):
  """
  Given the euclidean distance transform of a label ("Distance to Boundary Function"), 
  convert it into a skeleton with scale and const TEASAR parameters. 

  DBF: Result of the euclidean distance transform. Must represent a single label.
  scale: during the "rolling ball" invalidation phase, multiply the DBF value by this.
  const: during the "rolling ball" invalidation phase, this is the minimum radius in voxels.
  anisotropy: (x,y,z) relative scaling factors for distance
  max_boundary_distance: skip labels that have a DBF maximum value greater than this
    (e.g. for skipping somas). This value should be in nanometers, but if you are using
    this outside its original context it could be voxels.
  pdrf_scale: scale factor in front of dbf, used to weight dbf over euclidean distance (higher to pay more attention to dbf) (default 5000)
  pdrf_exponent: exponent in dbf formula on distance from edge (default 16.. one of 1,2,4,8,16,32..)

  Based on the algorithm by:

  M. Sato, I. Bitter, M. Bender, A. Kaufman, and M. Nakajima. 
  "TEASAR: tree-structure extraction algorithm for accurate and robust skeletons"  
    Proc. the Eighth Pacific Conference on Computer Graphics and Applications. Oct. 2000.
    doi:10.1109/PCCGA.2000.883951 (https://ieeexplore.ieee.org/document/883951/)

  Returns: Skeleton object
  """
  dbf_max = np.max(DBF)

  # > 5000 nm, gonna be a soma or blood vessel
  if dbf_max > max_boundary_distance:
    somata_coords = np.unravel_index(np.argmax(DBF), DBF.shape)
    invalidated, labels = igneous.skeletontricks.roll_invalidation_ball(
      labels, DBF, [ somata_coords ], scale=1, const=0, 
      anisotropy=anisotropy
    )
    
  any_voxel = igneous.skeletontricks.first_label(labels)   
  if any_voxel is None: 
    return Skeleton()

  # "4.4 DAF:  Compute distance from any voxel field"
  # Compute DAF, but we immediately convert to the PDRF
  # The extremal point of the PDRF is a valid root node
  # even if the DAF is computed from an arbitrary pixel.
  DAF = igneous.dijkstra.euclidean_distance_field(
    np.asfortranarray(labels), any_voxel, anisotropy=anisotropy)
  root = igneous.skeletontricks.find_target(labels, DAF)
  DAF = igneous.dijkstra.euclidean_distance_field(
    np.asfortranarray(labels), root, anisotropy=anisotropy)

  # Add p(v) to the DAF (pp. 4, section 4.5)
  # "4.5 PDRF: Compute penalized distance from root voxel field"
  # Let M > max(DBF)
  # p(v) = 5000 * (1 - DBF(v) / M)^16
  # 5000 is chosen to allow skeleton segments to be up to 3000 voxels
  # long without exceeding floating point precision.

  # IMPLEMENTATION NOTE: 
  # Appearently repeated *= is much faster than "** f(16)" 
  # 12,740.0 microseconds vs 4 x 560 = 2,240 microseconds (5.69x)

  # More clearly written:
  # PDRF = 5000 * ((1 - DBF * M) ** 16)

  DBF[DBF == 0] = np.inf
  f = lambda x: np.float32(x)
  M = 1 / (dbf_max ** 1.01)
 
  pdrf_exponent_lg2 = log(pdrf_exponent,2)

  # if pdrf_exponent is small enough and a perfect factor of 2, use iterative multiplication
  if ((pdrf_exponent_lg2<50000) & ((pdrf_exponent_lg2 % 1) == 0)): 
    PDRF = (f(1) - (DBF * M)) # ^0
    for k in range(pdrf_exponent_lg2):
      PDRF *= PDRF # ^dbf_exponent
  else: # otherwise fall back to regular exponent
    PDRF =  (f(1) - (DBF * M))**pdrf_exponent
  PDRF *= f(pdrf_scale)
  PDRF += DAF
  del DAF

  paths = []
  valid_labels = np.count_nonzero(labels)
    
  # Use dijkstra propogation w/o a target to generate a field of
  # pointers from each voxel to its parent. Then we can rapidly
  # compute multiple paths by simply hopping pointers using path_from_parents
  parents = igneous.dijkstra.parental_field(np.asfortranarray(PDRF), root)

  invalid_vertices = {}

  while valid_labels > 0:
    target = igneous.skeletontricks.find_target(labels, PDRF)
    path = igneous.dijkstra.path_from_parents(parents, target)
    invalidated, labels = igneous.skeletontricks.roll_invalidation_ball(
      labels, DBF, path, scale, const, 
      anisotropy=anisotropy, invalid_vertices=invalid_vertices,
    )
    valid_labels -= invalidated
    paths.append(path)
    for vertex in path:
      invalid_vertices[tuple(vertex)] = True

  del invalid_vertices

  skel_verts, skel_edges = path_union(paths)
  skel_radii = DBF[skel_verts[::3], skel_verts[1::3], skel_verts[2::3]]

  skel_verts = skel_verts.astype(np.float32).reshape( (skel_verts.size // 3, 3) )
  skel_edges = skel_edges.reshape( (skel_edges.size // 2, 2)  )

  return Skeleton(skel_verts, skel_edges, skel_radii)

def path_union(paths):
  """
  Given a set of paths with a common root, attempt to join them
  into a tree at the first common linkage.
  """
  tree = defaultdict(set)
  tree_id = {}
  vertices = []

  ct = 0
  for path in paths:
    for i in range(path.shape[0] - 1):
      parent = tuple(path[i, :].tolist())
      child = tuple(path[i + 1, :].tolist())
      tree[parent].add(child)
      if not parent in tree_id:
        tree_id[parent] = ct
        vertices.append(parent)
        ct += 1
      if not child in tree:
        tree[child] = set()
      if not child in tree_id:
        tree_id[child] = ct
        vertices.append(child)
        ct += 1 

  root = tuple(paths[0][0,:].tolist())
  edges = []

  # Note: Chose iterative rather than recursive solution
  # because somas can cause stack overflows for small TEASAR
  # parameters.
  stack = [ root ]

  while len(stack) > 0:
    parent = stack.pop()
    for child in tree[parent]:
      edges.append([ tree_id[parent], tree_id[child] ])
      stack.append(child)

  npv = np.zeros((len(vertices) * 3,), dtype=np.uint32)
  for i, vertex in enumerate(vertices):
    npv[ 3 * i + 0 ] = vertex[0]
    npv[ 3 * i + 1 ] = vertex[1]
    npv[ 3 * i + 2 ] = vertex[2]

  npe = np.zeros((len(edges) * 2,), dtype=np.uint32)
  for i, edge in enumerate(edges):
    npe[ 2 * i + 0 ] = edges[i][0]
    npe[ 2 * i + 1 ] = edges[i][1]

  return npv, npe

def xy_path_projection(paths, labels, N=0):
  if type(paths) != list:
    paths = [ paths ]

  projection = np.zeros( (labels.shape[0], labels.shape[1] ), dtype=np.uint8)
  outline = labels.any(axis=-1).astype(np.uint8) * 77
  outline = outline.reshape( (labels.shape[0], labels.shape[1] ) )
  projection += outline
  for path in paths:
    for coord in path:
      projection[coord[0], coord[1]] = 255

  projection = Image.fromarray(projection.T, 'L')
  N = str(N).zfill(3)
  mkdir('./saved_images/projections')
  projection.save('./saved_images/projections/{}.png'.format(N), 'PNG')

