import networkx as nx
import numpy as np

from scipy import spatial
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import dijkstra
import scipy.sparse.csgraph as csgraph

from cloudvolume.lib import Bbox
from cloudvolume import PrecomputedSkeleton

import igneous.skeletontricks

## Public API of Module

def trim_skeleton(skeleton, dust_threshold=100, tick_threshold=1500):
  skeleton = remove_dust(skeleton, dust_threshold) # edges
  skeleton = remove_loops(skeleton)
  skeleton = connect_pieces(skeleton)
  skeleton = remove_ticks(skeleton, tick_threshold)
  return skeleton

## Implementation Details Below

def combination_pairs(n):
  pairs = np.array([])

  for i in range(n):
    for j in range(n-i-1):
      pairs = np.concatenate((pairs, np.array([i, i+j+1 ])))

  pairs = np.reshape(pairs,[ pairs.shape[0] // 2, 2 ])
  return pairs.astype(np.uint16)

def find_connected(nodes, edges):
  s = nodes.shape[0] 
  nodes = np.unique(edges).astype(np.uint32)

  conn_mat = lil_matrix((s, s), dtype=np.bool)
  conn_mat[edges[:,0], edges[:,1]] = 1

  n, l = csgraph.connected_components(conn_mat, directed=False)
  
  l_nodes = l[nodes]
  l_list = np.unique(l_nodes)
  return [ l == i for i in l_list  ]

def remove_dust(skeleton, dust_threshold):
  """dust_threshold in # of edges"""
  nodes = skeleton.vertices
  edges = skeleton.edges 

  if skeleton.empty():
    return skeleton

  connected = find_connected(nodes, edges)

  for i in range(len(connected)):
    path = connected[i] # [ T, T, F, F, T, T ] etc

    if np.sum(path) < dust_threshold:
      path_nodes = np.where(path)[0]

      for j in range(len(path_nodes)):
        del_row_idx, del_col_idx = np.where(edges == path_nodes[j])
        edges = np.delete(edges, del_row_idx, 0)

  skeleton.edges = edges
  return skeleton.consolidate()

def connect_pieces(skeleton):
  if skeleton.empty():
    return skeleton

  nodes = skeleton.vertices
  edges = skeleton.edges
  radii = skeleton.radii

  all_connected = True
  while all_connected:
    connected = find_connected(nodes, edges)
    pairs = combination_pairs(len(connected))

    all_connected = False
    for i in range(pairs.shape[0]):
      path_piece = connected[pairs[i,0]]
      nodes_piece = nodes[path_piece].astype(np.float32)
      nodes_piece_idx = np.where(path_piece)[0]

      path_tree = connected[pairs[i,1]]
      nodes_tree = nodes[path_tree]
      nodes_tree_idx = np.where(path_tree)[0]
      tree = spatial.cKDTree(nodes_tree)

      (dist, idx) = tree.query(nodes_piece)
      min_dist = np.min(dist)

      if min_dist < 50:
        min_dist_idx = int(np.where(dist == min_dist)[0][0])
        start_idx = nodes_piece_idx[min_dist_idx]
        end_idx = nodes_tree_idx[idx[min_dist_idx]]

        # test if line between points exits object
        if (radii[start_idx] + radii[end_idx]) >= min_dist:
          new_edge = np.array([[ start_idx, end_idx ]])
          edges = np.concatenate((edges, new_edge), axis=0)
          all_connected = True
          break

  skeleton.edges = edges
  return skeleton.consolidate()

def remove_ticks(skeleton, threshold):
  """
  Simple merging of individual TESAR cubes results in lots of little 
  ticks due to the edge effect. We can remove them by thresholding
  the path length from a given branch to the "main body" of the neurite.

  If TEASAR parameters were chosen such that they allowed for spines to
  be traced, this is also an opportunity to correct for that.

  O(N^2) in the number of branches, but it should be possible to make
  this faster by being more intelligent about recomputation.

  Parameters:
    threshold: The maximum length in nanometers that may be culled.

  Returns: tick free skeleton
  """
  if skeleton.empty():
    return skeleton

  edges = np.copy(skeleton.edges)

  def extract_tick(current_node):
    edge_row_idx, edge_col_idx = np.where(edges == current_node)

    path = np.array([], dtype=np.uint32)
    distance = 0
    single_piece = False
    while edge_row_idx.shape[0] == 1 and distance < threshold:
      next_node = edges[edge_row_idx, 1 - edge_col_idx][0]
      path = np.concatenate((path, edge_row_idx))

      vertices = np.array([ skeleton.vertices[current_node], skeleton.vertices[next_node] ], dtype=np.float32)
      distance += np.linalg.norm(vertices[1,:] - vertices[0,:])

      prev_row_idx = edge_row_idx
      prev_col_idx = 1 - edge_col_idx
      current_node = next_node
      
      edge_row_idx, edge_col_idx = np.where(edges == current_node)

      if edge_row_idx.shape[0] == 1:
        single_piece = True
        break

      next_row_idx = np.setdiff1d(edge_row_idx, prev_row_idx)
      next_col_idx = edge_col_idx[np.where(edge_row_idx == next_row_idx[0])[0]]

      edge_row_idx = next_row_idx 
      edge_col_idx = next_col_idx

    return path, single_piece, distance

  while True:
    unique_nodes, unique_counts = np.unique(edges, return_counts=True)
    end_idx = np.where(unique_counts == 1)[0]
    potentials = []

    for idx in end_idx:
      current_node = unique_nodes[idx]
      path, single_piece, distance = extract_tick(current_node)

      if distance < threshold and not single_piece:
        potentials.append([ path, distance ])

    if len(potentials) == 0:
      break

    unique_counts = { unique_nodes[i]: unique_counts[i] for i in range(unique_counts.shape[0]) }
    potentials = sorted(potentials, key=lambda x: x[1])
    path, distance = potentials[0]
    edges = np.delete(edges, path, axis=0)

  skeleton.edges = edges
  return skeleton.consolidate()

def remove_loops(skeleton):
  if skeleton.empty():
    return skeleton

  skels = []
  for component in skeleton.components():
    skels.append(_remove_loops(component))

  return PrecomputedSkeleton.simple_merge(skels).consolidate()

def _remove_loops(skeleton):
  nodes = skeleton.vertices
  G = nx.Graph()
  G.add_edges_from(skeleton.edges)
  
  while True: # Loop until all cycles are removed
    edges = np.array(list(G.edges), dtype=np.int32)
    edges_cycle = igneous.skeletontricks.find_cycle(edges)

    if len(edges_cycle) == 0:
      break

    edges_cycle = np.array(edges_cycle, dtype=np.uint32)
    edges_cycle = np.sort(edges_cycle, axis=1)

    nodes_cycle = np.unique(edges_cycle)
    nodes_cycle = nodes_cycle.astype(np.int64)
    
    unique_nodes, unique_counts = np.unique(edges, return_counts=True)
    branch_nodes = unique_nodes[ unique_counts >= 3 ]

    # branch cycles are cycle nodes that coincide with a branch point
    branch_cycle = nodes_cycle[np.isin(nodes_cycle,branch_nodes)]
    branch_cycle = branch_cycle.astype(np.int64)

    if branch_cycle.shape[0] == 1:
      branch_cycle_point = nodes[branch_cycle, :]
      cycle_points = nodes[nodes_cycle, :]

      dist = np.sum((cycle_points - branch_cycle_point) ** 2, 1)
      end_node = nodes_cycle[np.argmax(dist)]

      G.remove_edges_from(edges_cycle)
      G.add_edge(branch_cycle[0], end_node)

    elif branch_cycle.shape[0] == 2:
      path = nx.shortest_path(G, branch_cycle[0], branch_cycle[1])

      edge_path = path2edge(path)
      edge_path = np.sort(edge_path, axis=1)

      row_valid = np.ones(edges_cycle.shape[0])
      for i in range(edge_path.shape[0]):
        row_valid -= (edges_cycle[:,0] == edge_path[i,0]) * (edges_cycle[:,1] == edge_path[i,1])

      row_valid = row_valid.astype(np.bool)
      edge_path = edges_cycle[row_valid,:]

      G.remove_edges_from(edge_path)

    elif branch_cycle.shape[0] == 0:
      G.remove_edges_from(edges_cycle)

    else:
      branch_cycle_points = nodes[branch_cycle,:]

      centroid = np.mean(branch_cycle_points, axis=0)
      dist = np.sum((nodes - centroid) ** 2, 1)
      intersect_node = np.argmin(dist)
      intersect_point = nodes[intersect_node,:]

      G.remove_edges_from(edges_cycle)

      new_edges = np.zeros((branch_cycle.shape[0], 2))
      new_edges[:,0] = branch_cycle
      new_edges[:,1] = intersect_node

      if np.isin(intersect_node, branch_cycle):
        idx = np.where(branch_cycle == intersect_node)
        new_edges = np.delete(new_edges, idx, 0)

      G.add_edges_from(new_edges)

  skeleton.vertices = nodes
  skeleton.edges = np.array(list(G.edges), dtype=np.uint32)
  return skeleton

def path2edge(path):
  """
  path: sequence of nodes

  Returns: sequence separated into edges
  """
  edges = np.zeros([len(path) - 1, 2], dtype=np.uint32)
  for i in range(len(path)-1):
    edges[i,0] = path[i]
    edges[i,1] = path[i+1]
  return edges
