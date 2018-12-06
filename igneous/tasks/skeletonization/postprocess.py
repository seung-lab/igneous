import networkx as nx
import numpy as np

from scipy import spatial
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import dijkstra
import scipy.sparse.csgraph as csgraph

from cloudvolume.lib import Bbox
from cloudvolume import PrecomputedSkeleton

## Public API of Module

def trim_skeleton(skeleton, dust_threshold=100, tick_threshold=100):
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

def remove_edges(edges, predecessor, start_idx, end_idx):
  edges = np.sort(edges)

  path = [end_idx]
  current_idx = end_idx
  
  while current_idx != start_idx:
    pred_idx = predecessor[start_idx, current_idx]
    
    current_edge = np.sort(np.array([pred_idx, current_idx]))
    current_edge_idx = np.where((edges[:,0]==current_edge[0])*(edges[:,1]==current_edge[1]))[0]
    edges = np.delete(edges,current_edge_idx,0)

    current_idx = pred_idx
    path.append(pred_idx)

  path = np.array(path)
  return edges, path

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

def remove_row(array, rows2remove):
  array = np.sort(array, axis=1)
  rows2remove = np.sort(rows2remove, axis=1)

  for i in range(rows2remove.shape[0]):
    idx = find_row(array,rows2remove[i,:])

    if np.sum(idx == -1) == 0:
      array = np.delete(array, idx, axis=0)

  return array

def interpolate_line(point1, point2):
  n_step = 10

  NDIM = point1.size
  int_points = np.zeros((n_step,NDIM))
  
  point1 = point1.astype(np.float32)
  point2 = point2.astype(np.float32)
  for i in range(n_step):
    a = i + 1
    b = n_step - i
    int_points[i,:] = (a*point2 + b*point1) / (a + b)

  int_points = np.round(int_points)
  return np.unique(int_points, axis=0)

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

def remove_ticks(skeleton, threshold=150):
  if skeleton.empty():
    return skeleton

  edges = skeleton.edges
  path_all = np.ones(1)

  while path_all.shape[0] != 0:
    unique_nodes, unique_counts = np.unique(edges, return_counts=True)

    end_idx = np.where(unique_counts == 1)[0]

    path_all = np.array([])
    for i in range(end_idx.shape[0]):
      idx = end_idx[i]
      current_node = unique_nodes[idx]

      edge_row_idx, edge_col_idx = np.where(edges == current_node)

      path = np.array([])
      single_piece = 0
      while edge_row_idx.shape[0] == 1 and path.shape[0] < threshold:
        
        next_node = edges[edge_row_idx,1-edge_col_idx]
        path = np.concatenate((path,edge_row_idx))

        prev_row_idx = edge_row_idx
        prev_col_idx = 1-edge_col_idx
        current_node = next_node
        
        edge_row_idx, edge_col_idx = np.where(edges == current_node)

        if edge_row_idx.shape[0] == 1:
          single_piece = 1
          break

        next_row_idx = np.setdiff1d(edge_row_idx,prev_row_idx)
        next_col_idx = edge_col_idx[np.where(edge_row_idx == next_row_idx[0])[0]]

        edge_row_idx = next_row_idx 
        edge_col_idx = next_col_idx

      if path.shape[0] < threshold and single_piece == 0:
        path_all = np.concatenate((path_all, path))
     
    edges = np.delete(edges, path_all, axis=0)

  skeleton.edges = edges
  return skeleton.consolidate()

def remove_loops(skeleton):
  if skeleton.empty():
    return skeleton

  nodes = skeleton.vertices
  edges = skeleton.edges
  edges = np.sort(edges, axis=1)
  
  while True: # Loop until all cycles are removed
    G = nx.Graph()

    G.add_edges_from(edges)

    try: 
      edges_cycle = nx.find_cycle(G, orientation='ignore')
    except nx.exception.NetworkXNoCycle:
      break

    edges_cycle = np.array(edges_cycle)
    edges_cycle = np.sort(edges_cycle, axis=1)

    nodes_cycle = np.unique(edges_cycle)
    nodes_cycle = nodes_cycle.astype(np.int64)
    
    unique_nodes, unique_counts = np.unique(edges, return_counts=True)
    branch_nodes = unique_nodes[ unique_counts >= 3 ]

    branch_cycle = nodes_cycle[np.isin(nodes_cycle,branch_nodes)]
    branch_cycle = branch_cycle.astype(np.int64)

    if branch_cycle.shape[0] == 1:
      branch_cycle_point = nodes[branch_cycle,:]

      cycle_points = nodes[nodes_cycle,:]

      dist = np.sum((cycle_points - branch_cycle_point) ** 2, 1)
      end_node = nodes_cycle[np.argmax(dist)]

      edges = remove_row(edges, edges_cycle)

      new_edge = np.array([branch_cycle[0],end_node])
      new_edge = np.reshape(new_edge,[1,2])
      edges = np.concatenate((edges,new_edge), 0)

    elif branch_cycle.shape[0] == 2:
      path = nx.shortest_path(G,branch_cycle[0],branch_cycle[1])

      edge_path = path2edge(path)
      edge_path = np.sort(edge_path, axis=1)

      row_valid = np.ones(edges_cycle.shape[0])
      for i in range(edge_path.shape[0]):
        row_valid -= (edges_cycle[:,0] == edge_path[i,0]) * (edges_cycle[:,1] == edge_path[i,1])

      row_valid = row_valid.astype(np.bool)
      edge_path = edges_cycle[row_valid,:]

      edges = remove_row(edges, edge_path)

    elif branch_cycle.shape[0] == 0:
      edges = remove_row(edges, edges_cycle)

    else:
      branch_cycle_points = nodes[branch_cycle,:]

      centroid = np.mean(branch_cycle_points, axis=0)
      dist = np.sum((nodes - centroid) ** 2, 1)
      intersect_node = np.argmin(dist)
      intersect_point = nodes[intersect_node,:]

      edges = remove_row(edges, edges_cycle)      

      new_edges = np.zeros((branch_cycle.shape[0], 2))
      new_edges[:,0] = branch_cycle
      new_edges[:,1] = intersect_node

      if np.isin(intersect_node, branch_cycle):
        idx = np.where(branch_cycle == intersect_node)
        new_edges = np.delete(new_edges, idx, 0)

      edges = np.concatenate((edges,new_edges), 0)

  skeleton.vertices = nodes
  skeleton.edges = edges.astype(np.uint32)
  return skeleton.consolidate()

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

def find_row(array, row):
  """
  array: array to search for
  row: row to find

  Returns: row indices
  """
  row = np.array(row)

  if array.shape[1] != row.size:
    raise ValueError("Dimensions do not match!")
  
  NDIM = array.shape[1]
  valid = np.zeros(array.shape, dtype='bool')

  for i in range(NDIM):
    valid[:,i] = array[:,i] == row[i]

  row_loc = np.zeros([ array.shape[0], 1 ])

  if NDIM == 2:
    row_loc = valid[:,0] * valid[:,1]
  elif NDIM == 3:
    row_loc = valid[:,0] * valid[:,1] * valid[:,2]

  idx = np.where(row_loc==1)[0]
  if len(idx) == 0:
    idx = -1
  return idx