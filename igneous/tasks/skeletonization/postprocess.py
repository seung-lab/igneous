import networkx as nx
import numpy as np

from scipy import spatial
from scipy.sparse import lil_matrix
import scipy.sparse.csgraph as csgraph

from cloudvolume.lib import Bbox

from .definitions import Skeleton, Nodes

## Public API of Module

def crop_skeleton(skeleton, bound):
  """Cropping is required to eliminate edge effects."""
  if type(bound) == Bbox:
    bound = np.array(bound.to_list()).reshape( (2,3) )

  nodes_valid_mask = get_valid(skeleton.nodes, bound)
  nodes_valid_idx = np.where(nodes_valid_mask)[0]

  edges_valid_mask = np.isin(skeleton.edges, nodes_valid_idx)
  edges_valid_idx = edges_valid_mask[:,0] * edges_valid_mask[:,1] 
  skeleton.edges = edges[edges_valid_idx,:]
  return consolidate_skeleton(skeleton)

def trim_skeleton(skeleton, ptcloud):
  skeleton = remove_dust(skeleton, 100) # 100 edges
  skeleton = remove_loops(skeleton)
  skeleton = connect_pieces(skeleton, ptcloud)
  skeleton = remove_ticks(skeleton)
  return skeleton

def merge_skeletons(skeleton1, skeleton2):
  nodes1 = skeleton1.nodes
  nodes2 = skeleton2.nodes

  edges1 = skeleton1.edges
  edges2 = skeleton2.edges

  tree1 = spatial.cKDTree(nodes1)

  nodes2 = nodes2.astype('float32')
  (dist, nodes1_idx) = tree1.query(nodes2)

  graph2 = edges2sparse(nodes2, edges2)
  
  overlap = dist == 0
  nodes2_overlap = np.where(overlap)[0]

  edges2 = remove_overlap_edges(nodes2_overlap, edges2) 

  connected = find_connected(nodes2, edges2)

  conn_mat1 = edges2sparse(nodes1, edges1)
  conn_mat2 = edges2sparse(nodes2, edges2)
  dist_mat1, pred1 = shortest_path(conn_mat1, directed=False, return_predecessors=True)

  for i in range(len(connected)):
    path = connected[i]
    path_idx = np.where(path)[0]

    conn_mat_path = conn_mat2[path,:][:,path]
    
    end_nodes2 = np.where(np.sum(conn_mat_path, 0)==1)[1]
    
    end_idx = path_idx[end_nodes2]

    end_points = nodes2[end_idx,:]
    
    end_nodes1 = np.zeros(end_points.shape[0], dtype='int')
    for j in range(end_points.shape[0]):
      p_end = end_points[j,:]

      node_end = find_row(nodes1, p_end)

      end_nodes1[j] = node_end

    if np.sum(end_nodes1<0) > 0:
      continue

    c = 1
    pairs = combination_pairs(end_nodes1.shape[0])
    for j in range(pairs.shape[0]):
      c = c * np.isfinite(dist_mat1[end_nodes1[pairs[j,0]],end_nodes1[pairs[j,1]]])

    if c==1:
      edges2 = remove_overlap_edges(path_idx, edges2)

  skeleton2.edges = edges2
  skeleton2 = consolidate_skeleton(skeleton2)
  return skeleton1, skeleton2

## Implementation Details Below

def get_valid(points, bound):
  return (points[:,0] > bound[0,0]) \
    * (points[:,0] < bound[1,0]) \
    * (points[:,1] > bound[0,1]) \
    * (points[:,1] < bound[1,1]) \
    * (points[:,2] > bound[0,2]) \
    * (points[:,2] < bound[1,2])

def edges2sparse(nodes, edges):
  s = nodes.shape[0]
  conn_mat = lil_matrix((s, s), dtype=bool)
  conn_mat[edges[:,0],edges[:,1]] = 1
  conn_mat[edges[:,1],edges[:,0]] = 1

  return conn_mat

def find_connected(nodes, edges):
  s = nodes.shape[0] 
  nodes = np.unique(edges)
  nodes = nodes.astype('int')

  conn_mat = lil_matrix((s, s), dtype=bool)
  conn_mat[edges[:,0],edges[:,1]] = 1

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

def remove_overlap_edges(nodes_overlap, edges):
  edge_overlap = np.isin(edges, nodes_overlap)
  del_idx = np.where(edge_overlap[:,0] * edge_overlap[:,1])
  edges = np.delete(edges,del_idx,0)
  return edges

def remove_dust(skeleton, dust_threshold):
  """dust_threshold in # of edges"""
  nodes = skeleton.nodes
  edges = skeleton.edges 

  connected = find_connected(nodes, edges)

  for i in range(len(connected)):
    path = connected[i]

    if np.sum(path) < dust_threshold:
      path_nodes = np.where(path)[0]

      for j in range(len(path_nodes)):
        del_row_idx, del_col_idx = np.where(edges == path_nodes[j])
        edges = np.delete(edges, del_row_idx, 0)

  skeleton.edges = edges
  return consolidate_skeleton(skeleton)

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
  
  point1 = point1.astype('float32')
  point2 = point2.astype('float32')
  for i in range(n_step):
    a = i + 1
    b = n_step - i
    int_points[i,:] = (a*point2 + b*point1) / (a + b)

  int_points = np.round(int_points)
  return np.unique(int_points, axis=0)

def connect_pieces(skeleton, ptcloud):
  nodes = skeleton.nodes
  edges = skeleton.edges

  all_connected = 1
  while all_connected == 1:
    connected = find_connected(nodes, edges)

    n_connected = len(connected)

    pairs = combination_pairs(n_connected)

    all_connected = 0
    for i in range(pairs.shape[0]):
      path_piece = connected[pairs[i,0]]
      nodes_piece = nodes[path_piece]
      nodes_piece = nodes_piece.astype('float32')
      nodes_piece_idx = np.where(path_piece)[0]

      path_tree = connected[pairs[i,1]]
      nodes_tree = nodes[path_tree]
      nodes_tree_idx = np.where(path_tree)[0]
      tree = spatial.cKDTree(nodes_tree)

      (dist, idx) = tree.query(nodes_piece)

      min_dist = np.min(dist)

      if min_dist < 50:
    
        min_dist_idx = int(np.where(dist==min_dist)[0][0])
        start_idx = nodes_piece_idx[min_dist_idx]
        end_idx = nodes_tree_idx[idx[min_dist_idx]]

        int_points = interpolate_line(nodes[start_idx,:],nodes[end_idx,:])
  
        for k in range(int_points.shape[0]):
          in_seg = np.sum(~np.any(ptcloud - int_points[k,:], axis=1))
          if in_seg == 0:
            break

        if in_seg:
          new_edge = np.array([start_idx,end_idx])
          new_edge = np.reshape(new_edge,[1,2])
          edges = np.concatenate((edges,new_edge),0)
          all_connected += 1
          break

  skeleton.edges = edges
  return consolidate_skeleton(skeleton)

def remove_ticks(skeleton, threshold=150):
  edges = skeleton.edges
  path_all = np.ones(1)

  while path_all.shape[0] != 0:
    unique_nodes, unique_counts = np.unique(edges, return_counts=True)

    end_idx = np.where(unique_counts==1)[0]

    path_all = np.array([])
    for i in range(end_idx.shape[0]):
      idx = end_idx[i]
      current_node = unique_nodes[idx]

      edge_row_idx, edge_col_idx = np.where(edges==current_node)

      path = np.array([])
      single_piece = 0
      while edge_row_idx.shape[0] == 1 and path.shape[0] < threshold:
        
        next_node = edges[edge_row_idx,1-edge_col_idx]
        path = np.concatenate((path,edge_row_idx))

        prev_row_idx = edge_row_idx
        prev_col_idx = 1-edge_col_idx
        current_node = next_node
        
        edge_row_idx, edge_col_idx = np.where(edges==current_node)

        if edge_row_idx.shape[0] == 1:
          single_piece = 1
          break

        next_row_idx = np.setdiff1d(edge_row_idx,prev_row_idx)
        next_col_idx = edge_col_idx[np.where(edge_row_idx==next_row_idx[0])[0]]

        edge_row_idx = next_row_idx 
        edge_col_idx = next_col_idx

      if path.shape[0] < threshold and single_piece == 0:
        path_all = np.concatenate((path_all,path))
     
    edges = np.delete(edges,path_all,axis=0)

  skeleton.edges = edges
  return consolidate_skeleton(skeleton)

def remove_loops(skeleton):
  nodes = skeleton.nodes
  edges = skeleton.edges
  edges = np.sort(edges, axis=1)
  
  cycle_exists = 1

  while cycle_exists == 1:
    G = nx.Graph()

    for i in range(edges.shape[0]):
      G.add_edge(edges[i,0], edges[i,1])

    try: 
      edges_cycle = nx.find_cycle(G, orientation='ignore')
    except:
      cycle_exists = 0
      continue

    edges_cycle = np.array(edges_cycle)
    edges_cycle = np.sort(edges_cycle, axis=1)

    nodes_cycle = np.unique(edges_cycle)
    nodes_cycle = nodes_cycle.astype('int')
    
    unique_nodes, unique_counts = np.unique(edges, return_counts=True)
    branch_nodes = unique_nodes[ unique_counts >= 3 ]

    branch_cycle = nodes_cycle[np.isin(nodes_cycle,branch_nodes)]
    branch_cycle = branch_cycle.astype('int')

    if branch_cycle.shape[0] == 1:
      branch_cycle_point = nodes[branch_cycle,:]

      cycle_points = nodes[nodes_cycle,:]

      dist = np.sum((cycle_points - branch_cycle_point)**2, 1)
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
        row_valid = row_valid - (edges_cycle[:,0]==edge_path[i,0])*(edges_cycle[:,1]==edge_path[i,1])

      row_valid = row_valid.astype('bool')
      edge_path = edges_cycle[row_valid,:]

      edges = remove_row(edges, edge_path)

    elif branch_cycle.shape[0] == 0:
      edges = remove_row(edges, edges_cycle)

    else:
      branch_cycle_points = nodes[branch_cycle,:]

      centroid = np.mean(branch_cycle_points, axis=0)
      dist = np.sum((nodes - centroid)**2, 1)
      intersect_node = np.argmin(dist)
      intersect_point = nodes[intersect_node,:]

      edges = remove_row(edges, edges_cycle)      

      new_edges = np.zeros((branch_cycle.shape[0],2))
      new_edges[:,0] = branch_cycle
      new_edges[:,1] = intersect_node

      if np.isin(intersect_node, branch_cycle):
        idx = np.where(branch_cycle==intersect_node)
        new_edges = np.delete(new_edges, idx, 0)

      edges = np.concatenate((edges,new_edges), 0)

  skeleton.nodes = nodes
  skeleton.edges = edges
  return consolidate_skeleton(skeleton)

def consolidate_skeleton(skeleton):
  nodes = skeleton.nodes 
  edges = skeleton.edges
  radii = skeleton.radii

  if nodes.shape[0] == 0 or edges.shape[0] == 0:
    return Skeleton()
  
  # Remove duplicate nodes
  unique_nodes, unique_idx, unique_counts = np.unique(nodes, axis=0, return_index=True, return_counts=True)
  unique_edges = np.copy(edges)

  dup_idx = np.where(unique_counts>1)[0]
  for i in range(dup_idx.shape[0]):
    dup_node = unique_nodes[dup_idx[i],:]
    dup_node_idx = find_row(nodes, dup_node)

    for j in range(dup_node_idx.shape[0]-1):
      start_idx, end_idx = np.where(edges==dup_node_idx[j+1])
      unique_edges[start_idx, end_idx] = unique_idx[dup_idx[i]]

  # Remove unnecessary nodes
  eff_node_list = np.unique(unique_edges)
  eff_node_list = eff_node_list.astype('int')
  
  eff_nodes = nodes[eff_node_list]
  eff_radii = radii[eff_node_list]

  eff_edges = np.copy(unique_edges)
  for i, node in enumerate(eff_node_list, 0):
    row_idx, col_idx = np.where(unique_edges==node)
    eff_edges[row_idx,col_idx] = i

  eff_edges = np.unique(eff_edges, axis=0)

  skeleton.nodes = eff_nodes
  skeleton.edges = eff_edges
  skeleton.radii = eff_radii
  return skeleton
