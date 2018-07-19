"""
Skeletonization algorithm based on TEASAR (Sato et al. 2000).

Primary Author: Alex Bae
Integrating Author: Will Silversmith
Affiliation: Seung Lab, Princeton Neuroscience Institue
Date: June 2018
"""
import numpy as np

from scipy import ndimage
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix

from .definitions import Skeleton, Nodes, find_row, path2edge
from .postprocess import consolidate_skeleton

VERBOSE = False
def debug(txt):
  if VERBOSE:
    print(txt)

def skeletonize(
    point_cloud, parameters=[ 10, 10 ], dsmp_resolution=[1,1,1], 
    init_root=[], init_dest=[], soma=False 
  ):
  """
  Required:
    point_cloud: N x 3 XYZ int32 numpy array
  
  Optional:
    dsmp_resolution: downsample resolution (used for soma processing)
    parameters: [ m, b ]
      TEASAR bbox point elimination parameters.
      mx + b where x is the distance to boundary. Points within this radius get culled.
      Smaller values are better at detecting small branches like spines.
    init_roots: N x 3 array of initial root coordinates (for somas)
    init_dest: N x 3 array of inital dest coordinates (for somas)
    soma: (bool) Process somas?
   
  Returns: Skeleton()
  """
  if point_cloud.shape[0] == 0:
    return Skeleton()
  
  # If initial roots is empty, take the first point
  init_root = np.array(init_root)
  init_dest = np.array(init_dest)

  if len(init_root) == 0:
    init_root = point_cloud[0,:]
  else:
    for i in range(init_root.shape[0]):
      root = init_root[i,:]
      root_idx = find_row(point_cloud, root)

      if root_idx == -1:
        dist = np.sum((point_cloud - root) ** 2, 1)
        root = point_cloud[np.argmin(dist),:]
        init_root[i,:] = root 

  for i in range(init_dest.shape[0]):
    dest = init_dest[i,:]
    dest_idx = find_row(point_cloud, dest)

    if dest_idx == -1:
      dist = np.sum((point_cloud - dest) ** 2, 1)
      dest = point_cloud[np.argmin(dist),:]
      init_dest[i,:] = dest 

  # Downsample points
  if sum(dsmp_resolution) > 3:
    debug(">>>>> Downsample...")
    point_cloud = downsample_points(point_cloud, dsmp_resolution)
    init_root = downsample_points(init_root, dsmp_resolution)

    if init_dest.shape[0] != 0:
      init_dest = downsample_points(init_dest, dsmp_resolution)

  # Convert coordinates to bounding box
  min_bound = np.min(point_cloud, axis=0)
  point_cloud = point_cloud - min_bound + 1
  init_root = init_root - min_bound + 1

  if init_dest.shape[0] != 0:
    init_dest = init_dest - min_bound + 1

  # Skeletonize chunk surrounding object
  debug(">>>>> Building skeleton...")
  skeleton = TEASAR(point_cloud, parameters, init_root, init_dest, soma)
  
  # Convert coordinates back into original coordinates
  if skeleton.nodes.shape[0] != 0:
    skeleton.nodes = upsample_points(skeleton.nodes + min_bound - 1, dsmp_resolution)

  return skeleton

def downsample_points(points, dsmp_resolution):
  """
  points: n x 3 point coordinates
  dsmp_resolution: [x, y, z] downsample resolution

  Returns: n x 3 downsampled point coordinates
  """
  if len(points.shape) == 1:
    points = np.reshape(points, (1, 3))

  dsmp_resolution = np.array(dsmp_resolution, dtype='float')
  point_downsample = points / dsmp_resolution
  point_downsample = np.round(point_downsample)
  point_downsample = np.unique(point_downsample, axis=0)
  return point_downsample.astype('uint16')

def upsample_points(points, dsmp_resolution):
  """
  points: n x 3 point coordinates
  dsmp_resolution: [x, y, z] downsample resolution

  Returns: n x 3 upsampled point coordinates
  """
  dsmp_resolution = np.array(dsmp_resolution)
  point_upsample = points * dsmp_resolution
  return point_upsample.astype('uint16')
 
def find_path(predecessor, end, start=[]):
  """
  Required:
    predecessor: n x n array of predecessors of shortest path from i to j
    end: destination node
  
  Optional:
    start: start node (Not necessary if the predecessor array is 1D array)
  
  Returns: n x 1 array consisting nodes in path
  """
  path_list = [end]
  pred = end

  while True:
    if len(predecessor.shape) > 1:
      pred = predecessor[start,pred]
    else:
      pred = predecessor[pred]

    if pred == -9999:
      break
    else:
      path_list.append(pred)

  path_list.reverse()
  return np.array(path_list)

def thr_linear(x, linear_parameters, threshold):
  """
  x: function input
  parameters: [slope, constant]
  threshold: threshold of cutoff
    
  Returns: interpolation 
  """
  slope, const = linear_parameters
  return min(x * slope + const, threshold)

def reorder_nodes(nodes, edges):
  """
  nodes: list of node numbers
  edges: list of edges

  Returns: edges with reordered node numbers
  """
  edges_reorder = np.zeros(edges.shape)
  for i in range(edges.shape[0]):
    edges_reorder[i,0] = np.where(nodes == edges[i,0])[0]
    edges_reorder[i,1] = np.where(nodes == edges[i,1])[0]

  return edges_reorder

def array2point(array, object_id=None):
  """
  array: array with labels
  object_id: object label to extract point cloud

  Return: n x 3 point coordinates 
  """

  if object_id is None:
    object_coord = np.where(array > 0)
  else:
    object_coord = np.where(array == object_id)

  object_x = object_coord[0]
  object_y = object_coord[1]
  object_z = object_coord[2]

  points = np.zeros([ len(object_x), 3 ], dtype='uint32')
  points[:,0] = object_x
  points[:,1] = object_y
  points[:,2] = object_z

  return points

def create_TEASAR_edges(object_points, DBF, max_bound, soma):
  n = object_points.shape[0]  
  NDIM = object_points.shape[1]

  object_nodes = Nodes(object_points, max_bound)

  # Penalty weight for the edges

  # Paper calls for:
  # M > np.max(DBF)
  # p_v = 5000 * ((1 - DBF / M) ** 16)
  # values are calibrated to precision of float32

  M = np.max(DBF) ** 1.01
  p_v = 100000 * ((1 - DBF / M) ** 16)
  p_v = p_v.astype(np.float32)

  # 26-connectivity
  nhood_26 = np.zeros([3,3,3], dtype='bool')
  nhood_26 = np.where(nhood_26 == 0)

  nhood = np.zeros([ nhood_26[0].size, 3 ], dtype=np.float16)
  for i in range(NDIM):
    nhood[:,i] = nhood_26[i]
  nhood = nhood - 1
  nhood = np.delete(nhood,find_row(nhood, [0,0,0]), axis=0)

  n_nhood = nhood.shape[0]  
  nhood_weight = np.sum(nhood ** 2, axis=1) ** 0.5

  nhood_points = np.zeros([n, 3], dtype=np.float16)
  nhood_nodes = np.ones([n, n_nhood], dtype=np.int32) * -1 
  edge_dist = np.zeros([n, n_nhood], dtype=np.float16)
  edge_weight = np.zeros([n, n_nhood], dtype=np.float32)

  objpts16 = object_points.astype(np.float16)

  debug("Setting edge weight...")
  for i in range(n_nhood):
    nhood_points = objpts16 + nhood[i,:]
    valid = np.all(nhood_points >= 0, axis=1) * np.all(nhood_points < max_bound, axis=1)

    # turn valid points into node ids
    nhood_nodes[valid,i] = object_nodes.sub2node(nhood_points[valid,:])

    valid = nhood_nodes[:,i] != -1
    edge_dist[valid,i] = nhood_weight[i]

    valid_idx = np.where(valid)[0]

    if soma:
      edge_weight[valid,i] = nhood_weight[i] * p_v[object_points[valid_idx,0], object_points[valid_idx,1], object_points[valid_idx,2]]
    else:
      edge_weight[valid,i] = p_v[object_points[valid_idx,0], object_points[valid_idx,1], object_points[valid_idx,2]]

  return (nhood_nodes, edge_dist, edge_weight)

def create_TEASAR_graph(object_points, DBF, max_bound, soma):
  """object_points = point cloud, DBF = distance based transform"""
  nhood_nodes, edge_dist, edge_weight = create_TEASAR_edges(object_points, DBF, max_bound, soma)

  if np.max(edge_weight) < np.finfo(np.float16).max:
    edge_weight = edge_weight.astype(np.float16)

  valid_edge = np.where((nhood_nodes != -1))
  valid_edge = ( valid_edge[0].astype(np.int32), valid_edge[1].astype(np.int32) )

  n = object_points.shape[0]
  rowcol = (valid_edge[0], nhood_nodes[valid_edge[0], valid_edge[1]])

  debug("Creating graph...")
  G_dist = csr_matrix(
    (edge_dist[valid_edge[0], valid_edge[1]], rowcol), 
      shape=(n,n),
      dtype='float16',
    )

  G = csr_matrix(
    (edge_weight[valid_edge[0], valid_edge[1]], rowcol), 
      shape=(n,n))

  return G_dist, G

def TEASAR(
    object_points, parameters, init_root=np.array([]), 
    init_dest=np.array([]), soma=False
  ):
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
  NDIM = object_points.shape[1]
  object_points = object_points.astype('uint32')

  max_bound = np.max(object_points, axis=0) + 2

  bin_im = np.zeros(max_bound, dtype='bool')
  bin_im[object_points[:,0], object_points[:,1], object_points[:,2]] = True

  n = object_points.shape[0]
  debug('Number of points ::::: ' + str(n))

  # Distance to the boundary map
  debug("Creating DBF...")

  # Might be possible to implement faster version of DBF function
  # than in numpy. Can include anisotropy.
  DBF = ndimage.distance_transform_edt(bin_im).astype(np.float32)

  G_dist, G = create_TEASAR_graph(object_points, DBF, max_bound, soma=soma)

  object_nodes = Nodes(object_points, max_bound)
  root_nodes = object_nodes.sub2node(init_root)
  n_root = root_nodes.shape[0]
  
  is_disconnected = np.ones(n, dtype='bool')
  
  r = 0
  c = 0 
  nodes = np.array([])
  edges = np.array([])

  # When destination nodes are not set.
  if init_dest.shape[0] == 0:
    while np.any(is_disconnected):
      debug("Processing connected component...")
      # Calculate distance map from the root node
      if r <= n_root - 1: 
        root = root_nodes[r]
        
        D_Gdist = dijkstra(G_dist, directed=True, indices=root)

        cnt_comp = ~np.isinf(D_Gdist)
        is_disconnected = is_disconnected * ~cnt_comp

        cnt_comp = np.where(cnt_comp)[0]
        cnt_comp_im = np.zeros(max_bound, dtype='bool')
        cnt_comp_im[object_points[cnt_comp,0], object_points[cnt_comp,1], object_points[cnt_comp,2]] = 1

      # Set separate root node for broken pieces
      else:
        root = np.where(is_disconnected == 1)[0][0]

        D_Gdist = dijkstra(G_dist, directed=True, indices=root)

        cnt_comp = ~np.isinf(D_Gdist)
        is_disconnected = is_disconnected * ~cnt_comp

        cnt_comp = np.where(cnt_comp)[0]
        cnt_comp_im = np.zeros(max_bound, dtype='bool')
        cnt_comp_im[object_points[cnt_comp,0], object_points[cnt_comp,1], object_points[cnt_comp,2]] = 1

      # Graph shortest path in the weighted graph
      D_G, pred_G = dijkstra(G, directed=True, indices=root, return_predecessors=True)

      # Build skeleton and remove pieces that are completed.
      # Iterate until entire connected component is completed.

      if cnt_comp.shape[0] < 5000:
        r = r + 1 
        continue

      path_list = [];
      while np.any(cnt_comp):
        debug("Finding path...")
        dest_node = cnt_comp[np.where(D_Gdist[cnt_comp] == np.max(D_Gdist[cnt_comp]))[0][0]]

        path = find_path(pred_G, dest_node)
        path_list.append(path)

        for i in range(len(path)):
          path_node = path[i]
          path_point = object_points[path_node,:]

          d = thr_linear(DBF[path_point[0], path_point[1], path_point[2]], parameters, 500)
          
          cube_min = np.zeros(3, dtype=np.uint32)
          cube_min = path_point - d
          cube_min[cube_min < 0] = 0
          cube_min = cube_min.astype(np.uint32)
          
          cube_max = np.zeros(3, dtype=np.uint32)
          cube_max = path_point + d
          cube_max[cube_max > max_bound] = max_bound[cube_max > max_bound]
          cube_max = cube_max.astype(np.uint32)

          cnt_comp_im[cube_min[0]:cube_max[0], cube_min[1]:cube_max[1], cube_min[2]:cube_max[2]] = 0

        cnt_comp_sub = array2point(cnt_comp_im)
        cnt_comp = object_nodes.sub2node(cnt_comp_sub)

      for i in range(len(path_list)):
        path = path_list[i]

        if c + i == 0:
          nodes = path
          edges = path2edge(path)

        else:
          nodes = np.concatenate((nodes, path))
          edges_path = path2edge(path)
          edges = np.concatenate((edges, edges_path))

      r = r + 1 
      c = c + 1

  # When destination nodes are set. 
  else:
    dest_nodes = object_nodes.sub2node(init_dest)

    path_list = []
    for r in range(root_nodes.shape[0]):
      root = root_nodes[r]

      D_G, pred_G = dijkstra(G, directed=True, indices=root, return_predecessors=True)

      for i in range(dest_nodes.shape[0]):
        dest = dest_nodes[i]

        if np.isfinite(D_G[dest]):
          path = find_path(pred_G, dest)
          path_list.append(path)


    for i in range(len(path_list)):
      path = path_list[i]

      # if soma:
      #   path = np.delete(path,np.arange(1,int(path.shape[0]*0.4)))

      if i == 0:
        nodes = path
        edges = path2edge(path)
      else:
        nodes = np.concatenate((nodes,path))
        edges_path = path2edge(path)
        edges = np.concatenate((edges,edges_path))
          
  if nodes.shape[0] == 0 or edges.shape[0] == 0:
    return Skeleton()
  
  # Consolidate nodes and edges
  nodes = np.unique(nodes)
  edges = np.unique(edges, axis=0)

  skel_nodes = object_points[nodes,:]
  skel_edges = reorder_nodes(nodes,edges)
  skel_edges = skel_edges.astype('uint32')
  skel_radii = DBF[skel_nodes[:,0], skel_nodes[:,1], skel_nodes[:,2]]

  skeleton = Skeleton(skel_nodes, skel_edges, skel_radii)
  return consolidate_skeleton(skeleton)

