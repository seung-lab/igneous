import numpy as np

class Skeleton:
  def __init__(self, 
    nodes=np.array([]), edges=np.array([]), 
    radii=np.array([]), root=np.array([])
  ):

    if nodes.size == 0:
      nodes = nodes.reshape( (0, 3) )

    if edges.size == 0:
      edges = edges.reshape( (0, 2) )

    self.nodes = nodes
    self.edges = edges
    self.radii = radii
    self.root = root

  def empty(self):
    return self.nodes.size == 0 or self.edges.size == 0

class Nodes:
  def __init__(self, coord, max_bound):
    n = coord.shape[0]

    coord = coord.astype(np.int32)
    self.max_bound = max_bound.astype(np.int32)

    idx = coord[:,0] + max_bound[0] * (coord[:,1] + max_bound[1] * coord[:,2])

    idx2node = np.ones(np.prod(max_bound), dtype=np.int32) * -1
    idx2node[idx] = np.arange(coord.shape[0], dtype=np.int32)
    self.node = idx2node

  def sub2idx(self, sub_array):
    if len(sub_array.shape) == 1:
      sub_array = np.reshape(sub_array,(1,3))

    sub_array = sub_array.astype('uint32')
    max_bound = self.max_bound
    return sub_array[:,0] + max_bound[0] * (sub_array[:,1] + max_bound[1] * sub_array[:,2])

  def sub2node(self, sub_array):
    idx_array = self.sub2idx(sub_array)
    return self.node[idx_array].astype('int32')

def path2edge(path):
  """
  path: sequence of nodes

  Returns: sequence separated into edges
  """
  edges = np.zeros([len(path)-1,2], dtype='uint32')
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