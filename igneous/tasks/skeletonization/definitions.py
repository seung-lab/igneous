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