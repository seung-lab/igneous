import numpy as np

class Skeleton:
  def __init__(self, 
    nodes=np.array([]), edges=np.array([]), 
    radii=np.array([]), root=np.array([])
  ):

    self.nodes = nodes
    self.edges = edges
    self.radii = radii
    self.root = root

  def empty(self):
    return self.nodes.size == 0 or self.edges.size == 0

class Nodes:
  def __init__(self, coord, max_bound):
    n = coord.shape[0]
    coord = coord.astype('uint32')
    self.max_bound = max_bound.astype('uint32')

    idx = np.zeros(n, dtype='uint64')
    idx = coord[:,0] + max_bound[0] * coord[:,1] + max_bound[0] * max_bound[1] * coord[:,2]
    self.idx = idx

    idx2node = np.ones(np.prod(max_bound))*-1
    idx2node[idx] = np.arange(coord.shape[0], dtype='int64')
    self.node = idx2node

  def sub2idx(self, sub_array):
    if len(sub_array.shape) == 1:
      sub_array = np.reshape(sub_array,(1,3))

    sub_array = sub_array.astype('uint32')

    max_bound = self.max_bound
    idx = np.zeros(sub_array.shape[0])
    idx = sub_array[:,0] + max_bound[0]*sub_array[:,1] + max_bound[0]*max_bound[1]*sub_array[:,2]
    return idx

  def sub2node(self, sub_array):
    if len(sub_array.shape) == 1:
      sub_array = np.reshape(sub_array,(1,3))

    sub_array = sub_array.astype('uint32')
    max_bound = self.max_bound
    idx_array = sub_array[:,0] + max_bound[0]*sub_array[:,1] + max_bound[0]*max_bound[1]*sub_array[:,2]

    node = self.node[idx_array]
    return node.astype('int64')