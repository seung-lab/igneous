""" 
Graphene is based on an octree data structure. We mesh at
the lowest agglomerated level of the octree (L2) and then 
create stitched meshes at each higher level of the tree 
recursively starting from those L2 meshes. This scheme allows
use to rapidly reconfigure small regions that change while 
proofreading.

The goal of meshing is to render each L2 label into a distinct
mesh. However, in order for the stitching to work, as in a flat 
segmentation, we need to extend one voxel into the neighboring 
chunk. 

Each L2 chunk represents a self contained proofread agglomeration
of the L1 watershed regions over an area of typically 256x256x512
voxels. Actual proofreading extends across L2 chunks, but this is
represented by higher levels of the octree. Therefore, adjacent
L2 chunks have completely different labels.

In order to create accurate merging between chunks, we'll need to
relabel the overlap region using the lowest parent chunk ids between
the primary chunk and each individual overlap region (therefore each
overlap region may require a different chunk graph level). The reason
we can't just label them all with the root ID and mesh is that levels
higher than the lowest level may introduce non-local information. For
instance, a self touch on the boundary that would have been kept apart
at L3 would be unified at the root. The chunk position determines what 
the lowest root is. For example, a chunk located at the origin will 
require an L3 overlap merge. A chunk located one unit from the origin
will require an L4 merge. A chunk located three units from the origin 
will require an L5 merge, and so on according to the fan-out of the octree.

Once the chunk is relabeled using the correct parent, some odd cases 
crop up.  For example, due to information in the overlap region, L2 
labels that were formerly independent connected components may be 
merged via a chain of one or more L3+ relabeled overlap labels. Therefore,
we apply the following rules:

1. If the L2 label does not touch the overlap region, use that label.
2. Each overlap supervoxel is merged with neighboring labels that share 
  a parent. Each resulting connected component is remapped using the minimum 
  covered L2 label from the primary chunk. If no such label exists it 
  is mapped to 0.

"""
from typing import Sequence
import numpy as np
import time
import collections
from functools import lru_cache
import datetime
import pytz

import networkx as nx

import cc3d

from tqdm import tqdm

import cloudvolume
from cloudvolume.lib import Bbox, Vec, xyzrange
from cloudvolume import CloudVolume
import fastremap

UTC = pytz.UTC

def remap_segmentation(
  cv, chunk_x, chunk_y, chunk_z, mip=2, 
  overlap_vx=1, time_stamp=None, progress=False
):
  ws_cv = CloudVolume(
    cv.meta.cloudpath, mip=mip, 
    progress=progress, fill_missing=cv.fill_missing
  )
  mip_diff = mip - cv.meta.watershed_mip

  mip_chunk_size = np.array(
    cv.meta.graph_chunk_size, dtype=np.int
  ) / np.array([2 ** mip_diff, 2 ** mip_diff, 1])
  mip_chunk_size = mip_chunk_size.astype(np.int)

  offset = Vec(chunk_x, chunk_y, chunk_z) * mip_chunk_size
  bbx = Bbox(offset, offset + mip_chunk_size + overlap_vx)
  if cv.meta.chunks_start_at_voxel_offset:
    bbx += ws_cv.voxel_offset
  bbx = Bbox.clamp(bbx, ws_cv.bounds)

  seg = ws_cv[bbx][..., 0]

  if not np.any(seg):
    return seg

  sv_remapping, unsafe_dict = get_lx_overlapping_remappings(
    cv, chunk_x, chunk_y, chunk_z, 
    time_stamp=time_stamp, progress=progress
  )

  seg = fastremap.mask_except(seg, list(sv_remapping.keys()), in_place=True)
  fastremap.remap(seg, sv_remapping, preserve_missing_labels=True, in_place=True)

  for unsafe_root_id in tqdm(unsafe_dict.keys(), desc="Unsafe Relabel", disable=(not progress)):
    bin_seg = seg == unsafe_root_id

    if np.sum(bin_seg) == 0:
        continue

    cc_seg = cc3d.connected_components(bin_seg)
    for i_cc in range(1, np.max(cc_seg) + 1):
      bin_cc_seg = cc_seg == i_cc

      overlaps = []
      overlaps.extend(np.unique(seg[-2, :, :][bin_cc_seg[-1, :, :]]))
      overlaps.extend(np.unique(seg[:, -2, :][bin_cc_seg[:, -1, :]]))
      overlaps.extend(np.unique(seg[:, :, -2][bin_cc_seg[:, :, -1]]))
      overlaps = np.unique(overlaps)

      linked_l2_ids = overlaps[np.in1d(overlaps, unsafe_dict[unsafe_root_id])]

      if len(linked_l2_ids) == 0:
        seg[bin_cc_seg] = 0
      else:
        seg[bin_cc_seg] = linked_l2_ids[0]

  return seg

@lru_cache(maxsize=200)
def get_higher_to_lower_remapping(cv, chunk_id, time_stamp):
  """ 
  Retrieves lx node id to sv id mappping

  :param cv: CloudVolumeGraphene object
  :param chunk_id: np.uint64
  :param time_stamp: datetime object
  :return: dictionary
  """
  rr_chunk = cv.get_chunk_mappings(chunk_id=chunk_id, timestamp=time_stamp)

  # This for-loop ensures that only the latest lx_ids are considered
  # The order by id guarantees the time order (only true for same neurons
  # but that is the case here).
  lx_remapping = {}
  all_lower_ids = set()
  for k in sorted(rr_chunk.keys(), reverse=True):
    this_child_ids = rr_chunk[k]

    if this_child_ids[0] in all_lower_ids:
      continue

    all_lower_ids.update(list(this_child_ids))
    lx_remapping[k] = this_child_ids

  return lx_remapping


@lru_cache(maxsize=None)
def get_root_lx_remapping(cv, chunk_id, stop_layer, time_stamp):
  """
  Retrieves root to l2 node id mapping

  :param cv: CloudVolumeGraphene object
  :param chunk_id: np.uint64
  :param stop_layer: int
  :param time_stamp: datetime object
  :return: multiples
  """
  lx_id_remap = get_higher_to_lower_remapping(
    cv, chunk_id, time_stamp=time_stamp
  )

  lx_ids = np.array(list(lx_id_remap.keys()))
  root_ids = cv.get_roots(lx_ids, stop_layer=stop_layer)

  return lx_ids, np.array(root_ids), lx_id_remap


# @lru_cache(maxsize=None)
def get_lx_overlapping_remappings(
  cv, chunk_x, chunk_y, chunk_z, 
  time_stamp=None, progress=False
):
  """ 
  Retrieves sv id to layer mapping for chunk with overlap in positive
    direction (one chunk)

  :param cv: CloudVolumeGraphene object
  :param chunk_x: np.uint64
  :param chunk_y: np.uint64
  :param chunk_z: np.uint64
  :param time_stamp: datetime object
  :return: multiples
  """
  if time_stamp is None:
    time_stamp = datetime.datetime.utcnow()

  if isinstance(time_stamp, datetime.datetime) and time_stamp.tzinfo is None:
    time_stamp = UTC.localize(time_stamp)

  chunk_layer = 2
  chunk_id = cv.meta.encode_label(chunk_layer, chunk_x, chunk_y, chunk_z, 0)

  neigh_chunk_ids = []
  neigh_parent_chunk_ids = []

  # Collect neighboring chunks and their parent chunk ids
  # We only need to know about the parent chunk ids to figure the lowest
  # common chunk
  # Notice that the first neigh_chunk_id is equal to `chunk_id`.
  for x in range(chunk_x, chunk_x + 2):
    for y in range(chunk_y, chunk_y + 2):
      for z in range(chunk_z, chunk_z + 2):
        
        neigh_chunk_id = cv.meta.encode_label(layer=2, x=x, y=y, z=z, segid=0)
        neigh_chunk_ids.append(neigh_chunk_id)
        neigh_parent_chunk_ids.append(
          get_parent_chunk_ids(cv, neigh_chunk_id)
        )

  # Find lowest common chunk
  # finding the layer that has the lowest common parent of
  # these 8 neighboring chunks
  # Layer agreement will equal 0 if that is layer 3, 1 if
  # that is layer 4, etc
  neigh_parent_chunk_ids = np.array(neigh_parent_chunk_ids)
  layer_agreement = np.all(
    (neigh_parent_chunk_ids - neigh_parent_chunk_ids[0]) == 0, axis=0
  )
  stop_layer = np.where(layer_agreement)[0][0] + chunk_layer

  # Find the parent in the lowest common chunk for each l2 id. These parent
  # ids are referred to as root ids even though they are not necessarily the
  # root id.
  neigh_lx_ids = []
  neigh_lx_id_remap = {}
  neigh_root_ids = []

  safe_lx_ids = []
  unsafe_lx_ids = []
  unsafe_root_ids = []

  # This loop is the main bottleneck
  for neigh_chunk_id in tqdm(neigh_chunk_ids, disable=(not progress), desc="Neighbor"):
    # print(f"Neigh: {neigh_chunk_id} --------------")

    lx_ids, root_ids, lx_id_remap = get_root_lx_remapping(
      cv, neigh_chunk_id, stop_layer, time_stamp=time_stamp
    )
    neigh_lx_ids.extend(lx_ids)
    neigh_lx_id_remap.update(lx_id_remap)
    neigh_root_ids.extend(root_ids)

    if neigh_chunk_id == chunk_id:
      # The first neigh_chunk_id is the one we are interested in. All lx
      # ids that share no root id with any other lx id are "safe", meaning
      # that we can easily obtain the complete remapping (including
      # overlap) for these. All other ones have to be resolved using the
      # segmentation.
      _, u_idx, c_root_ids = np.unique(
        neigh_root_ids, return_counts=True, return_index=True
      )

      safe_lx_ids = lx_ids[u_idx[c_root_ids == 1]]
      unsafe_lx_ids = lx_ids[~np.in1d(lx_ids, safe_lx_ids)]
      unsafe_root_ids = np.unique(root_ids[u_idx[c_root_ids != 1]])

  lx_root_dict = dict(zip(neigh_lx_ids, neigh_root_ids))
  root_lx_dict = collections.defaultdict(list)

  # Future sv id -> lx mapping
  sv_ids = []
  lx_ids_flat = []

  # Do safe ones first
  for i_root_id in range(len(neigh_root_ids)):
    root_lx_dict[neigh_root_ids[i_root_id]].append(neigh_lx_ids[i_root_id])

  for lx_id in safe_lx_ids:
    root_id = lx_root_dict[lx_id]
    for neigh_lx_id in root_lx_dict[root_id]:
      lx_sv_ids = neigh_lx_id_remap[neigh_lx_id]
      sv_ids.extend(lx_sv_ids)
      lx_ids_flat.extend([lx_id] * len(neigh_lx_id_remap[neigh_lx_id]))

  # For the unsafe ones we can only do the in chunk svs
  # But we will map the out of chunk svs to the root id and store the
  # hierarchical information in a dictionary
  for lx_id in unsafe_lx_ids:
    sv_ids.extend(neigh_lx_id_remap[lx_id])
    lx_ids_flat.extend([lx_id] * len(neigh_lx_id_remap[lx_id]))

  unsafe_dict = collections.defaultdict(list)
  for root_id in unsafe_root_ids:
    if np.sum(~np.in1d(root_lx_dict[root_id], unsafe_lx_ids)) == 0:
      continue

    for neigh_lx_id in root_lx_dict[root_id]:
      unsafe_dict[root_id].append(neigh_lx_id)

      if neigh_lx_id in unsafe_lx_ids:
        continue

      sv_ids.extend(neigh_lx_id_remap[neigh_lx_id])
      lx_ids_flat.extend([root_id] * len(neigh_lx_id_remap[neigh_lx_id]))

  # Combine the lists for a (chunk-) global remapping
  sv_remapping = dict(zip(sv_ids, lx_ids_flat))

  return sv_remapping, unsafe_dict

def get_parent_chunk_ids(cv, label):
  """ 
  Creates list of chunk parent ids

  :param label: np.uint64
  :return: np.ndarray
  """
  parent_chunk_layers = range(
    int(cv.meta.decode_layer_id(label)) + 1, int(cv.meta.n_layers) + 1
  )

  chunk_coord = Vec(*cv.meta.decode_chunk_position(label))

  parent_chunk_ids = [ cv.meta.decode_chunk_id(label) ]
  for layer in parent_chunk_layers:
    chunk_coord = chunk_coord // cv.meta.fan_out
    chunk_id = cv.meta.encode_label(
      layer, chunk_coord[0], chunk_coord[1], chunk_coord[2], segid=0
    )
    parent_chunk_ids.append(chunk_id)

  return np.array(parent_chunk_ids, dtype=np.uint64)
