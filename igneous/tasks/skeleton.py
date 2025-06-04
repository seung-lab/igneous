from typing import Optional, Sequence, Dict, List

from functools import reduce
import itertools
import json
import mmap
import pickle
import posixpath
import os
import re
from collections import defaultdict

from tqdm import tqdm

import numpy as np

import mapbuffer
from mapbuffer import MapBuffer, IntMap
import cloudfiles
from cloudfiles import CloudFiles, CloudFile

import cloudvolume
from cloudvolume import CloudVolume, Skeleton, paths
from cloudvolume.lib import Vec, Bbox, sip, xyzrange
from cloudvolume.datasource.precomputed.sharding import synthesize_shard_files

import cc3d
import crackle
import fastmorph
import fastremap
import kimimaro

from taskqueue import RegisteredTask, queueable

SEGIDRE = re.compile(r'/(\d+):.*?$')

def filename_to_segid(filename):
  matches = SEGIDRE.search(filename)
  if matches is None:
    raise ValueError("There was an issue with the fragment filename: " + filename)

  segid, = matches.groups()
  return int(segid)

def strip_integer_attributes(skeletons):
  for skel in skeletons:
    skel.extra_attributes = [ 
    attr for attr in skel.extra_attributes 
    if attr['data_type'] in ('float32', 'float64')
  ]
  return skeletons

class SkeletonTask(RegisteredTask):
  """
  Stage 1 of skeletonization.

  Convert chunks of segmentation into chunked skeletons and point clouds.
  They will be merged in the stage 2 task SkeletonMergeTask.
  """
  def __init__(
    self, cloudpath:str, 
    shape:Sequence[int], offset:Sequence[int], 
    mip:int, teasar_params:dict, will_postprocess:bool,
    info:dict = None, 
    object_ids:Optional[Sequence[int]] = None,
    mask_ids:Optional[Sequence[int]] = None,
    fix_branching:bool = True,
    fix_borders:bool = True,
    fix_avocados:bool = False,
    fill_holes:int = 0,
    dust_threshold:int = 1000, 
    progress:bool = False,
    parallel:int = 1,
    fill_missing:bool = False,
    sharded:bool = False,
    frag_path:Optional[str] = None, 
    spatial_index:bool = True,
    spatial_grid_shape:Optional[Sequence[int]] = None,
    synapses:Optional[Sequence[Sequence[float]]] = None, 
    dust_global:bool = False,
    cross_sectional_area:bool = False,
    cross_sectional_area_smoothing_window:int = 1,
    cross_sectional_area_shape_delta:int = 150,
    dry_run:bool = False,
    strip_integer_attributes:bool = True,
    fix_autapses:bool = False,
    timestamp:Optional[int] = None,
    root_ids_cloudpath:Optional[str] = None,
  ):
    super().__init__(
      cloudpath, shape, offset, mip, 
      teasar_params, will_postprocess, 
      info, object_ids, mask_ids,
      fix_branching, fix_borders, 
      fix_avocados, fill_holes,
      dust_threshold, progress, parallel,
      fill_missing, bool(sharded), frag_path, bool(spatial_index),
      spatial_grid_shape, synapses, bool(dust_global),
      bool(cross_sectional_area), int(cross_sectional_area_smoothing_window),
      int(cross_sectional_area_shape_delta),
      bool(dry_run), bool(strip_integer_attributes),
      bool(fix_autapses), timestamp,
      root_ids_cloudpath,
    )
    if isinstance(self.frag_path, str):
      self.frag_path = cloudfiles.paths.normalize(self.frag_path)
    self.bounds = Bbox(offset, Vec(*shape) + Vec(*offset))
    self.index_bounds = Bbox(offset, Vec(*spatial_grid_shape) + Vec(*offset))

    # aggressive morphological hole filling has a 1-2vx 
    # edge effect that needs to be cropped away
    self.hole_filling_padding = (self.fill_holes >= 3) * 2

  def execute(self):
    # For graphene volumes, if we've materialized the root IDs
    # into a static archive, let's use that because it's way more
    # efficient for fetching root IDs.
    cloudpath = self.cloudpath
    if self.root_ids_cloudpath:
      cloudpath = self.root_ids_cloudpath

    lru_bytes = 0
    lru_encoding = 'same'

    if self.cross_sectional_area:
      lru_bytes = self.bounds.size() + 2 * self.cross_sectional_area_shape_delta
      lru_bytes = lru_bytes[0] * lru_bytes[1] * lru_bytes[2] * 8 // 500
      lru_encoding = 'crackle'

    vol = CloudVolume(
      cloudpath,
      mip=self.mip,
      bounded=(self.hole_filling_padding == 0),
      info=self.info,
      cdn_cache=False,
      parallel=self.parallel,
      fill_missing=self.fill_missing,
      lru_bytes=lru_bytes,
      lru_encoding=lru_encoding,
    )
    bbox = Bbox.clamp(self.bounds, vol.bounds)
    index_bbox = Bbox.clamp(self.index_bounds, vol.bounds)

    bbox.minpt -= self.hole_filling_padding
    bbox.maxpt += self.hole_filling_padding

    path = vol.info.get("skeletons", "skeletons")
    if self.frag_path is None:
      path = vol.meta.join(self.cloudpath, path)
    else:
      # if the path is to a volume root, follow the info instructions,
      # otherwise place the files exactly where frag path says to
      test_path = CloudFiles(self.frag_path).join(self.frag_path, "info")
      test_info = CloudFile(test_path).get_json()
      if test_info is not None and 'scales' in test_info:
        path = CloudFiles(self.frag_path).join(self.frag_path, path)
      else:
        path = self.frag_path

    all_labels = vol.download(
      bbox.to_slices(), 
      agglomerate=True, 
      timestamp=self.timestamp
    )
    all_labels = all_labels[:,:,:,0]

    if self.mask_ids:
      all_labels = fastremap.mask(all_labels, self.mask_ids)

    extra_targets_after = {}
    if self.synapses:
      extra_targets_after = kimimaro.synapses_to_targets(
        all_labels, self.synapses
      )

    dust_threshold = self.dust_threshold
    if self.dust_global and dust_threshold > 0:
      dust_threshold = 0
      all_labels = self.apply_global_dust_threshold(vol, all_labels)

    if self.fill_holes and self.fix_autapses:
      raise ValueError("fill_holes is not currently compatible with fix_autapses")

    voxel_graph = None
    if self.fix_autapses:
      voxel_graph = self.voxel_connectivity_graph(vol, bbox, all_labels)

    skeletons = self.skeletonize(
      all_labels, 
      vol, 
      dust_threshold, 
      extra_targets_after, 
      voxel_graph,
    )
    del all_labels

    if self.cross_sectional_area: # This is expensive!
      skeletons = self.compute_cross_sectional_area(vol, bbox, skeletons)

    # voxel centered (+0.5) and uses more accurate bounding box from mip 0
    corrected_offset = (bbox.minpt.astype(np.float32) - vol.meta.voxel_offset(self.mip) + 0.5) * vol.meta.resolution(self.mip)
    corrected_offset += vol.meta.voxel_offset(0) * vol.meta.resolution(0)

    for segid, skel in skeletons.items():
      skel.vertices[:] += corrected_offset

    if self.synapses:
      for segid, skel in skeletons.items():
        terminal_nodes = skel.vertices[ skel.terminals() ]

        for i, vert in enumerate(terminal_nodes):
          vert = vert / vol.resolution - self.bounds.minpt
          vert = tuple(np.round(vert).astype(int))
          if vert in extra_targets_after.keys():
            skel.vertex_types[i] = extra_targets_after[vert]
    
    # old versions of neuroglancer don't
    # support int attributes
    if self.strip_integer_attributes:
      strip_integer_attributes(skeletons.values())

    if self.dry_run:
      return skeletons

    if self.sharded:
      self.upload_batch(vol, path, index_bbox, skeletons)
    else:
      self.upload_individuals(vol, path, bbox, skeletons)

    if self.spatial_index:
      self.upload_spatial_index(vol, path, index_bbox, skeletons)

  def _do_operation(self, all_labels, fn):
    if self.fill_holes > 0:
      filled_labels, hole_labels = fastmorph.fill_holes(
        all_labels,
        remove_enclosed=True,
        return_removed=True,
        fix_borders=(self.fill_holes >= 2),
        morphological_closing=(self.fill_holes >= 3),
      )

      if self.fill_holes >= 3:
        hp = self.hole_filling_padding
        all_labels = np.asfortranarray(all_labels[hp:-hp,hp:-hp,hp:-hp])
        filled_labels= np.asfortranarray(filled_labels[hp:-hp,hp:-hp,hp:-hp])

      all_labels = crackle.compress(all_labels)
      skeletons = fn(filled_labels)
      del filled_labels

      all_labels = crackle.decompress(all_labels)
      hole_labels = all_labels * np.isin(all_labels, list(hole_labels))
      del all_labels

      hole_skeletons = fn(hole_labels)
      skeletons.update(hole_skeletons)
      del hole_labels
      del hole_skeletons
    else:
      skeletons = fn(all_labels)

    return skeletons

  def skeletonize(
    self, 
    all_labels:np.ndarray, 
    vol:CloudVolume, 
    dust_threshold:int, 
    extra_targets_after:dict, 
    voxel_graph:np.ndarray,
  ) -> dict:
    def do_skeletonize(labels):
      return kimimaro.skeletonize(
        labels, self.teasar_params, 
        object_ids=self.object_ids, 
        anisotropy=vol.resolution,
        dust_threshold=dust_threshold, 
        progress=self.progress, 
        fix_branching=self.fix_branching,
        fix_borders=self.fix_borders,
        fix_avocados=self.fix_avocados,
        fill_holes=False, # moved this logic into SkeletonTask / fastmorph
        parallel=self.parallel,
        extra_targets_after=extra_targets_after.keys(),
        voxel_graph=voxel_graph,
      )

    return self._do_operation(all_labels, do_skeletonize)

  def voxel_connectivity_graph(
    self, 
    vol:CloudVolume, 
    bbox:Bbox, 
    root_labels:np.ndarray,
  ) -> np.ndarray:

    if vol.meta.path.format != "graphene":
      vol = CloudVolume(
        self.cloudpath, mip=self.mip, 
        info=self.info, cdn_cache=False,
        parallel=self.parallel, 
        fill_missing=self.fill_missing,
      )

    if vol.meta.path.format != "graphene":
      raise ValueError("Can't extract a voxel connectivity graph from non-graphene volumes.")

    layer_2 = vol.download(
      bbox, 
      stop_layer=2,
      agglomerate=True,
      timestamp=self.timestamp,
    )[...,0]

    graph_chunk_size = np.array(vol.meta.graph_chunk_size) / vol.meta.downsample_ratio(vol.mip)
    graph_chunk_size = graph_chunk_size.astype(int)

    shape = bbox.size()[:3]
    sgx, sgy, sgz = list(np.ceil(shape / graph_chunk_size).astype(int))

    vcg = cc3d.voxel_connectivity_graph(layer_2, connectivity=26)
    del layer_2

    # the proper way to do this would be to get the lowest the L3..LN root
    # as needed, but the lazy way to do this is to get the root labels
    # which will retain a few errors, but overall the error rate should be
    # over 100x less. We need to shade in the sides of the connectivity graph
    # with edges that represent the connections between the adjacent boxes.

    root_vcg = cc3d.voxel_connectivity_graph(root_labels, connectivity=26)
    clamp_box = Bbox([0,0,0], shape)

    for gx,gy,gz in xyzrange([sgx, sgy, sgz]):
      bbx = Bbox((gx,gy,gz), (gx+1, gy+1, gz+1))
      bbx *= graph_chunk_size
      bbx = Bbox.clamp(bbx, clamp_box)

      slicearr = []
      for i in range(3):
        bbx1 = bbx.clone()
        bbx1.maxpt[i] = bbx1.minpt[i] + 1
        slicearr.append(bbx1)

        bbx1 = bbx.clone()
        bbx1.minpt[i] = bbx1.maxpt[i] - 1
        slicearr.append(bbx1)

      for bbx1 in slicearr:
        vcg[bbx1.to_slices()] = root_vcg[bbx1.to_slices()] 

    return vcg

  def compute_cross_sectional_area(self, vol, bbox, skeletons):
    if len(skeletons) == 0:
      return skeletons

    # Why redownload a bigger image? In order to avoid clipping the
    # cross sectional areas on the edges.
    delta = int(self.cross_sectional_area_shape_delta)

    big_bbox = bbox.clone()
    big_bbox.grow(delta)
    big_bbox = Bbox.clamp(big_bbox, vol.bounds)

    big_bbox.minpt -= self.hole_filling_padding
    big_bbox.maxpt += self.hole_filling_padding

    all_labels = vol[big_bbox][...,0]

    delta = bbox.minpt - big_bbox.minpt

    # place the skeletons in exactly the same position
    # in the enlarged image
    for skel in skeletons.values():
      skel.vertices += delta * vol.resolution

    if self.mask_ids:
      all_labels = fastremap.mask(all_labels, self.mask_ids)

    def do_cross_section(labels):
      return kimimaro.cross_sectional_area(
        labels, skeletons,
        anisotropy=vol.resolution,
        smoothing_window=self.cross_sectional_area_smoothing_window,
        progress=self.progress,
        in_place=True,
        fill_holes=False,
      )

    skeletons = self._do_operation(all_labels, do_cross_section)
    del all_labels

    # move the vertices back to their old smaller image location
    for skel in skeletons.values():
      skel.vertices -= delta * vol.resolution

    return self.repair_cross_sectional_area_contacts(vol, bbox, skeletons)

  def repair_cross_sectional_area_contacts(self, vol, bbox, skeletons):
    from dbscan import DBSCAN

    boundaries = [
      bbox.minpt.x == vol.bounds.minpt.x,
      bbox.maxpt.x == vol.bounds.maxpt.x,
      bbox.minpt.y == vol.bounds.minpt.y,
      bbox.maxpt.y == vol.bounds.maxpt.y,
      bbox.minpt.z == vol.bounds.minpt.z,
      bbox.maxpt.z == vol.bounds.maxpt.z,
    ]

    if all(boundaries):
      return skeletons

    invalid_repairs = 0
    for i, bnd in enumerate(boundaries):
      invalid_repairs |= (bnd << i)

    invalid_repairs = (~np.uint8(invalid_repairs)) & np.uint8(0b00111111)

    # We want to repair any skeleton that has a contact with the
    # edge except those that are contacting the volume boundary due to futility

    repair_skels = []
    for skel in skeletons.values():
      contacts = skel.cross_sectional_area_contacts & invalid_repairs
      if np.any(contacts):
        repair_skels.append(skel)

    delta = int(self.cross_sectional_area_shape_delta)

    shape = bbox.size3()
    
    def reprocess_skel(pts, skel):
      pts_bbx = Bbox.from_points(pts)

      pts_bbx_vol = pts_bbx + bbox.minpt
      center = pts_bbx_vol.center().astype(int)
      skel_bbx = Bbox(center, center+1)
      skel_bbx.grow(delta + shape // 2)

      skel_bbx = Bbox.clamp(skel_bbx, vol.bounds)

      skel_bbx.minpt -= self.hole_filling_padding
      skel_bbx.maxpt += self.hole_filling_padding

      binary_image = vol.download(
        skel_bbx, mip=vol.mip, label=skel.id
      )[...,0]

      diff = bbox.minpt - skel_bbx.minpt
      skel.vertices += diff * vol.resolution

      # we binarized the label for memory's sake, 
      # so need to harmonize that with the skeleton ID
      segid = skel.id
      skel.id = 1

      if self.fill_holes > 0:
        binary_image = fastmorph.fill_holes(
          binary_image,
          fix_borders=(self.fill_holes >= 2),
          morphological_closing=(self.fill_holes >= 3),
        )
        if self.fill_holes >= 3:
          hp = self.hole_filling_padding
          binary_image = np.asfortranarray(binary_image[hp:-hp,hp:-hp,hp:-hp])

      kimimaro.cross_sectional_area(
        binary_image, skel,
        anisotropy=vol.resolution,
        smoothing_window=self.cross_sectional_area_smoothing_window,
        progress=self.progress,
        in_place=True,
        fill_holes=False,
        repair_contacts=True,
      )
      skel.id = segid
      skel.vertices -= diff * vol.resolution

    for skel in repair_skels:
      verts = (skel.vertices // vol.resolution).astype(int)
      reprocess_skel(verts, skel)

      pts = verts[skel.cross_sectional_area_contacts > 0]
      if len(pts) == 0:
        continue

      labels, core_samples_mask = DBSCAN(pts, eps=5, min_samples=2)
      uniq = fastremap.unique(labels)
      for lbl in uniq:
        reprocess_skel(pts[labels == lbl], skel)

    return skeletons

  def apply_global_dust_threshold(self, vol, all_labels):
    path = vol.meta.join(self.cloudpath, vol.key, 'stats', 'voxel_counts.im')
    cf = CloudFile(path)
    memcf = CloudFile(path.replace(f"{cf.protocol}://", "mem://"))

    if not cf.exists():
      raise FileNotFoundError(f"Cannot apply global dust threshold without {path}")

    buf = None
    if memcf.exists():
      buf = memcf.get()
    else:
      cloudfiles.clear_memory()

    if buf is None:
      if cf.protocol != "file":
        buf = cf.get()
        memcf.put(buf, compress='zstd')
      else:
        buf = cf

    mb = IntMap(buf)
    uniq = fastremap.unique(all_labels)

    valid_objects = []
    for label in uniq:
      if label == 0:
        continue
      if mb[label] >= self.dust_threshold:
        valid_objects.append(label)

    return fastremap.mask_except(all_labels, valid_objects)

  def upload_batch(self, vol, path, bbox, skeletons):
    mbuf = MapBuffer(
      skeletons, compress="br", 
      tobytesfn=lambda skel: skel.to_precomputed()
    )

    cf = CloudFiles(path, progress=vol.progress)
    cf.put(
      path="{}.frags".format(bbox.to_filename()),
      content=mbuf.tobytes(),
      compress=None,
      content_type="application/x-mapbuffer",
      cache_control=False,
    )

  def upload_individuals(self, vol, path, bbox, skeletons):
    skeletons = skeletons.values()

    if not self.will_postprocess:
      vol.skeleton.upload(skeletons)
      return 

    bbox = bbox * vol.resolution
    cf = CloudFiles(path, progress=vol.progress)
    cf.puts(
      (
        (
          f"{skel.id}:{bbox.to_filename()}",
          pickle.dumps(skel)
        )
        for skel in skeletons
      ),
      compress='gzip',
      content_type="application/python-pickle",
      cache_control=False,
    )

  def upload_spatial_index(self, vol, path, bbox, skeletons):
    spatial_index = {}
    for segid, skel in tqdm(skeletons.items(), disable=(not vol.progress), desc="Extracting Bounding Boxes"):
      segid_bbx = Bbox.from_points( skel.vertices )
      spatial_index[segid] = segid_bbx.to_list()

    bbox = bbox.astype(vol.resolution.dtype) * vol.resolution
    precision = vol.skeleton.spatial_index.precision
    cf = CloudFiles(path, progress=vol.progress)
    cf.put_json(
      path=f"{bbox.to_filename(precision)}.spatial",
      content=spatial_index,
      compress='gzip',
      cache_control=False,
    )

class UnshardedSkeletonMergeTask(RegisteredTask):
  """
  Stage 2 of skeletonization.

  Merge chunked TEASAR skeletons into a single skeleton.

  If we parallelize using prefixes single digit prefixes ['0','1',..'9'] all meshes will
  be correctly processed. But if we do ['10','11',..'99'] meshes from [0,9] won't get
  processed and need to be handle specifically by creating tasks that will process
  a single mesh ['0:','1:',..'9:']
  """
  def __init__(
      self, cloudpath, prefix, 
      crop=0, dust_threshold=4000, max_cable_length=None,
      tick_threshold=6000, delete_fragments=False
    ):
    super(UnshardedSkeletonMergeTask, self).__init__(
      cloudpath, prefix, crop, 
      dust_threshold, max_cable_length,
      tick_threshold, delete_fragments
    )
    self.max_cable_length = float(max_cable_length) if max_cable_length is not None else None

  def execute(self):
    self.vol = CloudVolume(self.cloudpath, cdn_cache=False)
    self.vol.mip = self.vol.skeleton.meta.mip

    fragment_filenames = self.get_filenames()
    skels = self.get_skeletons_by_segid(fragment_filenames)

    skeletons = []
    for segid, frags in skels.items():
      skeleton = self.fuse_skeletons(frags)
      if self.max_cable_length is None or skeleton.cable_length() <= self.max_cable_length:
        skeleton = kimimaro.postprocess(
          skeleton, self.dust_threshold, self.tick_threshold
        )
      skeleton.id = segid
      skeletons.append(skeleton)

    self.vol.skeleton.upload(skeletons)
    
    if self.delete_fragments:
      cf = CloudFiles(self.cloudpath, progress=True)
      cf.delete(fragment_filenames)

  def get_filenames(self):
    prefix = '{}/{}'.format(self.vol.skeleton.path, self.prefix)

    cf = CloudFiles(self.cloudpath, progress=True)
    return [ _ for _ in cf.list(prefix=prefix) ]

  def get_skeletons_by_segid(self, filenames):
    cf = CloudFiles(self.cloudpath, progress=False)
    skels = cf.get(filenames)

    skeletons = defaultdict(list)
    for skel in skels:
      try:
        segid = filename_to_segid(skel['path'])
      except ValueError:
        # Typically this is due to preexisting fully
        # formed skeletons e.g. skeletons_mip_3/1588494
        continue

      skeletons[segid].append( 
        (
          Bbox.from_filename(skel['path']),
          pickle.loads(skel['content'])
        )
      )

    return skeletons

  def fuse_skeletons(self, skels):
    if len(skels) == 0:
      return Skeleton()

    bbxs = [ item[0] for item in skels ]
    skeletons = [ item[1] for item in skels ]

    skeletons = self.crop_skels(bbxs, skeletons)
    skeletons = [ s for s in skeletons if not s.empty() ]

    if len(skeletons) == 0:
      return Skeleton()

    return Skeleton.simple_merge(skeletons).consolidate()

  def crop_skels(self, bbxs, skeletons):
    cropped = [ s.clone() for s in skeletons ]

    if self.crop <= 0:
      return cropped
    
    for i in range(len(skeletons)):
      bbx = bbxs[i]
      bbx = bbx.astype(self.vol.resolution.dtype) 
      bbx.minpt += self.crop * self.vol.resolution
      bbx.maxpt -= self.crop * self.vol.resolution

      if bbx.volume() <= 0:
        continue

      cropped[i] = cropped[i].crop(bbx)

    return cropped

class ShardedSkeletonMergeTask(RegisteredTask):
  def __init__(
    self, cloudpath, shard_no, 
    dust_threshold=4000, 
    tick_threshold=6000, 
    frag_path=None, 
    cache=False,
    spatial_index_db=None, 
    max_cable_length=None,
    dry_run=False,
  ):
    super(ShardedSkeletonMergeTask, self).__init__(
      cloudpath, shard_no,  
      dust_threshold, tick_threshold, frag_path, cache, spatial_index_db,
      max_cable_length
    )
    self.progress = False
    self.max_cable_length = float(max_cable_length) if max_cable_length is not None else None
    self.dry_run = dry_run

  def execute(self):
    # cache is necessary for local computation, but on GCE download is very fast
    # so cache isn't necessary.
    cv = CloudVolume(
      self.cloudpath, 
      progress=self.progress,
      spatial_index_db=self.spatial_index_db,
      cache=self.cache
    )

    # This looks messy because we are trying to avoid retaining
    # unnecessary memory. In the original iteration, this was 
    # using 50 GB+ memory on minnie65. With changes to this
    # and the spatial_index, we are getting it down to something reasonable.
    locations = self.locations_for_labels(
      labels_for_shard(cv, self.shard_no, self.progress), 
      cv
    )
    filenames = set(itertools.chain(*locations.values()))
    labels = set(locations.keys())
    del locations
    skeletons = self.get_unfused(labels, filenames, cv, self.frag_path)
    del labels
    del filenames
    skeletons = self.process_skeletons(skeletons, in_place=True)

    if len(skeletons) == 0:
      return (skeletons, None)

    shard_files = synthesize_shard_files(cv.skeleton.reader.spec, skeletons)

    if len(shard_files) != 1:
      raise ValueError(
        "Only one shard file should be generated per task. Expected: {} Got: {} ".format(
          str(self.shard_no), ", ".join(shard_files.keys())
      ))

    if self.dry_run:
      return (skeletons, shard_files)

    cf = CloudFiles(cv.skeleton.meta.layerpath, progress=self.progress)
    cf.puts( 
      ( (fname, data) for fname, data in shard_files.items() ),
      compress=False,
      content_type='application/octet-stream',
      cache_control='no-cache',      
    )

  def process_skeletons(self, unfused_skeletons, in_place=False):
    skeletons = {}
    if in_place:
      skeletons = unfused_skeletons

    for label in tqdm(unfused_skeletons.keys(), desc="Postprocessing", disable=(not self.progress)):
      skels = unfused_skeletons[label]
      skel = Skeleton.simple_merge(skels)
      skel.id = label
      skel.extra_attributes = [ 
        attr for attr in skel.extra_attributes \
        if attr['data_type'] == 'float32' 
      ]
      skel = skel.consolidate()
      if self.max_cable_length is not None and skel.cable_length() > self.max_cable_length:
        skeletons[label] = skel.to_precomputed()
      else:
        skeletons[label] = kimimaro.postprocess(
          skel, 
          dust_threshold=self.dust_threshold, # voxels 
          tick_threshold=self.tick_threshold, # nm
        ).to_precomputed()

    return skeletons

  def get_unfused(self, labels, filenames, cv, frag_path):
    skeldirfn = lambda loc: cv.meta.join(cv.skeleton.meta.skeleton_path, loc)
    filenames = [ skeldirfn(loc) for loc in filenames ]

    block_size = 50

    if len(filenames) < block_size:
      blocks = [ filenames ]
      n_blocks = 1
    else:
      n_blocks = max(len(filenames) // block_size, 1)
      blocks = sip(filenames, block_size)

    frag_prefix = frag_path or cv.cloudpath
    local_input = False
    if paths.extract(frag_prefix).protocol == "file":
       local_input = True
       frag_prefix = frag_prefix.replace("file://", "", 1)

    all_skels = defaultdict(list)
    for filenames_block in tqdm(blocks, desc="Filename Block", total=n_blocks, disable=(not self.progress)):
      if local_input:
        all_files = {}
        for filename in filenames_block:
          all_files[filename] = open(os.path.join(frag_prefix, filename), "rb")
      else:
        all_files = { 
          filename: CloudFile(cv.meta.join(frag_prefix, filename), cache_meta=True)
          for filename in filenames_block 
        } 
      
      for filename, content in tqdm(all_files.items(), desc="Scanning Fragments", disable=(not self.progress)):
        fragment = MapBuffer(content, frombytesfn=Skeleton.from_precomputed)

        for label in labels:
          try:
            skel = fragment[label]
            skel.id = label
            all_skels[label].append(skel)
          except KeyError:
            continue

        if hasattr(content, "close"):
          content.close()

    return all_skels

  def locations_for_labels(self, labels, cv):
    SPATIAL_EXT = re.compile(r'\.spatial$')
    index_filenames = cv.skeleton.spatial_index.file_locations_per_label(labels)
    for label, locations in index_filenames.items():
      for i, location in enumerate(locations):
        bbx = Bbox.from_filename(re.sub(SPATIAL_EXT, '', location))
        bbx /= cv.meta.resolution(cv.skeleton.meta.mip)
        index_filenames[label][i] = bbx.to_filename() + '.frags'
    return index_filenames

def labels_for_shard(cv, shard_no, progress):
  """
  Try to fetch precalculated labels from `$shardno.labels` (faster) otherwise, 
  compute which labels are applicable to this shard from the shard index (much slower).
  """
  labels = CloudFiles(cv.skeleton.meta.layerpath).get_json(shard_no + '.labels')
  if labels is not None:
    return labels

  labels = cv.skeleton.spatial_index.query(cv.bounds * cv.resolution)
  spec = cv.skeleton.reader.spec

  return [ 
    lbl for lbl in tqdm(labels, desc="Computing Shard Numbers", disable=(not progress))  \
    if spec.compute_shard_location(lbl).shard_number == shard_no 
  ]

@queueable
def ShardedFromUnshardedSkeletonMergeTask(
  src:str,
  dest:str,
  shard_no:str,
  cache_control:bool = False,
  skel_dir:Optional[str] = None,
  progress:bool = False,
):
  cv_src = CloudVolume(src)

  if skel_dir is None and 'skeletons' in cv.info:
    skel_dir = cv.info['skeletons']

  cv_dest = CloudVolume(dest, skel_dir=skel_dir, progress=progress)

  labels = labels_for_shard(cv_dest, shard_no, progress)
  skeletons = cv_src.skeleton.get(labels)
  del labels

  if len(skeletons) == 0:
    return

  skeletons = strip_integer_attributes(skeletons)
  skeletons = { skel.id: skel.to_precomputed() for skel in skeletons }
  shard_files = synthesize_shard_files(cv_dest.skeleton.reader.spec, skeletons)

  if len(shard_files) != 1:
    raise ValueError(
      "Only one shard file should be generated per task. Expected: {} Got: {} ".format(
        str(shard_no), ", ".join(shard_files.keys())
    ))

  cf = CloudFiles(cv_dest.skeleton.meta.layerpath, progress=progress)
  cf.puts( 
    ( (fname, data) for fname, data in shard_files.items() ),
    compress=False,
    content_type='application/octet-stream',
    cache_control='no-cache',      
  )

@queueable
def DeleteSkeletonFilesTask(
  cloudpath:str,
  prefix:str,
  skel_dir:Optional[str] = None
):
  cv = CloudVolume(cloudpath, skel_dir=skel_dir)
  cf = CloudFiles(cv.skeleton.meta.layerpath)
  cf.delete(cf.list(prefix=prefix))


@queueable
def TransferSkeletonFilesTask(
  src:str,
  dest:str,
  prefix:str,
  skel_dir:Optional[str] = None
):
  cv_src = CloudVolume(src)
  cv_dest = CloudVolume(dest, skel_dir=skel_dir)

  cf_src = CloudFiles(cv_src.skeleton.meta.layerpath)
  cf_dest = CloudFiles(cv_dest.skeleton.meta.layerpath)

  cf_src.transfer_to(cf_dest, paths=cf_src.list(prefix=prefix))
