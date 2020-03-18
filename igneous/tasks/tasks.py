from __future__ import print_function

from collections import defaultdict
from collections.abc import Sequence

try:
  from StringIO import cStringIO as BytesIO
except ImportError:
  from io import BytesIO

import json
import math
import os
import random
import re
import pickle
# from tempfile import NamedTemporaryFile  # used by BigArrayTask

# from backports import lzma               # used by HyperSquareTask
# import blosc                             # used by BigArrayTask
# import h5py                              # used by BigArrayTask

import numpy as np
from tqdm import tqdm

from cloudvolume import CloudVolume
from cloudvolume.storage import Storage, SimpleStorage
from cloudvolume.lib import min2, Vec, Bbox, mkdir, jsonify
from taskqueue import RegisteredTask

from igneous import chunks, downsample_scales

import DracoPy
import fastremap
import tinybrain
import zmesh

def downsample_and_upload(
    image, bounds, vol, ds_shape, 
    mip=0, axis='z', skip_first=False,
    sparse=False
  ):
    ds_shape = min2(vol.volume_size, ds_shape[:3])

    # sometimes we downsample a base layer of 512x512
    # into underlying chunks of 64x64 which permits more scales
    underlying_mip = (mip + 1) if (mip + 1) in vol.available_mips else mip
    underlying_shape = vol.mip_underlying(underlying_mip).astype(np.float32)
    toidx = {'x': 0, 'y': 1, 'z': 2}
    preserved_idx = toidx[axis]
    underlying_shape[preserved_idx] = float('inf')

    # Need to use ds_shape here. Using image bounds means truncated
    # edges won't generate as many mip levels
    fullscales = downsample_scales.compute_plane_downsampling_scales(
      size=ds_shape,
      preserve_axis=axis,
      max_downsampled_size=int(min(*underlying_shape)),
    )
    factors = downsample_scales.scale_series_to_downsample_factors(fullscales)

    if len(factors) == 0:
      print("No factors generated. Image Shape: {}, Downsample Shape: {}, Volume Shape: {}, Bounds: {}".format(
          image.shape, ds_shape, vol.volume_size, bounds)
      )

    vol.mip = mip
    if not skip_first:
      vol[bounds.to_slices()] = image

    if len(factors) == 0:
      return

    num_mips = len(factors)

    mips = []
    if vol.layer_type == 'image':
      mips = tinybrain.downsample_with_averaging(image, factors[0], num_mips=num_mips)
    elif vol.layer_type == 'segmentation':
      mips = tinybrain.downsample_segmentation(
        image, factors[0], 
        num_mips=num_mips, sparse=sparse
      )
    else:
      mips = tinybrain.downsample_with_striding(image, factors[0], num_mips=num_mips)

    new_bounds = bounds.clone()
   
    for factor3 in factors:
      vol.mip += 1
      new_bounds //= factor3
      mipped = mips.pop(0)
      new_bounds.maxpt = new_bounds.minpt + Vec(*mipped.shape[:3])
      vol[new_bounds] = mipped

def cache(task, cloudpath):
  layer_path, filename = os.path.split(cloudpath)

  classname = task.__class__.__name__
  lcldir = mkdir(os.path.join('/tmp/', classname))
  lclpath = os.path.join(lcldir, filename)

  if os.path.exists(lclpath):
    with open(lclpath, 'rb') as f:
      filestr = f.read()
  else:
    with Storage(layer_path, n_threads=0) as stor:
      filestr = stor.get_file(filename)

    with open(lclpath, 'wb') as f:
      f.write(filestr)

  return filestr

class IngestTask(RegisteredTask):
  """Ingests and does downsampling.
     We want tasks execution to be independent of each other, so that no synchronization is
     required.
     The downsample scales should be such that the lowest resolution chunk should be able
     to be produce from the data available.
  """

  def __init__(self, chunk_path, chunk_encoding, layer_path):
    super(IngestTask, self).__init__(chunk_path, chunk_encoding, layer_path)
    self.chunk_path = chunk_path
    self.chunk_encoding = chunk_encoding
    self.layer_path = layer_path

  def execute(self):
    volume = CloudVolume(self.layer_path, mip=0)
    bounds = Bbox.from_filename(self.chunk_path)
    image = self._download_input_chunk(bounds)
    image = chunks.decode(image, self.chunk_encoding)
    # BUG: We need to provide some kind of ds_shape independent of the image
    # otherwise the edges of the dataset may not generate as many mip levels.
    downsample_and_upload(image, bounds, volume, mip=0,
                          ds_shape=image.shape[:3])

  def _download_input_chunk(self, bounds):
    storage = Storage(self.layer_path, n_threads=0)
    relpath = 'build/{}'.format(bounds.to_filename())
    return storage.get_file(relpath)


class DeleteTask(RegisteredTask):
  """Delete a block of images inside a layer on all mip levels."""

  def __init__(self, layer_path, shape, offset, mip=0, num_mips=5):
    super(DeleteTask, self).__init__(layer_path, shape, offset, mip, num_mips)
    self.layer_path = layer_path
    self.shape = Vec(*shape)
    self.offset = Vec(*offset)
    self.mip = mip
    self.num_mips = num_mips

  def execute(self):
    vol = CloudVolume(self.layer_path, mip=self.mip)

    highres_bbox = Bbox( self.offset, self.offset + self.shape )

    top_mip = min(vol.available_mips[-1], self.mip + self.num_mips)

    for mip in range(self.mip, top_mip + 1):
      vol.mip = mip
      bbox = vol.bbox_to_mip(highres_bbox, self.mip, mip)
      bbox = bbox.round_to_chunk_size(vol.underlying, offset=vol.bounds.minpt)
      bbox = Bbox.clamp(bbox, vol.bounds)

      if bbox.volume() == 0: 
        continue

      vol.delete(bbox)

class BlackoutTask(RegisteredTask):
  def __init__(
    self, cloudpath, mip, shape, offset, 
    value=0, non_aligned_writes=False
  ):
    super(BlackoutTask, self).__init__(
      cloudpath, mip, shape, 
      offset, value, non_aligned_writes
    )
    self.shape = Vec(*self.shape)
    self.offset = Vec(*self.offset)

  def execute(self):
    vol = CloudVolume(self.cloudpath, self.mip, non_aligned_writes=self.non_aligned_writes)
    bounds = Bbox(self.offset, self.shape + self.offset)
    bounds = Bbox.clamp(bounds, vol.bounds)
    img = np.zeros(bounds.size3(), dtype=vol.dtype) + self.value
    vol[bounds] = img

class TouchTask(RegisteredTask):
  def __init__(self, cloudpath, mip, shape, offset):
    super(TouchTask, self).__init__(cloudpath, mip, shape, offset)
  
  def execute(self):
    # This could be made more sophisticated using exists
    vol = CloudVolume(self.cloudpath, self.mip, fill_missing=False)
    bounds = Bbox(self.offset, self.shape + self.offset)
    bounds = Bbox.clamp(bounds, vol.bounds)
    image = vol[bounds]

class QuantizeTask(RegisteredTask):
  def __init__(self, source_layer_path, dest_layer_path, shape, offset, mip, fill_missing=False):
    super(QuantizeTask, self).__init__(
        source_layer_path, dest_layer_path, shape, offset, mip, fill_missing)
    self.source_layer_path = source_layer_path
    self.dest_layer_path = dest_layer_path
    self.shape = Vec(*shape)
    self.offset = Vec(*offset)
    self.fill_missing = fill_missing
    self.mip = mip

  def execute(self):
    srcvol = CloudVolume(self.source_layer_path, mip=self.mip,
                         fill_missing=self.fill_missing)

    bounds = Bbox(self.offset, self.shape + self.offset)
    bounds = Bbox.clamp(bounds, srcvol.bounds)

    image = srcvol[bounds.to_slices()][:, :, :, :1]  # only use x affinity
    image = (image * 255.0).astype(np.uint8)

    destvol = CloudVolume(self.dest_layer_path, mip=self.mip)
    downsample_and_upload(image, bounds, destvol, self.shape, mip=self.mip, axis='z')


class MeshTask(RegisteredTask):
  def __init__(self, shape, offset, layer_path, **kwargs):
    """
    Convert all labels in the specified bounding box into meshes
    via marching cubes and quadratic edge collapse (github.com/seung-lab/zmesh).

    Required:
      shape: (sx,sy,sz) size of task
      offset: (x,y,z) offset from (0,0,0)
      layer_path: neuroglancer/cloudvolume dataset path

    Optional:
      lod: (uint) level of detail to record these meshes at
      mip: (uint) level of the resolution pyramid to download segmentation from
      simplification_factor: (uint) try to reduce the number of triangles in the 
        mesh by this factor (but constrained by max_simplification_error)
      max_simplification_error: The maximum physical distance that
        simplification is allowed to move a triangle vertex by. 
      mesh_dir: which subdirectory to write the meshes to (overrides info file location)
      remap_table: agglomerate segmentation before meshing using { orig_id: new_id }
      generate_manifests: (bool) if it is known that the meshes generated by this 
        task will not be cropped by the bounding box, avoid needing to run a seperate
        MeshManifestTask pass by generating manifests on the spot.

      These two options are used to allow sufficient overlap for trivial mesh stitching
      between adjacent tasks.

        low_padding: (uint) expand the bounding box by this many pixels by subtracting
          this padding from the minimum point of the bounding box on all axes.
        high_padding: (uint) expand the bounding box by this many pixels adding
          this padding to the maximum point of the bounding box on all axes.

      parallel_download: (uint: 1) number of processes to use during the segmentation download
      cache_control: (str: None) specify the cache-control header when uploading mesh files
      dust_threshold: (uint: None) don't bother meshing labels strictly smaller than this number of voxels.
      encoding: (str) 'precomputed' (default) or 'draco'
      draco_compression_level: (uint: 1) only applies to draco encoding
      draco_create_metadata: (bool: False) only applies to draco encoding
      progress: (bool: False) show progress bars for meshing 
      object_ids: (list of ints) if specified, only mesh these ids
      fill_missing: (bool: False) replace missing segmentation files with zeros instead of erroring
      spatial_index: (bool: False) generate a JSON spatial index of which meshes are available in
        a given bounding box. 
      sharded: (bool: False) If True, upload all meshes together as a single pickled 
        fragment file. 
      timestamp: (int: None) (graphene only) use the segmentation existing at this
        UNIX timestamp.
    """
    super(MeshTask, self).__init__(shape, offset, layer_path, **kwargs)
    self.shape = Vec(*shape)
    self.offset = Vec(*offset)
    self.layer_path = layer_path
    self.options = {
      'cache_control': kwargs.get('cache_control', None),
      'draco_compression_level': kwargs.get('draco_compression_level', 1),
      'draco_create_metadata': kwargs.get('draco_create_metadata', False),
      'dust_threshold': kwargs.get('dust_threshold', None),
      'encoding': kwargs.get('encoding', 'precomputed'),
      'fill_missing': kwargs.get('fill_missing', False),
      'generate_manifests': kwargs.get('generate_manifests', False),
      'high_padding': kwargs.get('high_padding', 1),
      'low_padding': kwargs.get('low_padding', 0),
      'lod': kwargs.get('lod', 0),
      'max_simplification_error': kwargs.get('max_simplification_error', 40),
      'simplification_factor': kwargs.get('simplification_factor', 100),
      'mesh_dir': kwargs.get('mesh_dir', None),
      'mip': kwargs.get('mip', 0),
      'object_ids': kwargs.get('object_ids', None),
      'parallel_download': kwargs.get('parallel_download', 1),
      'progress': kwargs.get('progress', False),
      'remap_table': kwargs.get('remap_table', None),
      'spatial_index': kwargs.get('spatial_index', False),
      'sharded': kwargs.get('sharded', False),
      'timestamp': kwargs.get('timestamp', None),
    }
    supported_encodings = ['precomputed', 'draco']
    if not self.options['encoding'] in supported_encodings:
      raise ValueError('Encoding {} is not supported. Options: {}'.format(
        self.options['encoding'], ', '.join(supported_encodings)
      ))
    self._encoding_to_compression_dict = {
      'precomputed': True,
      'draco': False,
    }

    self.options['remap_table'] = {
      int(k): int(v) for k, v in self.options['remap_table'].items()
    }

  def execute(self):
    self._volume = CloudVolume(
      self.layer_path, self.options['mip'], bounded=False,
      parallel=self.options['parallel_download'], 
      fill_missing=self.options['fill_missing']
    )
    self._bounds = Bbox(self.offset, self.shape + self.offset)
    self._bounds = Bbox.clamp(self._bounds, self._volume.bounds)

    self.progress = bool(self.options['progress'])

    self._mesher = zmesh.Mesher(self._volume.resolution)

    # Marching cubes loves its 1vx overlaps.
    # This avoids lines appearing between
    # adjacent chunks.
    data_bounds = self._bounds.clone()
    data_bounds.minpt -= self.options['low_padding']
    data_bounds.maxpt += self.options['high_padding']

    self._mesh_dir = self.get_mesh_dir()

    if self.options['encoding'] == 'draco':
      self.draco_encoding_settings = self._compute_draco_encoding_settings()

    # chunk_position includes the overlap specified by low_padding/high_padding
    # agglomerate only applies to graphene volumes, no-op for precomputed
    data = self._volume.download(
      data_bounds, agglomerate=True, 
      timestamp=self.options['timestamp']
    )

    if not np.any(data):
      return

    data = self._remove_dust(data, self.options['dust_threshold'])
    data = self._remap(data)

    if self.options['object_ids']:
      data = fastremap.mask_except(data, self.options['object_ids'], in_place=True)

    data, renumbermap = fastremap.renumber(data, in_place=True)
    renumbermap = { v:k for k,v in renumbermap.items() }
    self._compute_meshes(data, renumbermap)

  def get_mesh_dir(self):
    if self.options['mesh_dir'] is not None:
      return self.options['mesh_dir']
    elif 'mesh' in self._volume.info:
      return self._volume.info['mesh']
    else:
      raise ValueError("The mesh destination is not present in the info file.")

  def _compute_draco_encoding_settings(self):
    def calculate_quantization_bits_and_range(min_quantization_range, max_draco_bin_size, draco_quantization_bits=None):
        if draco_quantization_bits is None:
            draco_quantization_bits = np.ceil(
                np.log2(min_quantization_range / max_draco_bin_size + 1))
        num_draco_bins = 2 ** draco_quantization_bits - 1
        draco_bin_size = np.ceil(min_quantization_range / num_draco_bins)
        draco_quantization_range = draco_bin_size * num_draco_bins
        if draco_quantization_range < min_quantization_range + draco_bin_size:
            if draco_bin_size == max_draco_bin_size:
                return calculate_quantization_bits_and_range(min_quantization_range, max_draco_bin_size, draco_quantization_bits + 1)
            else:
                draco_bin_size = draco_bin_size + 1
                draco_quantization_range = draco_quantization_range + num_draco_bins
        return draco_quantization_bits, draco_quantization_range, draco_bin_size

    min_quantization_range = max((self.shape + self.options['low_padding'] + self.options['high_padding']) * self._volume.resolution)
    max_draco_bin_size = np.floor(min(self._volume.resolution) / np.sqrt(2))
    draco_quantization_bits, draco_quantization_range, draco_bin_size = calculate_quantization_bits_and_range(
        min_quantization_range, max_draco_bin_size)
    draco_quantization_origin = self.offset - (self.offset % draco_bin_size)
    return {
        'quantization_bits': draco_quantization_bits,
        'compression_level': self.options['draco_compression_level'],
        'quantization_range': draco_quantization_range,
        'quantization_origin': draco_quantization_origin,
        'create_metadata': self.options['draco_create_metadata']
    }

  def _remove_dust(self, data, dust_threshold):
    if dust_threshold:
      segids, pxct = fastremap.unique(data, return_counts=True)
      dust_segids = [ sid for sid, ct in zip(segids, pxct) if ct < int(dust_threshold) ]
      data = fastremap.mask(data, dust_segids, in_place=True)

    return data

  def _remap(self, data):
    if self.options['remap_table'] is None:
      return data 

    remap = self.options['remap_table']
    remap[0] = 0

    data = fastremap.mask_except(data, list(remap.keys()), in_place=True)
    data = fastremap.remap(data, remap, in_place=True)
    return data

  def _compute_meshes(self, data, renumbermap):
    data = data[:, :, :, 0].T
    self._mesher.mesh(data)
    del data

    bounding_boxes = {}
    meshes = {}

    for obj_id in tqdm(self._mesher.ids(), disable=(not self.progress), desc="Mesh"):
      remapped_id = renumbermap[obj_id]
      mesh_binary, mesh_bounds = self._create_mesh(obj_id)
      bounding_boxes[remapped_id] = mesh_bounds.to_list()
      meshes[remapped_id] = mesh_binary

    if self.options['sharded']:
      self._upload_batch(meshes, self._bounds)
    else:
      self._upload_individuals(meshes, self.options['generate_manifests'])

    if self.options['spatial_index']:
      self._upload_spatial_index(self._bounds, bounding_boxes)

  def _upload_batch(self, meshes, bbox):
    with SimpleStorage(self.layer_path, progress=self.options['progress']) as stor:
      # Create mesh batch for postprocessing later
      stor.put_file(
        file_path="{}/{}.frags".format(self._mesh_dir, bbox.to_filename()),
        content=pickle.dumps(skeletons),
        compress='gzip',
        content_type="application/python-pickle",
        cache_control=False,
      )

  def _upload_individuals(self, mesh_binaries, generate_manifests):
    with Storage(self.layer_path) as storage:
      for segid, mesh_binary in mesh_binaries.items():
        storage.put_file(
          file_path='{}/{}:{}:{}'.format(
            self._mesh_dir, segid, self.options['lod'],
            self._bounds.to_filename()
          ),
          content=mesh_binary,
          compress=self._encoding_to_compression_dict[self.options['encoding']],
          cache_control=self.options['cache_control']
        )

        if generate_manifests:
          fragments = []
          fragments.append('{}:{}:{}'.format(
            segid, self.options['lod'],
            self._bounds.to_filename())
          )

          storage.put_file(
            file_path='{}/{}:{}'.format(
              self._mesh_dir, segid, self.options['lod']
            ),
            content=json.dumps({"fragments": fragments}),
            content_type='application/json',
            cache_control=self.options['cache_control']
          )


  def _create_mesh(self, obj_id):
    mesh = self._mesher.get_mesh(
      obj_id,
      simplification_factor=self.options['simplification_factor'],
      max_simplification_error=self.options['max_simplification_error']
    )

    self._mesher.erase(obj_id)

    resolution = self._volume.resolution
    offset = self._bounds.minpt - self.options['low_padding']
    mesh.vertices[:] += offset * resolution

    mesh_bounds = Bbox(
      np.amin(mesh.vertices, axis=0), 
      np.amax(mesh.vertices, axis=0)
    )

    if self.options['encoding'] == 'draco':
      mesh_binary = DracoPy.encode_mesh_to_buffer(
        mesh.vertices.flatten('C'), mesh.faces.flatten('C'), 
        **self.draco_encoding_settings
      )
    elif self.options['encoding'] == 'precomputed':
      mesh_binary = mesh.to_precomputed()

    return mesh_binary, mesh_bounds

  def _upload_spatial_index(self, bbox, mesh_bboxes):
    with SimpleStorage(self.layer_path, progress=self.options['progress']) as stor:
      stor.put_file(
        file_path="{}/{}.spatial".format(self._mesh_dir, bbox.to_filename()),
        content=jsonify(mesh_bboxes).encode('utf8'),
        compress='gzip',
        content_type="application/json",
        cache_control=False,
      )

class MeshManifestTask(RegisteredTask):
  """
  Finalize mesh generation by post-processing chunk fragment
  lists into mesh fragment manifests.
  These are necessary for neuroglancer to know which mesh
  fragments to download for a given segid.

  If we parallelize using prefixes single digit prefixes ['0','1',..'9'] all meshes will
  be correctly processed. But if we do ['10','11',..'99'] meshes from [0,9] won't get
  processed and need to be handle specifically by creating tasks that will process
  a single mesh ['0:','1:',..'9:']
  """

  def __init__(self, layer_path, prefix, lod=0, mesh_dir=None):
    super(MeshManifestTask, self).__init__(layer_path, prefix)
    self.layer_path = layer_path
    self.lod = lod
    self.prefix = prefix
    self.mesh_dir = mesh_dir

  def execute(self):
    with Storage(self.layer_path) as storage:
      self._info = json.loads(storage.get_file('info').decode('utf8'))

      if self.mesh_dir is None and 'mesh' in self._info:
        self.mesh_dir = self._info['mesh']

      self._generate_manifests(storage)

  def _get_mesh_filenames_subset(self, storage):
    prefix = '{}/{}'.format(self.mesh_dir, self.prefix)
    segids = defaultdict(list)

    for filename in storage.list_files(prefix=prefix):
      filename = os.path.basename(filename)
      # `match` implies the beginning (^). `search` matches whole string
      matches = re.search(r'(\d+):(\d+):', filename)

      if not matches:
        continue

      segid, lod = matches.groups()
      segid, lod = int(segid), int(lod)

      if lod != self.lod:
        continue

      segids[segid].append(filename)

    return segids

  def _generate_manifests(self, storage):
    segids = self._get_mesh_filenames_subset(storage)
    for segid, frags in segids.items():
      storage.put_file(
          file_path='{}/{}:{}'.format(self.mesh_dir, segid, self.lod),
          content=json.dumps({"fragments": frags}),
          content_type='application/json',
      )


# class BigArrayTask(RegisteredTask):
#   def __init__(self, layer_path, chunk_path, chunk_encoding, version):
#     super(BigArrayTask, self).__init__(
#         layer_path, chunk_path, chunk_encoding, version)
#     self.layer_path = layer_path
#     self.chunk_path = chunk_path
#     self.chunk_encoding = chunk_encoding
#     self.version = version

#   def execute(self):
#     self._parse_chunk_path()
#     self._storage = Storage(self.layer_path)
#     self._download_input_chunk()
#     self._upload_chunk()

#   def _parse_chunk_path(self):
#     if self.version == 'zfish_v0/affinities':
#       match = re.match(r'^.*/bigarray/block_(\d+)-(\d+)_(\d+)-(\d+)_(\d+)-(\d+)_1-3.h5$',
#                        self.chunk_path)
#     elif self.version == 'zfish_v0/image' or self.version == 'pinky_v0/image':
#       match = re.match(r'^.*/bigarray/(\d+):(\d+)_(\d+):(\d+)_(\d+):(\d+)$',
#                        self.chunk_path)
#     else:
#       raise NotImplementedError(self.version)

#     (self._xmin, self._xmax,
#      self._ymin, self._ymax,
#      self._zmin, self._zmax) = match.groups()

#     self._xmin = int(self._xmin)
#     self._xmax = int(self._xmax)
#     self._ymin = int(self._ymin)
#     self._ymax = int(self._ymax)
#     self._zmin = int(self._zmin)
#     self._zmax = int(self._zmax)
#     self._filename = self.chunk_path.split('/')[-1]

#   def _download_input_chunk(self):
#     string_data = self._storage.get_file(
#         os.path.join('bigarray', self._filename))
#     if self.version == 'zfish_v0/affinities':
#       self._data = self._decode_hdf5(string_data)
#     elif self.version == 'zfish_v0/image':
#       self._data = self._decode_blosc(string_data, shape=[2048, 2048, 128])
#     elif self.version == 'pinky_v0/image':
#       self._data = self._decode_blosc(string_data, shape=[2048, 2048, 64])
#     else:
#       raise NotImplementedError(self.version)

#   def _decode_blosc(self, string, shape):
#     seeked = blosc.decompress(string[10:])
#     arr = np.fromstring(seeked, dtype=np.uint8).reshape(
#         shape[::-1]).transpose((2, 1, 0))
#     return np.expand_dims(arr, 3)

#   def _decode_hdf5(self, string):
#     with NamedTemporaryFile(delete=False) as tmp:
#       tmp.write(string)
#       tmp.close()
#       with h5py.File(tmp.name, 'r') as h5:
#         return np.transpose(h5['img'][:], axes=(3, 2, 1, 0))

#   def _upload_chunk(self):
#     if self.version == 'zfish_v0/affinities':
#       shape = [313472, 193664, 1280]
#       offset = [14336, 11264, 16384]
#     elif self.version == 'zfish_v0/image':
#       shape = [69632, 34816, 1280]
#       offset = [14336, 12288, 16384]
#     elif self.version == 'pinky_v0/image':
#       shape = [100352, 55296, 1024]
#       offset = [2048, 14336, 16384]
#     else:
#       raise NotImplementedError(self.version)

#     xmin = self._xmin - offset[0] - 1
#     xmax = min(self._xmax - offset[0], shape[0])
#     ymin = self._ymin - offset[1] - 1
#     ymax = min(self._ymax - offset[1], shape[1])
#     zmin = self._zmin - offset[2] - 1
#     zmax = min(self._zmax - offset[2], shape[2])

#     # bigarray chunk has padding to fill the volume
#     chunk = self._data[:xmax-xmin, :ymax-ymin, :zmax-zmin, :]
#     filename = 'build/{:d}-{:d}_{:d}-{:d}_{:d}-{:d}'.format(
#         xmin, xmax, ymin, ymax, zmin, zmax)
#     encoded = self._encode(chunk, self.chunk_encoding)
#     self._storage.put_file(filename, encoded)
#     self._storage.wait_until_queue_empty()

#   def _encode(self, chunk, encoding):
#     if encoding == "jpeg":
#       return chunks.encode_jpeg(chunk)
#     elif encoding == "npz":
#       return chunks.encode_npz(chunk)
#     elif encoding == "npz_uint8":
#       chunk = chunk * 255
#       chunk = chunk.astype(np.uint8)
#       return chunks.encode_npz(chunk)
#     elif encoding == "raw":
#       return chunks.encode_raw(chunk)
#     else:
#       raise NotImplementedError(encoding)


# class HyperSquareTask(RegisteredTask):
#   def __init__(self, bucket_name, dataset_name, layer_name,
#                volume_dir, layer_type, overlap, resolution):

#     self.bucket_name = bucket_name
#     self.dataset_name = dataset_name
#     self.layer_name = layer_name
#     self.volume_dir = volume_dir
#     self.layer_type = layer_type
#     self.overlap = Vec(*overlap)

#     self.resolution = Vec(*resolution)

#     self._volume_cloudpath = 'gs://{}/{}'.format(
#         self.bucket_name, self.volume_dir)
#     self._bucket = None
#     self._metadata = None
#     self._bounds = None

#   def execute(self):
#     client = storage.Client.from_service_account_json(
#         lib.credentials_path(), project=lib.GCLOUD_PROJECT_NAME
#     )
#     self._bucket = client.get_bucket(self.bucket_name)
#     self._metadata = meta = self._download_metadata()

#     self._bounds = Bbox(
#         meta['physical_offset_min'],  # in voxels
#         meta['physical_offset_max']
#     )

#     shape = Vec(*meta['chunk_voxel_dimensions'])
#     shape = Vec(shape.x, shape.y, shape.z, 1)

#     if self.layer_type == 'image':
#       dtype = meta['image_type'].lower()
#       cube = self._materialize_images(shape, dtype)
#     elif self.layer_type == 'segmentation':
#       dtype = meta['segment_id_type'].lower()
#       cube = self._materialize_segmentation(shape, dtype)
#     else:
#       dtype = meta['affinity_type'].lower()
#       return NotImplementedError("Don't know how to get the images for this layer.")

#     self._upload_chunk(cube, dtype)

#   def _download_metadata(self):
#     cloudpath = '{}/metadata.json'.format(self.volume_dir)
#     metadata = self._bucket.get_blob(cloudpath).download_as_string()
#     return json.loads(metadata)

#   def _materialize_segmentation(self, shape, dtype):
#     segmentation_path = '{}/segmentation.lzma'.format(self.volume_dir)
#     seg_blob = self._bucket.get_blob(segmentation_path)
#     return self._decode_lzma(seg_blob.download_as_string(), shape, dtype)

#   def _materialize_images(self, shape, dtype):
#     cloudpaths = ['{}/jpg/{}.jpg'.format(self.volume_dir, i)
#                   for i in xrange(shape.z)]
#     datacube = np.zeros(shape=shape, dtype=np.uint8)  # x,y,z,channels

#     prefix = '{}/jpg/'.format(self.volume_dir)

#     blobs = self._bucket.list_blobs(prefix=prefix)
#     for blob in blobs:
#       z = int(re.findall(r'(\d+)\.jpg', blob.name)[0])
#       imgdata = blob.download_as_string()
#       # Hypersquare images are each situated in the xy plane
#       # so the shape should be (width,height,1)
#       shape = self._bounds.size3()
#       shape.z = 1
#       datacube[:, :, z, :] = chunks.decode_jpeg(imgdata, shape=tuple(shape))

#     return datacube

#   def _decode_lzma(self, string_data, shape, dtype):
#     arr = lzma.decompress(string_data)
#     arr = np.fromstring(arr, dtype=dtype)
#     return arr.reshape(shape[::-1]).T

#   def _upload_chunk(self, datacube, dtype):
#     vol = CloudVolume(self.dataset_name, self.layer_name, mip=0)
#     hov = self.overlap / 2  # half overlap, e.g. 32 -> 16 in e2198
#     img = datacube[hov.x:-hov.x, hov.y:-hov.y,
#                    hov.z:-hov.z, :]  # e.g. 256 -> 224
#     bounds = self._bounds.clone()

#     # the boxes are offset left of zero by half overlap, so no need to
#     # compensate for weird shifts. only upload the non-overlap region.

#     downsample_and_upload(image, bounds, vol, ds_shape=img.shape)
#     vol[bounds.to_slices()] = img


class HyperSquareConsensusTask(RegisteredTask):
  """
  Import an Eyewire consensus into neuroglancer by combining
  database information encoded as JSON files with pre-ingested
  Hypersquare.

  The result of the remapping is that all human traced cells should
  be present and identifiable by their Eyewire cells.id number.
  The remaining segments should be the same segid but reencoded
  from 16 to 32 bits such that their high bits encode the
  tasks.segmentation_id according to some mapping that fits
  into 16 bits. It's usually:

    tasks.segmentation_id - min(tasks.segmentation_id for that dataset)

  The consensus map file should be uploaded into the neuroglancer
  data layer directory corresponding to the destination layer being
  processed. The contents of the file are JSON encoded and look like:

  { VOLUMEID: { CELLID: [segids] } }
  """

  def __init__(self, src_path, dest_path, ew_volume_id,
               consensus_map_path, shape, offset):

    super(HyperSquareConsensusTask, self).__init__(
        src_path, dest_path, ew_volume_id,
        consensus_map_path, shape, offset
    )
    self.src_path = src_path
    self.dest_path = dest_path
    self.consensus_map_path = consensus_map_path
    self.shape = Vec(*shape)
    self.offset = Vec(*offset)
    self.ew_volume_id = int(ew_volume_id)

  def execute(self):
    bounds = Bbox(self.offset, self.shape + self.offset)
    srcvol = CloudVolume(self.src_path, fill_missing=True)
    destvol = CloudVolume(self.dest_path)

    consensus = cache(self, self.consensus_map_path).decode('utf8')
    consensus = json.loads(consensus)
    try:
      consensus = consensus[str(self.ew_volume_id)]
    except KeyError:
      print("Black Region", bounds, self.ew_volume_id)
      consensus = {}

    segidmap = self.build_segid_map(consensus, destvol.dtype)

    try:
      image = srcvol[bounds.to_slices()]
    except ValueError:
      print("Skipping", bounds)
      zeroshape = list(bounds.size3()) + [srcvol.num_channels]
      image = np.zeros(shape=zeroshape, dtype=destvol.dtype)

    image = image.astype(destvol.dtype)
    # Merge equivalent segments, non-consensus segments are black
    consensus_image = segidmap[image]

    # Write volume ID to high bits of extended bit width image
    volume_segid_image = image | (self.ew_volume_id << 16)

    # Zero out segid 0 in the high bits so neuroglancer interprets them as empty
    volume_segid_image *= np.logical_not(np.logical_not(image))

    # Final image is consensus keyed by cell ID (C), i.e. 0xCCCCCCCC.
    # Non-empty non-consensus segments are written as:
    # | 16 bit volume_id (V) | 16 bit seg_id (S) |, i.e. 0xVVVVSSSS
    # empties are 0x00000000
    final_image = consensus_image + (consensus_image == 0) * volume_segid_image

    destvol[bounds.to_slices()] = final_image

  def build_segid_map(self, consensus, dtype):
    segidmap = np.zeros(shape=(2 ** 16), dtype=dtype)

    for cellid in consensus:
      for segid in consensus[cellid]:
        segidmap[segid] = int(cellid)

    return segidmap


class ContrastNormalizationTask(RegisteredTask):
  """TransferTask + Contrast Correction based on LuminanceLevelsTask output."""
  # translate = change of origin

  def __init__(
    self, src_path, dest_path, levels_path, shape, 
    offset, mip, clip_fraction, fill_missing, 
    translate, minval, maxval
  ):

    super(ContrastNormalizationTask, self).__init__(
      src_path, dest_path, levels_path, shape, offset, 
      mip, clip_fraction, fill_missing, translate,
      minval, maxval
    )
    self.src_path = src_path
    self.dest_path = dest_path
    self.shape = Vec(*shape)
    self.offset = Vec(*offset)
    self.fill_missing = fill_missing
    self.translate = Vec(*translate)
    self.mip = int(mip)
    if isinstance(clip_fraction, Sequence):
      assert len(clip_fraction) == 2
      self.lower_clip_fraction = float(clip_fraction[0])
      self.upper_clip_fraction = float(clip_fraction[1])
    else:
      self.lower_clip_fraction = self.upper_clip_fraction = float(clip_fraction)

    self.minval = minval 
    self.maxval = maxval

    self.levels_path = levels_path if levels_path else self.src_path

    assert 0 <= self.lower_clip_fraction <= 1
    assert 0 <= self.upper_clip_fraction <= 1
    assert self.lower_clip_fraction + self.upper_clip_fraction <= 1

  def execute(self):
    srccv = CloudVolume(
        self.src_path, fill_missing=self.fill_missing, mip=self.mip)
    destcv = CloudVolume(
        self.dest_path, fill_missing=self.fill_missing, mip=self.mip)

    bounds = Bbox(self.offset, self.shape[:3] + self.offset)
    bounds = Bbox.clamp(bounds, srccv.bounds)
    image = srccv[bounds.to_slices()].astype(np.float32)

    zlevels = self.fetch_z_levels()

    nbits = np.dtype(srccv.dtype).itemsize * 8
    maxval = float(2 ** nbits - 1)

    for z in range(bounds.minpt.z, bounds.maxpt.z):
      imagez = z - bounds.minpt.z
      zlevel = zlevels[imagez]
      (lower, upper) = self.find_section_clamping_values(
          zlevel, self.lower_clip_fraction, 1 - self.upper_clip_fraction)
      if lower == upper:
        continue
      img = image[:, :, imagez]
      img = (img - float(lower)) * (maxval / (float(upper) - float(lower)))
      image[:, :, imagez] = img

    image = np.round(image)

    minval = self.minval if self.minval is not None else 0.0
    maxval = self.maxval if self.maxval is not None else maxval

    image = np.clip(image, minval, maxval).astype(destcv.dtype)

    bounds += self.translate
    downsample_and_upload(image, bounds, destcv, self.shape, mip=self.mip)

  def find_section_clamping_values(self, zlevel, lowerfract, upperfract):
    filtered = np.copy(zlevel)

    # remove pure black from frequency counts as
    # it has no information in our images
    filtered[0] = 0

    cdf = np.zeros(shape=(len(filtered),), dtype=np.uint64)
    cdf[0] = filtered[0]
    for i in range(1, len(filtered)):
      cdf[i] = cdf[i - 1] + filtered[i]

    total = cdf[-1]

    if total == 0:
      return (0, 0)

    lower = 0
    for i, val in enumerate(cdf):
      if float(val) / float(total) > lowerfract:
        break
      lower = i

    upper = 0
    for i, val in enumerate(cdf):
      if float(val) / float(total) > upperfract:
        break
      upper = i

    return (lower, upper)

  def fetch_z_levels(self):
    bounds = Bbox(self.offset, self.shape[:3] + self.offset)

    levelfilenames = [
      'levels/{}/{}'.format(self.mip, z) \
      for z in range(bounds.minpt.z, bounds.maxpt.z)
    ]
    
    with Storage(self.levels_path) as stor:
      levels = stor.get_files(levelfilenames)

    errors = [ 
      level['filename'] \
      for level in levels if level['content'] == None
    ]

    if len(errors):
      raise Exception(", ".join(
          errors) + " were not defined. Did you run a LuminanceLevelsTask for these slices?")

    levels = [(
      int(os.path.basename(item['filename'])),
      json.loads(item['content'].decode('utf-8'))
    ) for item in levels ]

    levels.sort(key=lambda x: x[0])
    levels = [x[1] for x in levels]
    return [ np.array(x['levels'], dtype=np.uint64) for x in levels ]


class LuminanceLevelsTask(RegisteredTask):
  """Generate a frequency count of luminance values by random sampling. Output to $PATH/levels/$MIP/$Z"""

  def __init__(self, src_path, levels_path, shape, offset, coverage_factor, mip):
    super(LuminanceLevelsTask, self).__init__(
      src_path, levels_path, shape, 
      offset, coverage_factor, mip
    )
    self.src_path = src_path
    self.shape = Vec(*shape)
    self.offset = Vec(*offset)
    self.coverage_factor = coverage_factor
    self.mip = int(mip)
    self.levels_path = levels_path

    assert 0 < coverage_factor <= 1, "Coverage Factor must be between 0 and 1"

  def execute(self):
    srccv = CloudVolume(self.src_path, mip=self.mip, fill_missing=True)

    # Accumulate a histogram of the luminance levels
    nbits = np.dtype(srccv.dtype).itemsize * 8
    levels = np.zeros(shape=(2 ** nbits,), dtype=np.uint64)

    bounds = Bbox(self.offset, self.shape[:3] + self.offset)
    bounds = Bbox.clamp(bounds, srccv.bounds)

    bboxes = self.select_bounding_boxes(bounds)
    for bbox in bboxes:
      img2d = srccv[bbox.to_slices()].reshape((bbox.volume()))
      cts = np.bincount(img2d)
      levels[0:len(cts)] += cts.astype(np.uint64)

    covered_area = sum([bbx.volume() for bbx in bboxes])

    bboxes = [(bbox.volume(), bbox.size3()) for bbox in bboxes]
    bboxes.sort(key=lambda x: x[0])
    biggest = bboxes[-1][1]

    output = {
      "levels": levels.tolist(),
      "patch_size": biggest.tolist(),
      "num_patches": len(bboxes),
      "coverage_ratio": covered_area / self.shape.rectVolume(),
    }

    path = self.levels_path if self.levels_path else self.src_path
    path = os.path.join(path, 'levels')
    with Storage(path, n_threads=0) as stor:
      stor.put_json(
        file_path="{}/{}".format(self.mip, self.offset.z),
        content=output,
        cache_control='no-cache'
      )

  def select_bounding_boxes(self, dataset_bounds):
    # Sample patches until coverage factor is satisfied. 
    # Ensure the patches are non-overlapping and random.
    sample_shape = Bbox((0, 0, 0), (2048, 2048, 1))
    area = self.shape.rectVolume()

    total_patches = int(math.ceil(area / sample_shape.volume()))
    N = int(math.ceil(float(total_patches) * self.coverage_factor))

    # Simplification: We are making patch selection against a discrete
    # grid instead of a continuous space. This removes the influence of
    # overlap in a less complex fashion.
    patch_indicies = set()
    while len(patch_indicies) < N:
      ith_patch = random.randint(0, (total_patches - 1))
      patch_indicies.add(ith_patch)

    gridx = int(math.ceil(self.shape.x / sample_shape.size3().x))

    bboxes = []
    for i in patch_indicies:
      patch_start = Vec(i % gridx, i // gridx, 0)
      patch_start *= sample_shape.size3()
      patch_start += self.offset
      bbox = Bbox(patch_start, patch_start + sample_shape.size3())
      bbox = Bbox.clamp(bbox, dataset_bounds)
      bboxes.append(bbox)
    return bboxes

class TransferTask(RegisteredTask):
  # translate = change of origin
  def __init__(
    self, src_path, dest_path, 
    mip, shape, offset, 
    translate=(0,0,0),
    fill_missing=False, 
    skip_first=False, 
    skip_downsamples=False,
    delete_black_uploads=False, 
    background_color=0,
    sparse=False,
    axis='z'
  ):
    super(TransferTask, self).__init__(
      src_path, dest_path, 
      mip, shape, offset, 
      translate, fill_missing, 
      skip_first, skip_downsamples,
      delete_black_uploads, background_color,
      sparse, axis
    )
    self.src_path = src_path
    self.dest_path = dest_path
    self.mip = mip
    self.shape = Vec(*shape)
    self.offset = Vec(*offset)
    self.fill_missing = bool(fill_missing)
    self.translate = Vec(*translate)
    self.delete_black_uploads = bool(delete_black_uploads)
    self.sparse = bool(sparse)
    self.skip_first = bool(skip_first)
    self.skip_downsamples = bool(skip_downsamples)
    self.background_color = background_color
    self.axis = axis
    self.sparse = sparse

  def execute(self):
    srccv = CloudVolume(
      self.src_path, fill_missing=self.fill_missing,
      mip=self.mip, bounded=False
    )
    destcv = CloudVolume(
      self.dest_path, fill_missing=self.fill_missing,
      mip=self.mip, delete_black_uploads=self.delete_black_uploads,
      background_color=self.background_color
    )

    dst_bounds = Bbox(self.offset, self.shape + self.offset)
    dst_bounds = Bbox.clamp(dst_bounds, destcv.bounds)
    src_bounds = dst_bounds - self.translate
    image = srccv[src_bounds.to_slices()]

    if self.skip_downsamples:
      destcv[dst_bounds] = image
    else:
      downsample_and_upload(
        image, dst_bounds, destcv,
        self.shape, mip=self.mip,
        skip_first=self.skip_first,
        sparse=self.sparse, axis=self.axis
      )

class DownsampleTask(TransferTask):
  def __init__(
    self, layer_path, mip, shape, offset, 
    fill_missing=False, axis='z', sparse=False,
    delete_black_uploads=False, background_color=0,
    dest_path=None
  ):
    self.layer_type = layer_path

    if dest_path is None:
      dest_path = layer_path

    super(DownsampleTask, self).__init__(
      layer_path, dest_path, 
      mip, shape, offset, 
      translate=(0,0,0),
      fill_missing=fill_missing, 
      skip_first=True, 
      skip_downsamples=False,
      delete_black_uploads=delete_black_uploads, 
      background_color=background_color,
      sparse=sparse,
      axis=axis
    )



class WatershedRemapTask(RegisteredTask):
  """
  Take raw watershed output and using a remapping file,
  generate an aggregated segmentation.

  The remap array is a key:value mapping where the
  array index is the key and the value is the contents.

  You can find a script to convert h5 remap files into npy
  files in pipeline/scripts/remap2npy.py

  Required:
      map_path: path to remap file. Must be in npy or npz format.
      src_path: path to watershed layer
      dest_path: path to new layer
      shape: size of volume to remap
      offset: voxel offset into dataset
  """

  def __init__(self, map_path, src_path, dest_path, shape, offset):
    super(WatershedRemapTask, self).__init__(
        map_path, src_path, dest_path, shape, offset)
    self.map_path = map_path
    self.src_path = src_path
    self.dest_path = dest_path
    self.shape = Vec(*shape)
    self.offset = Vec(*offset)

  def execute(self):
    srccv = CloudVolume(self.src_path)
    destcv = CloudVolume(self.dest_path)

    bounds = Bbox(self.offset, self.shape + self.offset)
    bounds = Bbox.clamp(bounds, srccv.bounds)

    remap = self._get_map()
    watershed_data = srccv[bounds.to_slices()]

    # Here's how the remapping works. Numpy has a special
    # indexing that can be used to perform the remap.
    # The remap array is a key:value mapping where the
    # array index is the key and the value is the contents.
    # The watershed_data array contains only data values that
    # are within the length of the remap array.
    #
    # e.g.
    #
    # remap = np.array([1,2,3]) # i.e. 0=>1, 1=>2, 1=>3
    # vals = np.array([0,1,1,1,2,0,2,1,2])
    #
    # remap[vals] # array([1, 2, 2, 2, 3, 1, 3, 2, 3])

    image = remap[watershed_data]
    downsample_and_upload(image, bounds, destcv, self.shape)

  def _get_map(self):
    file = BytesIO(cache(self, self.map_path))
    remap = np.load(file)
    file.close()
    return remap

class MaskAffinitymapTask(RegisteredTask):
    """
    black out the affinitymap regions according to a mask. The mask could be 
    lower resolution in a higher mip level. The affinitymap correspond with 
    zero intensive voxels in the mask will be blacked out. 
    """
    def __init__(self, aff_input_layer_path, aff_output_layer_path, aff_mip, 
                 mask_layer_path, mask_mip, bounds):
        super().__init__(aff_input_layer_path, aff_output_layer_path, aff_mip, 
                         mask_layer_path, mask_mip, bounds)
        self.aff_input_layer_path = aff_input_layer_path 
        self.aff_output_layer_path = aff_output_layer_path 
        self.aff_mip = aff_mip 
        self.mask_layer_path = mask_layer_path
        self.mask_mip = mask_mip 

        self.aff_slices = bounds.to_slices()

    def execute(self):
        self._read_mask()
        self._read_affinity_map()
        self._mask_affinity_map()
        self._upload_affinity_map()

    def _read_affinity_map(self):
        print("download affinity map chunk...")
        if np.all(self.mask==0):
            print("the mask is all black, fill affinitymap with zeros directly")
            sz = (3,) + tuple(s.stop-s.start for s in self.aff_slices) 
            self.aff = np.zeros( sz, dtype='float32' )
            return 

        vol = CloudVolume(self.aff_input_layer_path, bounded=False, fill_missing=True,
                          progress=True, mip=self.aff_mip)
        # the slices did not contain the channel dimension
        self.aff = vol[self.aff_slices[::-1] + (slice(0,3),)]
        self.aff = np.transpose(self.aff)

    def _read_mask(self):
        print("download mask chunk...")
        vol = CloudVolume(self.mask_layer_path, bounded=False, fill_missing=True,
                          progress=True, mip=self.mask_mip)
        self.xyfactor = 2**(self.mask_mip - self.aff_mip)
        # only scale the indices in XY plane 
        self.mask_slices = tuple(slice(a.start//self.xyfactor, a.stop//self.xyfactor) 
                                 for a in self.aff_slices[1:3])
        self.mask_slices = (self.aff_slices[0],) + self.mask_slices 

        # the slices did not contain the channel dimension
        print("mask slices: {}".format(self.mask_slices))
        self.mask = vol[self.mask_slices[::-1]]
        self.mask = np.transpose(self.mask)
        print("shape of mask: {}".format(self.mask.shape))
        self.mask = np.squeeze(self.mask, axis=0)

    def _mask_affinity_map(self):
        if np.all(self.mask):
            print("mask elements are all positive, return directly")
            #return
        if not np.any(self.aff):
            print("affinitymap all black, return directly")
            return 

        print("perform masking ...")
        # use c++ backend 
        # from datatools import mask_affiniy_map 
        # mask_affinity_map(self.aff, self.mask)
        
        assert np.any(self.mask)
        print("upsampling mask ...")
        # upsampling factor in XY plane 
        mask = np.zeros(self.aff.shape[1:], dtype=self.mask.dtype)
        for offset in np.ndindex((self.xyfactor, self.xyfactor)):
            mask[:, np.s_[offset[0]::self.xyfactor], np.s_[offset[1]::self.xyfactor]] = self.mask 

        assert mask.shape == self.aff.shape[1:]
        assert np.any(self.mask)
        np.multiply(self.aff[0,:,:,:], mask, out=self.aff[0,:,:,:])
        np.multiply(self.aff[1,:,:,:], mask, out=self.aff[1,:,:,:])
        np.multiply(self.aff[2,:,:,:], mask, out=self.aff[2,:,:,:])
        assert np.any(self.aff)

    def _upload_affinity_map(self):
        print("upload affinity map chunk...")
        print("output path: {}".format(self.aff_output_layer_path))
        vol = CloudVolume(self.aff_output_layer_path, compress='gzip', 
                          fill_missing=True, bounded=False, autocrop=True, 
                          mip=self.aff_mip, progress=True)
        self.aff = np.transpose(self.aff)
        vol[self.aff_slices[::-1]+(slice(0,3),)] = self.aff 


class InferenceTask(RegisteredTask):
    """
    run inference like ChunkFlow.jl
    1. cutout image using cloudvolume
    2. run inference
    3. crop the margin to make the output aligned with cloud storage backend
    4. upload to cloud storage using cloudvolume

    Note that I always use z,y,x in python, but cloudvolume use x,y,z for indexing.
    So I always do a reverse of slices before indexing.
    """
    def __init__(self, image_layer_path, convnet_path, mask_layer_path, output_layer_path,
            output_offset, output_shape, patch_size, patch_overlap,
            cropping_margin_size, output_key='output', num_output_channels=3, 
                 image_mip=1, output_mip=1, mask_mip=3):
        
        super().__init__(image_layer_path, convnet_path, mask_layer_path, output_layer_path,
                output_offset, output_shape, patch_size, patch_overlap, 
                cropping_margin_size, output_key, num_output_channels, 
                image_mip, output_mip, mask_mip)
        
        output_shape = Vec(*output_shape)
        output_offset = Vec(*output_offset)
        self.image_layer_path = image_layer_path
        self.convnet_path = convnet_path
        self.mask_layer_path = mask_layer_path 
        self.output_layer_path = output_layer_path
        self.output_bounds = Bbox(output_offset, output_shape + output_offset)
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.cropping_margin_size = cropping_margin_size
        self.output_key = output_key
        self.num_output_channels = num_output_channels
        self.image_mip = image_mip
        self.output_mip = output_mip
        self.mask_mip = mask_mip 
    
    def execute(self):
        self._read_mask()
        # if the mask is black, no need to run inference 
        if np.all(self.mask == 0):
            return 
        self._read_image()
        self._inference()
        self._crop()
        if self.mask: 
            self._mask_output()

        self._upload_output()

    def _read_mask(self):
        if self.mask_layer_path is None or not self.mask_layer_path: 
            print('no mask layer path defined')
            self.mask = None 
            return 
        print("download mask chunk...")
        vol = CloudVolume(self.mask_layer_path, bounded=False, fill_missing=True,
                          progress=True, mip=self.mask_mip)
        self.xyfactor = 2**(self.mask_mip - self.output_mip)
        # only scale the indices in XY plane 
        self.mask_slices = tuple(slice(a.start//self.xyfactor, a.stop//self.xyfactor) 
                                 for a in self.output_bounds.to_slices()[1:3])
        self.mask_slices = (self.output_bounds.to_slices()[0],) + self.mask_slices 

        # the slices did not contain the channel dimension
        print("mask slices: {}".format(self.mask_slices))
        self.mask = vol[self.mask_slices[::-1]]
        self.mask = np.transpose(self.mask)
        print("shape of mask: {}".format(self.mask.shape))
        self.mask = np.squeeze(self.mask, axis=0)

    def _mask_output(self):
        if np.all(self.mask):
            print("mask elements are all positive, return directly")
            #return
        if not np.any(self.output):
            print("output volume is all black, return directly")
            return 

        print("perform masking ...")
        # use c++ backend 
        # from datatools import mask_affiniy_map 
        # mask_affinity_map(self.aff, self.mask)
        
        assert np.any(self.mask)
        print("upsampling mask ...")
        # upsampling factor in XY plane 
        mask = np.zeros(self.output.shape[1:], dtype=self.mask.dtype)
        for offset in np.ndindex((self.xyfactor, self.xyfactor)):
            mask[:, np.s_[offset[0]::self.xyfactor], np.s_[offset[1]::self.xyfactor]] = self.mask 

        assert mask.shape == self.output.shape[1:]
        assert np.any(self.mask)
        np.multiply(self.output[0,:,:,:], mask, out=self.output[0,:,:,:])
        np.multiply(self.output[1,:,:,:], mask, out=self.output[1,:,:,:])
        np.multiply(self.output[2,:,:,:], mask, out=self.output[2,:,:,:])
        assert np.any(self.output)

    def _read_image(self):
        self.vol = CloudVolume(self.image_layer_path, bounded=False, fill_missing=False,
                               progress=True, mip=self.image_mip, parallel=True)
        output_slices = self.output_bounds.to_slices()
        self.input_slices = tuple(slice(s.start - m, s.stop + m) for s, m in
                                  zip(output_slices, self.cropping_margin_size))
        # always reverse the indexes since cloudvolume use x,y,z indexing
        self.image = self.vol[self.input_slices[::-1]]
        # the cutout is fortran ordered, so need to transpose and make it C order
        self.image = np.transpose(self.image)
        self.image = np.ascontiguousarray(self.image)
        assert self.image.shape[0] == 1
        self.image = np.squeeze(self.image, axis=0)

    def _inference(self):
        # prepare for inference
        from chunkflow.block_inference_engine import BlockInferenceEngine
        from chunkflow.frameworks.pznet_patch_inference_engine import PZNetPatchInferenceEngine
        patch_engine = PZNetPatchInferenceEngine(self.convnet_path)
        self.block_inference_engine = BlockInferenceEngine(
            patch_inference_engine=patch_engine,
            patch_size=self.patch_size,
            overlap=self.patch_overlap,
            output_key=self.output_key,
            output_channels=self.num_output_channels)


        # inference engine input is a OffsetArray rather than normal numpy array
        # it is actually a numpy array with global offset
        from chunkflow.offset_array import OffsetArray

        input_offset = tuple(s.start for s in self.input_slices)
        input_chunk = OffsetArray(self.image, global_offset=input_offset)
        self.output = self.block_inference_engine(input_chunk)

    def _crop(self):
        self.output = self.output[:,
                                  self.cropping_margin_size[0] : -self.cropping_margin_size[0],
                                  self.cropping_margin_size[1] : -self.cropping_margin_size[1],
                                  self.cropping_margin_size[2] : -self.cropping_margin_size[2]]

    def _upload_output(self):
        vol = CloudVolume(self.output_layer_path, compress='gzip', fill_missing=True,
                          bounded=False, autocrop=True, mip=self.image_mip, progress=True)
        output_slices = self.output_bounds.to_slices()
        self.output = np.transpose(self.output)
        vol[output_slices[::-1]+(slice(0,self.output.shape[-1]),)] = self.output

