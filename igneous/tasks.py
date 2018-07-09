from __future__ import print_function

from collections import defaultdict

try:
  from StringIO import cStringIO as BytesIO
except ImportError:
  from io import BytesIO

import json
import math
import os
import random
import re
# from tempfile import NamedTemporaryFile  # used by BigArrayTask

# from backports import lzma               # used by HyperSquareTask
# import blosc                             # used by BigArrayTask
# import h5py                              # used by BigArrayTask

import numpy as np
from tqdm import tqdm

from cloudvolume import CloudVolume, Storage
from cloudvolume.lib import min2, Vec, Bbox, mkdir
from taskqueue import RegisteredTask

from igneous import chunks, downsample, downsample_scales
from igneous import Mesher  # broken out for ease of commenting out


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
    toidx = { 'x': 0, 'y': 1, 'z': 2 }
    preserved_idx = toidx[axis]
    underlying_shape[preserved_idx] = float('inf')

    # Need to use ds_shape here. Using image bounds means truncated
    # edges won't generate as many mip levels
    fullscales = downsample_scales.compute_plane_downsampling_scales(
      size=ds_shape,
      preserve_axis=axis,
      max_downsampled_size=int(min(*underlying_shape)),
  )
  factors = downsample.scale_series_to_downsample_factors(fullscales)

  if len(factors) == 0:
    print("No factors generated. Image Shape: {}, Downsample Shape: {}, Volume Shape: {}, Bounds: {}".format(
        image.shape, ds_shape, vol.volume_size, bounds)
    )

  downsamplefn = downsample.method(vol.layer_type, sparse=sparse)

  vol.mip = mip
  if not skip_first:
    vol[bounds.to_slices()] = image

  new_bounds = bounds.clone()

  for factor3 in factors:
    vol.mip += 1
    image = downsamplefn(image, factor3)
    new_bounds //= factor3
    new_bounds.maxpt = new_bounds.minpt + Vec(*image.shape[:3])
    vol[new_bounds.to_slices()] = image


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


class PrintTask(RegisteredTask):
  """For testing the task_execution.py script."""

  def __init__(self, index):
    super(PrintTask, self).__init__(index)
    self.index = index

  def execute(self):
    print(self.index)


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

  def __init__(self, layer_path, shape, offset):
    super(DeleteTask, self).__init__(layer_path, shape, offset)
    self.layer_path = layer_path
    self.shape = Vec(*shape)
    self.offset = Vec(*offset)

  def execute(self):
    vol = CloudVolume(self.layer_path)

    highres_bbox = Bbox(self.offset, self.offset + self.shape)
    for mip in vol.available_mips:
      vol.mip = mip
      slices = vol.slices_from_global_coords(highres_bbox.to_slices())
      bbox = Bbox.from_slices(slices).round_to_chunk_size(
          vol.underlying, offset=vol.bounds.minpt)
      vol.delete(bbox)


class DownsampleTask(RegisteredTask):
  def __init__(self, 
    layer_path, mip, shape, offset, 
    fill_missing=False, axis='z', sparse=False):

    super(DownsampleTask, self).__init__(
      layer_path, mip, shape, offset, 
      fill_missing, axis, sparse
    )
    self.layer_path = layer_path
    self.mip = mip
    self.shape = Vec(*shape)
    self.offset = Vec(*offset)
    self.fill_missing = fill_missing
    self.axis = axis
    self.sparse = sparse

  def execute(self):
    vol = CloudVolume(self.layer_path, self.mip,
                      fill_missing=self.fill_missing)
    bounds = Bbox(self.offset, self.shape + self.offset)
    bounds = Bbox.clamp(bounds, vol.bounds)
    image = vol[ bounds.to_slices() ]
    downsample_and_upload(
      image, bounds, vol, 
      self.shape, self.mip, self.axis, 
      skip_first=True, sparse=self.sparse
    )

class QuantizeAffinitiesTask(RegisteredTask):
  def __init__(self, source_layer_path, dest_layer_path, shape, offset, fill_missing=False):
    super(QuantizeAffinitiesTask, self).__init__(
        source_layer_path, dest_layer_path, shape, offset, fill_missing)
    self.source_layer_path = source_layer_path
    self.dest_layer_path = dest_layer_path
    self.shape = Vec(*shape)
    self.offset = Vec(*offset)
    self.fill_missing = fill_missing

  def execute(self):
    srcvol = CloudVolume(self.source_layer_path, mip=0,
                         fill_missing=self.fill_missing)

    bounds = Bbox(self.offset, self.shape + self.offset)
    bounds = Bbox.clamp(bounds, srcvol.bounds)

    image = srcvol[bounds.to_slices()][:, :, :, :1]  # only use x affinity
    image = (image * 255.0).astype(np.uint8)

    destvol = CloudVolume(self.dest_layer_path, mip=0)
    downsample_and_upload(image, bounds, destvol, self.shape, mip=0, axis='z')


class MeshTask(RegisteredTask):
  def __init__(self, shape, offset, layer_path, **kwargs):
    super(MeshTask, self).__init__(shape, offset, layer_path, **kwargs)
    self.shape = Vec(*shape)
    self.offset = Vec(*offset)
    self.layer_path = layer_path
    self.options = {
        'lod': kwargs.get('lod', 0),
        'mip': kwargs.get('mip', 0),
        'simplification_factor': kwargs.get('simplification_factor', 100),
        'max_simplification_error': kwargs.get('max_simplification_error', 40),
        'remap_table': kwargs.get('remap_table', None),
        'generate_manifests': kwargs.get('generate_manifests', False),
        'low_padding': kwargs.get('low_padding', 1),
        'high_padding': kwargs.get('high_padding', 1)
    }

  def execute(self):
    self._volume = CloudVolume(
        self.layer_path, self.options['mip'], bounded=False)
    self._bounds = Bbox(self.offset, self.shape + self.offset)
    self._bounds = Bbox.clamp(self._bounds, self._volume.bounds)

    self._mesher = Mesher(self._volume.resolution)

    # Marching cubes loves its 1vx overlaps.
    # This avoids lines appearing between
    # adjacent chunks.
    data_bounds = self._bounds.clone()
    data_bounds.minpt -= self.options['low_padding']
    data_bounds.maxpt += self.options['high_padding']

    self._mesh_dir = None
    if 'meshing' in self._volume.info:
      self._mesh_dir = self._volume.info['meshing']
    elif 'mesh' in self._volume.info:
      self._mesh_dir = self._volume.info['mesh']

    if not self._mesh_dir:
      raise ValueError("The mesh destination is not present in the info file.")

    # chunk_position includes the overlap specified by low_padding/high_padding
    self._data = self._volume[data_bounds.to_slices()]
    self._remap()
    self._compute_meshes()

  def _remap(self):
    if self.options['remap_table'] is not None:
      actual_remap = {
          int(k): int(v) for k, v in self.options['remap_table'].items()
      }

      self._remap_list = [0] + list(actual_remap.values())
      enumerated_remap = {int(v): i for i, v in enumerate(self._remap_list)}

      do_remap = lambda x: enumerated_remap[actual_remap.get(x, 0)]
      self._data = np.vectorize(do_remap)(self._data)

  def _compute_meshes(self):
    with Storage(self.layer_path) as storage:
      data = self._data[:, :, :, 0].T
      self._mesher.mesh(data)
      for obj_id in self._mesher.ids():
        if self.options['remap_table'] is None:
          remapped_id = obj_id
        else:
          remapped_id = self._remap_list[obj_id]

        storage.put_file(
            file_path='{}/{}:{}:{}'.format(
                self._mesh_dir, remapped_id, self.options['lod'],
                self._bounds.to_filename()
            ),
            content=self._create_mesh(obj_id),
            compress=True,
        )

        if self.options['generate_manifests']:
          fragments = []
          fragments.append('{}:{}:{}'.format(remapped_id, self.options['lod'],
                                             self._bounds.to_filename()))

          storage.put_file(
              file_path='{}/{}:{}'.format(
                  self._mesh_dir, remapped_id, self.options['lod']),
              content=json.dumps({"fragments": fragments}),
              content_type='application/json'
          )

  def _create_mesh(self, obj_id):
    mesh = self._mesher.get_mesh(
        obj_id,
        simplification_factor=self.options['simplification_factor'],
        max_simplification_error=self.options['max_simplification_error']
    )
    vertices = self._update_vertices(
        np.array(mesh['points'], dtype=np.float32))
    vertex_index_format = [
        np.uint32(len(vertices) / 3), # Number of vertices (3 coordinates)
        vertices,
        np.array(mesh['faces'], dtype=np.uint32)
    ]
    return b''.join([array.tobytes() for array in vertex_index_format])

  def _update_vertices(self, points):
    # zi_lib meshing multiplies vertices by 2.0 to avoid working with floats,
    # but we need to recover the exact position for display
    # Note: points are already multiplied by resolution, but missing the offset
    points /= 2.0
    resolution = self._volume.resolution
    xmin, ymin, zmin = self._bounds.minpt
    points[0::3] = points[0::3] + xmin * resolution.x
    points[1::3] = points[1::3] + ymin * resolution.y
    points[2::3] = points[2::3] + zmin * resolution.z
    return points


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

  def __init__(self, layer_path, prefix, lod=0):
    super(MeshManifestTask, self).__init__(layer_path, prefix)
    self.layer_path = layer_path
    self.lod = lod
    self.prefix = prefix

  def execute(self):
    with Storage(self.layer_path) as storage:
      self._info = json.loads(storage.get_file('info').decode('utf8'))

      self.mesh_dir = None
      if 'meshing' in self._info:
        self.mesh_dir = self._info['meshing']
      elif 'mesh' in self._info:
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
    for segid, frags in tqdm(segids.items()):
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

  def __init__(self, src_path, dest_path, shape, offset, mip, clip_fraction, fill_missing, translate):
    super(ContrastNormalizationTask, self).__init__(src_path, dest_path,
                                                    shape, offset, mip, clip_fraction, fill_missing, translate)
    self.src_path = src_path
    self.dest_path = dest_path
    self.shape = Vec(*shape)
    self.offset = Vec(*offset)
    self.fill_missing = fill_missing
    self.translate = Vec(*translate)
    self.mip = int(mip)
    self.clip_fraction = float(clip_fraction)

    assert 0 <= self.clip_fraction <= 1

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
          zlevel, self.clip_fraction, 1 - self.clip_fraction)
      if lower == upper:
        continue
      img = image[:, :, imagez]
      img = (img - float(lower)) * (maxval / (float(upper) - float(lower)))
      image[:, :, imagez] = img

    image = np.round(image)
    image = np.clip(image, 0.0, maxval).astype(destcv.dtype)

    bounds += self.translate
    downsample_and_upload(image, bounds, destcv, self.shape)

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
    levelfilenames = ['levels/{}/{}'.format(self.mip, z)
                      for z in range(bounds.minpt.z, bounds.maxpt.z + 1)]
    with Storage(self.src_path) as stor:
      levels = stor.get_files(levelfilenames)

    errors = [level['filename']
              for level in levels if level['content'] == None]
    if len(errors):
      raise Exception(", ".join(
          errors) + " were not defined. Did you run a LuminanceLevelsTask for these slices?")

    levels = [(
        int(os.path.basename(item['filename'])),
        json.loads(item['content'].decode('utf-8'))
    ) for item in levels]
    levels.sort(key=lambda x: x[0])
    levels = [x[1] for x in levels]
    return [np.array(x['levels'], dtype=np.uint64) for x in levels]


class LuminanceLevelsTask(RegisteredTask):
  """Generate a frequency count of luminance values by random sampling. Output to $PATH/levels/$MIP/$Z"""

  def __init__(self, src_path, shape, offset, coverage_factor, mip):
    super(LuminanceLevelsTask, self).__init__(
        src_path, shape, offset, coverage_factor, mip)
    self.src_path = src_path
    self.shape = Vec(*shape)
    self.offset = Vec(*offset)
    self.coverage_factor = coverage_factor
    self.mip = int(mip)

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

    levels_path = os.path.join(self.src_path, 'levels')
    with Storage(levels_path, n_threads=0) as stor:
      stor.put_json(
          file_path="{}/{}".format(self.mip, self.offset.z),
          content=output,
          cache_control='no-cache'
      )

  def select_bounding_boxes(self, dataset_bounds):
    # Sample 1024x1024x1 patches until coverage factor is
    # satisfied. Ensure the patches are non-overlapping and
    # random.
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
  def __init__(self, src_path, dest_path, shape, offset, fill_missing, translate):
    super(TransferTask, self).__init__(
        src_path, dest_path, shape, offset, fill_missing, translate)
    self.src_path = src_path
    self.dest_path = dest_path
    self.shape = Vec(*shape)
    self.offset = Vec(*offset)
    self.fill_missing = fill_missing
    self.translate = Vec(*translate)

  def execute(self):
    srccv = CloudVolume(self.src_path, fill_missing=self.fill_missing)
    destcv = CloudVolume(self.dest_path, fill_missing=self.fill_missing)

    bounds = Bbox(self.offset, self.shape + self.offset)
    bounds = Bbox.clamp(bounds, srccv.bounds)
    image = srccv[bounds.to_slices()]
    bounds += self.translate
    downsample_and_upload(image, bounds, destcv, self.shape)


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
