from functools import reduce
import operator

from cloudvolume import CloudVolume
from cloudvolume.lib import Vec, xyzrange

from igneous.tasks import (
  WatershedRemapTask, 
  MaskAffinitymapTask, InferenceTask
)
from .common import (
  operator_contact, FinelyDividedTaskIterator, 
  get_bounds, num_tasks
)

__all__ = [
  "create_inference_tasks",
  "create_watershed_remap_tasks",
  "create_mask_affinity_map_tasks",
]

def create_inference_tasks(
    image_layer_path, convnet_path, 
    mask_layer_path, output_layer_path, output_block_start, output_block_size, 
    grid_size, patch_size, patch_overlap, cropping_margin_size,
    output_key='output', num_output_channels=3, 
    image_mip=1, output_mip=1, mask_mip=3
  ):
    """
    convnet inference block by block. The block coordinates should be aligned with 
    cloud storage. 
    """
    class InferenceTaskIterator():
      def __len__(self):
        return int(reduce(operator.mul, grid_size))
      def __iter__(self):
        for x, y, z in xyzrange(grid_size):
          output_offset = tuple(s+x*b for (s, x, b) in 
                                zip(output_block_start, (z, y, x), 
                                    output_block_size))
          yield InferenceTask(
              image_layer_path=image_layer_path,
              convnet_path=convnet_path,
              mask_layer_path=mask_layer_path,
              output_layer_path=output_layer_path,
              output_offset=output_offset,
              output_shape=output_block_size,
              patch_size=patch_size, 
              patch_overlap=patch_overlap,
              cropping_margin_size=cropping_margin_size,
              output_key=output_key,
              num_output_channels=num_output_channels,
              image_mip=image_mip,
              output_mip=output_mip,
              mask_mip=mask_mip
          )

        vol = CloudVolume(output_layer_path, mip=output_mip)
        vol.provenance.processing.append({
            'method': {
                'task': 'InferenceTask',
                'image_layer_path': image_layer_path,
                'convnet_path': convnet_path,
                'mask_layer_path': mask_layer_path,
                'output_layer_path': output_layer_path,
                'output_offset': output_offset,
                'output_shape': output_block_size,
                'patch_size': patch_size,
                'patch_overlap': patch_overlap,
                'cropping_margin_size': cropping_margin_size,
                'output_key': output_key,
                'num_output_channels': num_output_channels,
                'image_mip': image_mip,
                'output_mip': output_mip,
                'mask_mip': mask_mip,
            },
            'by': OPERATOR_CONTACT,
            'date': strftime('%Y-%m-%d %H:%M %Z'),
        })
        vol.commit_provenance()

    return InferenceTaskIterator()

def create_watershed_remap_tasks(
    map_path, src_layer_path, dest_layer_path, 
    shape=Vec(2048, 2048, 64)
  ):
  shape = Vec(*shape)
  vol = CloudVolume(src_layer_path)

  downsample_scales.create_downsample_scales(dest_layer_path, mip=0, ds_shape=shape)

  class WatershedRemapTaskIterator(FinelyDividedTaskIterator):
    def task(self, shape, offset):
      return WatershedRemapTask(
        map_path=map_path,
        src_path=src_layer_path,
        dest_path=dest_layer_path,
        shape=shape.clone(),
        offset=offset.clone(),
      )
    
    def on_finish(self):
      dvol = CloudVolume(dest_layer_path)
      dvol.provenance.processing.append({
        'method': {
          'task': 'WatershedRemapTask',
          'src': src_layer_path,
          'dest': dest_layer_path,
          'remap_file': map_path,
          'shape': list(shape),
        },
        'by': OPERATOR_CONTACT,
        'date': strftime('%Y-%m-%d %H:%M %Z'),
      }) 
      dvol.commit_provenance()

  return WatershedRemapTaskIterator(vol.bounds, shape)

# def create_bigarray_task(cloudpath):
#   """
#   Creates one task for each bigarray chunk present in the bigarray folder.
#   These tasks will convert the bigarray chunks into chunks that ingest tasks are able to understand.
#   """
#   class BigArrayTaskIterator():
#     def __iter__(self):    
#       with Storage(cloudpath) as storage:
#         for filename in storage.list_blobs(prefix='bigarray/'):
#           yield BigArrayTask(
#             chunk_path=storage.get_path_to_file('bigarray/'+filename),
#             chunk_encoding='npz', # npz_uint8 to convert affinites float32 affinties to uint8
#             version='{}/{}'.format(storage._path.dataset_name, storage._path.layer_name)
#           )
#   return BigArrayTaskIterator()

def create_mask_affinity_map_tasks(
    aff_input_layer_path, aff_output_layer_path, 
    aff_mip, mask_layer_path, mask_mip, output_block_start, 
    output_block_size, grid_size 
  ):
    """
    affinity map masking block by block. The block coordinates should be aligned with 
    cloud storage. 
    """

    class MaskAffinityMapTaskIterator():
      def __len__(self):
        return int(reduce(operator.mul, grid_size))
      def __iter__(self):
        for x, y, z in xyzrange(grid_size):
          output_bounds = Bbox.from_slices(tuple(slice(s+x*b, s+x*b+b)
                  for (s, x, b) in zip(output_block_start, (z, y, x), output_block_size)))
          yield MaskAffinitymapTask(
              aff_input_layer_path=aff_input_layer_path,
              aff_output_layer_path=aff_output_layer_path,
              aff_mip=aff_mip, 
              mask_layer_path=mask_layer_path,
              mask_mip=mask_mip,
              output_bounds=output_bounds,
          )

        vol = CloudVolume(output_layer_path, mip=aff_mip)
        vol.provenance.processing.append({
            'method': {
                'task': 'InferenceTask',
                'aff_input_layer_path': aff_input_layer_path,
                'aff_output_layer_path': aff_output_layer_path,
                'aff_mip': aff_mip,
                'mask_layer_path': mask_layer_path,
                'mask_mip': mask_mip,
                'output_block_start': output_block_start,
                'output_block_size': output_block_size, 
                'grid_size': grid_size,
            },
            'by': operator_contact(),
            'date': strftime('%Y-%m-%d %H:%M %Z'),
        })
        vol.commit_provenance()

    return MaskAffinityMapTaskIterator()