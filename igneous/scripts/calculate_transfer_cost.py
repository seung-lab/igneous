import numpy as np
from cloudvolume import CloudVolume
from cloudvolume.lib import Vec, Bbox
from collections import defaultdict

COMPRESSION_RATIO = 1.6
COST_PER_STORAGE_COST = {
  "STANDARD": .05,
  "COLDLINE": .1
}

def calculate_num_chunks_in_bounds(bounds, chunk_size):
  chunks_per_dim = (bounds.maxpt - bounds.minpt) / chunk_size
  return np.prod(np.ceil(np.array(chunks_per_dim)))


def to_file_size(size_in_bytes):
  units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
  cur_unit = 0
  cur_magnitude = size_in_bytes
  while cur_magnitude > 1000:
    cur_magnitude = cur_magnitude / 1024
    cur_unit = cur_unit + 1
    if cur_unit == len(units) - 1:
      break
  return cur_magnitude, units[cur_unit]

def calculate_transfer_cost(src_layer_path, dest_layer_path, 
    chunk_size=None, shape=Vec(2048*8, 2048*8, 64),
    bounds=None, mip=0,
    skip_downsamples=False,
    skip_first=False, skip_ds_mips=[],
    mip_to_storage_class={},
    use_dvol_info=False,
    is_inter_region=False):
      shape = Vec(*shape)
      if use_dvol_info:
        vol = CloudVolume(dest_layer_path, mip=mip)
        dvol = vol
      else:
        vol = CloudVolume(src_layer_path, mip=mip)
        dvol = CloudVolume(dest_layer_path, mip=mip)

      if bounds is None:
        bounds = vol.bounds.clone()

      num_chunks_per_storage_class = defaultdict(int)
      amount_data_per_storage_class = defaultdict(int)
      num_mips = len(vol.info['scales'])

      for cur_mip in range(mip, num_mips):
        if cur_mip in skip_ds_mips or (cur_mip == mip and skip_first):
          continue
        cur_bounds = vol.bbox_to_mip(bounds, mip=0, to_mip=mip)
        cur_bounds = Bbox.clamp(cur_bounds, dvol.bounds)
        cur_shape = vol.point_to_mip(shape, mip, cur_mip)
        cur_cs = vol.mip_chunk_size(cur_mip)
        if cur_cs[0] > cur_shape[0] or cur_cs[1] > cur_shape[1] or cur_cs[2] > cur_shape[2]:
          break
        num_chunks = calculate_num_chunks_in_bounds(cur_bounds, cur_cs)
        num_data = num_chunks * np.prod(cur_cs) / COMPRESSION_RATIO
        storage_class = mip_to_storage_class[mip] if mip in mip_to_storage_class else 'STANDARD'
        num_chunks_per_storage_class[storage_class] += num_chunks
        amount_data_per_storage_class[storage_class] += num_data

      cost = 0
      print('Class A Request Cost\n')
      for sc, num_chunks in num_chunks_per_storage_class.items():
        cost_for_sc = (num_chunks / 10000) * COST_PER_STORAGE_COST[sc]
        cost = cost + cost_for_sc
        print(f'{num_chunks:,} chunks created for storage class {sc}')
        print(f'cost for storage class {sc} = ${cost_for_sc:,.2f}')

      print(f'Total Class A Cost = ${cost:,.2f}\n')
      for sc, amount_data in amount_data_per_storage_class.items():
          file_size, unit = to_file_size(amount_data)
          print(f'Amount of data for storage class {sc}: {file_size:,.2f} {unit}\n')

      if is_inter_region:
        total_data = 0
        for sc, amount_data in amount_data_per_storage_class.items():
          total_data = total_data + amount_data
        egress_cost = .01 * (total_data / (1024 ** 3))
        print(f'Inter-region egress cost = ${egress_cost:,.2f}')
