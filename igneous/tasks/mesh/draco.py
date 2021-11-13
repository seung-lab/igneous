from typing import Tuple, Dict, Any

import numpy as np

Triple = Tuple[int,int,int]

def draco_encoding_settings(
  shape:Triple, offset:Triple, resolution:Triple,
  compression_level:int, create_metadata:bool,
  uses_new_draco_bin_size=False
) -> Dict[str,Any]:
  chunk_offset_nm = offset * resolution
  
  min_quantization_range = max(shape * resolution)
  if uses_new_draco_bin_size:
    max_draco_bin_size = np.floor(min(resolution) / 2)
  else:
    max_draco_bin_size = np.floor(min(resolution) / np.sqrt(2))

  (
    draco_quantization_bits,
    draco_quantization_range,
    draco_bin_size,
  ) = calculate_draco_quantization_bits_and_range(
    min_quantization_range, max_draco_bin_size
  )
  draco_quantization_origin = chunk_offset_nm - (chunk_offset_nm % draco_bin_size)
  return {
    "quantization_bits": draco_quantization_bits,
    "compression_level": compression_level,
    "quantization_range": draco_quantization_range,
    "quantization_origin": draco_quantization_origin,
    "create_metadata": create_metadata,
  }

def calculate_draco_quantization_bits_and_range(
  min_quantization_range:int, 
  max_draco_bin_size:int, 
  draco_quantization_bits=None
) -> tuple:
  """
  Computes draco parameters for integer quantizing the meshes.
  """
  if draco_quantization_bits is None:
    draco_quantization_bits = np.ceil(
      np.log2(min_quantization_range / max_draco_bin_size + 1)
    )
  num_draco_bins = 2 ** draco_quantization_bits - 1
  draco_bin_size = np.ceil(min_quantization_range / num_draco_bins)
  draco_quantization_range = draco_bin_size * num_draco_bins
  if draco_quantization_range < min_quantization_range + draco_bin_size:
    if draco_bin_size == max_draco_bin_size:
      return calculate_draco_quantization_bits_and_range(
        min_quantization_range, max_draco_bin_size, draco_quantization_bits + 1
      )
    else:
      draco_bin_size = draco_bin_size + 1
      draco_quantization_range = draco_quantization_range + num_draco_bins
  return draco_quantization_bits, draco_quantization_range, draco_bin_size