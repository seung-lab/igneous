import numpy as np

from igneous import downsample_scales


def test_plane_scales_xy():
  scales = downsample_scales.compute_plane_downsampling_scales( 
    (2048, 2048, 512), max_downsampled_size=128
  )

  assert len(scales) == 5
  assert scales[0] == (1,1,1)
  assert scales[1] == (2,2,1)
  assert scales[2] == (4,4,1)
  assert scales[3] == (8,8,1)
  assert scales[4] == (16,16,1)

  scales = downsample_scales.compute_plane_downsampling_scales( 
    (357, 2048, 512), max_downsampled_size=128
  )

  assert len(scales) == 2
  assert scales[0] == (1,1,1)
  assert scales[1] == (2,2,1)

  scales = downsample_scales.compute_plane_downsampling_scales( 
    (0, 2048, 512), max_downsampled_size=128
  )

  assert len(scales) == 1
  assert scales[0] == (1,1,1)


def test_plane_scales_yz():
  scales = downsample_scales.compute_plane_downsampling_scales( 
    (512, 2048, 2048), max_downsampled_size=128, preserve_axis='x'
  )

  assert len(scales) == 5
  assert scales[0] == (1,1,1)
  assert scales[1] == (1,2,2)
  assert scales[2] == (1,4,4)
  assert scales[3] == (1,8,8)
  assert scales[4] == (1,16,16)


  scales = downsample_scales.compute_plane_downsampling_scales( 
    (64, 2048, 2048), max_downsampled_size=128, preserve_axis='x'
  )

  assert len(scales) == 5
  assert scales[0] == (1,1,1)
  assert scales[1] == (1,2,2)
  assert scales[2] == (1,4,4)
  assert scales[3] == (1,8,8)
  assert scales[4] == (1,16,16)


def test_downsample_shape_from_memory_target():
  try:
    downsample_scales.downsample_shape_from_memory_target(1, 0, 1, 1, (2,2,1), 16)
    assert False
  except ValueError:
    pass

  shape = downsample_scales.downsample_shape_from_memory_target(1, 1, 1, 1, (1,1,1), 16)
  assert np.all(shape == (4,4,1))

  shape = downsample_scales.downsample_shape_from_memory_target(2, 1, 1, 1, (1,1,1), 16)
  assert np.all(shape == (2,2,1))

  shape = downsample_scales.downsample_shape_from_memory_target(1, 2, 1, 1, (1,1,1), 16)
  assert np.all(shape == (4,4,1))

  shape = downsample_scales.downsample_shape_from_memory_target(1, 256, 256, 64, (1,1,1), 20e6)
  assert np.all(shape == (512,512,64))

  shape = downsample_scales.downsample_shape_from_memory_target(8, 256, 256, 64, (1,1,1), 35e6)
  assert np.all(shape == (256,256,64))

  shape = downsample_scales.downsample_shape_from_memory_target(8, 256, 256, 64, (1,1,1), 3.5e9)
  assert np.all(shape == (2560,2560,64))

  shape = downsample_scales.downsample_shape_from_memory_target(1, 1, 1, 1, (2,2,1), 16)
  assert np.all(shape == (4,4,1))

  shape = downsample_scales.downsample_shape_from_memory_target(1, 1, 1, 1, (2,2,2), 64)
  assert np.all(shape == (4,4,4))

  try:
    downsample_scales.downsample_shape_from_memory_target(1, 1, 1, 1, (2,2,1), 0)
    assert False
  except ValueError:
    pass

  try:
    downsample_scales.downsample_shape_from_memory_target(8, 128, 128, 64, (2,2,1), 100)
    assert False
  except ValueError:
    pass

  shape = downsample_scales.downsample_shape_from_memory_target(8, 128, 128, 64, (2,2,1), 3.5e9)
  assert np.all(shape == (2048,2048,64))

  shape = downsample_scales.downsample_shape_from_memory_target(1, 128, 128, 64, (2,2,1), 3.5e9)
  assert np.all(shape == (4096,4096,64))

  shape = downsample_scales.downsample_shape_from_memory_target(8, 128, 128, 64, (2,2,2), 3.5e9)
  assert np.all(shape == (512,512,256))

  shape = downsample_scales.downsample_shape_from_memory_target(4, 100, 100, 50, (2,2,1), 3.5e9)
  assert np.all(shape == (3200,3200,50))





