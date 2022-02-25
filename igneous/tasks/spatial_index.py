from typing import Optional, Tuple, Union

import numpy as np
import scipy.ndimage

from cloudvolume import CloudVolume, Bbox, Vec
from cloudfiles import CloudFiles
from taskqueue import queueable

def find_objects(labels):
  """  
  scipy.ndimage.find_objects performs about 7-8x faster on C 
  ordered arrays, so we just do it that way and convert
  the results if it's in F order.
  """
  if labels.flags.c_contiguous:
    return scipy.ndimage.find_objects(labels)
  else:
    all_slices = scipy.ndimage.find_objects(labels.T)
    return [ (slcs and slcs[::-1]) for slcs in all_slices ]

@queueable
def SpatialIndexTask(
  cloudpath:str, 
  shape:Tuple[int,int,int], 
  offset:Tuple[int,int,int], 
  subdir:str,
  precision:int,
  mip:int = 0, 
  fill_missing:bool=False, 
  compress:Optional[Union[str,bool]] = 'gzip', 
) -> None:
  """
  The main way to add a spatial index is to use the MeshTask or SkeletonTask,
  but old datasets or broken datasets may need it to be 
  reconstituted. An alternative use is create the spatial index
  over a different area size than the mesh or skeleton task.
  """
  cv = CloudVolume(
    cloudpath, mip=mip, 
    bounded=False, fill_missing=fill_missing
  )
  cf = CloudFiles(cloudpath)

  bounds = Bbox(Vec(*offset), Vec(*shape) + Vec(*offset))
  bounds = Bbox.clamp(bounds, cv.bounds)

  data_bounds = bounds.clone()
  data_bounds.maxpt += 1 # match typical Marching Cubes overlap

  resolution = cv.resolution 

  # remap: old img -> img
  img, remap = cv.download(data_bounds, renumber=True)
  img = img[...,0]
  slcs = find_objects(img)
  del img
  reverse_map = { v:k for k,v in remap.items() } # img -> old img

  bboxes = {}
  for label, slc in enumerate(slcs):
    if slc is None:
      continue
    obj_bounds = Bbox.from_slices(slc)
    obj_bounds += Vec(*offset)
    obj_bounds *= Vec(*resolution, dtype=np.float32)
    bboxes[str(reverse_map[label+1])] = \
      obj_bounds.astype(resolution.dtype).to_list()

  bounds = bounds.astype(resolution.dtype) * resolution
  cf.put_json(
    cf.join(subdir, f"{bounds.to_filename(precision)}.spatial"),
    bboxes,
    compress=compress,
    cache_control=False,
  )