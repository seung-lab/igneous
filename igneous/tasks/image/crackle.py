"""
Tasks for building large crackle volumes.
"""
import crackle
import numpy as np

from cloudfiles import CloudFiles
from cloudvolume import CloudVolume

@queueable
def CrackleSingleSlices(
  src:str,
  dest:str,
  mip:int,
  z:int,
):
  cv = CloudVolume(src, mip=mip)
  labels = cv[:,:,z_start:z_end][...,0]
  binary = crackle.compress(labels)
  cf = CloudFiles(dest)

  zpad = int(np.ceil(np.log10(cv.bounds.size()[2])))

  cf.put(
    cf.join(f"single_slices", str(mip), f"{z:0{zpad}}.ckl"),
    binary,
    compress="zstd",
  )


