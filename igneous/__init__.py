from igneous._mesher import Mesher8, Mesher16, Mesher32, Mesher64
from taskqueue import MockTaskQueue, TaskQueue, RegisteredTask
from .tasks import *
# from tasks_watershed import WatershedTask
from cloudvolume import CloudVolume, Storage, EmptyVolumeException

def Precomputed(storage, scale_idx=0, fill=False):
  """Shim to provide backwards compatibility with Precomputed."""
  return CloudVolume(storage.layer_path, mip=scale_idx, fill_missing=fill)