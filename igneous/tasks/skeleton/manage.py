from typing import Optional

from cloudfiles import CloudFiles, CloudFile

from taskqueue import queueable

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
