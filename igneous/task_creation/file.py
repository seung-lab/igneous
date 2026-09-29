from typing import Optional, Union

from functools import partial

from cloudfiles import CloudFiles
from cloudfiles.lib import sip

from igneous.tasks import TransferFilesTask

def create_directory_transfer_tasks(
  src:str,
  dest:str,
  reencode:Optional[Union[str,bool]] = None,
  files_per_task:int = 1000,
  transfer_block_size:int = 64,
  progress:bool = False,
  allow_missing:bool = False,
):
  cf = CloudFiles(src)
  
  def FileTransferTaskIterator():
    for paths in sip(cf.list(), files_per_task):
      yield partial(TransferFilesTask, 
        src=src,
        dest=dest,
        paths=paths,
        reencode=reencode,
        block_size=transfer_block_size,
        progress=progress,
        allow_missing=allow_missing,
      )

  return FileTransferTaskIterator()


__all__  = [
  "create_directory_transfer_tasks",
]