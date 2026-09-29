from typing import Optional, Union

import cloudfiles
from cloudfiles import CloudFiles, CloudFile

from taskqueue import queueable

@queueable
def TransferFilesTask(
    src:str,
    paths:list[str],
    dest:str,
    reencode:Optional[Union[str,bool]],
    block_size:int,
    progress:bool = False,
    allow_missing:bool = False,
):
    """
    Transfer a set of file paths from source to destination.
    """
    # paths = None would cause massive write amplification as every worker
    # attempts to copy the entire directory tree.
    if paths is None:
        raise ValueError(f"paths is None. You must provide a set of paths.")

    return CloudFiles(src).transfer_to(
        dest, 
        paths=paths,
        reencode=reencode,
        block_size=block_size,
        progress=progress,
        allow_missing=allow_missing,
    )

