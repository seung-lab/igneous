from typing import Optional, Sequence, Dict, List

from collections import defaultdict
from functools import reduce, partial
import itertools
import json
import mmap
import os
import pickle
import posixpath
import re
import time

from tqdm import tqdm

import numpy as np

import cloudfiles
from cloudfiles import CloudFiles, CloudFile

import cloudvolume
from cloudvolume import CloudVolume, Skeleton, paths
from cloudvolume.lib import Vec, Bbox, sip, xyzrange
from cloudvolume.datasource.precomputed.sharding import synthesize_shard_files

import cc3d
import crackle
import fastmorph
import fastremap
import gaara

from taskqueue import queueable

def ThinWholeObjectTask(
	src_cloudpath:str,
	dest_cloudpath:str,
	label:int,
	mip:int,
	radius:bool = True,
    cross_sectional_area:bool = False,
    cross_sectional_area_smoothing_window:int = 6,
    timestamp:Optional[int] = None,
    progress:bool = False,
    threads:int = 1,
    bbox:Optional[Bbox] = None,
    dry_run:bool = False,
    format:Literal['ostd', 'swc', 'precomputed'] = 'ostd',
):
	"""
	Compute a skeleton using gaara's Palyági thinning.
	This is a topology preserving transformation.

	Note: requires a mesh 

	src_cloudpath: The image data source.
	dest_cloudpath: where to write the skeletons
	label: the segid of the skeleton to extract
	mip: the image mip level to extract the skeleton from
	radius: compute the distance transform and store the result
		at each vertex
	cross_sectional_area: compute the cross sectional area
		at each vertex, normals are guided by the skeleton direction
	cross_sectional_area_smoothing_window: number of vertices to average
		over to prevent rapid oscillations in direction
	progress: display progress bars
	threads: use this many threads to perform the thinning computation
	bbox: if specified, download the image from this bbox specified
		in voxels at the specified mip level
	timestamp: (graphene only) use this UNIX timestamp to fetch the
		correct segmentation
	"""
	assert label >= 0
	assert threads >= 0
	assert cross_sectional_area_smoothing_window >= 1
	assert format in ["swc", "precomputed", "ostd"]

	cv = CloudVolume(src_cloudpath, mip=mip, agglomerate=True)

	if bbox is None:
		mesh = cv.mesh.get(label)
		bbox = Bbox.from_points(mesh.vertices // cv.resolution)
		bbox.grow(1)
		del mesh

	binary_image = cv.download(bbox, label=label, timestamp=timestamp)

	skeleton, binary_image = gaara.skeletonize(
		binary_image,
		binary_image=True,
		in_place=True,
		anisotropy=cv.resolution,
		radius=radius,
		cross_section=(
			cross_sectional_area_smoothing_window
			if cross_sectional_area else 0
		),
		threads=threads,
	)

	if dry_run:
		return skeleton

	cf = CloudFiles(dest_cloudpath)

	if format == "ostd":
		cf.put(f"{label}.ostd", skeleton.to_ostd())
	elif format == "swc":
		cf.put(f"{label}.swc", skeleton.to_swc())
	elif format == "precomputed":
		cf.put(f"{label}", skeleton.to_precomputed())
	else:
		raise ValueError(f"Format {format} is not supported.")

