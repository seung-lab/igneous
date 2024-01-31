from functools import reduce
import math
import multiprocessing as mp
import json
import os
import posixpath
import sys
import time
import urllib
import webbrowser

import click
from cloudvolume import CloudVolume, Bbox
from cloudvolume.lib import max2
from cloudfiles import CloudFiles
import cloudfiles.paths
import numpy as np
from taskqueue import TaskQueue, LocalTaskQueue
from taskqueue.lib import toabs
from taskqueue.paths import get_protocol
from tqdm import tqdm

from igneous import task_creation as tc
from igneous import downsample_scales
from igneous.secrets import LEASE_SECONDS, SQS_REGION_NAME

from igneous_cli.humanbytes import format_bytes

def normalize_path(queuepath):
  if not get_protocol(queuepath):
    return "fq://" + toabs(queuepath)
  return queuepath

def intify(x):
  return None if x is None else int(x)

def normalize_encoding(encoding):
  if encoding == "cseg":
    return "compressed_segmentation"
  elif encoding == "ckl":
    return "crackle"
  elif encoding == "cpso":
    return "compresso"
  elif encoding == "auto":
    return None

  return encoding

ENCODING_HELP = "Which image encoding to use. Options: [all] raw, png; [images] jpeg; [segmentations] compressed_segmentation (cseg), compresso (cpso), crackle (ckl); [floats] fpzip, kempressed, zfpc"

def enqueue_tasks(ctx, queue, tasks):
  parallel = int(ctx.obj.get("parallel", 1))

  if queue is None:
    tq = LocalTaskQueue(parallel=parallel)
    tq.insert_all(tasks)
  else:
    tq = TaskQueue(normalize_path(queue))
    tq.insert(tasks, parallel=parallel)
  return tq

class TupleN(click.ParamType):
  """A command line option type consisting of 3 comma-separated integers."""
  name = 'tupleN'
  def convert(self, value, param, ctx):
    if isinstance(value, str):
      value = tuple(map(int, value.split(',')))
    return value

class Tuple34(click.ParamType):
  """A command line option type consisting of 3 comma-separated integers."""
  name = 'tuple34'
  def convert(self, value, param, ctx):
    if isinstance(value, str):
      try:
        value = tuple(map(int, value.split(',')))
      except ValueError:
        self.fail(f"'{value}' does not contain a comma delimited list of 3 or 4 integers.")
      if len(value) not in (3,4):
        self.fail(f"'{value}' does not contain a comma delimited list of 3 or 4 integers.")
    return value

class Tuple3(click.ParamType):
  """A command line option type consisting of 3 comma-separated integers."""
  name = 'tuple3'
  def convert(self, value, param, ctx):
    if isinstance(value, str):
      try:
        value = tuple(map(int, value.split(',')))
      except ValueError:
        self.fail(f"'{value}' does not contain a comma delimited list of 3 integers.")
      if len(value) != 3:
        self.fail(f"'{value}' does not contain a comma delimited list of 3 integers.")
    return value
  
class Tuple2(click.ParamType):
  """A command line option type consisting of 2 comma-separated integers."""
  name = 'tuple2'
  def convert(self, value, param, ctx):
    if isinstance(value, str):
      try:
        value = tuple(map(int, value.split(',')))
      except ValueError:
        self.fail(f"'{value}' does not contain a comma delimited list of 3 integers.")
      if len(value) != 2:
        self.fail(f"'{value}' does not contain a comma delimited list of 3 integers.")
    return value

class EncodingType(click.ParamType):
  name = 'encoding'
  def convert(self, value, param, ctx):
    return normalize_encoding(value)

class CompressType(click.ParamType):
  name = "compress"
  def convert(self, value, param, ctx):
    if value and value.lower() in ("none", "false"):
      value = False
    return value

class CloudPath(click.ParamType):
  name = "CloudPath"
  def convert(self, value, param, ctx):
    return cloudfiles.paths.normalize(value)

def compute_bounds(path, mip, xrange, yrange, zrange):
  bounds = None
  if xrange or yrange or zrange:
    bounds = CloudVolume(path).meta.bounds(mip)

  if xrange:
    bounds.minpt.x = xrange[0]
    bounds.maxpt.x = xrange[1]
  if yrange:
    bounds.minpt.y = yrange[0]
    bounds.maxpt.y = yrange[1]
  if zrange:
    bounds.minpt.z = zrange[0]
    bounds.maxpt.z = zrange[1]

  return bounds


@click.group()
@click.option("-p", "--parallel", default=1, help="Run with this number of parallel processes. If 0, use number of cores.")
@click.version_option(version="4.22.0")
@click.pass_context
def main(ctx, parallel):
  """
  CLI tool for managing igneous jobs.
  https://github.com/seung-lab/igneous

  Igneous is a tool for producing neuroglancer
  datasets. It scales to hundreds of teravoxels
  or more.

  Select an operation, dataset, and queue and
  tasks will be inserted into the queue. Queues
  can be either SQS or a filesystem directory.

  Then use "igneous execute $queue" to start
  processing that operation.

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version. Run "igneous license" for details.
  """
  parallel = int(parallel)
  if parallel == 0:
    parallel = mp.cpu_count()
  ctx.ensure_object(dict)
  ctx.obj["parallel"] = max(min(parallel, mp.cpu_count()), 1)

@main.command()
def license():
  """Prints the license for this library and cli tool."""
  path = os.path.join(os.path.dirname(__file__), 'LICENSE')
  with open(path, 'rt') as f:
    print(f.read())

@main.group("image")
def imagegroup():
  """
  Manipulate image volumes. (subgroup)

  Images are the base datastructure in
  Neuroglancer. This subgroup offers
  methods for downsampling, transfers,
  reencoding, rechunking, sharding,
  and contrast correction.
  """
  pass

@imagegroup.command()
@click.argument("path", type=CloudPath())
@click.option('--queue', default=None, help="AWS SQS queue or directory to be used for a task queue. e.g. sqs://my-queue or ./my-queue. See https://github.com/seung-lab/python-task-queue")
@click.option('--mip', default=0, help="Build upward from this level of the image pyramid. Default: 0")
@click.option('--fill-missing', is_flag=True, default=False, help="Interpret missing image files as background instead of failing.")
@click.option('--num-mips', default=None, type=int, help="Build this many additional pyramid levels. Each increment increases memory requirements per task 4-8x.")
@click.option('--encoding', type=EncodingType(), default="auto", help=ENCODING_HELP, show_default=True)
@click.option('--encoding-level', default=None, help="For some encodings (png level,jpeg quality,fpzip precision) a simple scalar value can adjust the compression efficiency.", show_default=True)
@click.option('--sparse', is_flag=True, default=False, help="Don't count black pixels in mode or average calculations. For images, eliminates edge ghosting in 2x2x2 downsample. For segmentation, prevents small objects from disappearing at high mip levels.")
@click.option('--chunk-size', type=Tuple3(), default=None, help="Chunk size of new layers. e.g. 128,128,64")
@click.option('--compress', default=None, help="Set the image compression scheme. Options: 'gzip', 'br'")
@click.option('--volumetric', is_flag=True, default=False, help="Use 2x2x2 downsampling.")
@click.option('--delete-bg', is_flag=True, default=False, help="Issue a delete instead of uploading a background tile. This is helpful on systems that don't like tiny files.")
@click.option('--bg-color', default=0, help="Determines which color is regarded as background. Default: 0")
@click.option('--sharded', is_flag=True, default=False, help="Generate sharded downsamples which reduces the number of files.")
@click.option('--memory', default=3.5e9, type=int, help="(sharded only) Task memory limit in bytes. Task shape will be chosen to fit and maximize downsamples.", show_default=True)
@click.option('--xrange', type=Tuple2(), default=None, help="If specified, set x-bounds for downsampling in terms of selected mip. By default the whole dataset is selected. The bounds must be chunk aligned to the task size (maybe mysterious... use igneous design to investigate). e.g. 0,1024.", show_default=True)
@click.option('--yrange', type=Tuple2(), default=None, help="If specified, set y-bounds for downsampling in terms of selected mip. By default the whole dataset is selected. The bounds must be chunk aligned to the task size (maybe mysterious... use igneous design to investigate). e.g. 0,1024", show_default=True)
@click.option('--zrange', type=Tuple2(), default=None, help="If specified, set z-bounds for downsampling in terms of selected mip. By default the whole dataset is selected. The bounds must be chunk aligned to the task size (maybe mysterious... use igneous design to investigate). e.g. 0,1", show_default=True)
@click.pass_context
def downsample(
  ctx, path, queue, mip, fill_missing, 
  num_mips, encoding, encoding_level, sparse, 
  chunk_size, compress, volumetric,
  delete_bg, bg_color, sharded, memory,
  xrange, yrange, zrange, 
):
  """
  Create an image pyramid for grayscale or labeled images.
  By default, we use 2x2x1 downsampling. 

  The levels of the pyramid are called "mips" (from the fake latin 
  "Multum in Parvo" or "many in small"). The base of the pyramid,
  the highest resolution layer, is mip 0. Each level of the pyramid
  is one mip level higher.

  The general strategy is to downsample starting from mip 0. This
  builds several levels. Once that job is complete, pass in the 
  current top mip level of the pyramid. This builds it even taller
  (referred to as "superdownsampling").
  """
  if sharded:
    if num_mips and num_mips != 1:
      print("igneous: sharded downsamples only support producing one mip at a time. num_mips set to 1")
    num_mips = 1

  factor = (2,2,1)
  if volumetric:
  	factor = (2,2,2)

  bounds = compute_bounds(path, mip, xrange, yrange, zrange)

  if sharded:
    tasks = tc.create_image_shard_downsample_tasks(
      path, mip=mip, fill_missing=fill_missing, 
      sparse=sparse, chunk_size=chunk_size,
      encoding=encoding, memory_target=memory,
      factor=factor, bounds=bounds, bounds_mip=mip,
      encoding_level=encoding_level,
    )
  else:
    tasks = tc.create_downsampling_tasks(
      path, mip=mip, fill_missing=fill_missing, 
      num_mips=num_mips, sparse=sparse, 
      chunk_size=chunk_size, encoding=encoding, 
      delete_black_uploads=delete_bg, 
      background_color=bg_color, 
      compress=compress,
      factor=factor, bounds=bounds,
      bounds_mip=mip,
      memory_target=memory,
      encoding_level=encoding_level,
    )

  enqueue_tasks(ctx, queue, tasks)

@imagegroup.command()
@click.argument("src", type=CloudPath())
@click.argument("dest", type=CloudPath())
@click.option('--queue', default=None, help="AWS SQS queue or directory to be used for a task queue. e.g. sqs://my-queue or ./my-queue. See https://github.com/seung-lab/python-task-queue")
@click.option('--mip', default=0, help="Build upward from this level of the image pyramid.", show_default=True)
@click.option('--translate', type=Tuple3(), default=(0, 0, 0), help="Translate the bounding box by X,Y,Z voxels in the new location.")
@click.option('--downsample/--skip-downsample', is_flag=True, default=True, help="Whether or not to produce downsamples from transfer tiles.", show_default=True)
@click.option('--fill-missing', is_flag=True, default=False, help="Interpret missing image files as background instead of failing.")
@click.option('--memory', default=3.5e9, type=int, help="Task memory limit in bytes. Task shape will be chosen to fit and maximize downsamples.", show_default=True)
@click.option('--max-mips', default=5, help="Maximum number of additional pyramid levels.", show_default=True)
@click.option('--encoding', type=EncodingType(), default="auto", help=ENCODING_HELP, show_default=True)
@click.option('--encoding-level', default=None, help="For some encodings (png level,jpeg quality,fpzip precision) a simple scalar value can adjust the compression efficiency.", show_default=True)
@click.option('--sparse', is_flag=True, default=False, help="Don't count black pixels in mode or average calculations. For images, eliminates edge ghosting in 2x2x2 downsample. For segmentation, prevents small objects from disappearing at high mip levels.", show_default=True)
@click.option('--shape', type=Tuple3(), default=None, help="(overrides --memory) Set the task shape in voxels. This also determines how many downsamples you get. e.g. 2048,2048,64", show_default=True)
@click.option('--chunk-size', type=Tuple3(), default=None, help="Chunk size of destination layer. e.g. 128,128,64", show_default=True)
@click.option('--compress', default="gzip", help="Set the image compression scheme. Options: 'none', 'gzip', 'br'", show_default=True)
@click.option('--volumetric', is_flag=True, default=False, help="Use 2x2x2 downsampling.")
@click.option('--delete-bg', is_flag=True, default=False, help="Issue a delete instead of uploading a background tile. This is helpful on systems that don't like tiny files.")
@click.option('--bg-color', default=0, help="Determines which color is regarded as background.", show_default=True)
@click.option('--sharded', is_flag=True, default=False, help="Generate a sharded dataset which reduces the number of files. Downsamples are not generated.", show_default=True)
@click.option('--dest-voxel-offset', type=Tuple3(), default=None, help="Set the voxel offset for this mip level.")
@click.option('--clean-info', is_flag=True, default=False, help="Scrub info file of mesh and skeleton fields.", show_default=True)
@click.option('--no-src-update', is_flag=True, default=False, help="Don't update the source provenance file with the transfer metadata.", show_default=True)
@click.option('--truncate-scales/--no-truncate-scales', is_flag=True, default=True, help="Truncate the info file scales to the mip specified.", show_default=True)
@click.option('--xrange', type=Tuple2(), default=None, help="If specified, set x-bounds for sampling in terms of selected bounds mip. By default the whole dataset is selected. The bounds must be chunk aligned to the task size e.g. 0,1024.", show_default=True)
@click.option('--yrange', type=Tuple2(), default=None, help="If specified, set y-bounds for sampling in terms of selected bounds mip. By default the whole dataset is selected. The bounds must be chunk aligned to the task size e.g. 0,1024", show_default=True)
@click.option('--zrange', type=Tuple2(), default=None, help="If specified, set z-bounds for sampling in terms of selected bounds mip. By default the whole dataset is selected. The bounds must be chunk aligned to the task size e.g. 0,1", show_default=True)
@click.option('--bounds-mip', default=None, type=int, help="Which mip level are xrange, yrange, and zrange specified in?", show_default=True)
@click.option('--cutout', is_flag=True, default=False, help="If bounds are specified and creating a new volume, restrict the new volume to the specified bounds.", show_default=True)
@click.pass_context
def xfer(
	ctx, src, dest, queue, translate, 
  downsample, mip, fill_missing, 
  memory, max_mips, shape, sparse, 
  encoding, encoding_level, chunk_size, compress,
  volumetric, delete_bg, bg_color, sharded,
  dest_voxel_offset, clean_info, no_src_update,
  truncate_scales, 
  xrange, yrange, zrange, bounds_mip, cutout
):
  """
  Copy, re-encode, or shard an image layer.

  It is crucial to choose a good task shape. The task
  shape must be a multiple of two of the destination
  image layer chunk size. Too small, and you'll have
  an inefficient transfer. Too big, and you'll run out
  of memory and also have an inefficient transfer.

  Downsamples will by default be automatically calculated
  from whatever material is available. For the default
  2x2x1 downsampling, larger XY dimension is desirable
  compared to Z as more downsamples can be computed for
  each 2x2 increase in the task size.

  Use the --memory flag to automatically compute the
  a reasonable task shape based on your memory limits.
  """
  factor = (2,2,1)
  if volumetric:
  	factor = (2,2,2)

  if compress and compress.lower() in ("none", "false"):
    compress = False

  if encoding and encoding.lower() in ("jpeg", "png"):
    compress = False

  if bounds_mip is None:
    bounds_mip = mip
  bounds = compute_bounds(src, bounds_mip, xrange, yrange, zrange)

  if sharded:
    tasks = tc.create_image_shard_transfer_tasks(
      src, dest,
      chunk_size=chunk_size, fill_missing=fill_missing, mip=mip, 
      dest_voxel_offset=dest_voxel_offset, translate=translate, 
      encoding=encoding, memory_target=memory, clean_info=clean_info,
      encoding_level=encoding_level, truncate_scales=truncate_scales,
      compress=compress, bounds=bounds, bounds_mip=bounds_mip,
      cutout=cutout,
    )
  else:
    tasks = tc.create_transfer_tasks(
      src, dest, 
      chunk_size=chunk_size, fill_missing=fill_missing, 
      dest_voxel_offset=dest_voxel_offset, translate=translate, 
      mip=mip, shape=shape, encoding=encoding, skip_downsamples=(not downsample),
      delete_black_uploads=delete_bg, background_color=bg_color,
      compress=compress, factor=factor, sparse=sparse,
      memory_target=memory, max_mips=max_mips, 
      clean_info=clean_info, no_src_update=no_src_update,
      encoding_level=encoding_level, truncate_scales=truncate_scales,
      bounds=bounds, bounds_mip=bounds_mip, cutout=cutout,
    )

  enqueue_tasks(ctx, queue, tasks)

@imagegroup.command("roi")
@click.argument("src", type=CloudPath())
@click.option('--suppress-faint', default=0, help="Voxels below this value are set to background.", show_default=True)
@click.option('--dust', default=10, help="Suppress connected components smaller than this number of voxels.", show_default=True)
@click.option('--progress', is_flag=True, default=False, help="Show progress bars.", show_default=True)
@click.option('--max-axial-len', type=int, default=512, help="If the XY size is larger than the square of this number, the image will be automatically downsampled in-memory.", show_default=True)
@click.option('--z-step', type=int, default=None, help="How far to step in z for each ROI evaluation. Defaults to entire z range.", show_default=True)
def image_roi(src, progress, suppress_faint, dust, z_step, max_axial_len):
  """Computes bounding box of non-empty image regions."""
  bboxes = tc.compute_rois(
    src, 
    progress=progress, 
    suppress_faint_voxels=suppress_faint, 
    dust_threshold=dust, 
    max_axial_length=max_axial_len,
    z_step=z_step,
  )
  print(f"{len(bboxes)} ROI detected. info file updated.")

@imagegroup.command("reorder")
@click.argument("src", type=CloudPath())
@click.argument("dest", type=CloudPath())
@click.option('--queue', default=None, help="AWS SQS queue or directory to be used for a task queue. e.g. sqs://my-queue or ./my-queue. See https://github.com/seung-lab/python-task-queue")
@click.option('--mip', default=0, help="Build upward from this level of the image pyramid.", show_default=True)
@click.option('--fill-missing', is_flag=True, default=False, help="Interpret missing image files as background instead of failing.")
@click.option('--encoding', type=EncodingType(), default="auto", help=ENCODING_HELP, show_default=True)
@click.option('--encoding-level', default=None, help="For some encodings (png level,jpeg quality,fpzip precision) a simple scalar value can adjust the compression efficiency.", show_default=True)
@click.option('--compress', default="br", help="Set the image compression scheme. Options: 'none', 'gzip', 'br'", show_default=True)
@click.option('--delete-bg', is_flag=True, default=False, help="Issue a delete instead of uploading a background tile. This is helpful on systems that don't like tiny files.")
@click.option('--bg-color', default=0, help="Determines which color is regarded as background.", show_default=True)
@click.option('--mapping-file', required=True, help="JSON filename containing a sparse mapping of each moved z to its new position. e.g. '\{ \"5\": 6 \}'")
@click.pass_context
def image_reorder(
  ctx, src, dest, 
  queue, mip,
  fill_missing, 
  encoding, encoding_level,
  compress, 
  delete_bg, bg_color,
  clean_info, 
  mapping_file,
):
  """
  Re-arrange z-slices.

  Performs a transfer but accepts a JSON
  object that specifies which slices to
  move where. Slices not specified are
  mapped to their original location.

  If a slice would be lost by this operation,
  e.g. via being overwritten, an error will
  be issued before executing.

  JSON object format for a 3 slice volume
  across a range of z = 0-2 inclusive:
  
  { "1": 2 } # error, 2 is lost: [0,null,1]
  { "0": 1, "1": 0 } # results in [1,0,2]
  """
  with open(mapping_file, "rt") as f:
    seq = json.loads(f.read())

  tasks = tc.create_reordering_tasks(
    src=src, dest=dest,
    mip=mip,
    sequence=seq,
    fill_missing=fill_missing,
    compress=compress,
    delete_black_uploads=delete_bg, 
    background_color=bg_color,
    encoding=encoding, 
    encoding_level=encoding_level,
  )

  enqueue_tasks(ctx, queue, tasks)

@imagegroup.group("voxels")
def voxelgroup():
  """Compute voxel counts per label."""
  pass

@voxelgroup.command("count")
@click.argument("path", type=CloudPath())
@click.option('--mip', default=0, help="Count this mip level of the image pyramid.", show_default=True)
@click.option('--queue', default=None, help="AWS SQS queue or directory to be used for a task queue. e.g. sqs://my-queue or ./my-queue. See https://github.com/seung-lab/python-task-queue")
@click.pass_context
def count_voxels(ctx, path, mip, queue):
  """Create voxel counting tasks.

  These tasks are 512x512x512 voxels and result
  in a JSON file that lives at:
  $cloudpath/$KEY/stats/voxel_counts/$BBOX.json
  """
  tasks = tc.create_voxel_counting_tasks(path, mip=mip)
  enqueue_tasks(ctx, queue, tasks)

@voxelgroup.command("sum")
@click.argument("path", type=CloudPath())
@click.option('--mip', default=0, help="Count this mip level of the image pyramid.", show_default=True)
@click.option('--compress', default=None, help="", show_default=True)
@click.pass_context
def sum_voxel_counts(ctx, path, mip, compress):
  """Accumulate counts from each task.

  Results are saved in a mapbuffer IntMap file:
  $cloudpath/$KEY/stats/voxel_counts.im
  """
  tc.accumulate_voxel_counts(
    path, mip=mip, compress=compress
  )

@imagegroup.group("contrast")
def contrastgroup():
  """Perform contrast correction on the image."""
  pass

@contrastgroup.command()
@click.argument("path", type=CloudPath())
@click.option('--queue', default=None, help="AWS SQS queue or directory to be used for a task queue. e.g. sqs://my-queue or ./my-queue. See https://github.com/seung-lab/python-task-queue")
@click.option('--mip', default=0, help="Build histogram from this level of the image pyramid.", show_default=True)
@click.option('--coverage', default=0.01, type=float, help="Fraction of the image to sample. Range: [0,1]", show_default=True)
@click.option('--xrange', type=Tuple2(), default=None, help="If specified, set x-bounds for sampling in terms of selected bounds mip. By default the whole dataset is selected. The bounds must be chunk aligned to the task size e.g. 0,1024.", show_default=True)
@click.option('--yrange', type=Tuple2(), default=None, help="If specified, set y-bounds for sampling in terms of selected bounds mip. By default the whole dataset is selected. The bounds must be chunk aligned to the task size e.g. 0,1024", show_default=True)
@click.option('--zrange', type=Tuple2(), default=None, help="If specified, set z-bounds for sampling in terms of selected bounds mip. By default the whole dataset is selected. The bounds must be chunk aligned to the task size e.g. 0,1", show_default=True)
@click.option('--bounds-mip', default=None, type=int, help="Bounds are written in terms of this image pyramid level.", show_default=True)
@click.pass_context
def histogram(
  ctx, path, queue, mip, coverage,
  xrange, yrange, zrange, bounds_mip
):
  """(1) Compute the histogram for each z-slice."""
  if bounds_mip is None:
    bounds_mip = mip
  bounds = compute_bounds(path, bounds_mip, xrange, yrange, zrange)

  tasks = tc.create_luminance_levels_tasks(
    path, 
    levels_path=None, 
    coverage_factor=coverage, 
    mip=mip, 
    bounds_mip=bounds_mip, 
    bounds=bounds,
  )

  enqueue_tasks(ctx, queue, tasks)

@contrastgroup.command()
@click.argument("src", type=CloudPath())
@click.argument("dest", type=CloudPath())
@click.option('--shape', default="2048,2048,64", type=Tuple3(), help="Size of individual tasks in voxels.", show_default=True)
@click.option('--translate', type=Tuple3(), default=(0, 0, 0), help="Translate the bounding box by X,Y,Z voxels in the new location.")
@click.option('--queue', default=None, help="AWS SQS queue or directory to be used for a task queue. e.g. sqs://my-queue or ./my-queue. See https://github.com/seung-lab/python-task-queue")
@click.option('--mip', default=0, help="Apply normalization to this level of the image pyramid.", show_default=True)
@click.option('--clip-fraction', default=0.01, type=float, help="Fraction of histogram on left and right sides to clip. Range: [0,1]", show_default=True)
@click.option('--fill-missing', is_flag=True, default=False, help="Interpret missing image files as background instead of failing.")
@click.option('--minval', default=None, help="Set left side of histogram as this value. (must be in range of datatype)", show_default=True)
@click.option('--maxval', default=None, help="Set right side of histogram as this value. (must be in range of datatype)", show_default=True)
@click.option('--xrange', type=Tuple2(), default=None, help="If specified, set x-bounds for sampling in terms of selected bounds mip. By default the whole dataset is selected. The bounds must be chunk aligned to the task size e.g. 0,1024.", show_default=True)
@click.option('--yrange', type=Tuple2(), default=None, help="If specified, set y-bounds for sampling in terms of selected bounds mip. By default the whole dataset is selected. The bounds must be chunk aligned to the task size e.g. 0,1024", show_default=True)
@click.option('--zrange', type=Tuple2(), default=None, help="If specified, set z-bounds for sampling in terms of selected bounds mip. By default the whole dataset is selected. The bounds must be chunk aligned to the task size e.g. 0,1", show_default=True)
@click.option('--bounds-mip', default=None, type=int, help="Bounds are written in terms of this image pyramid level.", show_default=True)
@click.pass_context
def equalize(
  ctx, src, dest, queue,
  shape, mip, clip_fraction,
  fill_missing, translate,
  minval, maxval,
  xrange, yrange, zrange, bounds_mip
):
  """(2) Apply histogram equalization to z-slices."""
  if bounds_mip is None:
    bounds_mip = mip
  bounds = compute_bounds(src, bounds_mip, xrange, yrange, zrange)

  tasks = tc.create_contrast_normalization_tasks(
    src, dest, levels_path=None,
    shape=shape, mip=mip, clip_fraction=clip_fraction,
    fill_missing=fill_missing, translate=translate,
    minval=minval, maxval=maxval, bounds=bounds,
    bounds_mip=bounds_mip
  )

  enqueue_tasks(ctx, queue, tasks)

@imagegroup.group("ccl")
def cclgroup():
  """
  Perform connected components labeling on the image.

  Result will be a 6-connected labeling of the input
  image. All steps must use the same task shape. If
  a threshold (lte and/or gte) is applied, it must be
  used with the same values in all tasks. This ensures
  that when the CCL is recomputed at each step, the same
  values result.

  Intermediate linkage and relabelign data are saved 
  in PATH/KEY/ccl/{faces,equivalences,relabel}/

  The largest image that can be handled would have 2^64 voxels
  (18 exavoxels, a bit larger than a whole mouse brain).

  Each of the steps are labeled with their sequence number.
  Their order is (1) Generate 3 back faces for each task with 
  1 voxel overlap (so they can be referenced by adjacent tasks)
  (2) Compute linkages between CCL tasks and save the results 
  in a database. (3) Compute a global union find from the linkage 
  data and from that a global relabeling scheme which is saved 
  in the database (4) Apply the relabeling scheme to the image.
  """
  pass

@cclgroup.command("faces")
@click.argument("src", type=CloudPath())
@click.option('--shape', default="512,512,512", type=Tuple3(), help="Size of individual tasks in voxels.", show_default=True)
@click.option('--mip', default=0, help="Apply to this level of the image pyramid.", show_default=True)
@click.option('--queue', default=None, help="AWS SQS queue or directory to be used for a task queue. e.g. sqs://my-queue or ./my-queue. See https://github.com/seung-lab/python-task-queue")
@click.option('--threshold-gte', default=None, help="Threshold source image using image >= value.", show_default=True)
@click.option('--threshold-lte', default=None, help="Threshold source image using image <= value.", show_default=True)
@click.option('--fill-missing', is_flag=True, default=False, help="Interpret missing image files as background instead of failing.", show_default=True)
@click.option('--dust', default=0, help="Delete objects smaller than this number of voxels within a cutout.", show_default=True)
@click.pass_context
def ccl_faces(
  ctx, src, mip, shape, queue,
  threshold_lte, threshold_gte,
  fill_missing, dust
):
  """(1) Generate back face images."""
  tasks = tc.create_ccl_face_tasks(
    src, mip, shape,
    threshold_lte=intify(threshold_lte),
    threshold_gte=intify(threshold_gte),
    fill_missing=fill_missing,
    dust_threshold=dust,
  )

  enqueue_tasks(ctx, queue, tasks)

@cclgroup.command("links")
@click.argument("src", type=CloudPath())
@click.option('--shape', default="512,512,512", type=Tuple3(), help="Size of individual tasks in voxels.", show_default=True)
@click.option('--mip', default=0, help="Apply to this level of the image pyramid.", show_default=True)
@click.option('--queue', default=None, help="AWS SQS queue or directory to be used for a task queue. e.g. sqs://my-queue or ./my-queue. See https://github.com/seung-lab/python-task-queue")
@click.option('--threshold-gte', default=None, help="Threshold source image using image >= value.", show_default=True)
@click.option('--threshold-lte', default=None, help="Threshold source image using image <= value.", show_default=True)
@click.option('--fill-missing', is_flag=True, default=False, help="Interpret missing image files as background instead of failing.", show_default=True)
@click.option('--dust', default=0, help="Delete objects smaller than this number of voxels within a cutout.", show_default=True)
@click.pass_context
def ccl_equivalences(
  ctx, src, mip, shape, queue,
  threshold_lte, threshold_gte,
  fill_missing, dust
):
  """(2) Generate links between tasks."""
  tasks = tc.create_ccl_equivalence_tasks(
    src, mip, shape,
    threshold_lte=intify(threshold_lte),
    threshold_gte=intify(threshold_gte),
    fill_missing=fill_missing,
    dust_threshold=dust,
  )

  enqueue_tasks(ctx, queue, tasks)

@cclgroup.command("calc-labels")
@click.argument("src", type=CloudPath())
@click.option('--mip', default=0, required=True, help="Apply to this level of the image pyramid.", show_default=True)
@click.option('--shape', default="512,512,512", type=Tuple3(), help="Size of individual tasks in voxels.", show_default=True)
@click.pass_context
def ccl_calc_labels(ctx, src, mip, shape):
  """(3) Compute and serialize a relabeling to the DB."""
  import igneous.tasks.image.ccl
  igneous.tasks.image.ccl.create_relabeling(src, mip, shape)

@cclgroup.command("relabel")
@click.argument("src", type=CloudPath())
@click.argument("dest", type=CloudPath())
@click.option('--shape', default="512,512,512", type=Tuple3(), help="Size of individual tasks in voxels.", show_default=True)
@click.option('--mip', default=0, help="Apply to this level of the image pyramid.", show_default=True)
@click.option('--chunk-size', type=Tuple3(), default=None, help="Chunk size of destination layer. e.g. 128,128,64")
@click.option('--encoding', type=EncodingType(), default="compresso", help="Which image encoding to use. Options: raw, cseg, compresso, crackle", show_default=True)
@click.option('--queue', default=None, help="AWS SQS queue or directory to be used for a task queue. e.g. sqs://my-queue or ./my-queue. See https://github.com/seung-lab/python-task-queue")
@click.option('--threshold-gte', default=None, help="Threshold source image using image >= value.", show_default=True)
@click.option('--threshold-lte', default=None, help="Threshold source image using image <= value.", show_default=True)
@click.option('--fill-missing', is_flag=True, default=False, help="Interpret missing image files as background instead of failing.", show_default=True)
@click.option('--dust', default=0, help="Delete objects smaller than this number of voxels within a cutout.", show_default=True)
@click.pass_context
def ccl_relabel(
  ctx, src, dest, 
  shape, mip, chunk_size, 
  encoding, queue,
  threshold_lte, threshold_gte,
  fill_missing, dust
):
  """(4) Finally relabel and write a CCL image."""
  tasks = tc.create_ccl_relabel_tasks(
    src, dest, 
    mip=mip, shape=shape, 
    chunk_size=chunk_size, encoding=encoding,
    threshold_lte=intify(threshold_lte),
    threshold_gte=intify(threshold_gte),
    fill_missing=fill_missing,
    dust_threshold=dust,
  )

  enqueue_tasks(ctx, queue, tasks)

@cclgroup.command("clean")
@click.argument("src", type=CloudPath())
@click.option('--mip', default=0, required=True, help="Apply to this level of the image pyramid.", show_default=True)
def ccl_clean(src, mip):
  """(last) Cleans up intermediate files."""
  import igneous.tasks.image.ccl
  igneous.tasks.image.ccl.clean_intermediate_files(src, mip)

@cclgroup.command("auto")
@click.argument("src", type=CloudPath())
@click.argument("dest", type=CloudPath())
@click.option('--shape', default="512,512,512", type=Tuple3(), help="Size of individual tasks in voxels.", show_default=True)
@click.option('--mip', default=0, help="Apply to this level of the image pyramid.", show_default=True)
@click.option('--chunk-size', type=Tuple3(), default=None, help="Chunk size of destination layer. e.g. 128,128,64")
@click.option('--encoding', default="compresso", help="Which image encoding to use. Options: raw, cseg, compresso, crackle", show_default=True)
@click.option('--queue', default=None, help="AWS SQS queue or directory to be used for a task queue. e.g. sqs://my-queue or ./my-queue. See https://github.com/seung-lab/python-task-queue")
@click.option('--clean/--no-clean', default=True, is_flag=True, help="Delete intermediate files on completion.", show_default=True)
@click.option('--threshold-gte', default=None, help="Threshold source image using image >= value.", show_default=True)
@click.option('--threshold-lte', default=None, help="Threshold source image using image <= value.", show_default=True)
@click.option('--fill-missing', is_flag=True, default=False, help="Interpret missing image files as background instead of failing.", show_default=True)
@click.option('--dust', default=0, help="Delete objects smaller than this number of voxels within a cutout.", show_default=True)
@click.pass_context
def ccl_auto(
  ctx, src, dest, 
  shape, mip, 
  chunk_size, encoding, 
  queue, clean,
  threshold_lte, threshold_gte,
  fill_missing, dust
):
  """
  For local volumes, execute all steps automatically.
  """
  parallel = int(ctx.obj.get("parallel", 1))
  args = (queue, None, LEASE_SECONDS, True, -1, True, False)

  tasks = tc.create_ccl_face_tasks(
    src, mip, shape,
    threshold_lte=intify(threshold_lte),
    threshold_gte=intify(threshold_gte),
    fill_missing=fill_missing,
    dust_threshold=dust,
  )
  enqueue_tasks(ctx, queue, tasks)
  if queue:
    parallel_execute_helper(parallel, args)

  tasks = tc.create_ccl_equivalence_tasks(
    src, mip, shape,
    threshold_lte=intify(threshold_lte),
    threshold_gte=intify(threshold_gte),
    fill_missing=fill_missing,
    dust_threshold=dust,
  )
  enqueue_tasks(ctx, queue, tasks)
  if queue:
    parallel_execute_helper(parallel, args)

  import igneous.tasks.image.ccl
  igneous.tasks.image.ccl.create_relabeling(src, mip, shape)

  tasks = tc.create_ccl_relabel_tasks(
    src, dest, 
    mip=mip, shape=shape, 
    chunk_size=chunk_size, encoding=encoding,
    threshold_lte=intify(threshold_lte),
    threshold_gte=intify(threshold_gte),
    fill_missing=fill_missing,
    dust_threshold=dust,
  )
  enqueue_tasks(ctx, queue, tasks)
  if queue:
    parallel_execute_helper(parallel, args)

  if clean:
    igneous.tasks.image.ccl.clean_intermediate_files(src, mip)

@main.command()
@click.argument("queue", type=str)
@click.option('--aws-region', default=SQS_REGION_NAME, help=f"AWS region in which the SQS queue resides.", show_default=True)
@click.option('--lease-sec', default=LEASE_SECONDS, help=f"Seconds to lease a task for.", type=int, show_default=True)
@click.option('--tally/--no-tally', is_flag=True, default=True, help="Tally completed fq tasks. Does not apply to SQS.", show_default=True)
@click.option('--min-sec', default=-1, help='Execute for at least this many seconds and quit after the last task finishes. Special values: (0) Run at most a single task. (-1) Loop forever (default).', type=float)
@click.option('-q', '--quiet', is_flag=True, default=False, help='Suppress task status messages.', show_default=True)
@click.option('-x', '--exit-on-empty', is_flag=True, default=False, help="Exit immediately when the queue is empty. Not reliable for SQS as measurements are approximate.", show_default=True)
@click.pass_context
def execute(
  ctx, queue, aws_region,
  lease_sec, tally, min_sec,
  exit_on_empty, quiet
):
  """Execute igneous tasks from a queue.

  The queue must be an AWS SQS queue or a FileQueue directory. 
  Examples: (SQS) sqs://my-queue (FileQueue) fq://./my-queue
  (the fq:// is optional).

  See https://github.com/seung-lab/python-task-queue
  """
  parallel = int(ctx.obj.get("parallel", 1))
  args = (queue, aws_region, lease_sec, tally, min_sec, exit_on_empty, quiet)
  parallel_execute_helper(parallel, args)

def parallel_execute_helper(parallel, args):
  if parallel == 1:
    execute_helper(*args)
    return

  pool = mp.Pool(processes=parallel)
  try:
    for _ in range(parallel):
      pool.apply_async(execute_helper, args)
    pool.close()
    pool.join()
  except KeyboardInterrupt:
    print("Interrupted. Exiting.")
    pool.terminate()
    pool.join()

def execute_helper(
  queue, aws_region, lease_sec, 
  tally, min_sec, exit_on_empty,
  quiet
):
  tq = TaskQueue(normalize_path(queue), region_name=aws_region)

  sqs_sec_to_wait = 120
  start_time = time.time()
  last_empty = False

  # Offical Amazon docs state that to determine if a queue is empty,
  # you have to test whether it's consistently empty for several 
  # minutes. So we pick 120 seconds to wait somewhat arbitrarily.
  # https://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSDeveloperGuide/confirm-queue-is-empty.html
  def is_empty():
    nonlocal start_time
    nonlocal sqs_sec_to_wait
    nonlocal last_empty

    if tq.path.protocol == "sqs":
      if tq.is_empty():
        if last_empty:
          elapsed_time = time.time() - start_time
          if elapsed_time >= sqs_sec_to_wait:
            return True
          else:
            return False
        else:
          print(f"Queue appearently empty. Waiting {sqs_sec_to_wait} sec. to confirm.")
          start_time = time.time()
          last_empty = True
          return False
      else:
        last_empty = False
        return False
    else:
      return tq.is_empty()

  def stop_after_elapsed_time(tries, elapsed_time):
    if exit_on_empty and is_empty():
      return True

    if min_sec < 0:
      return False
    return min_sec < elapsed_time

  if min_sec != 0:
    tq.poll(
      lease_seconds=lease_sec,
      verbose=(not quiet),
      tally=tally,
      stop_fn=stop_after_elapsed_time,
    )
  else:
    task = tq.lease(seconds=lease_sec)
    task.execute()

@main.group("mesh")
def meshgroup():
  """
  Create 3D meshes from a segmentation. (subgroup)

  Meshing is a two step process of forging then 
  merging. First the meshes are created from a
  regular grid of segmentation cutouts. Second,
  the pieces are glued together.
  """
  pass

@meshgroup.command("xfer")
@click.argument("src", type=CloudPath())
@click.argument("dest", type=CloudPath())
@click.option('--queue', required=True, help="AWS SQS queue or directory to be used for a task queue. e.g. sqs://my-queue or ./my-queue. See https://github.com/seung-lab/python-task-queue", type=str)
@click.option("--sharded", is_flag=True, default=False, help="Generate shard fragments instead of outputing mesh fragments.", show_default=True)
@click.option("--dir", "mesh_dir", type=str, default=None, help="Write meshes into this directory instead of the one indicated in the info file.")
@click.option('--magnitude', default=2, help="Split up the work with 10^(magnitude) prefix based tasks.", show_default=True)
@click.option('--mip', type=int, default=None, help="Manually specify which mip level the images were derived from.", show_default=True)
@click.option('--nlod', type=int, default=0, help="(sharded) Create additional levels of detail (lods).", show_default=True)
@click.pass_context
def mesh_xfer(
  ctx, src, dest, queue,
  sharded, mesh_dir, magnitude,
  mip, nlod
):
  """
  Transfer meshes to another location.
  Enables conversion of unsharded to sharded
  as well.
  """
  cv_src = CloudVolume(src)

  if not cv_src.mesh.meta.is_sharded() and sharded:
    tasks = tc.create_sharded_multires_mesh_from_unsharded_tasks(
      src, dest,
      num_lod=nlod, 
      mesh_dir=mesh_dir, 
      mip=mip,
    )
  else:
    tasks = tc.create_xfer_meshes_tasks(
      src, dest, 
      mesh_dir=mesh_dir, 
      magnitude=magnitude,
    )

  enqueue_tasks(ctx, queue, tasks)

@meshgroup.command("forge")
@click.argument("path", type=CloudPath())
@click.option('--queue', required=True, help="AWS SQS queue or directory to be used for a task queue. e.g. sqs://my-queue or ./my-queue. See https://github.com/seung-lab/python-task-queue", type=str)
@click.option('--mip', default=0, help="Perform meshing using this level of the image pyramid.", show_default=True)
@click.option('--shape', type=Tuple3(), default=(448, 448, 448), help="Set the task shape in voxels.", show_default=True)
@click.option('--simplify/--skip-simplify', is_flag=True, default=True, help="Enable mesh simplification.", show_default=True)
@click.option('--fill-missing', is_flag=True, default=False, help="Interpret missing image files as background instead of failing.")
@click.option('--max-error', default=40, help="Maximum simplification error in physical units.", show_default=True)
@click.option('--dust-threshold', default=None, help="Skip meshing objects smaller than this number of voxels within a cutout. No default limit. Typical value: 1000.", type=int)
@click.option('--dust-global/--dust-local', is_flag=True, default=False, help="Use global voxel counts for the dust threshold (when >0). To use this feature you must first compute the global voxel counts using the 'igneous image voxels' command.", show_default=True)
@click.option('--dir', default=None, help="Write meshes into this directory instead of the one indicated in the info file.")
@click.option('--compress', default="gzip", help="Set the image compression scheme. Options: 'none', 'gzip', 'br'", show_default=True)
@click.option('--spatial-index/--skip-spatial-index', is_flag=True, default=True, help="Create the spatial index.", show_default=True)
@click.option('--sharded', is_flag=True, default=False, help="Generate shard fragments instead of outputing mesh fragments.", show_default=True)
@click.option('--closed-edge/--open-edge', is_flag=True, default=True, help="Whether meshes are closed on the side that contacts the dataset boundary.", show_default=True)
@click.pass_context
def mesh_forge(
  ctx, path, queue, mip, shape, 
  simplify, fill_missing, max_error, 
  dust_threshold, dir, compress, 
  spatial_index, sharded, closed_edge,
  dust_global
):
  """
  (1) Synthesize meshes from segmentation cutouts.

  A large labeled image is divided into a regular
  grid. zmesh is applied to grid point, which performs
  marching cubes and a quadratic mesh simplifier.

  Note that using task shapes with axes less than
  or equal to 511x1023x511 (don't ask) will be more
  memory efficient as it can use a 32-bit mesher.

  zmesh is used: https://github.com/seung-lab/zmesh
  """
  if compress.lower() == "none":
    compress = False

  tasks = tc.create_meshing_tasks(
    path, mip, shape, 
    simplification=simplify, max_simplification_error=max_error,
    mesh_dir=dir, cdn_cache=False, dust_threshold=dust_threshold,
    object_ids=None, progress=False, fill_missing=fill_missing,
    encoding='precomputed', spatial_index=spatial_index, 
    sharded=sharded, compress=compress, closed_dataset_edges=closed_edge,
    dust_global=dust_global
  )

  enqueue_tasks(ctx, queue, tasks)

@meshgroup.command("merge")
@click.argument("path", type=CloudPath())
@click.option('--queue', required=True, help="AWS SQS queue or directory to be used for a task queue. e.g. sqs://my-queue or ./my-queue. See https://github.com/seung-lab/python-task-queue", type=str)
@click.option('--magnitude', default=2, help="Split up the work with 10^(magnitude) prefix based tasks. Default: 2 (100 tasks)")
@click.option('--nlod', default=0, help="(multires) How many extra levels of detail to create.", show_default=True)
@click.option('--vqb', default=16, help="(multires) Vertex quantization bits for stored model representation. 10 or 16 only.", show_default=True)
@click.option('--min-chunk-size', type=Tuple3(), default="256,256,256",  help="(multires) Sets the minimum chunk size of the highest resolution mesh fragment.", show_default=True)
@click.option('--dir', default=None, help="Write manifests into this directory instead of the one indicated in the info file.")
@click.pass_context
def mesh_merge(ctx, path, queue, magnitude, nlod, vqb, dir, min_chunk_size):
  """
  (2) Merge the mesh pieces produced from the forging step.

  The per-cutout mesh fragments are then assembled and
  merged. However, this process occurs by compiling 
  a list of fragment files and uploading a "mesh manifest"
  file that is an index for locating the fragments.
  """
  if nlod > 0:
    tasks = tc.create_unsharded_multires_mesh_tasks(
      path, num_lod=nlod, 
      magnitude=magnitude, mesh_dir=dir,
      vertex_quantization_bits=vqb,
      min_chunk_size=min_chunk_size,
    )
  else:
    tasks = tc.create_mesh_manifest_tasks(
      path, magnitude=magnitude, mesh_dir=dir
    )

  enqueue_tasks(ctx, queue, tasks)

@meshgroup.command("merge-sharded")
@click.argument("path", type=CloudPath())
@click.option('--queue', required=True, help="AWS SQS queue or directory to be used for a task queue. e.g. sqs://my-queue or ./my-queue. See https://github.com/seung-lab/python-task-queue", type=str)
@click.option('--nlod', default=1, help="Number of levels of detail to create.", type=int, show_default=True)
@click.option('--vqb', default=16, help="Vertex quantization bits. Can be 10 or 16.", type=int, show_default=True)
@click.option('--compress-level', default=7, help="Draco compression level.", type=int, show_default=True)
@click.option('--shard-index-bytes', default=2**13, help="Size in bytes to make the shard index.", type=int, show_default=True)
@click.option('--minishard-index-bytes', default=2**15, help="Size in bytes to make the minishard index.", type=int, show_default=True)
@click.option('--min-shards', default=1, help="Minimum number of shards to generate. Excess shards can make it easier to parallelize the merge process.", type=int, show_default=True)
@click.option('--max-labels-per-shard', default=1000, help="Maximum average number of labels per a shard.", type=int, show_default=True)
@click.option('--minishard-index-encoding', default="gzip", help="Minishard indices can be compressed. gzip or raw.", show_default=True)
@click.option('--spatial-index-db', default=None, help="CloudVolume generated SQL database for spatial index.", show_default=True)
@click.option('--min-chunk-size', type=Tuple3(), default="256,256,256",  help="(multires) Sets the minimum chunk size of the highest resolution mesh fragment.", show_default=True)
@click.pass_context
def mesh_sharded_merge(
  ctx, path, queue, 
  nlod, vqb, compress_level,
  shard_index_bytes, minishard_index_bytes, min_shards,
  max_labels_per_shard, minishard_index_encoding, 
  spatial_index_db, min_chunk_size
):
  """
  (2) Postprocess fragments into finished sharded multires meshes.

  Only use this command if you used the --sharded flag
  during the forging step. Some reasonable defaults
  are selected for a dataset with a few million labels,
  but for smaller or larger datasets they may not be
  appropriate.

  The shard and minishard index default sizes are set to
  accomodate efficient access for a 100 Mbps connection.
  """
  tasks = tc.create_sharded_multires_mesh_tasks(
    path, 
    num_lod=nlod,
    draco_compression_level=compress_level,
    vertex_quantization_bits=vqb,
    shard_index_bytes=shard_index_bytes,
    minishard_index_bytes=minishard_index_bytes,
    minishard_index_encoding=minishard_index_encoding,
    min_shards=min_shards,
    spatial_index_db=spatial_index_db,
    min_chunk_size=min_chunk_size,
    max_labels_per_shard=max_labels_per_shard,
  )

  enqueue_tasks(ctx, queue, tasks)

@meshgroup.command("rm")
@click.argument("path", type=CloudPath())
@click.option('--queue', required=True, help="AWS SQS queue or directory to be used for a task queue. e.g. sqs://my-queue or ./my-queue. See https://github.com/seung-lab/python-task-queue", type=str)
@click.option('--magnitude', default=2, help="Split up the work with 10^(magnitude) prefix based tasks. Default: 2 (100 tasks)")
@click.option('--dir', 'mesh_dir', default=None, help="Target this directory instead of the one indicated in the info file.")
@click.pass_context
def mesh_rm(ctx, path, queue, magnitude, mesh_dir):
  """
  Delete mesh files.
  """
  tasks = tc.create_mesh_deletion_tasks(
    path, magnitude=magnitude, mesh_dir=mesh_dir
  )

  enqueue_tasks(ctx, queue, tasks)

@meshgroup.command("clean")
@click.argument("src", type=CloudPath())
def mesh_clean(src):
  """
  Removes temporary files.
  e.g. .labels and .frags files.
  """
  cv = CloudVolume(src)
  cf = CloudFiles(cv.mesh.meta.layerpath, progress=True)

  if not cv.mesh.meta.is_sharded():
    def remove_paths():
      for path in cf.list():
        N = len(os.path.basename(path).split(":"))
        # temp file: segid:0:fragment_name
        if N == 3:
          yield path
  else:
    def remove_paths():
      for path in cf.list():
        _, ext = os.path.splitext(path)
        if ext in (".labels", ".frags"):
          yield path

  cf.delete(remove_paths())

@meshgroup.group("spatial-index")
def spatialindexgroup():
  """
  (subgroup) Create or download mesh spatial indices.
  """
  pass

@spatialindexgroup.command("create")
@click.argument("path", type=CloudPath())
@click.option('--queue', required=True, help="AWS SQS queue or directory to be used for a task queue. e.g. sqs://my-queue or ./my-queue. See https://github.com/seung-lab/python-task-queue", type=str)
@click.option('--shape', default="448,448,448", type=Tuple3(), help="Shape in voxels of each indexing task.", show_default=True)
@click.option('--mip', default=0, help="Perform indexing using this level of the image pyramid.", show_default=True)
@click.option('--fill-missing', is_flag=True, default=False, help="Interpret missing image files as background instead of failing.", show_default=True)
@click.pass_context
def mesh_spatial_index_create(ctx, path, queue, shape, mip, fill_missing):
  """
  Create a spatial index on a pre-existing mesh.

  Sometimes datasets were meshes without a
  spatial index or need it to be updated.
  This function provides a more efficient
  way to accomplish that than remeshing.
  """
  tasks = tc.create_spatial_index_mesh_tasks(
    cloudpath=path,
    shape=shape,
    mip=mip, 
    fill_missing=fill_missing,
  )

  enqueue_tasks(ctx, queue, tasks)

@spatialindexgroup.command("db")
@click.argument("path", type=CloudPath())
@click.argument("database")
@click.option('--progress', is_flag=True, default=False, help="Show progress bars.", show_default=True)
@click.option('--allow-missing', is_flag=True, default=False, help="Allow missing index files.", show_default=True)
@click.pass_context
def mesh_spatial_index_download(ctx, path, database, progress, allow_missing):
  """
  Download the mesh spatial index into a database.

  sqlite paths: sqlite://filename.db (prefix optional)
  mysql paths: mysql://{user}:{pwd}@{host}/{database}
  """
  parallel = int(ctx.obj.get("parallel", 1))
  cv = CloudVolume(path)
  cv.mesh.spatial_index.to_sql(
    database, 
    progress=progress,
    allow_missing=allow_missing, 
    parallel=parallel,
  )

@main.group("skeleton")
def skeletongroup():
  """
  Create skeletons from a segmentation.

  Skeletonizing is a two step process of forging 
  then merging. First the skeletons are created
  from a regular grid of segmentation cutouts.
  Second, the pieces are postprocessed and glued
  together.
  """
  pass

@skeletongroup.command("forge")
@click.argument("path", type=CloudPath())
@click.option('--queue', required=True, help="AWS SQS queue or directory to be used for a task queue. e.g. sqs://my-queue or ./my-queue. See https://github.com/seung-lab/python-task-queue", type=str)
@click.option('--mip', default=0, help="Perform skeletonizing using this level of the image pyramid.", show_default=True)
@click.option('--shape', type=Tuple3(), default=(512, 512, 512), help="Set the task shape in voxels.", show_default=True)
@click.option('--fill-missing', is_flag=True, default=False, help="Interpret missing image files as background instead of failing.", show_default=True)
@click.option('--fix-branching', is_flag=True, default=True, help="Trades speed for quality of branching at forks. Default: True")
@click.option('--fix-borders', is_flag=True, default=True, help="Allows trivial merging of single voxel overlap tasks. Only switch off for datasets that fit in a single task.", show_default=True)
@click.option('--fix-avocados', is_flag=True, default=False, help="Fixes somata where nuclei and cytoplasm have separate segmentations.", show_default=True)
@click.option('--fill-holes', is_flag=True, default=False, help="Preprocess each cutout to eliminate background holes and holes caused by entirely contained inclusions. Warning: May remove labels that are considered inclusions.", show_default=True)
@click.option('--dust-threshold', default=1000, help="Skip skeletonizing objects smaller than this number of voxels within a cutout.", type=int, show_default=True)
@click.option('--dust-global/--dust-local', is_flag=True, default=False, help="Use global voxel counts for the dust threshold (when >0). To use this feature you must first compute the global voxel counts using the 'igneous image voxels' command.", show_default=True)
@click.option('--spatial-index/--skip-spatial-index', is_flag=True, default=True, help="Create the spatial index.", show_default=True)
@click.option('--scale', default=4, help="Multiplies invalidation radius by distance from boundary.", type=float, show_default=True)
@click.option('--const', default=10, help="Adds constant amount to invalidation radius in physical units.", type=float, show_default=True)
@click.option('--soma-detect', default=1100, help="Consider objects with peak distances to boundary larger than this soma candidates. Physical units.", type=float, show_default=True)
@click.option('--soma-accept', default=3500, help="Accept soma candidates over this threshold and perform a one-time spherical invalidation around their peak value. Physical units.", type=float, show_default=True)
@click.option('--soma-scale', default=1.0, help="Scale factor for soma invalidation.", type=float, show_default=True)
@click.option('--soma-const', default=300, help="Const factor for soma invalidation.", type=float, show_default=True)
@click.option('--max-paths', default=None, help="Abort skeletonizing an object after this many paths have been traced.", type=float)
@click.option('--sharded', is_flag=True, default=False, help="Generate shard fragments instead of outputing skeleton fragments.", show_default=True)
@click.option('--labels', type=TupleN(), default=None, help="Skeletonize only this comma separated list of labels.", show_default=True)
@click.option('--cross-section', type=int, default=0, help="Compute the cross sectional area for each skeleton vertex. May add substantial computation time. Integer value is the normal vector rolling average smoothing window over vertices. 0 means off.", show_default=True)
@click.pass_context
def skeleton_forge(
  ctx, path, queue, mip, shape, 
  fill_missing, dust_threshold, dust_global, spatial_index,
  fix_branching, fix_borders, fix_avocados, 
  fill_holes, scale, const, soma_detect, soma_accept,
  soma_scale, soma_const, max_paths, sharded, labels,
  cross_section,
):
  """
  (1) Synthesize skeletons from segmentation cutouts.

  A large labeled image is divided into a regular
  grid. Kimimaro is applied to grid point, which performs
  a TEASAR based skeletonization.

  You can read more about the parameters here:
  https://github.com/seung-lab/kimimaro

  Tutorials are located here:

  - https://github.com/seung-lab/kimimaro/wiki/A-Pictorial-Guide-to-TEASAR-Skeletonization

  - https://github.com/seung-lab/kimimaro/wiki/Intuition-for-Setting-Parameters-const-and-scale

  A guide to how much this might cost is located here:

  - https://github.com/seung-lab/kimimaro/wiki/The-Economics:-Skeletons-for-the-People
  """
  teasar_params = {
    'scale': scale,
    'const': const, # physical units
    'pdrf_exponent': 4,
    'pdrf_scale': 100000,
    'soma_detection_threshold': soma_detect, # physical units
    'soma_acceptance_threshold': soma_accept, # physical units
    'soma_invalidation_scale': soma_scale,
    'soma_invalidation_const': soma_const, # physical units
    'max_paths': max_paths, # default None
  }

  tasks = tc.create_skeletonizing_tasks(
    path, mip, shape, 
    teasar_params=teasar_params, 
    fix_branching=fix_branching, fix_borders=fix_borders, 
    fix_avocados=fix_avocados, fill_holes=fill_holes,
    dust_threshold=dust_threshold, progress=False,
    parallel=1, fill_missing=fill_missing, 
    sharded=sharded, spatial_index=spatial_index,
    dust_global=dust_global, object_ids=labels,
    cross_sectional_area=(cross_section > 0),
    cross_sectional_area_smoothing_window=int(cross_section),
  )

  enqueue_tasks(ctx, queue, tasks)


@skeletongroup.command("merge")
@click.argument("path", type=CloudPath())
@click.option('--queue', required=True, help="AWS SQS queue or directory to be used for a task queue. e.g. sqs://my-queue or ./my-queue. See https://github.com/seung-lab/python-task-queue", type=str)
@click.option('--min-cable-length', default=1000, help="Skip objects smaller than this physical path length. Default: 1000 nm", type=float)
@click.option('--max-cable-length', default=None, help="Skip objects larger than this physical path length. Default: no limit", type=float)
@click.option('--tick-threshold', default=0, help="Remove small \"ticks\", or branches from the main skeleton one at a time from smallest to largest. Branches larger than this are preserved. Default: no elimination", type=float)
@click.option('--delete-fragments', is_flag=True, default=False, help="Delete the skeleton fragments from the first pass after upload is complete.")
@click.option('--magnitude', default=2, help="Split up the work with 10^(magnitude) prefix based tasks. Default: 2 (100 tasks)", type=int)
@click.pass_context
def skeleton_merge(
  ctx, path, queue, 
  min_cable_length, max_cable_length, 
  tick_threshold, delete_fragments, 
  magnitude
):
  """
  (2) Postprocess fragments into finished skeletons.
  """
  tasks = tc.create_unsharded_skeleton_merge_tasks(
    path, 
    magnitude=magnitude, 
    dust_threshold=min_cable_length,
    max_cable_length=max_cable_length,
    tick_threshold=tick_threshold,
    delete_fragments=delete_fragments,
  )

  enqueue_tasks(ctx, queue, tasks)

@skeletongroup.command("merge-sharded")
@click.argument("path", type=CloudPath())
@click.option('--queue', required=True, help="AWS SQS queue or directory to be used for a task queue. e.g. sqs://my-queue or ./my-queue. See https://github.com/seung-lab/python-task-queue", type=str)
@click.option('--min-cable-length', default=1000, help="Skip objects smaller than this physical path length.", type=float, show_default=True)
@click.option('--max-cable-length', default=None, help="Skip objects larger than this physical path length. Default: no limit", type=float)
@click.option('--tick-threshold', default=0, help="Remove small \"ticks\", or branches from the main skeleton one at a time from smallest to largest. Branches larger than this are preserved. Default: no elimination", type=float)
@click.option('--shard-index-bytes', default=2**13, help="Size in bytes to make the shard index.", type=int, show_default=True)
@click.option('--minishard-index-bytes', default=2**15, help="Size in bytes to make the minishard index.", type=int, show_default=True)
@click.option('--max-labels-per-shard', default=2000, help="Maximum average number of labels per a shard.", type=int, show_default=True)
@click.option('--min-shards', default=1, help="Minimum number of shards to generate. Excess shards can make it easier to parallelize the merge process.", type=int, show_default=True)
@click.option('--minishard-index-encoding', default="gzip", help="Minishard indices can be compressed. gzip or raw. Default: gzip")
@click.option('--data-encoding', default="gzip", help="Shard data can be compressed. gzip or raw. Default: gzip")
@click.option('--spatial-index-db', default=None, help="CloudVolume generated SQL database for spatial index.", show_default=True)
@click.pass_context
def skeleton_sharded_merge(
  ctx, path, queue, 
  min_cable_length, max_cable_length, 
  tick_threshold, 
  shard_index_bytes, minishard_index_bytes, 
  max_labels_per_shard, min_shards,
  minishard_index_encoding, data_encoding,
  spatial_index_db
):
  """
  (2) Postprocess fragments into finished skeletons.

  Only use this command if you used the --sharded flag
  during the forging step. Some reasonable defaults
  are selected for a dataset with a few million labels,
  but for smaller or larger datasets they may not be
  appropriate.

  The shard and minishard index default sizes are set to
  accomodate efficient access for a 100 Mbps connection.
  """
  tasks = tc.create_sharded_skeleton_merge_tasks(
    path, 
    dust_threshold=min_cable_length,
    max_cable_length=max_cable_length,
    tick_threshold=tick_threshold,
    shard_index_bytes=shard_index_bytes,
    minishard_index_bytes=minishard_index_bytes,
    min_shards=min_shards,
    minishard_index_encoding=minishard_index_encoding, 
    data_encoding=data_encoding,
    spatial_index_db=spatial_index_db,
    max_labels_per_shard=max_labels_per_shard,
  )

  enqueue_tasks(ctx, queue, tasks)

@imagegroup.command("rm")
@click.argument("path", type=CloudPath())
@click.option('--queue', required=True, help="AWS SQS queue or directory to be used for a task queue. e.g. sqs://my-queue or ./my-queue. See https://github.com/seung-lab/python-task-queue", type=str)
@click.option('--mip', default=0, help="Which mip level to start deleting from. Default: 0")
@click.option('--num-mips', default=5, help="The number of mip levels to delete at once. Default: 5")
@click.option('--shape', default=None, help="The size of each deletion task as a comma delimited list. Must be a multiple of the chunk size.", type=Tuple3())
@click.option('--xrange', type=Tuple2(), default=None, help="Only delete in this range of x values. Default: None. Must be a multiple of the chunk size.", show_default=True)
@click.option('--yrange', type=Tuple2(), default=None, help="Only delete in this range of y values. Default: None. Must be a multiple of the chunk size.", show_default=True)
@click.option('--zrange', type=Tuple2(), default=None, help="Only delete in this range of z values. Default: None. Must be a multiple of the chunk size.", show_default=True)
@click.pass_context
def delete_images(
  ctx, path, queue, 
  mip, num_mips, shape, xrange, yrange, zrange
):
  """
  Delete the image layer of a dataset.
  """
  bounds = compute_bounds(path, mip, xrange, yrange, zrange)

  tasks = tc.create_deletion_tasks(
    path, mip, num_mips=num_mips, shape=shape, bounds=bounds
  )
  enqueue_tasks(ctx, queue, tasks)

@skeletongroup.command("rm")
@click.argument("path", type=CloudPath())
@click.option('--queue', required=True, help="AWS SQS queue or directory to be used for a task queue. e.g. sqs://my-queue or ./my-queue. See https://github.com/seung-lab/python-task-queue", type=str)
@click.option('--magnitude', default=2, help="Split up the work with 10^(magnitude) prefix based tasks. Default: 2 (100 tasks)")
@click.option('--dir', 'skel_dir', default=None, help="Target this directory instead of the one indicated in the info file.")
@click.pass_context
def skeleton_rm(ctx, path, queue, magnitude, skel_dir):
  """
  Delete skeleton files.
  """
  tasks = tc.create_skeleton_deletion_tasks(
    path, magnitude=magnitude, skel_dir=skel_dir
  )

  enqueue_tasks(ctx, queue, tasks)


@skeletongroup.command("xfer")
@click.argument("src", type=CloudPath())
@click.argument("dest", type=CloudPath())
@click.option('--queue', required=True, help="AWS SQS queue or directory to be used for a task queue. e.g. sqs://my-queue or ./my-queue. See https://github.com/seung-lab/python-task-queue", type=str)
@click.option("--sharded", is_flag=True, default=False, help="Generate shard fragments instead of outputing mesh fragments.", show_default=True)
@click.option("--dir", "skel_dir", type=str, default=None, help="Write skeletons into this directory instead of the one indicated in the info file.")
@click.option('--magnitude', default=2, help="Split up the work with 10^(magnitude) prefix based tasks.", show_default=True)
@click.pass_context
def skel_xfer(
  ctx, src, dest, queue,
  sharded, skel_dir, magnitude
):
  """
  Transfer skeletons to another location.
  Enables conversion of unsharded to sharded
  as well.
  """
  cv_src = CloudVolume(src)

  if not cv_src.skeleton.meta.is_sharded() and sharded:
    tasks = tc.create_sharded_skeletons_from_unsharded_tasks(
      src, dest, skel_dir=skel_dir,
    )
  else:
    tasks = tc.create_xfer_skeleton_tasks(
      src, dest,
      skel_dir=skel_dir, 
      magnitude=magnitude,
    )

  enqueue_tasks(ctx, queue, tasks)

@skeletongroup.command("clean")
@click.argument("src", type=CloudPath())
def skel_clean(src):
  """
  Removes temporary files.
  e.g. .labels and .frags files.
  """
  cv = CloudVolume(src)
  cf = CloudFiles(cv.skeleton.meta.layerpath, progress=True)

  if not cv.skeleton.meta.is_sharded():
    def remove_paths():
      for path in cf.list():
        N = len(os.path.basename(path).split(":"))
        # temp file: segid:0:fragment_name
        if N == 3:
          yield path
  else:
    def remove_paths():
      for path in cf.list():
        _, ext = os.path.splitext(path)
        if ext in (".labels", ".frags"):
          yield path

  cf.delete(remove_paths())

@skeletongroup.group("spatial-index")
def spatialindexgroupskel():
  """
  (subgroup) Create or download mesh spatial indices.
  """
  pass

@spatialindexgroupskel.command("create")
@click.argument("path", type=CloudPath())
@click.option('--queue', required=True, help="AWS SQS queue or directory to be used for a task queue. e.g. sqs://my-queue or ./my-queue. See https://github.com/seung-lab/python-task-queue", type=str)
@click.option('--shape', default="512,512,512", type=Tuple3(), help="Shape in voxels of each indexing task.", show_default=True)
@click.option('--mip', default=0, help="Perform indexing using this level of the image pyramid.", show_default=True)
@click.option('--fill-missing', is_flag=True, default=False, help="Interpret missing image files as background instead of failing.", show_default=True)
@click.pass_context
def skel_spatial_index_create(ctx, path, queue, shape, mip, fill_missing):
  """
  Create a spatial index on pre-existing skeletons.

  Sometimes datasets were skeletonized without a
  spatial index or need it to be updated.
  This function provides a more efficient
  way to accomplish that than reskeletonizing.
  """
  tasks = tc.create_spatial_index_skeleton_tasks(
    cloudpath=path,
    shape=shape,
    mip=mip, 
    fill_missing=fill_missing,
  )
  enqueue_tasks(ctx, queue, tasks)

@spatialindexgroupskel.command("db")
@click.argument("path", type=CloudPath())
@click.argument("database")
@click.option('--progress', is_flag=True, default=False, help="Show progress bars.", show_default=True)
@click.option('--allow-missing', is_flag=True, default=False, help="Allow missing index files.", show_default=True)
@click.pass_context
def skel_spatial_index_download(ctx, path, database, progress, allow_missing):
  """
  Download the skeleton spatial index into a database.

  sqlite paths: sqlite://filename.db (prefix optional)
  mysql paths: mysql://{user}:{pwd}@{host}/{database}
  """
  parallel = int(ctx.obj.get("parallel", 1))
  cv = CloudVolume(path)
  cv.skeleton.spatial_index.to_sql(
    database, 
    progress=progress,
    allow_missing=allow_missing, 
    parallel=parallel,
  )

@main.group("design")
def designgroup():
  """
  Tools to aid the design of tasks or neuroglancer layers.
  """
  pass

@designgroup.command("bounds")
@click.argument("path", type=CloudPath())
@click.option('--mip', default=0, help="Select level of the image pyramid.", show_default=True)
def dsbounds(path, mip):
  """
  Detects the volume bounds and chunk size for
  an unsharded image volume. Useful when there
  is a corrupted info file.
  """
  cv = CloudVolume(path, mip=mip)
  cf = CloudFiles(path)

  bboxes = []
  for filename in tqdm(cf.list(prefix=cv.key), desc="Computing Bounds"):
    bboxes.append( Bbox.from_filename(filename) )

  bounds = Bbox.expand(*bboxes)
  chunk_size = list(reduce(max2, map(lambda bbox: bbox.size3(), bboxes)))

  print(f"Bounds: {bounds}")
  print(f"Volume: {list(bounds.size3())}")
  print(f"Chunks: {chunk_size}")

@designgroup.command("ds-memory")
@click.argument("path", type=CloudPath())
@click.argument("memory_bytes", type=float)
@click.option('--mip', default=0, help="Select level of the image pyramid.", show_default=True)
@click.option('--factor', default="2,2,1", type=Tuple3(), help="Downsample factor to use.", show_default=True)
@click.option('--verbose', is_flag=True, help="Print some additional information.")
@click.option('--max-mips', default=5, help="Maximum downsamples to generate from this shape.", show_default=True)
def dsmemory(path, memory_bytes, mip, factor, verbose, max_mips):
  """
  Compute the task shape that maximizes the number of
  downsamples for a given amount of memory.
  """
  cv = CloudVolume(path, mip=mip)

  data_width = np.dtype(cv.dtype).itemsize
  cx, cy, cz = cv.chunk_size
  memory_bytes = int(memory_bytes)

  shape = downsample_scales.downsample_shape_from_memory_target(
    data_width, 
    cx, cy, cz, factor, 
    memory_bytes, max_mips
  )

  num_downsamples = int(math.log2(max(shape / cv.chunk_size)))

  mem_used = memory_used(data_width, shape, factor)

  shape = [ str(x) for x in shape ]
  shape = ",".join(shape)

  if verbose:
    print(f"Data Width: {data_width} B")
    print(f"Factor: {factor}")
    print(f"Chunk Size: {cx}, {cy}, {cz} voxels")
    print(f"Memory Limit: {format_bytes(memory_bytes, True)}")
    print("-----")
    print(f"Optimized Shape: {shape}")
    print(f"Downsamples: {num_downsamples}")
    print(f"Memory Used*: {format_bytes(mem_used, True)}")
    print(
      "\n"
      "* memory used is for retaining the image "
      "and all downsamples.\nAdditional costs may be incurred "
      "from processing."
    )
  else:
    print(shape)

@designgroup.command("ds-shape")
@click.argument("path", type=CloudPath())
@click.argument("shape", type=Tuple3())
@click.option('--mip', default=0, help="Select level of the image pyramid.", show_default=True)
@click.option('--factor', default="2,2,1", type=Tuple3(), help="Downsample factor to use.", show_default=True)
def dsshape(path, shape, mip, factor):
  """
  Compute the approximate memory usage for a
  given downsample task shape.
  """
  cv = CloudVolume(path, mip=mip)
  data_width = np.dtype(cv.dtype).itemsize
  memory_bytes = memory_used(data_width, shape, factor)  
  print(format_bytes(memory_bytes, True))

def memory_used(data_width, shape, factor): 
  memory_bytes = data_width * shape[0] * shape[1] * shape[2]

  if factor not in ((1,1,1), (2,2,1), (2,1,2), (1,2,2), (2,2,2)):
    print(f"igneous: factor must be 2,2,1 or 2,2,2. Got: {factor}")
    sys.exit()

  # comes from a converging infinite series proof
  constant = factor[0] * factor[1] * factor[2]
  if constant != 1:
    memory_bytes *= constant / (constant - 1)

  return memory_bytes

@main.command("view")
@click.argument("path", type=CloudPath())
@click.option('--browser/--no-browser', default=True, is_flag=True, help="Open the dataset in the system's default web browser.")
@click.option('--port', default=1337, help="localhost server port for the file server.", show_default=True)
@click.option('--ng', default="https://neuroglancer-demo.appspot.com/", help="Alternative Neuroglancer webpage to use.", show_default=True)
def view(path, browser, port, ng):
  """
  Open an on-disk dataset for viewing in neuroglancer.
  """
  rgb_shader = """#uicontrol invlerp normalized
void main() {
  vec3 data = vec3(toNormalized(getDataValue(0)), toNormalized(getDataValue(1)), toNormalized(getDataValue(2)));
  emitRGB(data);
}"""
  
  cv = CloudVolume(path)

  if cv.meta.path.protocol == "file":
    cloudpath = f"http://localhost:{port}"
    layer_name = "igneous"
  else:
    cloudpath = cv.cloudpath
    layer_name = posixpath.basename(cloudpath)

  config = {
    "layers": [
      {
        "type": cv.layer_type,
        "source": f"precomputed://{cloudpath}",
        "tab": "source",
        "name": layer_name
      }
    ],
    "selectedLayer": {
      "visible": True,
      "layer": layer_name
    },
    "layout": "4panel"
  }

  if cv.num_channels == 3:
    config["layers"][0]["shader"] = rgb_shader

  fragment = urllib.parse.quote(json.dumps(config))

  url = f"{ng}#!{fragment}"
  if browser:
    webbrowser.open(url, new=2)

  if cv.meta.path.protocol == "file":
    cv.viewer(port=port)

@imagegroup.command()
@click.argument("src", type=CloudPath())
@click.argument("dest", type=CloudPath())
@click.option('--resolution', type=Tuple3(), default=(1,1,1), help="Resolution of the volume in nanometers along x,y, and z dimensions.", show_default=True)
@click.option('--offset', type=Tuple3(), default=(0,0,0), help="Voxel offset in x,y, and z.", show_default=True)
@click.option('--seg', is_flag=True, default=False, help="Sets layer type to segmentation (default image).", show_default=True)
@click.option('--encoding', type=EncodingType(), default="raw", help=ENCODING_HELP, show_default=True)
@click.option('--compress', type=CompressType(), default="br", help="Set the image compression scheme. Options: 'none', 'gzip', 'br'", show_default=True)
@click.option('--chunk-size', type=Tuple3(), default=(128,128,64), help="Chunk size of new layers. e.g. 128,128,64", show_default=True)
@click.option('--h5-dataset', default="main", help="Which h5 dataset to acccess (hdf5 imports only).", show_default=True)
@click.pass_context
def create(
  ctx, src, dest, 
  resolution, offset, 
  seg, encoding,
  compress, chunk_size,
  h5_dataset
):
  """Create a Precomputed volume from another data source.

  Supports: .npy files, .h5/.hdf5 files, and .ckl files
  
  Hopefully will support others such as TIFF in the future.
  """
  src = src.replace("file://", "")
  ext = normalize_file_ext(src)

  if ext == ".npy":
    with open(src, "rb") as f:
      shape, forder, dtype = read_array_header(f)
    arr = np.lib.format.open_memmap(src, dtype=dtype, shape=shape, fortran_order=forder, mode="r")
  elif ext == ".ckl":
    import crackle
    arr = crackle.util.aload(src)
    if arr.nbytes < int(1e9):
      arr = arr.decompress()
  elif ext in (".h5", ".hdf5"):
    import h5py
    file = h5py.File(src, 'r')
    arr = np.array(file[h5_dataset])
  else:
    print("Format not supported.")
    return

  CloudVolume.from_numpy(
    arr, dest,
    resolution=resolution,
    voxel_offset=offset,
    chunk_size=chunk_size,
    layer_type=("segmentation" if seg else "image"),
    encoding=encoding,
    compress=compress,
    progress=True,
  )

def normalize_file_ext(filename):
  filename, ext = os.path.splitext(filename)

  while True:
    filename, ext2 = os.path.splitext(filename)
    if ext2 in ('.ckl', '.cpso'):
      return ext2
    elif ext2 == '':
      return ext
    ext = ext2

# c/o https://stackoverflow.com/questions/64226337/is-there-a-way-to-read-npy-header-without-loading-the-whole-file
def read_array_header(fobj):
  version = np.lib.format.read_magic(fobj)
  func_name = 'read_array_header_' + '_'.join(str(v) for v in version)
  func = getattr(np.lib.format, func_name)
  return func(fobj)
