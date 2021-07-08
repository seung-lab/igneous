import math
import multiprocessing as mp
import os
import sys

import click
from cloudvolume import CloudVolume
import numpy as np
from taskqueue import TaskQueue
from taskqueue.lib import toabs
from taskqueue.paths import get_protocol

from igneous import task_creation as tc
from igneous import downsample_scales
from igneous.secrets import LEASE_SECONDS, SQS_REGION_NAME

from igneous_cli.humanbytes import format_bytes

def normalize_path(queuepath):
  if not get_protocol(queuepath):
    return "fq://" + toabs(queuepath)
  return queuepath

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
  

@click.group()
@click.option("-p", "--parallel", default=1, help="Run with this number of parallel processes. If 0, use number of cores.")
@click.version_option(version="0.4.1")
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

@main.command()
@click.argument("path")
@click.option('--queue', default=None, required=True, help="AWS SQS queue or directory to be used for a task queue. e.g. sqs://my-queue or ./my-queue. See https://github.com/seung-lab/python-task-queue")
@click.option('--mip', default=0, help="Build upward from this level of the image pyramid. Default: 0")
@click.option('--fill-missing', is_flag=True, default=False, help="Interpret missing image files as background instead of failing.")
@click.option('--num-mips', default=5, help="Build this many additional pyramid levels. Each increment increases memory requirements per task 4-8x.  Default: 5")
@click.option('--cseg', is_flag=True, default=False, help="Use the compressed_segmentation image chunk encoding scheme. Segmentation only.")
@click.option('--compresso', is_flag=True, default=False, help="Use the compresso image chunk encoding scheme. Segmentation only.")
@click.option('--sparse', is_flag=True, default=False, help="Don't count black pixels in mode or average calculations. For images, eliminates edge ghosting in 2x2x2 downsample. For segmentation, prevents small objects from disappearing at high mip levels.")
@click.option('--chunk-size', type=Tuple3(), default=None, help="Chunk size of new layers. e.g. 128,128,64")
@click.option('--compress', default=None, help="Set the image compression scheme. Options: 'gzip', 'br'")
@click.option('--volumetric', is_flag=True, default=False, help="Use 2x2x2 downsampling.")
@click.option('--delete-bg', is_flag=True, default=False, help="Issue a delete instead of uploading a background tile. This is helpful on systems that don't like tiny files.")
@click.option('--bg-color', default=0, help="Determines which color is regarded as background. Default: 0")
@click.pass_context
def downsample(
	ctx, path, queue, mip, fill_missing, 
	num_mips, cseg, compresso, sparse, 
	chunk_size, compress, volumetric,
	delete_bg, bg_color
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
  if cseg and compresso:
    print("igneous: must choose one of --cseg or --compresso")
    sys.exit()

  encoding = None
  if cseg:
    encoding = "compressed_segmentation"
  elif compresso:
    encoding = "compresso"

  factor = (2,2,1)
  if volumetric:
  	factor = (2,2,2)

  tasks = tc.create_downsampling_tasks(
    path, mip=mip, fill_missing=fill_missing, 
    num_mips=num_mips, sparse=sparse, 
    chunk_size=chunk_size, encoding=encoding, 
    delete_black_uploads=delete_bg, 
    background_color=bg_color, 
    compress=compress,
    factor=factor  	
  )

  parallel = int(ctx.obj.get("parallel", 1))
  tq = TaskQueue(normalize_path(queue))
  tq.insert(tasks, parallel=parallel)

@main.command()
@click.argument("src")
@click.argument("dest")
@click.option('--queue', default=None, required=True, help="AWS SQS queue or directory to be used for a task queue. e.g. sqs://my-queue or ./my-queue. See https://github.com/seung-lab/python-task-queue")
@click.option('--mip', default=0, help="Build upward from this level of the image pyramid.", show_default=True)
@click.option('--translate', type=Tuple3(), default=(0, 0, 0), help="Translate the bounding box by X,Y,Z voxels in the new location.")
@click.option('--downsample/--skip-downsample', is_flag=True, default=True, help="Whether or not to produce downsamples from transfer tiles.", show_default=True)
@click.option('--fill-missing', is_flag=True, default=False, help="Interpret missing image files as background instead of failing.")
@click.option('--memory', default=3.5e9, type=int, help="Task memory limit in bytes. Task shape will be chosen to fit and maximize downsamples.", show_default=True)
@click.option('--max-mips', default=5, help="Maximum number of additional pyramid levels.", show_default=True)
@click.option('--cseg', is_flag=True, default=False, help="Use the compressed_segmentation image chunk encoding scheme. Segmentation only.")
@click.option('--compresso', is_flag=True, default=False, help="Use the compresso image chunk encoding scheme. Segmentation only.")
@click.option('--sparse', is_flag=True, default=False, help="Don't count black pixels in mode or average calculations. For images, eliminates edge ghosting in 2x2x2 downsample. For segmentation, prevents small objects from disappearing at high mip levels.")
@click.option('--shape', type=Tuple3(), default=(2048, 2048, 64), help="(overrides --memory) Set the task shape in voxels. This also determines how many downsamples you get. e.g. 2048,2048,64")
@click.option('--chunk-size', type=Tuple3(), default=None, help="Chunk size of destination layer. e.g. 128,128,64")
@click.option('--compress', default="gzip", help="Set the image compression scheme. Options: 'gzip', 'br'", show_default=True)
@click.option('--volumetric', is_flag=True, default=False, help="Use 2x2x2 downsampling.")
@click.option('--delete-bg', is_flag=True, default=False, help="Issue a delete instead of uploading a background tile. This is helpful on systems that don't like tiny files.")
@click.option('--bg-color', default=0, help="Determines which color is regarded as background.", show_default=True)
@click.pass_context
def xfer(
	ctx, src, dest, queue, translate, downsample, mip, 
	fill_missing, memory, max_mips, 
  shape, sparse, cseg, compresso,
	chunk_size, compress, volumetric,
	delete_bg, bg_color
):
  """
  Transfer an image layer to another location.

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
  if cseg and compresso:
    print("igneous: must choose one of --cseg or --compresso")
    sys.exit()

  encoding = None
  if cseg:
    encoding = "compressed_segmentation"
  elif compresso:
    encoding = "compresso"

  factor = (2,2,1)
  if volumetric:
  	factor = (2,2,2)

  if compress and compress.lower() == "false":
    compress = False

  tasks = tc.create_transfer_tasks(
    src, dest, 
    chunk_size=chunk_size, fill_missing=fill_missing, 
    translate=translate, mip=mip, shape=shape,
    encoding=encoding, skip_downsamples=(not downsample),
    delete_black_uploads=delete_bg, background_color=bg_color,
    compress=compress, factor=factor, sparse=sparse,
    memory_target=memory, max_mips=max_mips
  )

  parallel = int(ctx.obj.get("parallel", 1))
  tq = TaskQueue(normalize_path(queue))
  tq.insert(tasks, parallel=parallel)

@main.command()
@click.argument("queue", type=str)
@click.option('--aws-region', default=SQS_REGION_NAME, help=f"AWS region in which the SQS queue resides. Default: {SQS_REGION_NAME}")
@click.option('--lease-sec', default=LEASE_SECONDS, help=f"Seconds to lease a task for. Default: {LEASE_SECONDS}", type=int)
@click.option('--tally/--no-tally', is_flag=True, default=True, help="Tally completed fq tasks. Does not apply to SQS.")
@click.option('--min-sec', default=-1, help='Execute for at least this many seconds and quit after the last task finishes. Special values: (0) Run at most a single task. (-1) Loop forever (default).', type=float)
@click.pass_context
def execute(ctx, queue, aws_region, lease_sec, tally, min_sec):
  """Execute igneous tasks from a queue.

  The queue must be an AWS SQS queue or a FileQueue directory. 
  Examples: (SQS) sqs://my-queue (FileQueue) fq://./my-queue
  (the fq:// is optional).

  See https://github.com/seung-lab/python-task-queue
  """
  parallel = int(ctx.obj.get("parallel", 1))
  args = (queue, aws_region, lease_sec, tally, min_sec)

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

def execute_helper(queue, aws_region, lease_sec, tally, min_sec):
  tq = TaskQueue(normalize_path(queue), region_name=aws_region)

  def stop_after_elapsed_time(elapsed_time):
    if min_sec < 0:
      return False
    return min_sec < elapsed_time

  if min_sec != 0:
    tq.poll(
      lease_seconds=lease_sec,
      verbose=True,
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

@meshgroup.command("forge")
@click.argument("path")
@click.option('--queue', required=True, help="AWS SQS queue or directory to be used for a task queue. e.g. sqs://my-queue or ./my-queue. See https://github.com/seung-lab/python-task-queue", type=str)
@click.option('--mip', default=0, help="Perform meshing using this level of the image pyramid. Default: 0")
@click.option('--shape', type=Tuple3(), default=(448, 448, 448), help="Set the task shape in voxels. Default: 448,448,448")
@click.option('--simplify/--skip-simplify', is_flag=True, default=True, help="Enable mesh simplification. Default: True")
@click.option('--fill-missing', is_flag=True, default=False, help="Interpret missing image files as background instead of failing.")
@click.option('--max-error', default=40, help="Maximum simplification error in physical units. Default: 40 nm")
@click.option('--dust-threshold', default=None, help="Skip meshing objects smaller than this number of voxels within a cutout. No default limit. Typical value: 1000.", type=int)
@click.option('--dir', default=None, help="Write meshes into this directory instead of the one indicated in the info file.")
@click.option('--compress', default=None, help="Set the image compression scheme. Options: 'gzip', 'br'")
@click.option('--spatial-index/--skip-spatial-index', is_flag=True, default=True, help="Create the spatial index.")
@click.pass_context
def mesh_forge(
  ctx, path, queue, mip, shape, 
  simplify, fill_missing, max_error, 
  dust_threshold, dir, compress, 
  spatial_index
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

  Sharded format not currently supports. Coming soon.
  """
  tasks = tc.create_meshing_tasks(
    path, mip, shape, 
    simplification=simplify, max_simplification_error=max_error,
    mesh_dir=dir, cdn_cache=False, dust_threshold=dust_threshold,
    object_ids=None, progress=False, fill_missing=fill_missing,
    encoding='precomputed', spatial_index=spatial_index, 
    sharded=False, compress=compress
  )

  parallel = int(ctx.obj.get("parallel", 1))
  tq = TaskQueue(normalize_path(queue))
  tq.insert(tasks, parallel=parallel)

@meshgroup.command("merge")
@click.argument("path")
@click.option('--queue', required=True, help="AWS SQS queue or directory to be used for a task queue. e.g. sqs://my-queue or ./my-queue. See https://github.com/seung-lab/python-task-queue", type=str)
@click.option('--magnitude', default=2, help="Split up the work with 10^(magnitude) prefix based tasks. Default: 2 (100 tasks)")
@click.option('--dir', default=None, help="Write manifests into this directory instead of the one indicated in the info file.")
@click.pass_context
def mesh_merge(ctx, path, queue, magnitude, dir):
  """
  (2) Merge the mesh pieces produced from the forging step.

  The per-cutout mesh fragments are then assembled and
  merged. However, this process occurs by compiling 
  a list of fragment files and uploading a "mesh manifest"
  file that is an index for locating the fragments.
  """
  tasks = tc.create_mesh_manifest_tasks(
    path, magnitude=magnitude, mesh_dir=dir
  )

  parallel = int(ctx.obj.get("parallel", 1))
  tq = TaskQueue(normalize_path(queue))
  tq.insert(tasks, parallel=parallel)


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
@click.argument("path")
@click.option('--queue', required=True, help="AWS SQS queue or directory to be used for a task queue. e.g. sqs://my-queue or ./my-queue. See https://github.com/seung-lab/python-task-queue", type=str)
@click.option('--mip', default=0, help="Perform skeletonizing using this level of the image pyramid. Default: 0")
@click.option('--shape', type=Tuple3(), default=(512, 512, 512), help="Set the task shape in voxels. Default: 512,512,512")
@click.option('--fill-missing', is_flag=True, default=False, help="Interpret missing image files as background instead of failing.")
@click.option('--fix-branching', is_flag=True, default=True, help="Trades speed for quality of branching at forks. Default: True")
@click.option('--fix-borders', is_flag=True, default=True, help="Allows trivial merging of single voxel overlap tasks. Only switch off for datasets that fit in a single task. Default: True")
@click.option('--fix-avocados', is_flag=True, default=False, help="Fixes somata where nuclei and cytoplasm have separate segmentations. Default: False")
@click.option('--fill-holes', is_flag=True, default=False, help="Preprocess each cutout to eliminate background holes and holes caused by entirely contained inclusions. Warning: May remove labels that are considered inclusions. Default: False")
@click.option('--dust-threshold', default=1000, help="Skip skeletonizing objects smaller than this number of voxels within a cutout. Default: 1000.", type=int)
@click.option('--spatial-index/--skip-spatial-index', is_flag=True, default=True, help="Create the spatial index.")
@click.option('--scale', default=4, help="Multiplies invalidation radius by distance from boundary.", type=float)
@click.option('--const', default=10, help="Adds constant amount to invalidation radius in physical units.", type=float)
@click.option('--soma-detect', default=1100, help="Consider objects with peak distances to boundary larger than this soma candidates. Physical units. Default: 1100 nm", type=float)
@click.option('--soma-accept', default=3500, help="Accept soma candidates over this threshold and perform a one-time spherical invalidation around their peak value. Physical units. Default: 3500 nm", type=float)
@click.option('--soma-scale', default=1.0, help="Scale factor for soma invalidation.", type=float)
@click.option('--soma-const', default=300, help="Const factor for soma invalidation.", type=float)
@click.option('--max-paths', default=None, help="Abort skeletonizing an object after this many paths have been traced.", type=float)
@click.option('--sharded', is_flag=True, default=False, help="Generate shard fragments instead of outputing skeleton fragments.")
@click.pass_context
def skeleton_forge(
  ctx, path, queue, mip, shape, 
  fill_missing, dust_threshold, spatial_index,
  fix_branching, fix_borders, fix_avocados, 
  fill_holes, scale, const, soma_detect, soma_accept,
  soma_scale, soma_const, max_paths, sharded
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
  )

  parallel = int(ctx.obj.get("parallel", 1))
  tq = TaskQueue(normalize_path(queue))
  tq.insert(tasks, parallel=parallel)


@skeletongroup.command("merge")
@click.argument("path")
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

  parallel = int(ctx.obj.get("parallel", 1))
  tq = TaskQueue(normalize_path(queue))
  tq.insert(tasks, parallel=parallel)

@skeletongroup.command("merge-sharded")
@click.argument("path")
@click.option('--queue', required=True, help="AWS SQS queue or directory to be used for a task queue. e.g. sqs://my-queue or ./my-queue. See https://github.com/seung-lab/python-task-queue", type=str)
@click.option('--min-cable-length', default=1000, help="Skip objects smaller than this physical path length. Default: 1000 nm", type=float)
@click.option('--max-cable-length', default=None, help="Skip objects larger than this physical path length. Default: no limit", type=float)
@click.option('--tick-threshold', default=0, help="Remove small \"ticks\", or branches from the main skeleton one at a time from smallest to largest. Branches larger than this are preserved. Default: no elimination", type=float)
@click.option('--preshift-bits', default=3, help="Shift LSB to try to increase number of hash collisions.", type=int)
@click.option('--minishard-bits', default=10, help="2^bits number of bays holding variable numbers of labels per shard.", type=int)
@click.option('--shard-bits', default=2, help="2^bits number of shard files to generate.", type=int)
@click.option('--minishard-index-encoding', default="gzip", help="Minishard indices can be compressed. gzip or raw. Default: gzip")
@click.option('--data-encoding', default="gzip", help="Shard data can be compressed. gzip or raw. Default: gzip")
@click.pass_context
def skeleton_sharded_merge(
  ctx, path, queue, 
  min_cable_length, max_cable_length, 
  tick_threshold, 
  preshift_bits, minishard_bits, shard_bits,
  minishard_index_encoding, data_encoding
):
  """
  (2) Postprocess fragments into finished skeletons.

  Only use this command if you used the --sharded flag
  during the forging step. Some reasonable defaults
  are selected for a dataset with a few million labels,
  but for smaller or larger datasets they may not be
  appropriate.
  """
  tasks = tc.create_sharded_skeleton_merge_tasks(
    path, 
    dust_threshold=min_cable_length,
    max_cable_length=max_cable_length,
    tick_threshold=tick_threshold,
    preshift_bits=preshift_bits, 
    minishard_bits=minishard_bits, 
    shard_bits=shard_bits,
    minishard_index_encoding=minishard_index_encoding, 
    data_encoding=data_encoding,
  )

  parallel = int(ctx.obj.get("parallel", 1))
  tq = TaskQueue(normalize_path(queue))
  tq.insert(tasks, parallel=parallel)

@main.group("rm")
def deletegroup():
  """
  Parallelize the deletion process for
  different kinds of data.
  """
  pass

@deletegroup.command("image")
@click.argument("path")
@click.option('--queue', required=True, help="AWS SQS queue or directory to be used for a task queue. e.g. sqs://my-queue or ./my-queue. See https://github.com/seung-lab/python-task-queue", type=str)
@click.option('--mip', default=0, help="Which mip level to start deleting from. Default: 0")
@click.option('--num-mips', default=5, help="The number of mip levels to delete at once. Default: 5")
@click.option('--shape', default=None, help="The size of each deletion task as a comma delimited list. Must be a multiple of the chunk size.", type=Tuple3())
@click.pass_context
def delete_images(
  ctx, path, queue, 
  mip, num_mips, shape
):
  """
  Delete the image layer of a dataset.
  """
  tasks = tc.create_deletion_tasks(
    path, mip, num_mips=num_mips, shape=shape
  )
  parallel = int(ctx.obj.get("parallel", 1))
  tq = TaskQueue(normalize_path(queue))
  tq.insert(tasks, parallel=parallel)

@main.group("design")
def designgroup():
  """
  Tools to aid the design of tasks or neuroglancer layers.
  """
  pass


@designgroup.command("ds-memory")
@click.argument("path")
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
@click.argument("path")
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
