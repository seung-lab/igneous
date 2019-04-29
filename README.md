[![Build Status](https://travis-ci.org/seung-lab/igneous.svg?branch=master)](https://travis-ci.org/seung-lab/igneous) [![SfN 2018 Poster](https://img.shields.io/badge/poster-SfN%202018-blue.svg)](https://drive.google.com/open?id=1RKtaAGV2f7F13opnkQfbp6YBqmoD3fZi)

# Igneous

Igneous is a [Kubernetes](https://kubernetes.io/), [SQS](https://aws.amazon.com/sqs/), and CloudVolume based pipeline for working with Neuroglancer's [Precomputed](https://github.com/google/neuroglancer/tree/master/src/neuroglancer/datasource/precomputed) volumes. It uses [CloudVolume](https://github.com/seung-lab/cloud-volume) for accessing data on AWS S3, Google Storage, or the local filesystem. It can operate in the cloud using a task queuing system or run locally. Originally by Nacho and Will.

## Pre-Built Docker Container

You can use this container for scaling big jobs horizontally or to experiment with Igneous within the container.  

https://hub.docker.com/r/seunglab/igneous/

## Installation

You'll need Python 2 or 3, pip, a C++ compiler (g++ or clang), and virtualenv. Igneous appears to have higher performance using Python 3. It's tested under Ubuntu 14.04, Ubuntu 16.04 and Mac OS High Sierra.

```bash
git clone git@github.com:seung-lab/igneous.git
cd igneous
virtualenv venv
source venv/bin/activate
pip install numpy
pip install -r requirements.txt
python setup.py develop
```

The installation will download the dependencies listed in requirements.txt and
compile a meshing extension written by Aleksander Zlateski. The mesher is used to visualize segments in neuroglancer's 3D viewer.  

## Sample Local Use

This generates meshes for an already-existing precomputed segmentation volume. It uses the
MockTaskQueue driver (which is the single local worker mode).

```python3
from taskqueue import LocalTaskQueue
import igneous.task_creation as tc

# Mesh on 8 cores, use True to use all cores
cloudpath = 'gs://bucket/dataset/labels'
with LocalTaskQueue(parallel=8) as tq:
  tasks = tc.create_meshing_tasks(cloudpath, mip=3, shape=(256, 256, 256))
  tq.insert_all(tasks)
  tasks = tc.create_mesh_manifest_tasks(cloudpath)
  tq.insert_all(tasks)
print("Done!")

```

## Sample Cloud Use

Igneous is intended to be used with Kubernetes (k8s). A pre-built docker container is located on DockerHub as `seunglab/igneous:master`. A sample `deployment.yml` (used with `kubectl create -f deployment.yml`) is located in the root of the repository.  

As Igneous is based on [CloudVolume](https://github.com/seung-lab/cloud-volume), you'll need to create a `google-secret.json` or `aws-secret.json` to access buckets located on these services. 

You'll need to create an Amazon SQS queue to store the tasks you generate. Google's TaskQueue was previously supported but the API changed. It may be supported in the future.

### Populating the SQS Queue

There's a bit of an art to achieving high performance on SQS. You can read more about it [here](https://github.com/seung-lab/python-task-queue#how-to-achieve-high-performance).

```python3
import sys
from taskqueue import TaskQueue
import igneous.task_creation as tc

cloudpath = sys.argv[1]

# Get qurl from the SQS queue metadata, visible on the web dashboard when you click on it.
with TaskQueue(queue_server='sqs', qurl="$URL") as tq:
  tasks = tc.create_downsampling_tasks(
    cloudpath, mip=0, 
    fill_missing=True, preserve_chunk_size=True
  )
  tq.insert_all(tasks)
print("Done!")
```

### Executing Tasks in the Cloud

The following instructions are for Google Container Engine, but AWS has similar tools.

```bash
# Create a Kubernetes cluster
# e.g. 

export PROJECT_NAME=example
export CLUSTER_NAME=example
export NUM_NODES=5 # arbitrary

# Create a Google Container Cluster
gcloud container --project $PROJECT_NAME clusters create $CLUSTER_NAME --zone "us-east1-b" --machine-type "n1-standard-16" --image-type "GCI" --disk-size "50" --scopes "https://www.googleapis.com/auth/compute","https://www.googleapis.com/auth/devstorage.full_control","https://www.googleapis.com/auth/taskqueue","https://www.googleapis.com/auth/logging.write","https://www.googleapis.com/auth/cloud-platform","https://www.googleapis.com/auth/servicecontrol","https://www.googleapis.com/auth/service.management.readonly","https://www.googleapis.com/auth/trace.append" --num-nodes $NUM_NODES --network "default" --enable-cloud-logging --no-enable-cloud-monitoring

# Bind the kubectl command to this cluster
gcloud config set container/cluster $CLUSTER_NAME

# Give the cluster permission to read and write to your bucket
# You only need to include services you'll actually use.
kubectl create secret generic secrets \
--from-file=$HOME/.cloudvolume/secrets/google-secret.json \
--from-file=$HOME/.cloudvolume/secrets/aws-secret.json \
--from-file=$HOME/.cloudvolume/secrets/boss-secret.json 

# Create a Kubernetes deployment
kubectl create -f deployment.yml # edit deployment.yml in root of repo

# Resizing the cluster
gcloud container clusters resize $CLUSTER_NAME --size=20 # arbitrary node count
kubectl scale deployment igneous --replicas=320 # 16 * nodes b/c n1-standard-16 has 16 cores

# Spinning down

# Important: This will leave the kubernetes master running which you
# will be charged for. You can also fully delete the cluster.
gcloud container clusters resize $CLUSTER_NAME --size=0
kubectl delete deployment igneous
```

## Capabilities

You can find the following tasks in `igneous/tasks/tasks.py` and can use them via editing or importing functions from `igneous/task_creation.py`. 

Capability               |Tasks                                          |Description                                                          
:-----:|:-----:|:-----:
Downsampling             |DownsampleTask                                 |Generate image hierarchies.                                          
Meshing                  |MeshTask, MeshManifestTask                     |Create object meshes viewable in Neuroglancer.                       
Skeletonize              |SkeletonTask, SkeletonMergeTask                |Create Neuroglancer viewable skeletons using a modified TEASAR algorithm.        
Transfer                 |TransferTask                                   |Copy data, supports rechunking and coordinate translation.           
Deletion                 |DeleteTask                                     |Delete a data layer.                                                 
Contrast Normalization   |LuminanceLevelsTask, ContrastNormalizationTask |Spread out slice histograms to fill value range.                     
Quantization             |QuantizeAffinitiesTask                         |Rescale values into 8-bit to make them easier to visualize.          
Remapping                |WatershedRemapTask                             |Remap segmentations to create agglomerated labels.                   
Eyewire Consensus Import |HyperSquareConsensusTask                       |Map Eyewire consensus into Neuroglancer.                             
Ingest                   |IngestTask                                     |(deprecated) Convert HDF5 into Precomputed format.                   
HyperSquare Ingest       |HyperSquareTask                                |(deprecated) Convert Eyewire's HyperSquare format into Precomputed.  


### Downsampling (DownsampleTask)

For any but the very smallest volumes, it's desirable to create smaller summary images of what may be multi-gigabyte 
2D slices. The purpose of these summary images is make it easier to visualize the dataset or to work with lower
resolution data in the context of a data processing (e.g. ETL) pipeline.

While these options are configurable by hacking the code, image type datasets are downsampled 
in an recursive hierarchy using 2x2x1 average pooling. Segmentation type datasets (i.e. human ground truth
or machine labels) are downsampled using 2x2x1 mode pooling in a recursive hierarchy using the [COUNTLESS
algorithm](https://towardsdatascience.com/countless-high-performance-2x-downsampling-of-labeled-images-using-python-and-numpy-e70ad3275589). This means 
that mip 1 segmentation labels are exact mode computations, but subsequent ones may not be. 

Whether image or segmentation type downsampling will be used is determined from the neuroglancer info file's "type" attribute.

```python3
tasks = create_downsampling_tasks(
    layer_path, 
    mip=0, # Start downsampling from this mip level (writes to next level up)
    fill_missing=False, # Ignore missing chunks and fill them with black
    axis='z', 
    num_mips=5, # number of downsamples to produce. Downloaded shape is chunk_size * 2^num_mip
    chunk_size=None, # manually set chunk size of next scales, overrides preserve_chunk_size
    preserve_chunk_size=True, # use existing chunk size, don't halve to get more downsamples
    sparse=False, # for sparse segmentation, allow inflation of pixels against background
    bounds=None, # mip 0 bounding box to downsample 
    encoding=None # e.g. 'raw', 'compressed_segmentation', etc
  )
```

1. layer_path 
	e.g. 'gs://bucket/dataset/layer'
2. mip
	Which level of the hierarchy to start from, 0 is highest resolution, -1 means use the top downsample.
3. fill_missing
	If a file chunk is missing, fill it with zeros instead of throwing an error.
4. num_mips
	How many mips to to generate in this operation? More mips can mean at least 2^num_mips times the underlying chunk size.  
5. preserve_chunk_size: 
	False: Use a fixed block size and generate downsamples with decreasing chunk size. 
	True: Use a fixed underlying chunk size and increase the size of the base block to accomodate it and num_mips.
6. sparse: 
	Only has an effect on segmentation type images.
	False: The dataset contains large continuous labeled areas (most connectomics datasets). Uses the [COUNTLESS 2D](https://towardsdatascience.com/countless-high-performance-2x-downsampling-of-labeled-images-using-python-and-numpy-e70ad3275589) algorithm.
	True: The dataset contains sparse labels that are disconnected. Use the [Stippled COUNTLESS 2D](https://medium.com/@willsilversmith/countless-2d-inflated-2x-downsampling-of-labeled-images-holding-zero-values-as-background-4d13a7675f2d) algorithm.

### Data Transfer / Rechunking (TransferTask)

A common task is to take a dataset that was set up as single slices (X by Y by 1) chunks. This is often appropriate
for image alignment or other single section based processing tasks. However, this is not optimal for Neuroglancer
visualization or for achieving the highest performance over TCP networking (e.g. with [CloudVolume](https://github.com/seung-lab/cloud-volume)). Therefore, it can make sense to rechunk the dataset to create deeper and overall larger chunks (e.g. 64x64x64, 128x128x32, 128x128x64). In some cases, it can also be desirable to translate the coordinate system of a data layer. 

The `TransferTask` will automatically run the first few levels of downsampling as well, making it easier to
visualize progress and reducing the amount of work a subsequent `DownsampleTask` will need to do.

Another use case is to transfer a neuroglancer dataset from one cloud bucket to another, but often the cloud
provider's transfer service will suffice, even across providers. 

```python3
tasks = create_transfer_tasks(src_layer_path, dest_layer_path, 
	shape=Vec(2048, 2048, 64), fill_missing=False, translate=(0,0,0))
```

### Deletion (DeleteTask)  

If you want to parallelize deletion of a data layer in a bucket beyond using e.g. `gsutil -m rm`, you can 
horizontally scale out deleting using these tasks. Note that the tasks assume that the information to be deleted
is chunk aligned and named appropriately.

```python3
tasks = create_deletion_tasks(layer_path)
```

### Meshing (MeshTask & MeshManifestTask)

*Requires compilation of `mesher.so` using a C++ compiler.*  

Meshing is a two stage process. First, the dataset is divided up into atomic chunks that will be meshed independently of
each other using the `MeshTask`. The resulting mesh fragments are uploaded to the destination layer's meshing directory 
(which might be named something like `mesh_mip_3_err_40`). Without additional processing, Neuroglancer has no way of 
knowing the names of these chunks (which will be named something like `$SEGID:0:$BOUNDING_BOX` e.g. `1052:0:0-512_0-512_0-512`). 
The `$BOUNDING_BOX` part of the name is arbitrary and is the convention used by igneous because it is convenient for debugging.

The second stage is running the `MeshManifestTask` which generates files named `$SEGID:0`. It contains a short JSON snippet that
looks like `{ "fragments": [ "1052:0:0-512_0-512_0-512" ] }`. This file tells neuroglancer which mesh files to download.  

```python3
tasks = create_meshing_tasks(layer_path, mip, shape=Vec(512, 512, 512)) # First Pass
tasks = create_mesh_manifest_tasks(layer_path, magnitude=3) # Second Pass
```

The parameters above are mostly self explainatory, but the magnitude parameter of create_mesh_manifest_tasks is a bit odd. What a MeshManifestTask does is iterate through a proportion of the files defined by a filename prefix. `magnitude` splits up the work by 
an additional 10^magnitude. A high magnitude (3-5+) is appropriate for horizontal scaling workloads while small magnitudes 
(1-2) are more suited for small volumes locally processed since there is overhead introduced by splitting up the work.  

In the future, a third stage might be introduced that fuses all the small fragments into a single file.

### Skeletonization (SkeletonTask, SkeletonMergeTask)

Igneous provides the engine for performing out-of-core skeletonization of labeled images. 
The in-core part of the algorithm is provided by the [Kimimaro](https://github.com/seung-lab/kimimaro) library.  

The strategy is to apply Kimimaro mass skeletonization to 50% overlapping chunks of the segmentation and then fuse them in a second pass. 

```python3
import igneous.task_creation as tc 

# First Pass: Generate Skeletons
tasks = tc.create_skeletonization_tasks(
    cloudpath, mip, 
    shape=(512, 512, 512), 
    # see Kimimaro's documentation for the below parameters
    teasar_params={ ... }, # controls skeletonization 
    object_ids=None, # object id mask if applicable
  )

# Second Pass: Fuse Skeletons
tasks = tc.create_skeleton_merge_tasks(
  layer_path, mip, 
  crop=0, # in voxels
  magnitude=3, # same as mesh manifests
  dust_threshold=4000, # in nm
  tick_threshold=6000, # in nm
  delete_fragments=False
)
```

### Contrast Normalization (LuminanceLevelsTask & ContrastNormalizationTask)

Sometimes a dataset's luminance values cluster into a tight band and make the image unnecessarily bright or dark and above all
low contrast. Sometimes the data may be 16 bit, but the values cluster all at the low end, making it impossible to even see without
using ImageJ / Fiji or another program that supports automatic image normalization. Furthermore, Fiji can only go so far on a 
Teravoxel or Petavoxel dataset. 

The object of these tasks are to first create a representative sample of the luminance levels of a dataset per a Z slice (i.e. a frequency count of gray values). This levels information is then used to perform per Z section contrast normalization. In the future, perhaps we will attempt global normalization. The algorithm currently in use reads the levels files for a given Z slice,
determines how much of the ends of the distribution to lop off, perhaps 1% on each side (you should plot the levels files for your own data as this is configurable, perhaps you might choose 0.5% or 0.25%). The low value is recentered at 0, and the high value is stretched to 255 (in the case of uint8s) or 65,535 (in the case of uint16).

```python3
# First Pass: Generate $layer_path/levels/$mip/
tasks = create_luminance_levels_tasks(layer_path, coverage_factor=0.01, shape=None, offset=(0,0,0), mip=0) 
# Second Pass: Read Levels to stretch value distribution to full coverage
tasks = create_contrast_normalization_tasks(src_path, dest_path, shape=None, mip=0, clip_fraction=0.01, fill_missing=False, translate=(0,0,0))
```

## Conclusion

It's possible something has changed or is not covered in this documentation. Please read `igneous/task_creation.py` and `igneous/tasks/tasks.py` for the most current information.  

Please post an issue or PR if you think something needs to be addressed.  

## Related Projects  

- [tinybrain](https://github.com/seung-lab/tinybrain) - Downsampling code for images and segmentations.
- [kimimaro](https://github.com/seung-lab/kimimaro) - Skeletonization of dense volumetric labels.
- [CloudVolume](https://github.com/seung-lab/cloud-volume) - IO for images, meshes, and skeletons.
- [python-task-queue](https://github.com/seung-lab/python-task-queue) - Parallelized dependency-free cloud task management.

