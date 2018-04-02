[![Build Status](https://travis-ci.org/seung-lab/igneous.svg?branch=master)](https://travis-ci.org/seung-lab/igneous)

# igneous
Igneous is a library useful for working with data in Neuroglancer's precomputed volumes. It uses CloudVolume for access to the data (on AWS S3, Google GS, or on the filesystem). It is meant to integrate with a task queueing system (but has a single-worker mode as well). Originally by Nacho and Will.

It supports the following jobs:
* ingest
* hypersquare ingest
* downsample
* deletion
* meshing
* transfer
* wastershed remap
* quantized affinity

As of present, read igneous/task_creation.py for specifics of these jobs.

# Sample use
This generates meshes for an already-existing precomputed segmentation volume. It uses the
MockTaskQueue driver (which is the single-worker mode).
```
from taskqueue import MockTaskQueue
import igneous.task_creation as tc

print("Making meshes...")
tc.create_meshing_tasks(       MockTaskQueue(), cfg.path, cfg.compression, shape=Vec(cfg.size, cfg.size, cfg.size))
print("Updating metadata...")
tc.create_mesh_manifest_tasks( MockTaskQueue(), cfg.path)
print("Done!")

```
