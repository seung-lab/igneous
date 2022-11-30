"""
Tasks for converting between different alignments
for the same dataset.
"""
from cloudvolume import CloudVolume
from cloudvolume.lib import Vec, Bbox

import cc3d
import fastremap

from . import sql

# ws: gs://neuroglancer/pinky100_v0/ws/pinky100_ca_com
# seg: gs://neuroglancer/pinky100_v185/seg/
# graphene: graphene://https://minnie.microns-daf.com/segmentation/table/pinky_nf_v2
# ccl: precomputed://gs://interalignment-pinky/pinky100_v185_ccl

def PreciseWatershedRemapTask(
  supervoxel_path, sv_mip, 
  segmentation_path, seg_mip,
  ccl_path, ccl_mip,
  shape, offset,
  db_path
):

  shape = Vec(*shape)
  offset = Vec(*offset)
  bounds = Bbox(offset, offset + shape)

  sv_cv = CloudVolume(supervoxel_path, mip=sv_mip)
  seg_cv = CloudVolume(segmentation_path, mip=seg_mip)
  ccl_cv = CloudVolume(ccl_path, mip=ccl_mip)

  bounds = Bbox.clamp(bounds, sv_cv.bounds)

  sv_labels = sv_cv[bounds]
  seg_labels = seg_cv[bounds]

  sv_seg_map = fastremap.component_map(sv_labels, seg_labels)
  del seg_labels

  ccl_labels = ccl_cv[bounds]
  sv_ccl_map = fastremap.component_map(sv_labels, ccl_labels)
  del ccl_labels

  sv_relabeled, mapping = fastremap.renumber(sv_labels)
  stats = cc3d.statistics(sv_relabeled)
  del sv_relabeled

  centroids = { 
    mapping[i]: pt for i, pt in enumerate(stats["centroids"]) 
  }

  rows = []
  for svid, pt in centroids.items():
    row = [
      int(svid), int(sv_seg_map[svid]), int(sv_ccl_map[svid]),
      int(pt[0]), int(pt[1]), int(pt[2])
    ]
    rows.append(row)

  conn = sql.connect(db_path)
  cur = conn.cursor()

  binds = ",".join(["?"] * 5)
  binds = f"({binds})"
  binds = ",".join([binds] * len(rows))

  cur.execute(f"""
    INSERT INTO 
    supervoxel_mapping (supervoxel_id, label_id, ccl_id, x, y, z) 
    VALUES {binds}
  """, rows)
  cur.commit()
  cur.close()
  conn.close()




