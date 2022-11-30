import cloudvolume

def connect(dbpath):
	return cloudvolume.datasource.precomputed.spatial_index\
		.connect(dbpath)