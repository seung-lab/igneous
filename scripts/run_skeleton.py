from taskqueue import LocalTaskQueue
import igneous.task_creation as tc
import cloudvolume
import copy
import argschema
import marshmallow as mm

class TeasarParams(argschema.schemas.DefaultSchema):
    dbf_exponent = argschema.fields.Int(required=False, default=16)
    dbf_scale = argschema.fields.Int(required=False, default=5000, description="scaling factor in front of modified dbf to weight against euclidean distance")
    scale = argschema.fields.Int(required=False, default=10, description="factor to multiply DBF when doing rolling ball elimination")
    const = argschema.fields.Int(required=False, default=25, description="offset to add to dbf*scale during rolling ball elimination")
    max_boundary_distance = argschema.fields.Int(required=False, default=5000, description = "maximium DBF to consider (throw out nuclei)")

class SkeletonParam(argschema.ArgSchema):
    teasar_params = argschema.fields.Nested(TeasarParams, default={}, description='parameters to pass to teasar')
    cloudpath = argschema.fields.Str(required=True, description="cloudvolume path")
    param_name_skeletons = argschema.fields.Bool(required=False, default=True, description="save skeletons in folder/bucket named from parameters used")
    mip = argschema.fields.Int(required=False, default=2, description="mipmap level to use")
    chunk_shape = argschema.fields.List(argschema.fields.Int, validate=mm.validate.Length(equal=3),required=False, default =[512,512,512])
    n_threads = argschema.fields.Int(required=False, default=4, description="number of processes in pool")


class SkeletonModule(argschema.ArgSchemaParser):
    default_schema = SkeletonParam

    def run(self):
        cv = cloudvolume.CloudVolume(self.args['cloudpath'])
        if self.args['param_name_skeletons']:
            info = copy.copy(cv.info)
            skel_folder_str='skeletons'+'_{}'*len(self.args['teasar_params'].keys())
            info['skeletons']=skel_folder_str.format(*[v for k,v in sorted(self.args['teasar_params'].items(), key=lambda x: x[0])])
        else:
            info = None
        with LocalTaskQueue(parallel=self.args['n_threads']) as tq:
             tc.create_skeletonizing_tasks(tq,
                                             self.args['cloudpath'],
                                             mip=self.args['mip'],
                                             shape=tuple(self.args['chunk_shape']),
                                             teasar_params=self.args['teasar_params'],
                                             info=info)


example_parameters = {
    "cloudpath":"file:///nas5/connectome/pinky40subvol"
}

if (__name__ == "__main__"):
    mod = SkeletonModule(input_data = example_parameters)
    mod.run()


