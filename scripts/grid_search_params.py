import itertools
from run_skeleton import SkeletonModule
import multiprocessing
# We must import this explicitly, it is not imported by the top-level
# multiprocessing module.
import multiprocessing.pool
import time



class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


def process_skeleton_params(params):
    localpath = "file:///nas5/connectome/pinky40subvol"
    dbf_exponent, dbf_scale, scale, const, max_boundary = params

    input_d = {
        'cloudpath':localpath,
        'teasar_params':{
            'dbf_exponent':dbf_exponent,
            'dbf_scale':dbf_scale,
            'scale':scale,
            'const':const,
            'max_boundary_distance':max_boundary
        }
    }
    #print(input_d)
    mod = SkeletonModule(input_data=input_d, args=[])
    mod.run()

def test():
    pool = MyPool(10)

    dbf_exps = [4, 8, 16, 20, 32]
    dbf_scales = [2000,5000,15000]
    scales= [2,10,20,40]
    consts = [25]
    max_boundaries = [5000]

    all_params = itertools.product(dbf_exps, dbf_scales, scales, consts, max_boundaries)
    aparams=[p for p in all_params]

    results=pool.map(process_skeleton_params, aparams)

    
    pool.close()
    pool.join()

if __name__ == '__main__':
    test()



