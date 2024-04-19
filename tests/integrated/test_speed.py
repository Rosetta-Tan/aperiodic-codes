import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.helpers import GeometryUtils, Tiling
from src.helpers_fast import GeometryUtils as GeometryUtilsFast
from src._backend.helpers import subdivide
from timeit import default_timer
from numba import jit

def log_call(function, stat_dict, alt_name=None):
    if alt_name is None:
        fn_name = function.__name__
    else:
        fn_name = alt_name

    def rtn(*args, **kwargs):
        if __debug__:
            print('beginning', fn_name)

        tick = default_timer()
        rtn_val = function(*args, **kwargs)
        tock = default_timer()

        if __debug__:
            print('completed', fn_name)

        stat_dict[fn_name] = tock-tick

        return rtn_val

    return rtn

def test_subdivide():
    for _ in range(10000):
        faces = [(0, 0, 2+1j, 2)]
        for _ in range(5):
            faces = GeometryUtils.subdivide(faces)

def test_subdivide_cython():
    for _ in range(10000):
        faces = [(0, 0, 2+1j, 2)]
        for _ in range(5):
            faces = subdivide(faces)

if __name__ == '__main__':
    stats = {}
    log_call(test_subdivide, stats)()
    print(stats)

    stats = {}
    log_call(test_subdivide_cython, stats)()
    print(stats)

    