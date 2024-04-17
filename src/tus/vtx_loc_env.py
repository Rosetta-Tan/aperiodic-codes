import numpy as np
import matplotlib.pyplot as plt
import os
from timeit import default_timer as timer
from ldpc.mod2 import row_basis, nullspace, rank
from typing import List, Tuple
from config import gen

if __name__ == '__main__':
    triangles = [(0, 0, 2, 2 + 1j)]
    # ctg = 0
    # A = 0.+0.j
    # B = 2.+1.j
    # C = 2.+0.j
    # triangles.append((ctg, A, B, C))
    # for _ in range(gen):
    #     triangles = subdivide(triangles)
    # vertices = get_vertices(triangles)
    # h = get_qc_code(triangles, vertices)
    # h = h.T
    # m, n = h.shape
    # logical_basis = row_basis(nullspace(h))
    # k = len(logical_basis)

    # print('shape of h = ', h.shape)
    # print('k = ', k)

    # savedir = '../../data/20240415_pinwheel_tus/gen={gen}'
    # subdir = f'gen={gen}'
    # if not os.path.exists(os.path.join(savedir, subdir)):
    #     os.makedirs(os.path.join(savedir, subdir))