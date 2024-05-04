import numpy as np
import matplotlib.pyplot as plt
import os
from timeit import default_timer as timer
from ldpc.mod2 import row_basis, nullspace, rank
from helpers_qc import *
from helpers_distance import *
import json
from itertools import product

wing_percentile = 0.45
readdir = '/Users/yitan/Library/CloudStorage/GoogleDrive-yitan@g.harvard.edu/My Drive/from_cannon/qmemory_simulation/data/qc_code/pinwheel/laplacian/'
readdir = os.path.join(readdir, 'antiparity=True_gen=6')
savedir = '/Users/yitan/Library/CloudStorage/GoogleDrive-yitan@g.harvard.edu/My Drive/from_cannon/qmemory_simulation/data/qc_code/pinwheel/laplacian/'
# savedir = os.path.join(savedir, f'antiparity={ANTIPARITY}_gen={gen}_wing={wing_percentile}')
savedir = os.path.join(savedir, 'antiparity=True_gen=4')
if not os.path.exists(savedir):
    os.makedirs(savedir)

# Load data
h = np.load(os.path.join(readdir, 'h.npy'))
vertices = np.load(os.path.join(readdir, 'vertices.npy'))
edges = np.load(os.path.join(readdir, 'edges.npy'))


def boundary_surgery_central(h, vertices, wing_percentile):
    upper_surgery_inds = []
    lower_surgery_inds = []
    # start from the top left corner
    def valid_upper_surgery(vertex):
        # check if the ith row is valid for upper surgery
        on_upper_boundary = np.abs(vertex.imag - 1) < 1e-5
        within_middle = vertex.real > wing_percentile*2 and vertex.real < 2-wing_percentile*2
        return on_upper_boundary and within_middle
    def valid_lower_surgery(vertex):
        # check if the ith row is valid for lower surgery
        on_lower_boundary = np.abs(vertex.imag) < 1e-5
        within_middle = vertex.real > wing_percentile*2 and vertex.real < 2-wing_percentile*2
        return on_lower_boundary and within_middle
    for i in range(h.shape[1]):
        if valid_upper_surgery(vertices[i]):
            upper_surgery_inds.append(i)
        if valid_lower_surgery(vertices[i]):
            lower_surgery_inds.append(i)
    # surgery
    h = np.delete(h, upper_surgery_inds+lower_surgery_inds, axis=0)
    return h

h = boundary_surgery_central(h, vertices, wing_percentile)
np.save(os.path.join(savedir, f'h_wing={wing_percentile}.npy'), h)
