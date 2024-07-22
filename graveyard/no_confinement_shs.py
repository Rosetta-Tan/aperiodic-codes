import numpy as np
import matplotlib.pyplot as plt
plt.style.use('../figures/norm.mplstyle')
import os
from timeit import default_timer as timer
from ldpc.mod2 import row_basis, nullspace, rank
from helpers_qc import *
from helpers_distance import *
import json
from itertools import product
from scipy.optimize import curve_fit
from collections import Counter

L = 20
samplesize = 1000
readdir = f'/Users/yitan/Library/CloudStorage/GoogleDrive-yitan@g.harvard.edu/My Drive/from_cannon/qmemory_simulation/data/shs/L={L}_pbc_samplesize={samplesize}'
readname = f'hclassical_shs_squarelattice_L={L}.txt'
savedir = f'/Users/yitan/Library/CloudStorage/GoogleDrive-yitan@g.harvard.edu/My Drive/from_cannon/qmemory_simulation/data/shs/L={L}_pbc_samplesize={samplesize}'
if not os.path.exists(savedir):
    os.makedirs(savedir)

def coords_to_ind(i, j, L):
    return i * L + j

def ind_to_coords(ind, L):
    return ind // L, ind % L

def shs_squarelattice_basegraph(L, save=False):
    pc = np.zeros((L, L, L, L), dtype=int) # adjacency matrix
    for i in range(L):
        for j in range(L):
            pc[i, j, (i+1)%L, j] = 1
            pc[i, j, (i-1)%L, j] = 1
            pc[i, j, i, (j+1)%L] = 1
            pc[i, j, i, (j-1)%L] = 1
            pc[(i+1)%L, j, i, j] = 1
            pc[(i-1)%L, j, i, j] = 1
            pc[i, (j+1)%L, i, j] = 1
            pc[i, (j-1)%L, i, j] = 1
    pc = pc.reshape(L**2, L**2)
    if save:
        # savedir = '/Users/yitan/Library/CloudStorage/GoogleDrive-yitan@g.harvard.edu/My Drive/from_cannon/qmemory_simulation/data/laplacian_code'
        savename = f'hclassical_shs_squarelattice_L={L}.txt'
        np.savetxt(os.path.join(savedir, savename), pc, fmt='%d')
    return pc

# generate code
h = shs_squarelattice_basegraph(L, save=True)

# read data from file
# h = np.loadtxt(os.path.join(readdir, readname))

m, n = h.shape

############################################################################################################
# Do confinement
# 1. Randomly sample points within the bulk
# 2. Specify a radius r
# 3. Get the points within the ball 
# 4. Randomly select points at a given total number within the ball
# 5. Act the parity check matrix on the vector formed by selected points within the ball
# 7. Repeat 1-5 for multiple times
############################################################################################################
rng = np.random.default_rng(seed=0)

def get_random_start_inds_within_bulk(r, L, n_start_points):
    # start_pts = rng.uniform(low=r, high=2.-r, size=n_start_points) + 1j * rng.uniform(low=r, high=1.-r, size=n_start_points)
    start_i = rng.integers(low=r, high=L-r, size=n_start_points)
    start_j = rng.integers(low=r, high=L-r, size=n_start_points)
    start_inds = start_i * L + start_j
    return start_inds

def get_vertex_inds_within_ball(start_ind, r, L):
    start_i, start_j = ind_to_coords(start_ind, L)
    vertex_inds = []
    for i in range(L):
        for j in range(L):
            if (i-start_i)**2 + (j-start_j)**2 <= r**2:
                vertex_inds.append(coords_to_ind(i, j, L))
    return np.array(vertex_inds)

def get_vertex_inds_within_square(start_ind, r, L):
    start_i, start_j = ind_to_coords(start_ind, L)
    vertex_inds = []
    for i in range(L):  
        for j in range(L):
            if np.abs(i-start_i) + np.abs(j-start_j) <= r:
                vertex_inds.append(coords_to_ind(i, j, L))
    return np.array(vertex_inds)

def sample_logical_ops(logical_basis, nsamples_logical):
    # rng = np.random.default_rng(0)
    k = logical_basis.shape[0]
    logical_ops = np.zeros((nsamples_logical, logical_basis.shape[1]))
    # sample size_logicals logical operators, each of which is a linear combination of logical_basis
    for i in range(nsamples_logical):
        while True:
            coeff = rng.choice([0, 1], size=k).reshape(1, -1)
            if not np.all(coeff==0):  # make sure the logical operator is nontrivial
                break
        logical_ops[i] = np.mod(coeff@logical_basis, 2)
    return logical_ops

###########################################################################################################
# Ball or square-like truncation of locicals
###########################################################################################################
r = 4.01
nsamples = 1
logical_basis = row_basis(nullspace(h))
logical_op = logical_basis[10]
start_inds = get_random_start_inds_within_bulk(r, L, n_start_points=samplesize)
for iball in range(len(start_inds)):
    # posones = np.intersect1d(get_vertex_inds_within_square(start_inds[iball], r, L), np.where(logical_op==1)[0])
    posones = np.intersect1d(get_vertex_inds_within_ball(start_inds[iball], r, L), np.where(logical_op==1)[0])
    error_vecs = np.zeros((nsamples, h.shape[1]))
    wt_e = len(posones)
    for isample in range(nsamples):
        posones_sampled_error = rng.choice(posones, size=wt_e, replace=False)
        error_vecs[isample, posones_sampled_error] = 1

    synd_vecs = np.mod(error_vecs@(h.T), 2)
    min_synd_weight = np.min(np.sum(synd_vecs, axis=1))
    min_synd_vec = synd_vecs[np.argmin(np.sum(synd_vecs, axis=1))]
    min_error_vec = error_vecs[np.argmin(np.sum(synd_vecs, axis=1))]

    obj = {
        'error_weight': wt_e,
        'min_synd_weight': min_synd_weight
    }
    # with open(os.path.join(savedir, f'confinement_data_r={r}_iball={iball}_1norm.json'), 'w') as f:
    #     json.dump(obj, f)
    with open(os.path.join(savedir, f'confinement_data_r={r}_iball={iball}_2norm.json'), 'w') as f:
        json.dump(obj, f)

    xs, ys = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')
    xs = xs.flatten()
    ys = ys.flatten()

    fig, ax = plt.subplots()
    ax.scatter(xs, ys, marker='o', color='blue', s=50, alpha=0.5, zorder=0)
    ax.scatter(xs[posones], ys[posones], marker='o', s=60, color='green', alpha=0.8, zorder=1)
    pos_excitations = np.where(min_synd_vec==1)[0]
    ax.scatter(xs[pos_excitations], ys[pos_excitations], marker='*', s=300, color='red', alpha=0.4, zorder=1)
    ax.set_aspect('equal')
    ax.set_axis_off()
    # savename = f'visualize_error_excitation_r={r}_iball={iball}_1norm.pdf'
    savename = f'visualize_error_excitation_r={r}_iball={iball}_2norm.pdf'
    savepath = os.path.join(savedir, savename)
    fig.savefig(savepath, bbox_inches='tight', pad_inches=0)

############################################################################################################
# combine confinement data
############################################################################################################

fig, ax = plt.subplots()
error_weights = []
synd_weights = []
min_synd_weights = []
rs = [2.01, 3.01, 4.01, 5.01, 6.01, 7.01, 8.01, 9.01]
for r in rs:
    start_inds = get_random_start_inds_within_bulk(r, L, n_start_points=samplesize)
    for iball in range(len(start_inds)):
        with open(os.path.join(savedir, f'confinement_data_r={r}_iball={iball}_1norm.json'), 'r') as f:
            data = json.load(f)
        error_weights.append(data['error_weight'])
        synd_weights.append(data['min_synd_weight'])
        min_synd_weights.append(data['min_synd_weight'])
ax.scatter(error_weights, min_synd_weights, s=100, edgecolors='k', zorder=10, c='#E2C1CE')
ax.set_ylim(-0.5, 10)
fig.tight_layout()
fig.set_size_inches(6, 6)
fig.savefig('../figures/shs_no_confinement_test_1norm.pdf')


fig, ax = plt.subplots()
wte_wts = {}
error_weights = []
synd_weights = []
min_synd_weights = []
# rs = [2.01, 2.51, 3.01, 3.51, 4.01, 4.51, 5.01, 5.51, 6.01, 6.51, 7.01, 7.51, 8.01, 8.51, 9.01, 9.51]
rs = [2.01, 3.01, 4.01, 5.01, 6.01, 7.01, 8.01, 9.01]
for r in rs:
    start_inds = get_random_start_inds_within_bulk(r, L, n_start_points=samplesize)
    for iball in range(len(start_inds)):
        with open(os.path.join(savedir, f'confinement_data_r={r}_iball={iball}_2norm.json'), 'r') as f:
            data = json.load(f)
        error_weights.append(data['error_weight'])
        synd_weights.append(data['min_synd_weight'])
        min_synd_weights.append(data['min_synd_weight'])
        wte_wts[data['error_weight']] = data['min_synd_weight']
# for each error weight, find the minimum syndrome weight
error_weights = np.array(error_weights)
synd_weights = np.array(synd_weights)
min_synd_weights = np.array(min_synd_weights)
error_weights_unique = np.unique(error_weights)
min_synd_weights_unique = np.zeros_like(error_weights_unique)

for i, wt in enumerate(error_weights_unique):
    min_synd_weights_unique[i] = np.min(synd_weights[error_weights==wt])

ax.scatter(error_weights_unique, min_synd_weights_unique, s=100, edgecolors='k', zorder=10, c='#E2C1CE')

# ax.scatter(error_weights, min_synd_weights, s=100, edgecolors='k', zorder=10, c='#E2C1CE')
ax.set_ylim(1, 250)
ax.loglog()
fig.tight_layout()
fig.set_size_inches(6, 6)
fig.savefig('../figures/shs_no_confinement_test_2norm.pdf')

# plt.show()

