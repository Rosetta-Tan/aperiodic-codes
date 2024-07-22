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

gen = 5
sep = 9
ANTIPARITY = True
readdir = '/Users/yitan/Library/CloudStorage/GoogleDrive-yitan@g.harvard.edu/My Drive/from_cannon/qmemory_simulation/data/qc_code/pinwheel/laplacian/'
readdir = os.path.join(readdir, f'antiparity={ANTIPARITY}_gen={gen}_sep={sep}')
savedir = '/Users/yitan/Library/CloudStorage/GoogleDrive-yitan@g.harvard.edu/My Drive/from_cannon/qmemory_simulation/data/qc_code/pinwheel/laplacian/'
savedir = os.path.join(savedir, f'antiparity={ANTIPARITY}_gen={gen}_sep={sep}')
if not os.path.exists(savedir):
    os.makedirs(savedir)

def boundary_surgery_evenly(h, vertices, sep=1):
    upper_surgery_inds = []
    right_surgery_inds = []
    lower_surgery_inds = []
    left_surgery_inds = []
    for i in range(h.shape[1]):
        if np.abs(vertices[i].imag - 1) < 1e-8:
            upper_surgery_inds.append(i)
        if np.abs(vertices[i].real - 2) < 1e-8:
            right_surgery_inds.append(i)
        if np.abs(vertices[i].imag) < 1e-8:
            lower_surgery_inds.append(i)
        if np.abs(vertices[i].real) < 1e-8:
            left_surgery_inds.append(i)
    upper_surgery_inds = list(sorted(upper_surgery_inds, key=lambda i: vertices[i].real))[:-1]
    right_surgery_inds = list(sorted(right_surgery_inds, key=lambda i: vertices[i].imag, reverse=True))[:-1]
    lower_surgery_inds = list(sorted(lower_surgery_inds, key=lambda i: vertices[i].real, reverse=True))[:-1]
    left_surgery_inds = list(sorted(left_surgery_inds, key=lambda i: vertices[i].imag))[:-1]
    surgery_inds = upper_surgery_inds + right_surgery_inds + lower_surgery_inds + left_surgery_inds
    surgery_inds = surgery_inds[::int(sep+1)]
    h = np.delete(h, surgery_inds, axis=0)
    return h, surgery_inds

# read data from file
h = np.load(os.path.join(readdir, 'h.npy'))
vertices = np.load(os.path.join(readdir, 'vertices.npy'))
edges = np.load(os.path.join(readdir, 'edges.npy'))
h, surgery_inds = boundary_surgery_evenly(h, vertices, sep=sep)
lowwt_logical_op = np.load(os.path.join(readdir, 'lowwt_logical_op.npy'))

new_to_old_inds = {}
for i, ind in enumerate(vertices):
    if i not in surgery_inds:
        new_to_old_inds[len(new_to_old_inds)] = i


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
global n_start_points
global n_within_ball
n_start_points = 6
n_within_ball = 20
r = 0.045

# def valid_start_point(pt, r):
#     judge_real = pt.real > r and pt.real < 2. - r
#     judge_imag = pt.imag > r and pt.imag < 1. - r
#     return judge_real and judge_imag

def get_random_start_points_within_bulk(r):
    start_pts = rng.uniform(low=r, high=2.-r, size=n_start_points) + 1j * rng.uniform(low=r, high=1.-r, size=n_start_points)
    return start_pts

def get_vertex_inds_within_ball(start_pt, r, vertices):
    vertices = np.asarray(vertices)
    dists = np.abs(start_pt - vertices)
    within_ball_inds = np.where(dists < r)[0]
    return within_ball_inds

def sample_logical_ops(logical_basis, nsamples_logical):
    # rng = np.random.default_rng(0)
    k = logical_basis.shape[0]
    logical_ops = np.zeros((nsamples_logical, logical_basis.shape[1]))
    # sample size_logicals logical operators, each of which is a linear combination of logical_basis
    for i in range(nsamples_logical):
        while True:
            coeffs = rng.choice([0, 1], size=k).reshape(1, -1)
            if not np.all(coeffs==0):  # make sure the logical operator is nontrivial
                break
        logical_ops[i] = np.mod(coeffs@logical_basis, 2)
    return logical_ops


start_pts = get_random_start_points_within_bulk(r)

# print(f'Number of start points: {len(start_pts)}')
# print(f'Number of points within ball: {len(get_vertex_inds_within_ball(start_pts[0], r, vertices))}')
# print(f'Number of points within ball: {len(get_random_points_within_ball(start_pts[0], r, vertices, n_within_ball))}')

############################################################################################################
# low weight error regime
############################################################################################################
# posones = get_vertex_inds_within_ball(start_pts[0], r, vertices)
# error_vecs = np.zeros((2**len(posones), h.shape[1]))
# error_vecs[:, posones] = np.asarray(list(product([0, 1], repeat=len(posones))))
# for i, start_pt in enumerate(start_pts[1:]):
#     posones = get_vertex_inds_within_ball(start_pt, r, vertices)
#     error_vecs_tmp = np.zeros((2**len(posones), h.shape[1]))
#     error_vecs_tmp[:, posones] = np.asarray(list(product([0, 1], repeat=len(posones))))
#     error_vecs = np.vstack((error_vecs, error_vecs_tmp))

# error_weights = np.sum(error_vecs, axis=1)
# max_error_weight = int(np.max(error_weights))
# synd_vecs = np.mod(error_vecs@(h.T), 2)
# synd_weights = np.sum(synd_vecs, axis=1)

# min_synd_weight = []
# for error_weight in range(max_error_weight):
#     min_synd_weight.append(np.min(synd_weights[error_weights==error_weight]))

# np.save(os.path.join(savedir, 'lowwt_error_weights.npy'), error_weights)
# np.save(os.path.join(savedir, 'lowwt_synd_weights.npy'), synd_weights)
# np.save(os.path.join(savedir, 'lowwt_min_synd_weights.npy'), min_synd_weight)

# fig, ax = plt.subplots()
# ax.scatter(range(max_error_weight), min_synd_weight, s=100, edgecolors='k', zorder=10)
# fig.set_size_inches(6, 6)
# fig.savefig(os.path.join(savedir, f'error_weight_vs_synd_weight_min.png'), dpi=300)
# plt.show()

############################################################################################################
# high weight error regime
###########################################################################################################
nsamples = 1000
r = 0.35
start_pts = get_random_start_points_within_bulk(r)

for iball in range(len(start_pts)):
    posones = np.intersect1d(get_vertex_inds_within_ball(start_pts[iball], r, vertices), np.where(lowwt_logical_op==1)[0])
    error_vecs = np.zeros((nsamples, h.shape[1]))
    wt_e = len(posones)
    for isample in range(nsamples):
        posones_sampled_error = rng.choice(posones, size=wt_e, replace=False)
        error_vecs[isample, posones_sampled_error] = 1
    # for i, start_pt in enumerate(start_pts[1:], start=1):
    #     posones = np.intersect1d(get_vertex_inds_within_ball(start_pts[i], r, vertices), np.where(lowwt_logical_op==1)[0])
    #     error_vecs_tmp = np.zeros((nsamples, h.shape[1]))
    #     for isample in range(nsamples):
    #         posones_sampled_error = rng.choice(posones, size=wt_e, replace=False)
    #         error_vecs_tmp[isample, posones_sampled_error] = 1
    #     error_vecs = np.vstack((error_vecs, error_vecs_tmp))

    synd_vecs = np.mod(error_vecs@(h.T), 2)
    min_synd_weight = np.min(np.sum(synd_vecs, axis=1))
    min_synd_vec = synd_vecs[np.argmin(np.sum(synd_vecs, axis=1))]
    min_error_vec = error_vecs[np.argmin(np.sum(synd_vecs, axis=1))]

    obj = {
        'error_weight': wt_e,
        'min_synd_weight': min_synd_weight
    }
    with open(os.path.join(savedir, f'higher_wt_confinement_data_r={r}_iball={iball}.json'), 'w') as f:
        json.dump(obj, f)


    # '''Visualize low-weight logical operators'''
    # xs = np.array([v.real for v in vertices])
    # ys = np.array([v.imag for v in vertices])
    # fig, ax = plt.subplots()
    # # # annotate the points
    # # for ipt in range(len(xs)):
    # #     ax.annotate(ipt, (xs[ipt], ys[ipt]), zorder=2)
    # ax.scatter(xs, ys, marker='o', color='blue', s=50, alpha=0.5, zorder=0)
    # # ax.scatter(xs[posones], ys[posones], marker='*', s=300, color='pink', alpha=0.5, zorder=1)
    # pos_errors = np.where(min_error_vec==1)[0]
    # ax.scatter(xs[pos_errors], ys[pos_errors], marker='*', s=300, color='green', alpha=0.8, zorder=1)
    # pos_excitations = np.where(min_synd_vec==1)[0]
    # pos_excitations = np.array([new_to_old_inds[i] for i in pos_excitations])
    # ax.scatter(xs[pos_excitations], ys[pos_excitations], marker='*', s=300, color='red', alpha=0.4, zorder=1)

    # for edge in edges:
    #     ax.plot([edge[0].real, edge[1].real], [edge[0].imag, edge[1].imag], color='gray', alpha=0.5, zorder=0)
    # ax.set_aspect('equal')
    # ax.set_axis_off()
    # savename = f'visualize_error_excitation.pdf'
    # savepath = os.path.join(savedir, savename)
    # fig.set_size_inches(30,20)
    # fig.savefig(savepath, bbox_inches='tight', pad_inches=0)

############################################################################################################
# combine confinement data
############################################################################################################
lowwt_error_weights = list(np.load(os.path.join(readdir, 'lowwt_error_weights.npy')))
lowwt_error_weights = list(np.unique(lowwt_error_weights)[:-1])
lowwt_synd_weights = list(np.load(os.path.join(readdir, 'lowwt_synd_weights.npy')))
lowwt_min_synd_weights = list(np.load(os.path.join(readdir, 'lowwt_min_synd_weights.npy')))

error_weights = []
synd_weights = []
min_synd_weights = []

for r in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
    for iball in range(len(start_pts)):
        with open(os.path.join(savedir, f'higher_wt_confinement_data_r={r}_iball={iball}.json'), 'r') as f:
            data = json.load(f)
        error_weights.append(data['error_weight'])
        synd_weights.append(data['min_synd_weight'])
        min_synd_weights.append(data['min_synd_weight'])

np.save(os.path.join(savedir, 'error_weights.npy'), error_weights)
np.save(os.path.join(savedir, 'synd_weights.npy'), synd_weights)
np.save(os.path.join(savedir, 'min_synd_weight.npy'), min_synd_weights)


# curve fitting
def func(x, a, b):
    return a * x**b

popt, pcov = curve_fit(func, error_weights, min_synd_weights)

# also linear fitting the low weight data
lowwt_popt, lowwt_pcov = curve_fit(func, lowwt_error_weights, lowwt_min_synd_weights)

total_error_weights = lowwt_error_weights + error_weights
total_min_synd_weights = lowwt_min_synd_weights + min_synd_weights

fig, ax = plt.subplots()
ax.scatter(total_error_weights, total_min_synd_weights, s=100, edgecolors='k', zorder=10, c='#E2C1CE')
x_plot_high = np.linspace(np.power(10, 1.6), np.power(10, 2.8), 100)
ax.plot(x_plot_high, func(x_plot_high, *popt)*0.6, 'k--')
# x_plot_low = np.linspace(1, np.power(10, 1.1), 100)
# ax.plot(x_plot_low, func(x_plot_low, *lowwt_popt), 'k--')
ax.loglog()
fig.tight_layout()
fig.set_size_inches(6, 6)
# fig.savefig(os.path.join(savedir, f'error_weight_vs_synd_weight_min.pdf'))
fig.savefig('../figures/pinwheel_confinement.pdf')

plt.show()