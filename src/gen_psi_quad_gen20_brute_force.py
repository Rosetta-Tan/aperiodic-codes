import numpy as np
import matplotlib.pyplot as plt
from ldpc.mod2 import row_basis, nullspace, rank
from helpers_distance import get_classical_code_distance_time_limit, get_classical_code_distance_special_treatment
import json
import os
from itertools import product

readdir = '/Users/yitan/Library/CloudStorage/GoogleDrive-yitan@g.harvard.edu/My Drive/from_cannon/qmemory_simulation/data/qc_code/psi_tiling/'
savedir = '/Users/yitan/Library/CloudStorage/GoogleDrive-yitan@g.harvard.edu/My Drive/from_cannon/qmemory_simulation/data/qc_code/psi_tiling/gen_20/good_boundary1'
if not os.path.exists(savedir):
    os.makedirs(savedir)
data = np.load(os.path.join(readdir, 'psi_tiling_gen_20.npz'))

h = data['h']  # equivalent to face_to_vertex
vertices_pos = data['vertices_pos']
faces_pos = data['faces_pos']
edges = data['edges']
num_faces, num_vertices = h.shape

############################################################################################################
# post-processing
############################################################################################################

columns_to_remove = np.loadtxt('psi_tiling_columns_to_remove_gen20.txt', dtype=int)
columns_to_remove = np.unique(columns_to_remove)
columns_to_remove = sorted(columns_to_remove)

h = np.delete(h, columns_to_remove, axis=1)
rows_to_remove = []
for i in range(h.shape[0]):
    if np.sum(h[i, :]) < 3:
        rows_to_remove.append(i)
h = np.delete(h, rows_to_remove, axis=0)

# create a mapping between the old and new indices
old_to_new = {}
new_to_old = {}
new_index = 0
for i in range(num_vertices):
    if i not in columns_to_remove:
        old_to_new[i] = new_index
        new_to_old[new_index] = i
        new_index += 1

old_to_new = {}
new_to_old = {}
new_index = 0
for i in range(num_vertices):
    if i not in columns_to_remove:
        old_to_new[i] = new_index
        new_to_old[new_index] = i
        new_index += 1

print('(m,n) =', h.shape)
print('k = ', h.shape[1] - rank(h))
k = h.shape[1] - rank(h)


logical_basis = row_basis(nullspace(h))
print(logical_basis.shape)
logical_op_coeffs = np.asarray(list(product([0, 1], repeat=k)))


############################################################################################################
# visualize the patch
############################################################################################################
fig, ax = plt.subplots()

xs = vertices_pos[:, 0]
ys = vertices_pos[:, 1]
xs_after_removal = [x for i, x in enumerate(xs) if i not in columns_to_remove]
ys_after_removal = [y for i, y in enumerate(ys) if i not in columns_to_remove]
ax.scatter(xs_after_removal, ys_after_removal, marker='o')
# annotate the points
# for i in range(len(xs_after_removal)):
#     plt.annotate(new_to_old[i], (xs_after_removal[i], ys_after_removal[i]), zorder=2)
for i in range(len(xs)):
    plt.annotate(i, (xs[i], ys[i]), zorder=2)

for edge in edges:
    plt.plot([xs[edge[0]], xs[edge[1]]], [ys[edge[0]], ys[edge[1]]], color='gray', alpha=0.5, zorder=0)

edges_after_removal = []
for edge in edges:
    if edge[0] not in columns_to_remove and edge[1] not in columns_to_remove:
        edges_after_removal.append([edge[0], edge[1]])

for edge in edges_after_removal:
        plt.plot([xs[edge[0]], xs[edge[1]]], [ys[edge[0]], ys[edge[1]]], color='black', zorder=0)
ax.set_aspect('equal')
ax.set_axis_off()
fig.set_size_inches(50,50)
fig.savefig(os.path.join(savedir, 'visualize_patch_good_boundary1.pdf'))


############################################################################################################
# visualize the logical operators
############################################################################################################

for i in range(logical_op_coeffs.shape[0]):
# for i in range(2):
    logical_op = (logical_op_coeffs[i]@logical_basis).flatten()
    # d_bound, logical_op = get_classical_code_distance_special_treatment(h, target_weight=target_weight)
    # print(f'logical_op {i}: ', logical_op)
    pos_ones = np.where(logical_op == 1)[0]
    # print(f'pos_ones {i}: ', pos_ones)
    pos_ones = [new_to_old[j] for j in pos_ones]
    print('positions of one in the logical op (original indices): ', pos_ones)

    # visualize the points
    fig, ax = plt.subplots()
    # xs_after_removal = [x for i, x in enumerate(xs) if i not in columns_to_remove]
    # ys_after_removal = [y for i, y in enumerate(ys) if i not in columns_to_remove]
    ax.scatter(xs_after_removal, ys_after_removal, marker='o')
    # # annotate the points
    # for i in range(len(xs)):
    #     plt.annotate(i, (xs[i], ys[i]), zorder=2)
    ax.scatter(xs[pos_ones], ys[pos_ones], marker='o', color='pink', zorder=1)

    for edge in edges:
        plt.plot([xs[edge[0]], xs[edge[1]]], [ys[edge[0]], ys[edge[1]]], color='gray', alpha=0.5, zorder=0)

    edges_after_removal = []
    for edge in edges:
        if edge[0] not in columns_to_remove and edge[1] not in columns_to_remove:
            edges_after_removal.append([edge[0], edge[1]])

    for edge in edges_after_removal:
        plt.plot([xs[edge[0]], xs[edge[1]]], [ys[edge[0]], ys[edge[1]]], color='black', zorder=0)
    ax.set_aspect('equal')
    ax.set_axis_off()
    ax.set_title(f'logical operator {i}')
    fig.set_size_inches(30,30)
    savename = f'logical_op_{i}.pdf'
    savepath = os.path.join(savedir, savename)
    plt.savefig(savepath, bbox_inches='tight', pad_inches=0)


