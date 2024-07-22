import numpy as np
import matplotlib.pyplot as plt
import os
from timeit import default_timer as timer
from ldpc.mod2 import row_basis, nullspace, rank
from helpers_qc import *
from helpers_distance import *
import json
from itertools import product


readdir = '/Users/yitan/Library/CloudStorage/GoogleDrive-yitan@g.harvard.edu/My Drive/from_cannon/qmemory_simulation/data/qc_code/pinwheel/laplacian/'
readdir = os.path.join(readdir, 'antiparity=True_gen=6')
savedir = '/Users/yitan/Library/CloudStorage/GoogleDrive-yitan@g.harvard.edu/My Drive/from_cannon/qmemory_simulation/data/qc_code/pinwheel/laplacian/'
savedir = os.path.join(savedir, 'antiparity=True_gen=6_wing=0.45')
if not os.path.exists(savedir):
    os.makedirs(savedir)

# Load data
h = np.load(os.path.join(readdir, 'h_wing=0.45.npy'))
vertices = np.load(os.path.join(readdir, 'vertices.npy'))
edges = np.load(os.path.join(readdir, 'edges.npy'))


############################################################################################################
# Compute code properties
############################################################################################################
m, n = h.shape
logical_basis = row_basis(nullspace(h))
k = len(logical_basis)

print('shape of h = ', h.shape)
print('k = ', k)
    
d_bound, lowwt_logical_op = get_classical_code_distance_special_treatment(h=h, target_weight=get_classical_code_distance_time_limit(h, time_limit=60))
print('d_bound = ', d_bound)

############################################################################################################
# Visualize logicals
############################################################################################################
xs = np.array([v.real for v in vertices])
ys = np.array([v.imag for v in vertices])

'''Visualize all logical operators'''
# logical_op_coeffs = np.asarray(list(product([0, 1], repeat=len(logical_basis))))
# for i in range(len(logical_op_coeffs)):
#     logical_op = np.mod((logical_op_coeffs[i]@logical_basis).flatten(), 2)
#     pos_ones = np.where(logical_op == 1)[0]
#     if len(pos_ones) > h.shape[1]//8:
#         continue 
#     fig, ax = plt.subplots()
#     ax.scatter(xs, ys, marker='o', s=50, color='blue', alpha=0.5, zorder=0)
#     ax.scatter(xs[pos_ones], ys[pos_ones], marker='*', s=300, color='pink', zorder=1)
#     for edge in edges:
#         plt.plot([edge[0].real, edge[1].real], [edge[0].imag, edge[1].imag], color='gray', alpha=0.5, zorder=0)

#     ax.set_aspect('equal')
#     ax.set_axis_off()
#     ax.set_title(f'logical operator {i}')
#     fig.set_size_inches(30,30)
#     savename = f'logical_op_{i}.pdf'
#     savepath = os.path.join(savedir, savename)
#     fig.savefig(savepath, bbox_inches='tight', pad_inches=0)


'''Visualize low-weight logical operators'''
fig, ax = plt.subplots()
pos_ones = np.where(lowwt_logical_op == 1)[0]
ax.scatter(np.array([v.real for v in vertices]), np.array([v.imag for v in vertices]), marker='o')
# # annotate the points
# for ipt in range(len(xs)):
#     ax.annotate(ipt, (xs[ipt], ys[ipt]), zorder=2)
ax.scatter(xs, ys, marker='o', color='blue', s=50, alpha=0.5, zorder=0)
ax.scatter(xs[pos_ones], ys[pos_ones], marker='*', s=300, color='pink', zorder=1)

for edge in edges:
    ax.plot([edge[0].real, edge[1].real], [edge[0].imag, edge[1].imag], color='gray', alpha=0.5, zorder=0)
ax.set_aspect('equal')
ax.set_axis_off()
savename = f'low_weight_logical_op.pdf'
savepath = os.path.join(savedir, savename)
fig.set_size_inches(150,150)
fig.savefig(savepath, bbox_inches='tight', pad_inches=0)

############################################################################################################
# Save data
############################################################################################################
data = {
    'm': int(h.shape[0]),
    'n': int(h.shape[1]),
    'k': int(k),
    'd_bound': int(d_bound)
}
with open(os.path.join(savedir, 'data.json'), 'w') as f:
    json.dump(data, f)

np.save(os.path.join(savedir, 'lowwt_logical_op.npy'), lowwt_logical_op)

# plt.show()