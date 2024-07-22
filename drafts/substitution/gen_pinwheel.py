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

gen = 5
wing_percentile = 0.4
sep = 10  # p = (sep+1), the periodicity of the boundary surgery
ANTIPARITY = True
readdir = os.path.join('/Users/yitan/Coding/aperiodic-codes/data/pinwheel')
readdir = os.path.join(readdir, f'anti={ANTIPARITY}_gen={gen}_sep={sep}')
# readdir = os.path.join(readdir, f'antiparity={ANTIPARITY}_gen={gen}_sep={sep}')
# # readdir = os.path.join(readdir, f'antiparity={ANTIPARITY}_gen={gen}_wing={wing_percentile}')

# savedir = '/Users/yitan/Google Drive/My Drive/from_cannon/qmemory_simulation/data/qc_code/pinwheel/laplacian'
savedir = '/Users/yitan/Coding/aperiodic-codes/data/pinwheel/'
# savedir = os.path.join(savedir, f'antiparity={ANTIPARITY}_gen={gen}')
savedir = os.path.join(savedir, f'anti={ANTIPARITY}_gen={gen}_sep={sep}')
# savedir = os.path.join(savedir, f'antiparity={ANTIPARITY}_gen={gen}_sep={sep}_debug')
# savedir = os.path.join(savedir, f'antiparity={ANTIPARITY}_gen={gen}_sep={sep}_longdir')
# savedir = os.path.join(savedir, f'antiparity={ANTIPARITY}_gen={gen}_wing={wing_percentile}')
if not os.path.exists(savedir):
    os.makedirs(savedir)

def subdivide(triangles):
    result = []
    for ctg, A, B, C in triangles:
        if ctg == 0:
            P1 = A + 2/5*(B-A)
            P2 = A + 4/5*(B-A)
            P3 = 0.5*(A+C)
            P4 = 0.5*(P2+C)

            F1 = (0, A, P3, P1)
            F2 = (0, P2, P3, P1)
            F3 = (0, P3, P2, P4)
            F4 = (0, P3, C, P4)
            F5 = (0, C, B, P2)
            result += [F1, F2, F3, F4, F5] 
    return result

def close(a, b):
    return np.linalg.norm(a-b) < 1e-5

def get_vertices(faces):
    vertices = []
    for face in faces:
        vertices.append(face[1])
        vertices.append(face[2])
        vertices.append(face[3])
    vertices_new = []
    for v in vertices:
        if not any(close(v, v2) for v2 in vertices_new):
            vertices_new.append(v)
    return vertices_new

def get_edges(faces, vertices):
    def vertices_on_face(face, vertices):
        vs_on_f = [face[0]] # ctg
        for v in vertices:
            if close(face[1], v):
                vs_on_f.append(v)
        for v in vertices:
            if close(face[2], v):
                vs_on_f.append(v)
        for v in vertices:
            if close(face[3], v):
                vs_on_f.append(v)
        for v in vertices:
            if close((face[1]+face[3])/2, v):
                vs_on_f.append(v)
        return vs_on_f

    edges = []
    for face in faces:
        vs_on_f = vertices_on_face(face, vertices)
        if len(vs_on_f) == 4:
            edges.append((vs_on_f[1], vs_on_f[2]))
            edges.append((vs_on_f[2], vs_on_f[3]))
            edges.append((vs_on_f[3], vs_on_f[1]))
        if len(vs_on_f) == 5:
            edges.append((vs_on_f[1], vs_on_f[2]))
            edges.append((vs_on_f[2], vs_on_f[3]))
            edges.append((vs_on_f[1], vs_on_f[4]))
            edges.append((vs_on_f[3], vs_on_f[4]))
    return edges

def get_laplacian_code(vertices, edges):
    h = np.zeros((len(vertices), len(vertices)), dtype=int)
    for iv, v in enumerate(vertices):
        for e in edges:
            if close(v, e[0]):
                # find the other vertex of the edge
                for iv2, v2 in enumerate(vertices):
                    if close(v2, e[1]):
                        h[iv, iv2] = 1
                        break
            elif close(v, e[1]):
                # find the other vertex of the edge
                for iv2, v2 in enumerate(vertices):
                    if close(v2, e[0]):
                        h[iv, iv2] = 1
                        break
            else:
                continue

    for i in range(len(vertices)):
        if ANTIPARITY:
            if np.sum(h[i,:]) % 2 == 0:
                h[i,i] = 1
        else:
            if np.sum(h[i,:]) % 2 == 1:
                h[i,i] = 1
    
    return h


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
    upper_surgery_inds = list(sorted(upper_surgery_inds, key=lambda i: vertices[i].real))[0:-1]
    right_surgery_inds = list(sorted(right_surgery_inds, key=lambda i: vertices[i].imag, reverse=True))[0:-1]
    lower_surgery_inds = list(sorted(lower_surgery_inds, key=lambda i: vertices[i].real, reverse=True))[0:-1]
    left_surgery_inds = list(sorted(left_surgery_inds, key=lambda i: vertices[i].imag))[0:-1]
    surgery_inds = upper_surgery_inds + right_surgery_inds + lower_surgery_inds + left_surgery_inds
    surgery_inds = surgery_inds[::int(sep+1)]
    h = np.delete(h, surgery_inds, axis=0)
    return h, surgery_inds

def boundary_surgery_evenly_debug(h, vertices):
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
    return upper_surgery_inds, right_surgery_inds, lower_surgery_inds, left_surgery_inds

def boundary_surgery_evenly_long_direction(h, vertices, sep=1):
    upper_surgery_inds = []
    lower_surgery_inds = []
    for i in range(h.shape[1]):
        if np.abs(vertices[i].imag - 1) < 1e-8:
            upper_surgery_inds.append(i)
        if np.abs(vertices[i].imag) < 1e-8:
            lower_surgery_inds.append(i)
    upper_surgery_inds = list(sorted(upper_surgery_inds, key=lambda i: vertices[i].real))
    lower_surgery_inds = list(sorted(lower_surgery_inds, key=lambda i: vertices[i].real, reverse=True))
    upper_surgery_inds = upper_surgery_inds[0:-1:int(sep+1)]
    lower_surgery_inds = lower_surgery_inds[0:-1:int(sep+1)]
    surgery_inds = upper_surgery_inds + lower_surgery_inds
    h = np.delete(h, surgery_inds, axis=0)
    return h

############################################################################################################
# Generate code
############################################################################################################
triangles = []
ctg = 0
A = 0.+0.j
B = 2.+1.j
C = 2.+0.j
D = 0.+1.j
triangles.append((ctg, A, B, C))
triangles.append((ctg, B, A, D))
for _ in range(gen):
    triangles = subdivide(triangles)
vertices = get_vertices(triangles)
edges = get_edges(triangles, vertices)
h = get_laplacian_code(vertices, edges)
h, surgery_inds = boundary_surgery_evenly(h, vertices, sep=sep)

############################################################################################################
# load code from file
############################################################################################################
# h = np.load(os.path.join(readdir, 'h.npy'))
# vertices = np.load(os.path.join(readdir, 'vertices.npy'))
# edges = np.load(os.path.join(readdir, 'edges.npy'), allow_pickle=True)
# h, surgery_inds = boundary_surgery_evenly(h, vertices, sep=sep)
# # lowwt_logical_op = np.load(os.path.join(readdir, 'lowwt_logical_op.npy'))

############################################################################################################
# Compute code properties
############################################################################################################
m, n = h.shape
logical_basis = row_basis(nullspace(h))
k = len(logical_basis)
np.savetxt(os.path.join(savedir, 'logical_basis.txt'), logical_basis)

# print('shape of h = ', h.shape)
# print('k = ', k)
    
d_bound, lowwt_logical_op = get_classical_code_distance_special_treatment(h=h, target_weight=get_classical_code_distance_time_limit(h, time_limit=120))
# print('d_bound = ', d_bound)

############################################################################################################
# Visualize logicals
############################################################################################################
xs = np.array([v.real for v in vertices])
ys = np.array([v.imag for v in vertices])

# '''Visualize all logical operators'''
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


# '''Visualize low-weight logical operators'''
fig, ax = plt.subplots()
pos_ones = np.where(lowwt_logical_op == 1)[0]
# ax.scatter(np.array([v.real for v in vertices]), np.array([v.imag for v in vertices]), marker='o')
# # annotate the points
# for ipt in range(len(xs)):
#     ax.annotate(ipt, (xs[ipt], ys[ipt]), zorder=2)
ax.scatter(xs, ys, marker='o', color='#6168B0', s=300, edgecolors='k', zorder=1)
ax.scatter(xs[pos_ones], ys[pos_ones], marker='o', s=300, edgecolors='k', color='#F15C5B', zorder=2)
# ax.scatter(xs[pos_ones], ys[pos_ones], marker='o', s=300, edgecolors='k', color='#F15E22', zorder=2)

ax.scatter(xs[surgery_inds], ys[surgery_inds], marker='o', edgecolors='k', s=400, color='#DADADA', zorder=5)
# ax.scatter(xs[upper_surgery_inds], ys[upper_surgery_inds], marker='o', s=300, color='red', alpha=0.5, zorder=1)
# ax.scatter(xs[right_surgery_inds], ys[right_surgery_inds], marker='o', s=300, color='blue', alpha=0.5, zorder=2)
# ax.scatter(xs[lower_surgery_inds], ys[lower_surgery_inds], marker='o', s=300, color='green', alpha=0.5, zorder=3)
# ax.scatter(xs[left_surgery_inds], ys[left_surgery_inds], marker='o', s=300, color='purple', alpha=0.5, zorder=4)

for edge in edges:
    ax.plot([edge[0].real, edge[1].real], [edge[0].imag, edge[1].imag], lw=2, color='black', alpha=0.9, zorder=0)
ax.set_aspect('equal')
ax.set_axis_off()
savename = f'low_weight_logical_op.pdf'
savepath = os.path.join(savedir, savename)
# fig.set_size_inches(20,20)
# fig.set_size_inches(30,30)
fig.set_size_inches(60,60)
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
np.save(os.path.join(savedir, 'h.npy'), h)
# # np.save(os.path.join(savedir, 'vertices.npy'), vertices)
# # np.save(os.path.join(savedir, 'edges.npy'), edges)
# np.save(os.path.join(savedir, 'lowwt_logical_op.npy'), lowwt_logical_op)


# # plt.show()