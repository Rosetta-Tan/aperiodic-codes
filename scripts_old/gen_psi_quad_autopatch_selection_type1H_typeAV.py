import numpy as np
import matplotlib.pyplot as plt
from ldpc.mod2 import row_basis, nullspace, rank
from helpers_distance import get_classical_code_distance_time_limit, get_classical_code_distance_special_treatment
from helpers_qc import *
import os
from itertools import product
from shapely.geometry import Point, MultiPoint
from shapely.geometry.polygon import Polygon
from collections import Counter
from math import ceil, floor
from timeit import default_timer as timer

def zoom_factory(ax, max_xlim, max_ylim, base_scale = 2.):
    def zoom_fun(event):
        # get the current x and y limits
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        xdata = event.xdata # get event x location
        ydata = event.ydata # get event y location
        if event.button == 'up':
            # deal with zoom in
            scale_factor = 1/base_scale
            # x_scale = scale_factor / 2
            x_scale = scale_factor
        elif event.button == 'down':
            # deal with zoom out
            scale_factor = base_scale
            # x_scale = scale_factor * 2
            x_scale = scale_factor
        else:
            # deal with something that should never happen
            scale_factor = 1
            print(event.button)
        # set new limits
        new_width = (cur_xlim[1] - cur_xlim[0]) * x_scale
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

        if xdata - new_width * (1 - relx) > max_xlim[0]:
            x_min = xdata - new_width * (1 - relx)
        else:
            x_min = max_xlim[0]
        if xdata + new_width * (relx) < max_xlim[1]:
            x_max = xdata + new_width * (relx)
        else:
            x_max = max_xlim[1]
        if ydata - new_height * (1 - rely) > max_ylim[0]:
            y_min = ydata - new_height * (1 - rely)
        else:
            y_min = max_ylim[0]
        if ydata + new_height * (rely) < max_ylim[1]:
            y_max = ydata + new_height * (rely)
        else:
            y_max = max_ylim[1]
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.figure.canvas.draw()

    fig = ax.get_figure() # get the figure of interest
    # attach the call back
    fig.canvas.mpl_connect('scroll_event',zoom_fun)

    #return the function
    return zoom_fun

'''Gen15'''
# # readdir = '/Users/yitan/Library/CloudStorage/GoogleDrive-yitan@g.harvard.edu/My Drive/from_cannon/qmemory_simulation/data/qc_code/psi_tiling/'
# # savedir = '/Users/yitan/Library/CloudStorage/GoogleDrive-yitan@g.harvard.edu/My Drive/from_cannon/qmemory_simulation/data/qc_code/psi_tiling/gen_15/good_boundary'\
# readdir = '..\data\qc_code\psi_tiling'
# savedir = '..\\figures\qc_code\psi_tiling\gen_15\\boundary_trial2'
# data = np.load(os.path.join(readdir, 'psi_tiling_gen_15.npz'))
'''Gen 19'''
# readdir = '/Users/yitan/Library/CloudStorage/GoogleDrive-yitan@g.harvard.edu/My Drive/from_cannon/qmemory_simulation/data/qc_code/psi_tiling/'
# savedir = '/Users/yitan/Library/CloudStorage/GoogleDrive-yitan@g.harvard.edu/My Drive/from_cannon/qmemory_simulation/data/qc_code/psi_tiling/gen_19/type5_typeC'
# # readdir = '..\data\qc_code\psi_tiling'
# readdir = "G:\My Drive\\from_cannon\qmemory_simulation\data\qc_code\psi_tiling"
# # savedir = '..\\figures\qc_code\psi_tiling\gen_19\\type_5_horizontal'
# savedir = "G:\My Drive\\from_cannon\qmemory_simulation\data\qc_code\psi_tiling\gen_19\\type5_typeC"
# # savedir = "G:\My Drive\\from_cannon\qmemory_simulation\data\qc_code\psi_tiling\gen_19\\type5_localmodtypeC"
# # savedir = "G:\My Drive\\from_cannon\qmemory_simulation\data\qc_code\psi_tiling\gen_19\\type5_typeC_corrected"
# # savedir = "G:\My Drive\\from_cannon\qmemory_simulation\data\qc_code\psi_tiling\gen_19\\localmodtype5_typeC_1771"
# # savedir = "G:\My Drive\\from_cannon\qmemory_simulation\data\qc_code\psi_tiling\gen_19\\type6_typeC"
# data = np.load(os.path.join(readdir, 'psi_tiling_gen_19.npz'))
'''Gen 20'''
readdir = '/Users/yitan/Library/CloudStorage/GoogleDrive-yitan@g.harvard.edu/My Drive/from_cannon/qmemory_simulation/data/qc_code/psi_tiling/'
savedir = '/Users/yitan/Library/CloudStorage/GoogleDrive-yitan@g.harvard.edu/My Drive/from_cannon/qmemory_simulation/data/qc_code/psi_tiling/gen_20/autopatch'
# readdir = '..\data\qc_code\psi_tiling'
# # savedir = '..\\figures\qc_code\psi_tiling\gen_20\\type_5_horizontal'
# # savedir = "G:\My Drive\\from_cannon\qmemory_simulation\data\qc_code\psi_tiling\gen_20\\type5_typeC"
# # savedir = "G:\My Drive\\from_cannon\qmemory_simulation\data\qc_code\psi_tiling\gen_20\\type5_localmodtypeC"
# # savedir = "G:\My Drive\\from_cannon\qmemory_simulation\data\qc_code\psi_tiling\gen_20\\type5_typeC_corrected"
# savedir = "G:\My Drive\\from_cannon\qmemory_simulation\data\qc_code\psi_tiling\gen_20\\localmodtype5_typeC_1771"
# # savedir = "G:\My Drive\\from_cannon\qmemory_simulation\data\qc_code\psi_tiling\gen_20\\type6_typeC"
data = np.load(os.path.join(readdir, 'psi_tiling_gen_20.npz'))
'''Gen 21'''
# # readdir = '/Users/yitan/Library/CloudStorage/GoogleDrive-yitan@g.harvard.edu/My Drive/from_cannon/qmemory_simulation/data/qc_code/psi_tiling/'
# # savedir = '/Users/yitan/Library/CloudStorage/GoogleDrive-yitan@g.harvard.edu/My Drive/from_cannon/qmemory_simulation/data/qc_code/psi_tiling/gen_21/good_boundary'
# # readdir = '..\data\qc_code\psi_tiling'
# readdir = "G:\My Drive\\from_cannon\qmemory_simulation\data\qc_code\psi_tiling"
# savedir = "G:\My Drive\\from_cannon\qmemory_simulation\data\qc_code\psi_tiling\gen_21\\type5_typeC"
# data = np.load(os.path.join(readdir, 'psi_tiling_gen_21.npz'))
if not os.path.exists(savedir):
    os.makedirs(savedir)


h = data['h']  # equivalent to face_to_vertex
vertices_pos = data['vertices_pos']
faces_pos = data['faces_pos']
edges = data['edges']
FACES = data['faces']
FACES = [(int(ctg.real), np.array([A.real, A.imag]), np.array([B.real, B.imag]), np.array([C.real, C.imag]), np.array([D.real, D.imag])) for ctg, A, B, C, D in FACES]
num_faces, num_vertices = h.shape
xs = vertices_pos[:, 0]
ys = vertices_pos[:, 1]
rng = np.random.default_rng(0)

############################################################################################################
# Visualize natural background
############################################################################################################

# fig, ax = plt.subplots()
# # annotate the points
# for i in range(len(xs)):
#     plt.annotate(i, (xs[i], ys[i]), zorder=2)

# ax.scatter(xs, ys, marker='o', s=10)

# for edge in edges:
#     plt.plot([xs[edge[0]], xs[edge[1]]], [ys[edge[0]], ys[edge[1]]], color='gray', alpha=0.5, zorder=0)

# for iface in range(num_faces):
#     polygon_inds = np.where(h[iface, :] == 1)[0]
#     polygon = MultiPoint(vertices_pos[polygon_inds]).convex_hull
#     if FACES[iface][0] == 0:
#         plt.fill(*polygon.exterior.xy, alpha=0.5, color='#4377BC')
#     elif FACES[iface][0] == 1:
#         plt.fill(*polygon.exterior.xy, alpha=0.5, color='#7C287D')
#     elif FACES[iface][0] == 2:
#         plt.fill(*polygon.exterior.xy, alpha=0.5, color='#93C83E')

# face_xs = faces_pos[:, 0]
# face_ys = faces_pos[:, 1]
# # plt.scatter(face_xs, face_ys, marker='s', color='red', zorder=0)
# # for i in range(len(faces_pos)):
# #     plt.annotate(i, (face_xs[i], face_ys[i]), zorder=2)``

# ax.set_aspect('equal')
# ax.set_axis_off()
# fig.tight_layout()
# fig.set_size_inches(40,40)
# fig.savefig(os.path.join(savedir, 'visualize_patch.pdf'), bbox_inches='tight', pad_inches=0)

############################################################################################################
# patch selection
############################################################################################################

def close(v1, v2):
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    return np.linalg.norm(v1-v2) < 1e-5

def get_face_inds(v):
    # get the vertex inex from the coordinates
    v = np.asarray(v)
    v_ind = np.where([close(v, v_i) for v_i in vertices_pos])[0][0]
    face_inds = []
    for face_ind in range(h.shape[0]):
        if h[face_ind, v_ind] == 1:
            face_inds.append(face_ind)
    return face_inds

def get_local_env_horizontal(v):
    face_inds = get_face_inds(v)
    face_ctg_counter = Counter([FACES[face_ind][0] for face_ind in face_inds])
    if face_ctg_counter[0] == 3 and face_ctg_counter[1] == 0 and face_ctg_counter[2] == 1:
        return 1
    elif face_ctg_counter[0] == 3 and face_ctg_counter[1] == 1 and face_ctg_counter[2] == 1:
        return 2
    elif face_ctg_counter[0] == 2 and face_ctg_counter[1] == 1 and face_ctg_counter[2] == 2:
        return 3
    elif face_ctg_counter[0] == 3 and face_ctg_counter[1] == 2 and face_ctg_counter[2] == 0:
        return 4
    elif face_ctg_counter[0] == 1 and face_ctg_counter[1] == 1 and face_ctg_counter[2] == 2:
        return 5
    else:
        return 0

def get_local_env_vertical(v):
    face_inds = get_face_inds(v)
    face_ctg_counter = Counter([FACES[face_ind][0] for face_ind in face_inds])
    if face_ctg_counter[0] == 3 and face_ctg_counter[1] == 0 and face_ctg_counter[2] == 1:
        return 1
    elif face_ctg_counter[0] == 3 and face_ctg_counter[1] == 1 and face_ctg_counter[2] == 1:
        return 2
    elif face_ctg_counter[0] == 2 and face_ctg_counter[1] == 1 and face_ctg_counter[2] == 2:
        return 3
    elif face_ctg_counter[0] == 3 and face_ctg_counter[1] == 2 and face_ctg_counter[2] == 0:
        return 4
    elif face_ctg_counter[0] == 1 and face_ctg_counter[1] == 2 and face_ctg_counter[2] == 0:
        return 5
    else:
        return 0

def face_matching_in_local_env(v, ctg, order):
    face_inds = get_face_inds(v)
    for face_ind in face_inds:
        if FACES[face_ind][0] == ctg and close(FACES[face_ind][order], np.asarray(v)):
            return face_ind

def move_right(v):
    local_env = get_local_env_horizontal(v)
    if local_env in [1,2,3,4]:
        face_ind = face_matching_in_local_env(v, ctg=0, order=1)
        return FACES[face_ind][3]
    elif local_env == 5:
        face_ind = face_matching_in_local_env(v, ctg=2, order=4)
        return FACES[face_ind][2]

def move_down(v):
    local_env = get_local_env_vertical(v)
    if local_env in [1,2,3,4]:
        face_ind = face_matching_in_local_env(v, ctg=0, order=2)
        return FACES[face_ind][4]
    elif local_env == 5:
        face_ind = face_matching_in_local_env(v, ctg=1, order=1)
        return FACES[face_ind][3]
    
def move_left(v):
    local_env = get_local_env_horizontal(v)
    if local_env in [1,3]:
        face_ind = face_matching_in_local_env(v, ctg=2, order=2)
        return FACES[face_ind][4]
    elif local_env in [2,4,5]:
        face_ind = face_matching_in_local_env(v, ctg=0, order=3)
        return FACES[face_ind][1]

def move_up(v):
    local_env = get_local_env_vertical(v)
    if local_env in [1,5]:
        face_ind = face_matching_in_local_env(v, ctg=0, order=4)
        return FACES[face_ind][2]
    elif local_env in [2,3,4]:
        face_ind = face_matching_in_local_env(v, ctg=1, order=3)
        return FACES[face_ind][1]
	
def get_boundary(v_start, step_h, step_v):
    v_start = np.asarray(v_start)
    v = v_start.copy()
    boundary = [v]
    # move right
    for _ in range(step_h):
        v = move_right(v)
        boundary.append(v)
    while get_local_env_horizontal(v) == 5:
        v = move_right(v)
        boundary.append(v)
    # move down
    for _ in range(step_v):
        v = move_down(v)
        boundary.append(v)
    while get_local_env_vertical(v) == 5:
        v = move_down(v)
        boundary.append(v)
    # move left
    for _ in range(step_h):
        v = move_left(v)
        boundary.append(v)
    while get_local_env_horizontal(v) == 5:
        v = move_left(v)
        boundary.append(v)
    # move up
    terminate = False
    for _ in range(step_v):
        v = move_up(v)
        boundary.append(v)
        if close(v, v_start):
            terminate = True
            break
    while not terminate:
        v = move_up(v)
        boundary.append(v)
        if close(v, v_start):
            terminate = True
            break
    # if not terminate:
    # assert close(v, v_start)
    boundary = boundary[:-1]
    return np.asarray(boundary)

'''
Manually select the starting point, and the lengths in the horizontal and vertical directions
'''

start_ind = 611
v_start = vertices_pos[start_ind]
assert get_local_env_horizontal(v_start) != 0
assert get_local_env_vertical(v_start) != 0
len_h = 20
len_v = 16
savedir = os.path.join(savedir, f'start_index={start_ind}_len_h={len_h}_step_v={len_v}')
if not os.path.exists(savedir):
    os.makedirs(savedir)
boundary_vertices = get_boundary(v_start, len_h, len_v)

############################################################################################################
# Check the correctness of patch selection
'''Visualize the boundary'''
fig, ax = plt.subplots()
ax.scatter(xs, ys, marker='o', s=10)
for edge in edges:
    plt.plot([xs[edge[0]], xs[edge[1]]], [ys[edge[0]], ys[edge[1]]], color='gray', alpha=0.5, zorder=0)
for iface in range(num_faces):
    polygon_inds = np.where(h[iface, :] == 1)[0]
    polygon = MultiPoint(vertices_pos[polygon_inds]).convex_hull
    if FACES[iface][0] == 0:
        plt.fill(*polygon.exterior.xy, alpha=0.5, color='#4377BC')
    elif FACES[iface][0] == 1:
        plt.fill(*polygon.exterior.xy, alpha=0.5, color='#7C287D')
    elif FACES[iface][0] == 2:
        plt.fill(*polygon.exterior.xy, alpha=0.5, color='#93C83E')
# plot the boundary vertices
for iv, v in enumerate(boundary_vertices):
    plt.scatter(v[0], v[1], marker='s', color='red', zorder=0)
    v_ind = np.where([close(v, v_i) for v_i in vertices_pos])[0][0]
    plt.annotate(str(v_ind), (v[0], v[1]), zorder=2)
# f = zoom_factory(ax, ax.get_xlim(), ax.get_ylim(), base_scale=1.5)
ax.set_aspect('equal')
ax.set_axis_off()
fig.tight_layout()
fig.set_size_inches(30, 30)
fig.savefig(os.path.join(savedir, 'visualize_boundary.pdf'), bbox_inches='tight', pad_inches=0)
############################################################################################################

shape = Polygon(boundary_vertices)
columns_to_remove = []
for iv, v in enumerate(vertices_pos):
    if not shape.contains(Point(v[0],v[1])) and all([(not close(v, v_boundary)) for v_boundary in boundary_vertices]):
        columns_to_remove.append(iv)
rows_to_remove = []

h = np.delete(h, columns_to_remove, axis=1)
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

# create a mapping between the old and new indices (for faces)
faces_old_to_new = {}
faces_new_to_old = {}
new_index = 0
for i in range(num_faces):
    if i not in rows_to_remove:
        faces_old_to_new[i] = new_index
        faces_new_to_old[new_index] = i
        new_index += 1

xs_after_removal = [x for i, x in enumerate(xs) if i not in columns_to_remove]
ys_after_removal = [y for i, y in enumerate(ys) if i not in columns_to_remove]

faces_xs_after_removal = [x for i, x in enumerate(faces_pos[0]) if i not in rows_to_remove]
faces_ys_after_removal = [y for i, y in enumerate(faces_pos[1]) if i not in rows_to_remove]

edges_after_removal = []
for edge in edges:
    if edge[0] not in columns_to_remove and edge[1] not in columns_to_remove:
        edges_after_removal.append([edge[0], edge[1]])

print('(m,n) =', h.shape)
print('k = ', h.shape[1] - rank(h))
k = h.shape[1] - rank(h)

# ############################################################################################################
# Check the correctness of patch selection
# '''Visulize the selected patch'''
# fig, ax = plt.subplots()
# ax.scatter(xs_after_removal, ys_after_removal, s=100, marker='o')

# # # annnotate all the points
# # for ipt in range(len(xs)):
# #     plt.annotate(ipt, (xs[ipt], ys[ipt]), zorder=2)

# # annotate the remaining points
# for ipt in range(len(xs)):
#     if ipt not in columns_to_remove:
#         plt.annotate(old_to_new[ipt], (xs[ipt], ys[ipt]), color='purple', zorder=2)

# # ax.scatter(faces_xs_after_removal, faces_ys_after_removal, marker='s', color='red', zorder=0)
# # # annotate the remaining faces
# # for iface in range(num_faces):
# #     if iface not in rows_to_remove:
# #         plt.annotate(faces_old_to_new[iface], (face_xs[iface], face_ys[iface]), zorder=2)

# for edge in edges:
#     plt.plot([xs[edge[0]], xs[edge[1]]], [ys[edge[0]], ys[edge[1]]], color='gray', alpha=0.5, zorder=0)

# edges_after_removal = []
# for edge in edges:
#     if edge[0] not in columns_to_remove and edge[1] not in columns_to_remove:
#         edges_after_removal.append([edge[0], edge[1]])

# for edge in edges_after_removal:
#     plt.plot([xs[edge[0]], xs[edge[1]]], [ys[edge[0]], ys[edge[1]]], color='black', zorder=0)

# # for ih in range(h.shape[0]):
# #     for jh in range(h.shape[1]):
# #         if h[ih,jh] == 1:
# #             ax.plot([faces_xs_after_removal[ih], xs_after_removal[jh]], [faces_ys_after_removal[ih], ys_after_removal[jh]], color='gray', linewidth=2, zorder=-1)

# plt.fill(*shape.exterior.xy, alpha=0.5, color='red')

# fig.tight_layout()
# ax.set_aspect('equal')
# ax.set_axis_off()
# ############################################################################################################


############################################################################################################
# Visualize logicals
############################################################################################################
'''Visualize all logical operators'''
logical_basis = row_basis(nullspace(h))
print(logical_basis.shape)
logical_op_coeffs = np.asarray(list(product([0, 1], repeat=len(logical_basis))))
for i in range(len(logical_op_coeffs)):
# for i in range(60,64):
    logical_op = np.mod((logical_op_coeffs[i]@logical_basis).flatten(), 2)
    pos_ones = np.where(logical_op == 1)[0]
    if len(pos_ones) > h.shape[1]//8:
        continue
    pos_ones = [new_to_old[j] for j in pos_ones]
    print(f'{i}-th logical op, positions of ones (original indices): ', pos_ones)

    # visualize the points
    fig, ax = plt.subplots()
    # ax.scatter(xs_after_removal, ys_after_removal, s=100, marker='o')
    ax.scatter(xs_after_removal, ys_after_removal, marker='o')
    
    # # annnotate all the points
    # for ipt in range(len(xs)):
    #     plt.annotate(ipt, (xs[ipt], ys[ipt]), zorder=2)
    
    # # annotate the remaining points
    # for ipt in range(len(xs)):
    #     if ipt not in columns_to_remove:
    #         plt.annotate(old_to_new[ipt], (xs[ipt], ys[ipt]), color='purple', zorder=2)

    # ax.scatter(faces_xs_after_removal, faces_ys_after_removal, marker='s', color='red', zorder=0)
    # # annotate the remaining faces
    # for iface in range(num_faces):
    #     if iface not in rows_to_remove:
    #         plt.annotate(faces_old_to_new[iface], (face_xs[iface], face_ys[iface]), zorder=2)
    
    # plot the codeword
    # ax.scatter(xs[pos_ones], ys[pos_ones], marker='o', s=150, color='pink', zorder=1)
    ax.scatter(xs[pos_ones], ys[pos_ones], marker='o', color='pink', zorder=1)

    for edge in edges:
        plt.plot([xs[edge[0]], xs[edge[1]]], [ys[edge[0]], ys[edge[1]]], color='gray', alpha=0.5, zorder=0)

    edges_after_removal = []
    for edge in edges:
        if edge[0] not in columns_to_remove and edge[1] not in columns_to_remove:
            edges_after_removal.append([edge[0], edge[1]])

    for edge in edges_after_removal:
        plt.plot([xs[edge[0]], xs[edge[1]]], [ys[edge[0]], ys[edge[1]]], color='black', zorder=0)

    # for ih in range(h.shape[0]):
    #     for jh in range(h.shape[1]):
    #         if h[ih,jh] == 1:
    #             ax.plot([faces_xs_after_removal[ih], xs_after_removal[jh]], [faces_ys_after_removal[ih], ys_after_removal[jh]], color='gray', linewidth=2, zorder=-1)
    ax.set_aspect('equal')
    ax.set_axis_off()
    ax.set_title(f'logical operator {i}')
    # fig.set_size_inches(30,30)
    savename = f'logical_op_{i}.pdf'
    savepath = os.path.join(savedir, savename)
    fig.savefig(savepath, bbox_inches='tight', pad_inches=0)


'''Visualize low-weight logical operators'''
fig, ax = plt.subplots()
d_bound, logical_op = get_classical_code_distance_special_treatment(h, target_weight=get_classical_code_distance_time_limit(h, time_limit=60))
pos_ones = np.where(logical_op == 1)[0]
pos_ones = [new_to_old[j] for j in pos_ones]
print('positions of one in the logical op (original indices): ', pos_ones)

for i in range(len(boundary_vertices)-1):
    ax.plot([boundary_vertices[i][0], boundary_vertices[i+1][0]], [boundary_vertices[i][1], boundary_vertices[i+1][1]], color='red', zorder=0)
ax.plot([boundary_vertices[-1][0], boundary_vertices[0][0]], [boundary_vertices[-1][1], boundary_vertices[0][1]], color='red', zorder=0)

# visualize the points
# xs_after_removal = [x for i, x in enumerate(xs) if i not in columns_to_remove]
# ys_after_removal = [y for i, y in enumerate(ys) if i not in columns_to_remove]
ax.scatter(xs_after_removal, ys_after_removal, s=100, marker='o')
# annotate the points
# for ipt in range(len(xs)):
#     ax.annotate(ipt, (xs[ipt], ys[ipt]), zorder=2)
ax.scatter(xs[pos_ones], ys[pos_ones], marker='o', s=150, color='pink', zorder=1)
# ax.scatter(xs[pos_ones], ys[pos_ones], marker='o', color='pink', zorder=1)

for edge in edges:
    ax.plot([xs[edge[0]], xs[edge[1]]], [ys[edge[0]], ys[edge[1]]], color='gray', alpha=0.5, zorder=0)

edges_after_removal = []
for edge in edges:
    if edge[0] not in columns_to_remove and edge[1] not in columns_to_remove:
        edges_after_removal.append([edge[0], edge[1]])

for edge in edges_after_removal:
    ax.plot([xs[edge[0]], xs[edge[1]]], [ys[edge[0]], ys[edge[1]]], color='black', zorder=0)
ax.set_aspect('equal')
ax.set_axis_off()
savename = f'low_weight_logical_op.pdf'
savepath = os.path.join(savedir, savename)
fig.set_size_inches(30,30)
fig.savefig(savepath, bbox_inches='tight', pad_inches=0)

# plt.show()