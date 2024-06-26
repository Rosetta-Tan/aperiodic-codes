import numpy as np
import matplotlib.pyplot as plt
from ldpc.mod2 import row_basis, nullspace, rank
from helpers_distance import get_classical_code_distance_time_limit, get_classical_code_distance_special_treatment
from helpers_qc import *
import os
from itertools import product
import time
from shapely.geometry import Point, MultiPoint
from shapely.geometry.polygon import Polygon

def tellme(s):
    print(s)
    plt.title(s)
    plt.draw()

def mouse_event(event):
    print(f'x: {event.xdata} and y: {event.ydata}')
    plt.scatter(event.xdata, event.ydata, marker='o', color='red', zorder=1)
    plt.draw()

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
# readdir = '/Users/yitan/Library/CloudStorage/GoogleDrive-yitan@g.harvard.edu/My Drive/from_cannon/qmemory_simulation/data/qc_code/psi_tiling/'
# savedir = '/Users/yitan/Library/CloudStorage/GoogleDrive-yitan@g.harvard.edu/My Drive/from_cannon/qmemory_simulation/data/qc_code/psi_tiling/gen_20/type5_typeC'
# savedir = '/Users/yitan/Library/CloudStorage/GoogleDrive-yitan@g.harvard.edu/My Drive/from_cannon/qmemory_simulation/data/qc_code/psi_tiling/gen_20/ensemble'
# readdir = '..\data\qc_code\psi_tiling'
# # savedir = '..\\figures\qc_code\psi_tiling\gen_20\\type_5_horizontal'
# # savedir = "G:\My Drive\\from_cannon\qmemory_simulation\data\qc_code\psi_tiling\gen_20\\type5_typeC"
# # savedir = "G:\My Drive\\from_cannon\qmemory_simulation\data\qc_code\psi_tiling\gen_20\\type5_localmodtypeC"
# # savedir = "G:\My Drive\\from_cannon\qmemory_simulation\data\qc_code\psi_tiling\gen_20\\type5_typeC_corrected"
# savedir = "G:\My Drive\\from_cannon\qmemory_simulation\data\qc_code\psi_tiling\gen_20\\localmodtype5_typeC_1771"
# # savedir = "G:\My Drive\\from_cannon\qmemory_simulation\data\qc_code\psi_tiling\gen_20\\type6_typeC"
# data = np.load(os.path.join(readdir, 'psi_tiling_gen_20.npz'))
'''Gen 21'''
readdir = '/Users/yitan/Library/CloudStorage/GoogleDrive-yitan@g.harvard.edu/My Drive/from_cannon/qmemory_simulation/data/qc_code/psi_tiling/'
savedir = '/Users/yitan/Library/CloudStorage/GoogleDrive-yitan@g.harvard.edu/My Drive/from_cannon/qmemory_simulation/data/qc_code/psi_tiling/gen_21/autopatch'
# # readdir = '..\data\qc_code\psi_tiling'
# readdir = "G:\My Drive\\from_cannon\qmemory_simulation\data\qc_code\psi_tiling"
# savedir = "G:\My Drive\\from_cannon\qmemory_simulation\data\qc_code\psi_tiling\gen_21\\type5_typeC"
data = np.load(os.path.join(readdir, 'psi_tiling_gen_21.npz'))
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
fig, ax = plt.subplots()
# annotate the points
for i in range(len(xs)):
    plt.annotate(i, (xs[i], ys[i]), zorder=2)

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

face_xs = faces_pos[:, 0]
face_ys = faces_pos[:, 1]
plt.scatter(face_xs, face_ys, marker='s', color='red', zorder=0)
for i in range(len(faces_pos)):
    plt.annotate(i, (face_xs[i], face_ys[i]), zorder=2)

ax.set_aspect('equal')
ax.set_axis_off()
fig.tight_layout()
fig.set_size_inches(40, 40)
fig.savefig(os.path.join(savedir, 'visualize_patch.pdf'))

############################################################################################################
# boundary cut by loading from file
############################################################################################################

boundary_vertices = np.loadtxt('psi_tiling_boundary_vertices_gen20_ensemble_trial.txt')
# boundary_vertices = np.loadtxt('psi_tiling_boundary_vertices_gen20_localmodtype5_typeC_171.txt')
# for i in range(len(boundary_vertices)-1):
#     ax.plot([boundary_vertices[i][0], boundary_vertices[i+1][0]], [boundary_vertices[i][1], boundary_vertices[i+1][1]], color='red', zorder=0)
# ax.plot([boundary_vertices[-1][0], boundary_vertices[0][0]], [boundary_vertices[-1][1], boundary_vertices[0][1]], color='red', zorder=0)
f = zoom_factory(ax, ax.get_xlim(), ax.get_ylim(), base_scale=1.5)
cid = fig.canvas.mpl_connect('button_press_event', mouse_event)

############################################################################################################
# boundary cut by ginput
############################################################################################################

# tellme('Select points on the boundary with mouse or whitespace. Points must go in clockwise direction. Press enter to finish')
# max_xlim = ax.get_xlim() # get current x_limits to set max zoom out
# max_ylim = ax.get_ylim() # get current y_limits to set max zoom out
# f = zoom_factory(ax, max_xlim, max_ylim, base_scale=1.5)
# boundary_vertices = plt.ginput(n=-1, timeout=-1)
# boundary_vertices = np.asarray(boundary_vertices)
# # print('boundary vertices: ', boundary_vertices)
# # np.savetxt('psi_tiling_boundary_vertices_gen15.txt', boundary_vertices, fmt='%f')
# # np.savetxt('psi_tiling_boundary_vertices_gen19.txt', boundary_vertices, fmt='%f')
# # np.savetxt('psi_tiling_boundary_vertices_gen20.txt', boundary_vertices, fmt='%f')
# np.savetxt('psi_tiling_boundary_vertices_gen21.txt', boundary_vertices, fmt='%f')
# # plot the boundary
# for i in range(len(boundary_vertices)-1):
#     plt.plot([boundary_vertices[i][0], boundary_vertices[i+1][0]], [boundary_vertices[i][1], boundary_vertices[i+1][1]], color='red', zorder=0)
# plt.plot([boundary_vertices[-1][0], boundary_vertices[0][0]], [boundary_vertices[-1][1], boundary_vertices[0][1]], color='red', zorder=0)

# plt.draw()

############################################################################################################
# Post processing
############################################################################################################

shape = Polygon(boundary_vertices)
columns_to_remove = [iv for iv, v in enumerate(vertices_pos) if not shape.contains(Point(v[0],v[1]))]
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

faces_xs_after_removal = [x for i, x in enumerate(face_xs) if i not in rows_to_remove]
faces_ys_after_removal = [y for i, y in enumerate(face_ys) if i not in rows_to_remove]

edges_after_removal = []
for edge in edges:
    if edge[0] not in columns_to_remove and edge[1] not in columns_to_remove:
        edges_after_removal.append([edge[0], edge[1]])

print('(m,n) =', h.shape)
print('k = ', h.shape[1] - rank(h))
k = h.shape[1] - rank(h)


# logical_basis = row_basis(nullspace(h))
# print(logical_basis.shape)
# logical_op_coeffs = np.asarray(list(product([0, 1], repeat=k)))


############################################################################################################
# visualize the boundary cut patch
############################################################################################################

# fig, ax = plt.subplots()

# for i in range(len(boundary_vertices)-1):
#     ax.plot([boundary_vertices[i][0], boundary_vertices[i+1][0]], [boundary_vertices[i][1], boundary_vertices[i+1][1]], color='red', zorder=0)
# ax.plot([boundary_vertices[-1][0], boundary_vertices[0][0]], [boundary_vertices[-1][1], boundary_vertices[0][1]], color='red', zorder=0)

# ax.scatter(xs_after_removal, ys_after_removal, marker='o')
# # annotate the points
# # for i in range(len(xs_after_removal)):
# #     plt.annotate(new_to_old[i], (xs_after_removal[i], ys_after_removal[i]), zorder=2)
# # for i in range(len(xs)):
# #     plt.annotate(i, (xs[i], ys[i]), zorder=2)

# for edge in edges:
#     plt.plot([xs[edge[0]], xs[edge[1]]], [ys[edge[0]], ys[edge[1]]], color='gray', alpha=0.5, zorder=0)

# for edge in edges_after_removal:
#         plt.plot([xs[edge[0]], xs[edge[1]]], [ys[edge[0]], ys[edge[1]]], color='black', zorder=0)

# # face_xs = faces_pos[:, 0]
# # face_ys = faces_pos[:, 1]
# # plt.scatter(face_xs, face_ys, marker='s', color='red', zorder=0)
# # for i in range(len(faces_pos)):
# #     plt.annotate(i, (face_xs[i], face_ys[i]), zorder=2)

# ax.set_aspect('equal')
# ax.set_axis_off()
# # fig.savefig(os.path.join(savedir, 'visualize_patch_good_boundary1.pdf'))

# ############################################################################################################
# # visualize the logical operators
# ############################################################################################################

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
    ax.scatter(xs_after_removal, ys_after_removal, s=100, marker='o')
    
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
    ax.scatter(xs[pos_ones], ys[pos_ones], marker='o', s=100, color='pink', zorder=1)

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
    fig.set_size_inches(30,30)
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
for ipt in range(len(xs)):
    ax.annotate(ipt, (xs[ipt], ys[ipt]), zorder=2)
ax.scatter(xs[pos_ones], ys[pos_ones], marker='o', s=100, color='pink', zorder=1)

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
