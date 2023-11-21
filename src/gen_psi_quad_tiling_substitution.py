import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from ldpc.mod2 import row_basis, nullspace, rank
from helpers_distance import get_classical_code_distance_time_limit, get_classical_code_distance_special_treatment
from helpers_qc import *
from scipy.optimize import root_scalar
from timeit import default_timer as timer
import os

# global variables
# savedir = '/Users/yitan/Library/CloudStorage/GoogleDrive-yitan@g.harvard.edu/My Drive/from_cannon/qmemory_simulation/data/qc_code/psi_tiling/'
savedir = '..\data\qc_code\psi_tiling'

psi = root_scalar(lambda x: x**3 - x**2 - 1, bracket=[1, 2], method='brentq').root
sqrt_psi = np.sqrt(psi)

prototype_A = 1+0j
prototype_B = root_scalar(lambda x: 8*x**3 - 8*x**2 + 2*x + 1, bracket=[-1, 0], method='brentq').root + \
            1j * root_scalar(lambda x: 64*x**6 + 32*x**4 + 4*x**2 - 31, bracket=[0, 1], method='brentq').root

prototype_C = root_scalar(lambda x: 8*x**3 - 8*x**2 - 2*x + 3, bracket=[-1, 0], method='brentq').root + \
            1j * root_scalar(lambda x: 64*x**6 + 224*x**4 + 196*x**2 - 31, bracket=[-1, 0], method='brentq').root
prototype_D = root_scalar(lambda x: 8*x**3 - 32*x**2 + 38*x - 11, bracket=[0, 1], method='brentq').root + \
            1j * root_scalar(lambda x: 64*x**6 + 224*x**4 + 196*x**2 - 31, bracket=[-1, 0], method='brentq').root
prototype_E = root_scalar(lambda x: 8*x**3 - 40*x**2 + 54*x - 9, bracket=[0, 1], method='brentq').root + \
            1j * root_scalar(lambda x: 64*x**6 + 608*x**4 + 1444*x**2 - 279, bracket=[0, 1], method='brentq').root
prototype_F = root_scalar(lambda x: 8*x**3 + 8*x**2 - 2*x - 3, bracket=[0, 1], method='brentq').root + \
            1j * root_scalar(lambda x: 64*x**6 + 224*x**4 + 196*x**2 - 31, bracket=[0, 1], method='brentq').root

def cartesian_to_barycentric(simplex, point):
    simplex = np.array([[pt.real, pt.imag] for pt in simplex])
    point = np.array([point.real, point.imag])
    dim = np.shape(simplex)
    var = sp.symbols(f'x:{dim[0]}')
    equations = [sum(var[i] * simplex[i][j] for i in range(dim[0])) - point[j] for j in range(dim[1])]
    equations.append(sum(var) - 1)
    sol = sp.solve(equations, var)
    return [float(sol[v]) for v in var]

def barycentric_to_cartesian(simplex, barycentric_coord):
    assert len(simplex) == len(barycentric_coord)
    return np.sum([barycentric_coord[i] * simplex[i] for i in range(len(simplex))])

ABC_to_E_barycentric = cartesian_to_barycentric([prototype_A, prototype_B, prototype_C], prototype_E)
ABC_to_F_barycentric = cartesian_to_barycentric([prototype_A, prototype_B, prototype_C], prototype_F)

def subdivide(faces):
    result = []
    for ctg, A, B, C, D in faces:
        if ctg == 0:
            E = barycentric_to_cartesian([A, B, C], ABC_to_E_barycentric)
            F = barycentric_to_cartesian([A, B, C], ABC_to_F_barycentric)
            result += [(0, B, C, D, E), (1, E, D, A, F)]
        elif ctg == 1:
            result += [(2, A, B, C, D)]
        elif ctg == 2:
            result += [(0, A, B, C, D)]
    return result

def get_geometric_center(face):
    return (face[1]+face[2]+face[3]+face[4])/4

def get_edges_repr_indices(faces, vertices):
    edges = []
    for face in faces:
        vs_on_f_indices = vertices_on_face_repr_indices(face, vertices)
        edges.append((vs_on_f_indices[1], vs_on_f_indices[2]))
        edges.append((vs_on_f_indices[2], vs_on_f_indices[3]))
        edges.append((vs_on_f_indices[3], vs_on_f_indices[4]))
        edges.append((vs_on_f_indices[4], vs_on_f_indices[1]))
    return np.asarray(edges)

def get_edges(faces, vertices):
    edges = []
    for face in faces:
        vs_on_f = vertices_on_face(face, vertices)
        edges.append((vs_on_f[1], vs_on_f[2]))
        edges.append((vs_on_f[2], vs_on_f[3]))
        edges.append((vs_on_f[3], vs_on_f[4]))
        edges.append((vs_on_f[4], vs_on_f[1]))
    return np.asarray(edges)

def get_qc_code(faces, vertices):
    h = np.zeros((len(faces), len(vertices)))
    for i, face in enumerate(faces):
        for j in range(len(vertices)):
            for k in range(1, len(face)):
                if close(face[k], vertices[j]):
                    h[i,j] = 1
    return h

####################################################################################################
start = timer()
gen = 22
faces = []
faces.append((0, prototype_A, prototype_B, prototype_C, prototype_D))
for _ in range(gen):
    faces = subdivide(faces)

faces_pos = [np.array([get_geometric_center(face).real, get_geometric_center(face).imag]) for face in faces]
vertices = get_vertices(faces)
edges = get_edges(faces, vertices)
edges_repr_indices = get_edges_repr_indices(faces, vertices)
h = get_qc_code(faces, vertices)
# h = h.T
print('h.shape = ', h.shape)
print('k = ', h.shape[1]-rank(h))
logical_op = []
d_bound, logical_op = get_classical_code_distance_special_treatment(h, target_weight=h.shape[1]//8)
# d_bound, logical_op = get_classical_code_distance_special_treatment(h, target_weight=get_classical_code_distance_time_limit(h, time_limit=30))
print('d_bound = ', d_bound)
fig, ax = draw_qc_code_logical(faces, vertices, edges, faces_pos, h, logical_op)
# fig, ax = draw_qc_transposecode_logical(faces, vertices, edges, faces_pos, h, logical_op)
end = timer()
print('Time elapsed: ', end-start, ' seconds')
plt.show()

####################################################################################################
# save data
vertices_pos = np.array([[v.real, v.imag] for v in vertices])
savename = f'psi_tiling_gen_{gen}.npz'
savepath = os.path.join(savedir, savename)
np.savez(savepath, vertices_pos=vertices_pos, faces_pos=faces_pos, edges=edges_repr_indices, h=h)