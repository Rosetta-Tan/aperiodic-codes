import logging
from dataclasses import dataclass
from typing import FrozenSet, Iterable, List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import stim
from numba import njit
from ldpc.mod2 import rank
from aperiodic_codes.cut_and_project.z2 import row_echelon, nullspace, row_basis
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


ADJ_QUBITS = frozenset((0 + 0.5j, 0.5 + 0j, 0 - 0.5j, -0.5 + 0j))

def torus(c: complex, *, d: int) -> complex:
    r = c.real % d
    i = c.imag % d
    return r + i * 1j

def gen_geometry(d: int):
    vertices = np.array([c[0] + 1j*c[1] for c in np.ndindex((d, d))])
    edges = np.concatenate([vertices + 0.5, vertices + 0.5j])
    faces = vertices + 0.5 + 0.5j
    return vertices, edges, faces

def x_stabilizer(c: complex, q2i: Dict[complex, int], *, d: int) -> stim.PauliString:
    # c is a complex number representing a vertex, thus center of an X stabilizer
    qubits = [torus(c + move, d=d) for move in ADJ_QUBITS]
    qubit_inds = [q2i[q] for q in qubits]
    return qubit_inds

def z_stabilizer(c: complex, q2i: Dict[complex, int], *, d) -> stim.PauliString:
    # c is a complex number representing a face, thus center of a Z stabilizer
    qubits = [torus(c + move, d=d) for move in ADJ_QUBITS]
    qubit_inds = [q2i[q] for q in qubits]
    return qubit_inds

def gen_parity_check(vertices: np.ndarray,
                     edges: np.ndarray,
                     faces: np.ndarray,
                     q2i: Dict[complex, int],
                     d: int) -> np.ndarray:
    hx = np.zeros((len(vertices), len(edges)), dtype=np.int64)
    hz = np.zeros((len(faces), len(edges)), dtype=np.int64)
    for iv, v in enumerate(vertices):
        hx[iv, x_stabilizer(v, q2i, d=d)] = 1
    for i_f, f in enumerate(faces):
        hz[i_f, z_stabilizer(f, q2i, d=d)] = 1
    return hx, hz

@njit('(int64[:,::1], bool_)', cache=True)
def row_echelon(matrix, full):
    num_rows, num_cols = matrix.shape
    the_matrix = np.copy(matrix)
    transform_matrix = np.identity(num_rows, dtype=np.int64)
    pivot_row = 0
    pivot_cols = np.empty(min(num_rows, num_cols), dtype=np.int64)
    pivot_cols_count = 0

    for col in range(num_cols):
        if the_matrix[pivot_row, col] != 1:
            swap_row_index = pivot_row + np.argmax(the_matrix[pivot_row:num_rows, col])
            if the_matrix[swap_row_index, col] == 1:
                tmp = np.copy(the_matrix[pivot_row])
                the_matrix[pivot_row] = the_matrix[swap_row_index]
                the_matrix[swap_row_index] = tmp

                tmp = np.copy(transform_matrix[pivot_row])
                transform_matrix[pivot_row] = transform_matrix[swap_row_index]
                transform_matrix[swap_row_index] = tmp

        if the_matrix[pivot_row, col]:
            if not full:  
                elimination_range = np.arange(pivot_row + 1, num_rows)
            else:
                elimination_range = np.concatenate((np.arange(pivot_row), np.arange(pivot_row + 1, num_rows)))

            mask = the_matrix[elimination_range, col] == 1
            the_matrix[elimination_range[mask]] = (the_matrix[elimination_range[mask]] + the_matrix[pivot_row]) % 2
            transform_matrix[elimination_range[mask]] = (transform_matrix[elimination_range[mask]] + transform_matrix[pivot_row]) % 2
            
            pivot_cols[pivot_cols_count] = col
            pivot_cols_count += 1
            pivot_row += 1

        if pivot_row >= num_rows:
            break

    matrix_rank = pivot_row
    return the_matrix, matrix_rank, transform_matrix, pivot_cols[:pivot_cols_count]

def gen_ns_log(h, filepath):
    """
    Nullspace log
    """
    transpose = np.ascontiguousarray(h.T, dtype=np.int64)
    m, n = transpose.shape
    _, matrix_rank, transform, _ = row_echelon(transpose, full=False)
    nspace = transform[matrix_rank:m]
    with open(filepath, mode='w') as f:
        for row in nspace:
            f.write(' '.join(map(str, row)) + '\n')
    return nspace

def visualize_ns(ns, edges, d):
    """
    Visualize nullspace
    """
    fig, ax = plt.subplots(d+1, d)
    for i in range(d+1):
        for j in range(d):
            ax[i, j].set_aspect('equal')
            ax[i, j].set_xlim(-0.5, d-0.5)
            ax[i, j].set_ylim(-0.5, d-0.5)
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
    for irow, row in enumerate(ns):
        for i, val in enumerate(row):
            if val:
                ax[irow//d, irow%d].scatter(edges[i].real, edges[i].imag, color='black')
        
    return fig, ax

def gen_ge_log(hx, hz, filepath):
    """
    Gaussian elimination log
    """
    # with open(filepath, mode='w') as f:
    ker_hx = nullspace(hx)
    im_hzT = row_basis(hz)

    log_stack = np.vstack([im_hzT,ker_hx])
    transpose = np.ascontiguousarray(log_stack.T, dtype=np.int64)
    pivots = row_echelon(transpose, full=False)[3]
    log_op_indices = [i for i in range(im_hzT.shape[0],log_stack.shape[0]) if i in pivots]
    log_ops = log_stack[log_op_indices]

    return log_ops
        

if __name__ == '__main__':
    d = 3
    vertices, edges, faces = gen_geometry(d)
    e2i = {e: i for i, e in enumerate(edges)}
    hx, hz = gen_parity_check(vertices, edges, faces, e2i, d)
    np.save('hx.npy', hx)
    np.save('hz.npy', hz)
    ns_x = nullspace(hx)
    ns_z = nullspace(hz)
    filepath = 'ns_x.log'
    gen_ns_log(hx, filepath)
    fig_ns_s, ax_ns_s = visualize_ns(ns_x, edges, d)
    fig_ns_s.savefig('ns_x.png')

    cnt = 0
    for i in range(len(ns_x)):
        for j in range(len(ns_z)):
            if np.sum((ns_x[i] @ ns_z[j].T) % 2) % 2 == 1:
                cnt += 1
                print(f'ns_x[{i}] and ns_z[{j}] do not commute')
    print(f'Number of non-commuting pairs: {cnt}')