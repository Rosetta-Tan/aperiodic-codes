'''
Obtain the self-dual graphical codes on 3D and-and-poject tiling.
'''
import numpy as np
from aperiodic_codes.cut_and_project.cnp_utils import *

def are_neighbor(pt1, pt2):  
    return np.sum(np.abs(pt1 - pt2)) == 1

def gen_adj_mat(cut_pts):
    '''
    connect neighboring points in cut_pts 
    according to connectivity in the 6D lattice:
    if two points are neighbors in the 6D lattice,
    then they are also connected by an edge in the 3D tiling
    Args:
        cut_pts: np.array, shape=(6, n_cut)
    '''
    assert cut_pts.shape[0] == 6  # shape (6, n_cut)
    n_cut = cut_pts.shape[1]
    adjacency_matrix = np.zeros((n_cut, n_cut), dtype=int)
    for i in range(n_cut):
        for j in range(i+1, n_cut):
            if are_neighbor(cut_pts[:, i], cut_pts[:, j]):
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1

    return adjacency_matrix

def get_adjacency_code(adjacency_matrix, anti=False):
    '''
    Return the adjacency matrix of the Laplacian code.
    '''
    parity_check_matrix = adjacency_matrix.astype(int)
    if anti:
        parity_check_matrix = (np.eye(adjacency_matrix.shape[0]) + \
                                parity_check_matrix) % 2
    return parity_check_matrix

def get_laplacian_code(adjacency_matrix, anti=False):
    '''
    Return the Laplacian matrix (mod 2) of the Laplacian code.
    '''
    parity_check_matrix = adjacency_matrix.astype(int)
    for i in range(adjacency_matrix.shape[0]):
        if anti:
            parity_check_matrix[i, i] = (np.sum(adjacency_matrix[i]) + 1) % 2
        else:
            parity_check_matrix[i, i] = np.sum(adjacency_matrix[i]) % 2
    return parity_check_matrix