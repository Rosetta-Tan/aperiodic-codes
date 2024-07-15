'''
Cut window is positive eigenspace
Obtain tiling in the negative eigenspace 
'''

from timeit import default_timer as timer
import numpy as np
from ldpc.mod2 import *
from three_dim_cnp import *

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

def get_classical_code_distance_time_limit(h, time_limit=10):
    if rank(h) == h.shape[1]:
        print('Code is full rank, no codewords')
        return np.inf
    else:
        start = timer()
        ker = nullspace(h)
        def find_min_weight_while_build(matrix):
            span = []
            min_hamming_weight = np.inf
            for ir, row in enumerate(matrix):
                print('debug: ir = ', ir, 
                      'current min_hamming_weight = ', 
                      min_hamming_weight, flush=True)  # debug
                row_hamming_weight = np.sum(row)
                if row_hamming_weight < min_hamming_weight:
                    min_hamming_weight = row_hamming_weight
                temp = [row]
                for element in span:
                    newvec = (row + element) % 2
                    temp.append(newvec)
                    newvec_hamming_weight = np.sum(newvec)
                    if newvec_hamming_weight < min_hamming_weight:
                        min_hamming_weight = newvec_hamming_weight
                    end = timer()
                    if (end - start) > time_limit:
                        print('Time limit reached, aborting ...')
                        return min_hamming_weight
                span = list(np.unique(temp + span, axis=0))
            assert len(span) == 2**len(matrix) - 1
            return min_hamming_weight
        min_hamming_weight = find_min_weight_while_build(ker)
        return min_hamming_weight

def get_classical_code_distance_special_treatment(h, target_weight):
    """_summary_

    Args:
        h (_type_): _description_
        target_weight (_type_): _description_

    Returns:
        _type_: _description_
    """
    if rank(h) == h.shape[1]:
        print('Code is full rank, no codewords')
        return np.inf
    else:
        start = timer()
        print('Code is not full rank, there are codewords')
        print('Computing codeword space basis ...')
        ker = nullspace(h)
        end = timer()
        print(f'Elapsed time for computing codeword space basis: {end-start} seconds', flush=True)
        print('len of ker: ', len(ker))
        print('Start finding minimum Hamming weight while buiding codeword space ...')
        start = end
        
        def find_min_weight_while_build(matrix):
            span = []
            min_hamming_weight = np.inf
            for ir, row in enumerate(matrix):
                row_hamming_weight = np.sum(row)
                if row_hamming_weight < min_hamming_weight:
                    min_hamming_weight = row_hamming_weight
                    if min_hamming_weight <= target_weight:
                        assert np.sum(row) == min_hamming_weight
                        return min_hamming_weight, row
                temp = [row]
                for element in span:
                    newvec = (row + element) % 2
                    temp.append(newvec)
                    newvec_hamming_weight = np.sum(newvec)
                    if newvec_hamming_weight < min_hamming_weight:
                        min_hamming_weight = newvec_hamming_weight
                        if min_hamming_weight <= target_weight:
                            assert np.sum(newvec) == min_hamming_weight
                            return min_hamming_weight, newvec
                span = list(np.unique(temp + span, axis=0))
            assert len(span) == 2**len(matrix) - 1
        min_hamming_weight, logical_op = find_min_weight_while_build(ker)
        end = timer()
        print(f'Elapsed time for finding minimum Hamming weight while buiding codeword space : {end-start} seconds', flush=True)
        return min_hamming_weight, logical_op
    
def draw_laplacian_code_logical(proj_pts, adjacency_matrix, logical_op):    
    # make 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # make three axes have the same scale
    ax.set_box_aspect([1,1,1])

    ax.scatter(proj_pts[0], proj_pts[1], proj_pts[2])
    # plot the edges
    for i in range(proj_pts.shape[1]):
        for j in range(i+1, proj_pts.shape[1]):
            if adjacency_matrix[i, j]:
                ax.plot([proj_pts[0, i], proj_pts[0, j]],
                        [proj_pts[1, i], proj_pts[1, j]],
                        [proj_pts[2, i], proj_pts[2, j]],
                        color='gray', alpha=0.5)

    # plot the logical operator
    ones = [i for i in range(len(logical_op)) if logical_op[i] == 1]
    ax.scatter(proj_pts[0, ones],
               proj_pts[1, ones],
               proj_pts[2, ones],
               color='red', s=50, zorder=100)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    return fig, ax

def calc_code(project_pts, adjacency_matrix,h):
    # get the minimum distance of the code
    min_hamming_weight = get_classical_code_distance_time_limit(h, time_limit=30)
    print(f'minimum Hamming weight: {min_hamming_weight}')
    
    if min_hamming_weight == np.inf:
        return None, None, None

    # get the logical operator
    _, logical_op = get_classical_code_distance_special_treatment(
        h, min_hamming_weight)

    # draw the code
    fig, ax = draw_laplacian_code_logical(project_pts, 
                                          adjacency_matrix,
                                          logical_op)
    
    return fig, ax, min_hamming_weight

def pipeline(n):
    project_pts, adjacency_matrix = tiling_neg(n)
    h = get_laplacian_code(adjacency_matrix, anti=False)
    fig1, ax1, d1 = calc_code(project_pts, adjacency_matrix, h)
    if d1:
        ax1.set_title(f'Laplacian code, d={d1}')

    h = get_laplacian_code(adjacency_matrix, anti=True)
    fig2, ax2, d2 = calc_code(project_pts, adjacency_matrix, h)
    if d2:
        ax2.set_title(f'Anti-Laplacian code, d={d2}')

    h = get_adjacency_code(adjacency_matrix, anti=False)
    fig3, ax3, d3 = calc_code(project_pts, adjacency_matrix, h)
    if d3:
        ax3.set_title(f'Adjacency code, d={d3}')

    h = get_adjacency_code(adjacency_matrix, anti=True)
    fig4, ax4, d4 = calc_code(project_pts, adjacency_matrix, h)
    if d4:
        ax4.set_title(f'Anti-Adjacency code, d={d4}')

    plt.show()

if __name__ == '__main__':
    n = 8
    pipeline(n)
