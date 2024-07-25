'''
Cut window is negative eigenspace
Obtain tiling in the postive eigenspace 
'''

from timeit import default_timer as timer
import numpy as np
from ldpc.mod2 import *
from cnp_utils import *
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
    '''
    Calculate the code distance of the classical code within the time limit.
    Return:
        k: int, the dimension of the code
        d: int, the minimum Hamming distance found within the time limit
    '''
    def _find_min_weight_while_build(matrix):
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
    
    if rank(h) == h.shape[1]:
        print('Code is full rank, no codewords')
        return 0, np.inf
    else:
        start = timer()
        ker = nullspace(h)
        k = len(ker)
        d = _find_min_weight_while_build(ker)
        return k, d

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

def pipeline_calc_code(project_pts, adjacency_matrix,h):
    # get the minimum distance of the code
    k, d = get_classical_code_distance_time_limit(h, time_limit=10)
    print(f'k={k}, d={d}')
    
    if d == np.inf:
        return None, None, None, None

    # get the logical operator
    _, logical_op = get_classical_code_distance_special_treatment(h, d)

    # draw the code
    fig, ax = draw_laplacian_code_logical(project_pts, 
                                          adjacency_matrix,
                                          logical_op)
    
    return fig, ax, k, d

def check_expander(low, high):
    _, adj_mat = tiling(low, high)
    assert np.allclose(adj_mat, adj_mat.T), 'Adjacency matrix is not symmetric'
    normalized_lap = np.zeros((adj_mat.shape[0], adj_mat.shape[1]))
    for i in range(adj_mat.shape[0]):
        normalized_lap[i, i] = np.sum(adj_mat[i])
        for j in range(adj_mat.shape[1]):
            if j != i and adj_mat[i, j]:
                normalized_lap[i, j] = -1.
    assert np.allclose(normalized_lap, normalized_lap.T), \
        'Normalized Laplacian matrix is not symmetric'
    s = np.linalg.eigvalsh(normalized_lap)
    assert np.allclose(s[0], 0), \
        'The smallest eigenvalue of the normalized Laplacian matrix is not zero'
    # check the spectral gap
    spectral_gap = s[1]
    return spectral_gap

def save_spectal_gap(out_file, **kwargs):
    low, high, n = kwargs['low'], kwargs['high'], kwargs['n']
    spectral_gap = kwargs['spectral_gap']
    with open(out_file, 'w') as f:
        f.write(f'{low}, {high}, {n}, {spectral_gap}\n')

def check_bipartite_expander(low, high):
    # FIXME: get a more pricise way of checking bipartite expander
    project_pts, adjacency_matrix = tiling(low, high)
    # parity-check matrix is equivalent to the adj. matrix of bipartite graph
    h = get_laplacian_code(adjacency_matrix, anti=True).astype(float)
    
    # normalize the bipartite adjacency matrix
    left_mul = np.zeros_like(h)
    for i in range(h.shape[0]):
        left_mul[i, i] = np.sqrt(1 / np.sum(h[i]))
    
    right_mul = np.zeros_like(h)
    for i in range(h.shape[0]):
        right_mul[i, i] = np.sqrt(1 / np.sum(h[:,i]))
    
    h_normalized = left_mul @ h @ right_mul
    s = np.linalg.svd(h_normalized, compute_uv=False)

    assert np.allclose(s[0], 1)
    # check the spectral gap
    spectral_gap = 1 - s[1]
    print(f'Spectral gap: {spectral_gap}')
    
if __name__ == '__main__':
    low = -5
    high = 6
    project_pts, adjacency_matrix = tiling(low, high)
    n = project_pts.shape[1]
    print(f'number of pints: {n}')
    spectral_gap = check_expander(low, high)
    save_spectal_gap('spectral_gap.txt',
                     low=low, high=high, n=n, spectral_gap=spectral_gap)
    print(f'Spectral gap: {spectral_gap}')

    # h = get_laplacian_code(adjacency_matrix, anti=False)
    # fig1, ax1, k1, d1 = pipeline_calc_code(project_pts, adjacency_matrix, h)
    # if d1:
    #     ax1.set_title(f'Laplacian code, [n,k,d]=[{n},{k1},{d1}]')
    #     fig1.savefig(f'lap_low={low}_high={high}.png')

    # h = get_laplacian_code(adjacency_matrix, anti=True)
    # fig2, ax2, k2, d2 = pipeline_calc_code(project_pts, adjacency_matrix, h)
    # if d2:
    #     ax2.set_title(f'Anti-Laplacian code, [n,k,d]=[{n},{k2},{d2}]')
    #     fig2.savefig(f'anti_lap_low={low}_high={high}.png')

    # h = get_adjacency_code(adjacency_matrix, anti=False)
    # fig3, ax3, k3, d3 = calc_code(project_pts, adjacency_matrix, h)
    # if d3:
    #     ax3.set_title(f'Adjacency code, [n,k,d]=[{n},{k3},{d3}]')
    #     ax3.savefig(f'adj_low={low}_high={high}.png')

    # h = get_adjacency_code(adjacency_matrix, anti=True)
    # fig4, ax4, k4, d4 = calc_code(project_pts, adjacency_matrix, h)
    # if d4:
    #     ax4.set_title(f'Anti-Adjacency code, [n,k,d]=[{n},{k4},{d4}]')
    #     ax4.savefig(f'anti_adj_low={low}_high={high}.png')

    plt.show()
    
    # pipeline(low, high)
