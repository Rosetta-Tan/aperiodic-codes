from timeit import default_timer as timer
import numpy as np
from ldpc.mod2 import *
from ldpc import mod2sparse

# FIXME: Reduce redundant computation, organize print statements

def get_classical_code_distance_time_limit(h, time_limit=10):
    """
    Calculate the code distance of the classical code within the time limit.

    Returns:
        k: int, the dimension of the code
        d: int, the minimum Hamming distance found within the time limit
    """
    def _find_min_weight_while_build(matrix):
            span = []
            min_hamming_weight = np.inf
            for ir, row in enumerate(matrix):
                print('debug: ir = ', ir, 
                      'current min_hamming_weight = ', 
                      min_hamming_weight, flush=True)  # debug
                row_hamming_weight = np.sum(row)
                if row_hamming_weight < min_hamming_weight:
                    print(np.squeeze(np.argwhere(row == 1)));
                    min_hamming_weight = row_hamming_weight
                temp = [row]
                for element in span:
                    newvec = (row + element) % 2
                    temp.append(newvec)
                    newvec_hamming_weight = np.sum(newvec)
                    if newvec_hamming_weight < min_hamming_weight:
                        print(np.squeeze(np.argwhere(newvec == 1)));
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

def compute_lz(hx,hz):
    #lz logical operators
    #lz\in ker{hx} AND \notin Im(Hz.T)

    ker_hx = nullspace(hx) #compute the kernel basis of hx
    im_hzT = row_basis(hz) #compute the image basis of hz.T

    #in the below we row reduce to find vectors in kx that are not in the image of hz.T.
    log_stack = np.vstack([im_hzT,ker_hx])
    pivots = row_echelon(log_stack.T)[3]
    log_op_indices = [i for i in range(im_hzT.shape[0],log_stack.shape[0]) if i in pivots]
    log_ops = log_stack[log_op_indices]

    return log_ops

def cplmtspace(h):
    """
    Compute the complement space of the row space of h (img h.T).
    Note: row space of h means all linear combinations of checks.

    Returns:
        bases: np.array, the bases of the complement space
    """
    if h.shape[0] != h.shape[1]:
        raise NotImplementedError('Only square matrices are supported')
    else:
        img_hT = row_basis(h)
        f2n = np.eye(h.shape[0], dtype=int)
        #in the below we row reduce to find vectors in ker_h that are not in the image of h.T.
        cplmt_stack = np.vstack([img_hT,f2n])
        pivots = row_echelon(cplmt_stack.T)[3]
        cplmt_indices = [i for i in range(img_hT.shape[0], cplmt_stack.shape[0]) if i in pivots]
        cplmt_vecs = cplmt_stack[cplmt_indices]
        return cplmt_vecs