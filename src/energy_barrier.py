import numpy as np
from ldpc.code_util import *
from bposd.css import *
from bposd.hgp import hgp
from numba import njit
import matplotlib.pyplot as plt
from timeit import default_timer as timer

"""
Greedy algorithm to search for the logical operator
with the lowest energy barrier.
"""

def read_pc(filepath):
    """
    Read parity check matrix from file.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    pc = []
    for line in lines:
        row = [int(x) for x in line.split()]
        pc.append(row)
    return np.array(pc, dtype=int)

hx = read_pc('../data/hx_outputBalancedProduct.txt')
hz = read_pc('../data/hz_outputBalancedProduct.txt')
qcode = css_code(hx, hz)
'''all kinds of parameters are already obtained during init,
except for the code distance.
The way bposd.stab.compute_code_distance() goes from GF2 to GF4,
which is not optimal for CSS codes.
For CSS codes, one can obtain code distance by
enumerating Lx and Lz spaces independently.
'''

################################################################
# input: parity check matrix
# 
################################################################
def greedy(log_type='X'):
    """Greedy algorithm to search for the lowest energy barrier logical operator
        - step1: pick one qubit which has the smallest number of connected stabilizers
        pick a qubit that minimizes (column weight - 0)
        - step2: find the qubits that are connected with the qubit picked in step1
        pick a qubit that minimizes (column weight - 2 * current shared stabilizer weight)
        - step3: repeat step2 until the logical operator is found
        - stop condition: distance_to_codespace(qvec) == 0

    Args:
        qcode (css_code): The CSS code to search.
        log_type (str): Type of logical operator to search. Defaults to 'X'.

    Returns:
        qvec (arr_like): The lowest energy barrier logical operator. It's not guaranteed to be found.
    """
    if log_type == 'X':
        h = qcode.hz  # Z-type error violates X-type stabilizers
        log_op_space = row_span(qcode.lx)
        stab_weights = np.sum(qcode.hz, axis=0)
    elif log_type == 'Z':
        h = qcode.hx
        log_op_space = row_span(qcode.lz)
        stab_weights = np.sum(qcode.hx, axis=0)

    @njit
    def distance_to_codespace(qvec):
        """
        Compute the minimum Hamming distance between a vector and the codespace.
        """
        return np.bitwise_or(qvec, log_op_space).sum(axis=1).min()

    qvec = np.zeros(qcode.N, dtype=int)
    visited = []

    'step1: initialization'
    qidx = np.argmin(stab_weights)
    '''Q: how many indices does argmin return if there is degeneracy?
       A: 1, the first one, so no need to worry about degeneracy
    '''
    qvec[qidx] = 1
    visited.append(qidx)
    synd = np.mod(h@qvec, 2)
    iter = 1

    'step2 - loop - end'
    while distance_to_codespace(qvec)!=0:
        iter += 1
        # print('Visited (before this round): ', visited)
        qubits_avail = np.where(qvec==0)[0]
        for qubit in qubits_avail:
            if qubit in visited:
                raise IndexError(f"??? qubit={qubit} already visited! iter={iter}")
        stabs_avail = h[:, qubits_avail]
        stab_weights_avail = stab_weights[qubits_avail]
        print('len(stab_weights_avail): ', len(stab_weights_avail))
        shared_synd_weights = np.bitwise_and(stabs_avail.T, synd).sum(axis=1)
        to_minimize = stab_weights_avail - 2*shared_synd_weights
        for i in list(sorted(visited)):
            to_minimize = np.insert(to_minimize, i, 1000000)  # arbitrary large number

        assert np.all(to_minimize[sorted(visited)] == 1000000), f"{to_minimize}\n, \
                                                    len: {len(to_minimize)}\n, \
                                                    the relevant elements in to_minimize: {to_minimize[sorted(visited)]}\n, \
                                                    where large number?: {np.where(to_minimize==10000000)[0]}"
        assert len(to_minimize) == qcode.N, f"len(to_minimize)={len(to_minimize)}, qcode.N={qcode.N}"
        qidx = np.argmin(to_minimize)
        if qidx in visited:
            raise IndexError(f"qidx={qidx} already visited! iter={iter}")
        qvec[qidx] = 1
        visited.append(qidx)
        print('Visited (after this round): ', visited)
        assert np.all(qvec[visited] == 1), f"iter={iter}"
        assert np.sum(qvec) == iter, f"iter={iter}"
        synd = np.mod(h@qvec, 2)
        print('%'*50)

def greedy_randomized(qcode, log_type='X'):
    """Greedy algorithm (randomized version)
        - step1: pick one qubit which has the smallest number of connected stabilizers
        pick a qubit that minimizes (column weight - 0)
        - step2: find the qubits that are connected with the qubit picked in step1
        pick a qubit that minimizes (column weight - 2 * current shared stabilizer weight)
        - step3: repeat step2 until the logical operator is found
        - stop condition: distance_to_codespace(qvec) == 0

    Args:
        qcode (css_code): The CSS code to search.
        log_type (str): Type of logical operator to search. Defaults to 'X'.

    Returns:
        qvec (arr_like): The lowest energy barrier logical operator. It's not guaranteed to be found.
    """
    if log_type == 'X':
        h = qcode.hz  # Z-type error violates X-type stabilizers
        log_op_space = row_span(qcode.lx)
        stab_weights = np.sum(qcode.hz, axis=0)
    elif log_type == 'Z':
        h = qcode.hx
        log_op_space = row_span(qcode.lz)
        stab_weights = np.sum(qcode.hx, axis=0)

    @njit
    def distance_to_codespace(qvec):
        """
        Compute the minimum Hamming distance between a vector and the codespace.
        """
        return np.bitwise_or(qvec, log_op_space).sum(axis=1).min()

    qvec = np.zeros(qcode.N, dtype=int)
    visited = []
    rng = np.random.default_rng(seed=0)

    'step1: intialization'
    qidx_stack = np.where(stab_weights==np.min(stab_weights))[0]
    '''Compared to np.argmin, this manually takes into account degeneracy.
    '''
    qidx = rng.choice(qidx_stack)
    qvec[qidx] = 1
    visited.append(qidx)
    synd = np.mod(h@qvec, 2)
    synd_weight = np.sum(synd)
    iter = 1

    'step2 - loop - end'
    while distance_to_codespace(qvec)!=0:
        iter += 1
        # print('Visited (before this round): ', visited)
        qubits_avail = np.where(qvec==0)[0]
        for qubit in qubits_avail:
            if qubit in visited:
                raise IndexError(f"??? qubit={qubit} already visited! iter={iter}")
        stabs_avail = h[:, qubits_avail]
        stab_weights_avail = np.sum(stabs_avail, axis=0)
        print('len(stab_weights_avail): ', len(stab_weights_avail))
        shared_synd_weights = np.bitwise_and(stabs_avail.T, synd).sum(axis=1)
        to_minimize = stab_weights_avail - 2*shared_synd_weights
        for i in list(sorted(visited)):
            to_minimize = np.insert(to_minimize, i, 1000000)  # arbitrary large number

        assert np.all(to_minimize[sorted(visited)] == 1000000), f"{to_minimize}\n, \
                                                    len: {len(to_minimize)}\n, \
                                                    the relevant elements in to_minimize: {to_minimize[sorted(visited)]}\n, \
                                                    where large number?: {np.where(to_minimize==10000000)[0]}"
        assert len(to_minimize) == qcode.N, f"len(to_minimize)={len(to_minimize)}, qcode.N={qcode.N}"
        qidx_stack = np.where(to_minimize==np.min(to_minimize))[0]
        qidx = rng.choice(qidx_stack)
        if qidx in visited:
            raise IndexError(f"qidx={qidx} already visited! iter={iter}")
        qvec[qidx] = 1
        visited.append(qidx)
        print('Visited (after this round): ', visited)
        assert np.all(qvec[visited] == 1), f"iter={iter}"
        assert np.sum(qvec) == iter, f"iter={iter}"
        synd = np.mod(h@qvec, 2)
        print('%'*50)

def dfs(qvec, synd, qvec_history, log_type='X'):
    """Modification of the greedy algorithm. Allow backtracking.
        DFS should be realized in a recursive way.

    Args:
        qcode (css_code): The CSS code to search.
        log_type (str): Type of logical operator to search. Defaults to 'X'.

    Returns:
        qvec (arr_like): A low energy barrier logical operator.'
        It's *guaranteed* to be found.
        But it's not necessary the lowest energy cost one.
    """
    if log_type == 'X':
        h = qcode.hz  # Z-type error violates X-type stabilizers
        log_op_space = row_span(qcode.lx)
        stab_weights = np.sum(qcode.hz, axis=0)
    elif log_type == 'Z':
        h = qcode.hx
        log_op_space = row_span(qcode.lz)
        stab_weights = np.sum(qcode.hx, axis=0)
    
    @njit
    def distance_to_codespace(qvec):
        """
        Compute the minimum Hamming distance between a vector and the codespace.
        """
        return np.bitwise_or(qvec, log_op_space).sum(axis=1).min()
    
    # boundary condition of recursion
    if distance_to_codespace(qvec)==0:
        found = True
        print('Found! The found qvec: ', qvec)
        return qvec, found
    if np.sum(qvec)==qcode.N:
        print('All the qubits are in!')
        found = False
        return qvec, found

    qubits_avail = np.where(qvec==0)[0]
    print('how many qubits are in (tree depth): ', len(qvec_history))
        #   'current qvec history: ', qvec_history)
    
    # qvec_history = np.where(qvec==1)[0].tolist()
    assert np.sum(qvec) == len(qvec_history)
    stabs_avail = h[:, qubits_avail]
    stab_weights_avail = np.sum(stabs_avail, axis=0)
    shared_synd_weights = np.bitwise_and(stabs_avail.T, synd).sum(axis=1)
    to_minimize = stab_weights_avail - 2*shared_synd_weights
    for i in list(sorted(qvec_history)):
        to_minimize = np.insert(to_minimize, i, 1000000)  # arbitrary large number

    qidx_stack = np.where(to_minimize==np.min(to_minimize))[0]
    for qidx in qidx_stack:
        qvec[qidx] = 1
        qvec_history.append(qidx)
        synd = np.mod(h@qvec, 2)
        qvec, found = dfs(qvec, synd, qvec_history, log_type)
        if found:
            return qvec, found
        else:
            qvec[qidx] = 0
            qvec_history.pop()
            synd = np.mod(h@qvec, 2)
    found = False
    return qvec, found

def bfs(log_type='X'):
    'BFS is not suitable the problem'
    if log_type == 'X':
        h = qcode.hz  # Z-type error violates X-type stabilizers
        log_op_space = row_span(qcode.lx)
        stab_weights = np.sum(qcode.hz, axis=0)
    elif log_type == 'Z':
        h = qcode.hx
        log_op_space = row_span(qcode.lz)
        stab_weights = np.sum(qcode.hx, axis=0)
    
    @njit
    def distance_to_codespace(qvec):
        """
        Compute the minimum Hamming distance between a vector and the codespace.
        """
        return np.bitwise_or(qvec, log_op_space).sum(axis=1).min()
    
    qvec = np.zeros(qcode.N, dtype=int)
    synd = np.zeros(h.shape[0], dtype=int)
    qidx_bfsqueue = []
    visited = []
    cnt = 0
    
    'initialization'
    qidx_choice = np.where(stab_weights==np.min(stab_weights))[0]
    '''Compared to np.argmin, this manually takes into account degeneracy.
    '''
    qidx = qidx_choice[0]
    qidx_bfsqueue.append(qidx)  #入队
    visited.append(qidx)

    'loop - end'
    while len(qidx_bfsqueue) != 0:
        now = qidx_bfsqueue[0]  #取队首元素
        qvec[now] = 1
        cnt += 1

        if distance_to_codespace(qvec)==0:
            return qvec
        
        # generate new status
        qubits_avail = np.where(qvec==0)[0]
        qubits_noavail = np.where(qvec==1)[0].tolist()
        stabs_avail = h[:, qubits_avail] 
        stab_weights_avail = np.sum(stabs_avail, axis=0)
        print('len(stab_weights_avail): ', len(stab_weights_avail))
        shared_synd_weights = np.bitwise_and(stabs_avail.T, synd).sum(axis=1)
        to_minimize = stab_weights_avail - 2*shared_synd_weights
        for i in list(sorted(qubits_noavail)):
            to_minimize = np.insert(to_minimize, i, 1000000)  # arbitrary large number
        assert np.all(to_minimize[qubits_noavail]==1000000)
        qidx_choice = np.where(stab_weights==np.min(stab_weights))[0]
        for qidx in qidx_choice:
            if qidx not in visited:
                qidx_bfsqueue.append(qidx)
                visited.append(qidx)

        qidx_bfsqueue.pop(0)  #出队
    



################################################################
# Run
################################################################
# greedy(qcode, log_type='Z')  # doesn't work
# greedy_randomized(qcode, log_type='Z')  # doesn't work
qvec = np.zeros(qcode.N, dtype=int)
synd = np.zeros(hx.shape[0], dtype=int)
qvec_history = []
qvec, found = dfs(qvec, synd, qvec_history, log_type='Z')
print('The found qvec: ', qvec)