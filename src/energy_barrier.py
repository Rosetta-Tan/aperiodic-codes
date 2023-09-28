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
    return np.array(pc, dtype=np.uint8)

hx = read_pc('../data/hx_outputBalancedProduct.txt')
hz = read_pc('../data/hz_outputBalancedProduct.txt')
qcode = css_code(hx, hz)  # all kinds of parameters are already obtained during init

################################################################
# input: parity check matrix
# output: logical operator with lowest energy barrier
# step1: pick one qubit which has the smallest number of connected stabilizers
#   pick a qubit that minimizes (column weight - 0)
# step2: find the qubits that are connected with the qubit picked in step1
#   pick a qubit that minimizes (column weight - 2 * current shared stabilizer weight)
# step3: repeat step2 until the logical operator is found
# stop condition: distance_to_codespace(qvec) == 0
################################################################
def greedy(qcode, synd_type='X'):
    """Greedy algorithm to search for the lowest energy barrier logical operator

    Args:
        qcode (css_code): The CSS code to be searched.
        stab_type (str): Type of syndromes. Defaults to 'X'.

    Returns:
        _type_: _description_
    """
    if synd_type == 'X':
        h = qcode.hx
        log_op_bases = qcode.lx
        log_op_space = row_span(qcode.lx)
        col_weights = np.sum(qcode.hx, axis=1)
    elif synd_type == 'Z':
        h = qcode.hz
        log_op_bases = qcode.lz
        log_op_space = row_span(qcode.lz)
        col_weights = np.sum(qcode.hz, axis=1)

    def distance_to_codespace(qvec):
        """
        Compute the minimum Hamming distance between a vector and the codespace.
        """
        return np.bitwise_or(qvec, log_op_space).sum(axis=1).min()

    qvec = np.zeros(qcode.N, dtype=np.uint8)
    # step1: intialization
    qidx = np.argmin(col_weights)
    qvec[qidx] = 1
    synd = np.mod(h@qvec, 2)
    synd_weight = np.sum(synd)
    print('qvec init: ', qvec)
    # step2 - end
    iter = 1
    print('true log op?', qvec in log_op_space)
    while (not qvec in log_op_space) and iter < 4:
        qubits_avail = np.where(qvec == 0)[0]
        # for qidx in qubits_avail:
        #     qvec = np.zeros(qcode.N, dtype=np.uint8)
        #     qvec[qidx] = 1
        #     synd = np.mod(h@qvec, 2)
        synds_avail = h[:, qubits_avail]
        col_weights_avail = np.sum(synds_avail, axis=1)
        shared_stab_weights = [np.sum(synd_avail & synd) for synd_avail in synds_avail]
        shared_stab_weights = np.array(shared_stab_weights)
        qidx = np.argmin(col_weights_avail - 2 * shared_stab_weights)
        qvec[qidx] = 1
        iter += 1
        assert np.sum(qvec) == iter
        print(qvec, np.sum(qvec))
        synd = np.mod(h@qvec, 2)
        synd_weight = np.sum(synd)
        print('iter: ', iter)


################################################################
# Debug
################################################################
# greedy(qcode, stab_type='X')
v = np.zeros(qcode.N, dtype=np.uint8)
v[5] = 1
print(distance_to_codespace(v, row_span(qcode.lx)))

# print(qcode.lz)
# print(row_span(qcode.lz).shape)