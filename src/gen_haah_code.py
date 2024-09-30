import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, lil_matrix
from ldpc.code_util import get_code_parameters
from bposd.hgp import hgp
from bposd.css import *
from numba import njit
import matplotlib.pyplot as plt
from sys import argv

# log2L = int(argv[1])
L = int(argv[1])
# assert np.log2(L).is_integer(), "L must be a power of 2."
n = 2*L**3

@njit
def pc_tensor(L):
    '''Numpy version should only be used when L < 2^4'''
    tz = np.zeros((L,L,L,L,L,L,2))
    tx = np.zeros((L,L,L,L,L,L,2))
    for i in range(L):
        for j in range(L):
            for k in range(L):
                'Z-type checks'
                tz[i,j,k,i,j,k] = [0, 0]
                tz[i,j,k,(i+1)%L,j,k] = [0, 1]
                tz[i,j,k,i,(j+1)%L,k] = [0, 1]
                tz[i,j,k,i,j,(k+1)%L] = [0, 1]
                tz[i,j,k,(i+1)%L,(j+1)%L,k] = [1, 0]
                tz[i,j,k,(i+1)%L,j,(k+1)%L] = [1, 0]
                tz[i,j,k,i,(j+1)%L,(k+1)%L] = [1, 0]
                tz[i,j,k,(i+1)%L,(j+1)%L,(k+1)%L] = [1, 1]
                'X-type checks'
                tx[i,j,k,i,j,k] = [1, 1]
                tx[i,j,k,(i+1)%L,j,k] = [0, 1]
                tx[i,j,k,i,(j+1)%L,k] = [0, 1]
                tx[i,j,k,i,j,(k+1)%L] = [0, 1]
                tx[i,j,k,(i+1)%L,(j+1)%L,k] = [1, 0]
                tx[i,j,k,(i+1)%L,j,(k+1)%L] = [1, 0]
                tx[i,j,k,i,(j+1)%L,(k+1)%L] = [1, 0]
                tx[i,j,k,(i+1)%L,(j+1)%L,(k+1)%L] = [0, 0]
    return tz, tx

@njit
def pc_matrix(L, tz, tx):
    '''Numpy version should only be used when L < 2^4'''
    hz = tz.reshape((L**3, L**3*2))
    hx = tx.reshape((L**3, L**3*2))
    return hz, hx

# @njit
def pc_check_sparse(L):
    '''Scipy version should be used when L >= 2^4'''
    hz = csr_matrix((L**3, L**3*2))
    hx = csr_matrix((L**3, L**3*2))
    @njit
    def idx_check(i, j, k):
        return i*L**2 + j*L + k
    @njit
    def idx_bit(i, j, k, sublat):
        return i*2*L**2 + j*2*L + 2*k + sublat
    for i in range(L):
        for j in range(L):
            for k in range(L):
                'Z-type checks'
                hz[idx_check(i,j,k), idx_bit((i+1)%L,j,k,1)] = 1
                hz[idx_check(i,j,k), idx_bit(i,(j+1)%L,k,1)] = 1
                hz[idx_check(i,j,k), idx_bit(i,j,(k+1)%L,1)] = 1
                hz[idx_check(i,j,k), idx_bit((i+1)%L,(j+1)%L,k,0)] = 1
                hz[idx_check(i,j,k), idx_bit((i+1)%L,j,(k+1)%L,0)] = 1
                hz[idx_check(i,j,k), idx_bit(i,(j+1)%L,(k+1)%L,0)] = 1
                hz[idx_check(i,j,k), idx_bit((i+1)%L,(j+1)%L,(k+1)%L,0)] = 1
                hz[idx_check(i,j,k), idx_bit((i+1)%L,(j+1)%L,(k+1)%L,1)] = 1
                'X-type checks'
                hx[idx_check(i,j,k), idx_bit(i,j,k,0)] = 1
                hx[idx_check(i,j,k), idx_bit(i,j,k,1)] = 1
                hx[idx_check(i,j,k), idx_bit((i+1)%L,j,k,1)] = 1
                hx[idx_check(i,j,k), idx_bit(i,(j+1)%L,k,1)] = 1
                hx[idx_check(i,j,k), idx_bit(i,j,(k+1)%L,1)] = 1
                hx[idx_check(i,j,k), idx_bit((i+1)%L,(j+1)%L,k,0)] = 1
                hx[idx_check(i,j,k), idx_bit((i+1)%L,j,(k+1)%L,0)] = 1
                hx[idx_check(i,j,k), idx_bit(i,(j+1)%L,(k+1)%L,0)] = 1
    return hz, hx

####################################################################################################
# Generate the parity check matrices
####################################################################################################
tz, tx = pc_tensor(L)
hz, hx = pc_matrix(L, tz, tx)
np.savetxt(f'../data/haah_code/hz_L{L}.txt', hz, fmt='%d')
np.savetxt(f'../data/haah_code/hx_L{L}.txt', hx, fmt='%d')

# hz, hx = pc_check_sparse(L)
# sp.save_npz(f'../data/haah_code/hz_L{L}.npz', hz)
# sp.save_npz(f'../data/haah_code/hx_L{L}.npz', hx)

####################################################################################################
# Test
####################################################################################################
'''Test ground state degeneracy'''
# log2GSD = 4*L-2
# qcode = css_code(hx=hx, hz=hz)
# assert qcode.K == log2GSD, f"Ground state degeneracy is wrong. qcode.K = {qcode.K}, log2GSD = {log2GSD}."