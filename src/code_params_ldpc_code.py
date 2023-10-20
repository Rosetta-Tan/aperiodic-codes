import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import chain, combinations, zip_longest
from operator import itemgetter
from ldpc.mod2 import *
from ldpc.code_util import *
from ldpc.mod2sparse import *
from bposd.hgp import hgp
from bposd.css import *
import os, sys
import argparse
from scipy.sparse import csr_matrix, save_npz, load_npz
from numba import njit, jit
from timeit import default_timer as timer
from itertools import product
import warnings
from numba.core.errors import NumbaWarning, NumbaDeprecationWarning, NumbaPendingDeprecationWarning

warnings.simplefilter('ignore', category=NumbaWarning)
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

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

parser = argparse.ArgumentParser()
parser.add_argument('--size', dest='s', type=int, required=True, help='multiplier of deg_check (deg_bit) to get n (m)')
parser.add_argument('--radius', dest='r', type=float, help='distance threshold for RGG code')
parser.add_argument('--seed', dest='seed', type=int, default=0, help='rng seed for generating RGG code')
parser.add_argument('--readdir', dest='readdir', type=str, default='/Users/yitan/Google Drive/My Drive/from_cannon/qmemory_simulation/data/ldpc_code')
parser.add_argument('--savedir', dest='savedir', type=str, default='/Users/yitan/Google Drive/My Drive/from_cannon/qmemory_simulation/data/ldpc_code')
args = parser.parse_args()
deg_bit = 4
deg_check = 5
size = args.s
r = args.r
seed = args.seed
n = deg_check*size
m = deg_bit*size
readdir = args.readdir
savedir = args.savedir
rescaled = True

size = 20
deg_bit, deg_check = 8, 10
n, m = size*deg_check, size*deg_bit
r = 0.6
rescaled=True
seed = 20

# readname = f'hclassical_noprledgelocal_n={n}_m={m}_degbit={deg_bit}_degcheck={deg_check}_seed={seed}.txt'
readname = f'local_ldpc_n={n}_m={m}_deg_bit={deg_bit}_deg_check={deg_check}_r={r}_rescaled={rescaled}_seed={seed}.txt'
# savename = f'codedistance_hclassical_hclassical_noprledgelocal_n={n}_m={m}_degbit={deg_bit}_degcheck={deg_check}_see=d{seed}.npy'
savename = f'codedistance_local_ldpc_n={n}_m={m}_deg_bit={deg_bit}_deg_check={deg_check}_r={r}_rescaled={rescaled}_seed={seed}.txt'
readpath = os.path.join(readdir, readname)
h = read_pc(readpath)
horginal = h.copy()
# some rows of h are all zeros, remove them
h = h[~np.all(horginal == 0, axis=1)]
# some columns of h are all zeros, remove them
h = h[:, ~np.all(horginal == 0, axis=0)]
print('h.shape: ', h.shape, '(m,n)', (m,n))
print(get_ldpc_params(h))

def get_classical_code_distance(h):
    if rank(h) == h.shape[1]:
        print('Code is full rank, no codewords')
        return np.inf
    else:
        start = timer()
        print('Code is not full rank, there are codewords')
        print('Computing codeword space basis ...')
        ker = nullspace(h)
        print('debug: ker = ', ker)
        end = timer()
        print(f'Elapsed time for computing codeword space basis: {end-start} seconds', flush=True)
        print('len of ker: ', len(ker))
        print('Start finding minimum Hamming weight while buiding codeword space ...')
        start = end
        # @jit
        def find_min_weight_while_build(matrix):
            span = []
            min_hamming_weight = np.inf
            for ir, row in enumerate(matrix):
                print('debug: ir = ', ir, 'current min_hamming_weight = ', min_hamming_weight, flush=True)  # debug
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
                span = list(np.unique(temp + span, axis=0))
            assert len(span) == 2**len(matrix) - 1
            return min_hamming_weight
        min_hamming_weight = find_min_weight_while_build(ker)
        end = timer()
        print(f'Elapsed time for finding minimum Hamming weight while buiding codeword space : {end-start} seconds', flush=True)
        
        return min_hamming_weight

distance_h = get_classical_code_distance(h)
distance_hT = get_classical_code_distance(h.T)
print(f'distance_h: {distance_h}')
print(f'distance_hT: {distance_hT}')

# savepath_h = os.path.join(savedir, f'codedistance_hclassical_rescaled_n{n}_m{m}_degbit{deg_bit}_degcheck{deg_check}_r{r}_seed{seed}.npy')
# savepath_hT = os.path.join(savedir, f'codedistance_transpose_hclassical_rescaled_n{n}_m{m}_degbit{deg_bit}_degcheck{deg_check}_r{r}_seed{seed}.npy')
# np.save(savepath_h, distance_h)
# np.save(savepath_hT, distance_hT)
