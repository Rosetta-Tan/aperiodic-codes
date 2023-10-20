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
parser.add_argument('--n', dest='n', type=int, required=True) # number of nodes
parser.add_argument('--deglo', dest='deglo', type=int, default=3, help='lower bound of degree of each node')
parser.add_argument('--degup', dest='degup', type=int, default=5, help='upper bound of degree of each node')
parser.add_argument('--p', dest='p', type=float, default=0.5)
parser.add_argument('--d', dest='d', type=int, default=6)
parser.add_argument('--seed', dest='seed', type=int, default=0, help='random seed')
parser.add_argument('--readdir', dest='readdir', type=str, default='/Users/yitan/Google Drive/My Drive/from_cannon/qmemory_simulation/data/laplacian_code')
parser.add_argument('--savedir', dest='savedir', type=str, default='/Users/yitan/Google Drive/My Drive/from_cannon/qmemory_simulation/data/laplacian_code')
args = parser.parse_args()
n = args.n
deglo = args.deglo
degup = args.degup
p = args.p
d = args.d
seed = args.seed
readdir = args.readdir
savedir = args.savedir
readname = f'hclassical_configurationmodel_n={n}_deglo={deglo}_degcheck={degup}_seed={seed}.txt'
savename = f'codedistance_hclassical_noprledgelocal_n={n}_deglo={deglo}_degcheck={degup}_seed={seed}.npy'
# readname = f'hclassical_erdosrenyi_n={n}_p={p}_seed={seed}.txt'
# savename = f'codedistance_hclassical_erdosrenyi_n={n}_p={p}_seed={seed}.npy'
# readname = f'hclassical_randomregular_n={n}_d={d}_seed={seed}.txt'
# savename = f'codedistance_hclassical_randomregular_n={n}_d={d}_seed={seed}.npy'
readpath = os.path.join(readdir, readname)
h = read_pc(readpath)

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

rank_h = rank(h)
print(f'rank_h: {rank_h}')
distance_h = get_classical_code_distance(h)
print(f'distance_h: {distance_h}')

savepath_h = os.path.join(savedir, savename)
np.save(savepath_h, distance_h)

