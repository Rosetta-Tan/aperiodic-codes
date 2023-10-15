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

def read_sparse_pc(filepath):
    """
    Read sparse parity check matrix from file.
    """
    pc_sparse = load_npz(filepath)
    # pc = pc_sparse.toarray()
    assert pc_sparse.dtype == bool
    return pc_sparse

parser = argparse.ArgumentParser()
parser.add_argument('--size1', dest='s1', type=int, required=True, help='size multiplier of RGG code 1')
parser.add_argument('--size2', dest='s2', type=int, required=True, help='size multiplier of RGG code 2')
parser.add_argument('--radius1', dest='r1', type=float, help='distance threshold for RGG code 1')
parser.add_argument('--radius2', dest='r2', type=float, help='distance threshold for RGG code 2')
parser.add_argument('--seed1', dest='seed1', type=int, default=0, help='rng seed for generating RGG code 1')
parser.add_argument('--seed2', dest='seed2', type=int, default=0, help='rng seed for generating RGG code 2')
args = parser.parse_args()
deg_bit = 4
deg_check = 5
deg_bit = 4
deg_check = 5
size1 = args.s1
size2 = args.s2
r1 = args.r1
r2 = args.r2
seed1 = args.seed1
seed2 = args.seed2
n1 = deg_check*size1
m1 = deg_bit*size1
n2 = deg_check*size2
m2 = deg_bit*size2
assert n1*deg_bit == m1*deg_check
assert n2*deg_bit == m2*deg_check
n = n1*n2+m1*m2
mx = m1*n2
mz = m2*n1
readdir = '/Users/yitan/Google Drive/My Drive/from_cannon/qmemory_simulation/data/rgg_code'
savedir = '/Users/yitan/Google Drive/My Drive/from_cannon/qmemory_simulation/data/rgg_code'
readpath_x = os.path.join(readdir, f'hxhgp_n1_{n1}_m1_{m1}_n2_{n2}_m2_{m2}_n_{n}_mx_{mx}_degbit{deg_bit}_degcheck{deg_check}_r1_{r1}_r2_{r2}_seed1_{seed1}_seed2_{seed2}.txt')
readpath_z = os.path.join(readdir, f'hzhgp_n1_{n1}_m1_{m1}_n2_{n2}_m2_{m2}_n_{n}_mz_{mz}_degbit{deg_bit}_degcheck{deg_check}_r1_{r1}_r2_{r2}_seed1_{seed1}_seed2_{seed2}.txt')
# readpath_x = os.path.join(readdir, f'hxhgp_rescaled_n1_{n1}_m1_{m1}_n2_{n2}_m2_{m2}_n_{n}_mx_{mx}_degbit{deg_bit}_degcheck{deg_check}_r1_{r1}_r2_{r2}_seed1_{seed1}_seed2_{seed2}.txt')
# readpath_z = os.path.join(readdir, f'hzhgp_rescaled_n1_{n1}_m1_{m1}_n2_{n2}_m2_{m2}_n_{n}_mz_{mz}_degbit{deg_bit}_degcheck{deg_check}_r1_{r1}_r2_{r2}_seed1_{seed1}_seed2_{seed2}.txt')
hx = read_pc(readpath_x)
hz = read_pc(readpath_z)

# log_x_with_z_stabilizer_redundancy = row_span(row_basis(hx))
# log_z_with_x_stabilizer_redundancy = row_span(row_basis(hz))

def get_dx(hx):
    """
    Get the code distance of the X check matrix dx.
    dx is defined as the minimum weight of all non-zero codewords of Hx.
    """
    cws = codewords(hx)
    log_space = row_span(cws)
    

