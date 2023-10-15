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
import os
from sys import argv
import argparse
from scipy.sparse import csr_matrix, save_npz

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
parser.add_argument('--size1', dest='s1', type=int, required=True, help='size multiplier of RGG code 1')
parser.add_argument('--size2', dest='s2', type=int, required=True, help='size multiplier of RGG code 2')
parser.add_argument('--radius1', dest='r1', type=float, help='distance threshold for RGG code 1')
parser.add_argument('--radius2', dest='r2', type=float, help='distance threshold for RGG code 2')
parser.add_argument('--seed1', dest='seed1', type=int, default=0, help='rng seed for generating RGG code 1')
parser.add_argument('--seed2', dest='seed2', type=int, default=0, help='rng seed for generating RGG code 2')
args = parser.parse_args()
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
readdir = '/Users/yitan/Google Drive/My Drive/from_cannon/qmemory_simulation/data/rgg_code'
savedir = '/Users/yitan/Google Drive/My Drive/from_cannon/qmemory_simulation/data/rgg_code'
# readpath1 = os.path.join(readdir, f'hclassical_n{n1}_m{m1}_degbit{deg_bit}_degcheck{deg_check}_r{r1}_seed{seed1}.txt')
# readpath2 = os.path.join(readdir, f'hclassical_n{n2}_m{m2}_degbit{deg_bit}_degcheck{deg_check}_r{r2}_seed{seed2}.txt')
readpath1 = os.path.join(readdir, f'hclassical_rescaled_n{n1}_m{m1}_degbit{deg_bit}_degcheck{deg_check}_r{r1}_seed{seed1}.txt')
readpath2 = os.path.join(readdir, f'hclassical_rescaled_n{n2}_m{m2}_degbit{deg_bit}_degcheck{deg_check}_r{r2}_seed{seed2}.txt')
h1 = read_pc(readpath1)
h2 = read_pc(readpath2)
qcode = hgp(h1=h1, h2=h2)
hx, hz = qcode.hx, qcode.hz
# savepath_x = os.path.join(savedir, f'hxhgp_n1_{n1}_m1_{m1}_n2_{n2}_m2_{m2}_n_{hx.shape[1]}_mx_{hx.shape[0]}_degbit{deg_bit}_degcheck{deg_check}_r1_{r1}_r2_{r2}_seed1_{seed1}_seed2_{seed2}.txt')
# savepath_z = os.path.join(savedir, f'hzhgp_n1_{n1}_m1_{m1}_n2_{n2}_m2_{m2}_n_{hx.shape[1]}_mz_{hz.shape[0]}_degbit{deg_bit}_degcheck{deg_check}_r1_{r1}_r2_{r2}_seed1_{seed1}_seed2_{seed2}.txt')
savepath_x = os.path.join(savedir, f'hxhgp_rescaled_n1_{n1}_m1_{m1}_n2_{n2}_m2_{m2}_n_{hx.shape[1]}_mx_{hx.shape[0]}_degbit{deg_bit}_degcheck{deg_check}_r1_{r1}_r2_{r2}_seed1_{seed1}_seed2_{seed2}.txt')
savepath_z = os.path.join(savedir, f'hzhgp_rescaled_n1_{n1}_m1_{m1}_n2_{n2}_m2_{m2}_n_{hx.shape[1]}_mz_{hz.shape[0]}_degbit{deg_bit}_degcheck{deg_check}_r1_{r1}_r2_{r2}_seed1_{seed1}_seed2_{seed2}.txt')
# save dense format (txt)
np.savetxt(savepath_x, hx, fmt='%d')
np.savetxt(savepath_z, hz, fmt='%d')

# save sparse format (npz)
# hx_sparse = csr_matrix(hx)
# hz_sparse = csr_matrix(hz)
# hx_sparse = hx_sparse.astype(bool)
# hz_sparse = hz_sparse.astype(bool)
# savepath_x_sparse = os.path.join(savedir, f'hxhgp_n1_{n1}_m1_{m1}_n2_{n2}_m2_{m2}_n_{hx.shape[1]}_mx_{hx.shape[0]}_degbit{deg_bit}_degcheck{deg_check}_r1_{r1}_r2_{r2}_seed1_{seed1}_seed2_{seed2}.npz')
# savepath_z_sparse = os.path.join(savedir, f'hzhgp_n1_{n1}_m1_{m1}_n2_{n2}_m2_{m2}_n_{hx.shape[1]}_mz_{hz.shape[0]}_degbit{deg_bit}_degcheck{deg_check}_r1_{r1}_r2_{r2}_seed1_{seed1}_seed2_{seed2}.npz')
# save_npz(savepath_x_sparse, hx_sparse)
# save_npz(savepath_z_sparse, hz_sparse)