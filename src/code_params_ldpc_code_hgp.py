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

parser = argparse.ArgumentParser()
parser.add_argument('--size1', dest='s1', type=int, required=True, help='size multiplier of RGG code 1')
parser.add_argument('--size2', dest='s2', type=int, required=True, help='size multiplier of RGG code 2')
parser.add_argument('--radius1', dest='r1', type=float, help='distance threshold for RGG code 1')
parser.add_argument('--radius2', dest='r2', type=float, help='distance threshold for RGG code 2')
parser.add_argument('--seed1', dest='seed1', type=int, default=0, help='rng seed for generating RGG code 1')
parser.add_argument('--seed2', dest='seed2', type=int, default=0, help='rng seed for generating RGG code 2')
parser.add_argument('--readdir', dest='readdir', type=str, default='/Users/yitan/Google Drive/My Drive/from_cannon/qmemory_simulation/data/rgg_code')
parser.add_argument('--savedir', dest='savedir', type=str, default='/Users/yitan/Google Drive/My Drive/from_cannon/qmemory_simulation/data/rgg_code')
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
readdir = args.readdir
savedir = args.savedir
readpath_h1 = os.path.join(readdir, f'codedistance_hclassical_rescaled_n{n1}_m{m1}_degbit{deg_bit}_degcheck{deg_check}_r{r1}_seed{seed1}.npy')
readpath_h1T = os.path.join(readdir, f'codedistance_transpose_hclassical_rescaled_n{n1}_m{m1}_degbit{deg_bit}_degcheck{deg_check}_r{r1}_seed{seed1}.npy')
readpath_h2 = os.path.join(readdir, f'codedistance_hclassical_rescaled_n{n1}_m{m1}_degbit{deg_bit}_degcheck{deg_check}_r{r1}_seed{seed1}.npy')
readpath_h2T = os.path.join(readdir, f'codedistance_transpose_hclassical_rescaled_n{n1}_m{m1}_degbit{deg_bit}_degcheck{deg_check}_r{r1}_seed{seed1}.npy')
d1 = np.load(readpath_h1, allow_pickle=True)
d1T = np.load(readpath_h1T, allow_pickle=True)
d2 = np.load(readpath_h2, allow_pickle=True)
d2T = np.load(readpath_h2T, allow_pickle=True)

print(f'd1: {d1}')
print(f'd1T: {d1T}')
print(f'd2: {d2}')
print(f'd2T: {d2T}')

d = np.min([d1, d1T, d2, d2T])
savepath = os.path.join(savedir, f'codedistance_hgp_rescaled_n1={n1}_m1={m1}_n2={n2}_m2={m2}_degbit={deg_bit}_degcheck={deg_check}_r1={r1}_r2={r2}_seed1={seed1}_seed2={seed2}.npy')
np.save(savepath, d)


