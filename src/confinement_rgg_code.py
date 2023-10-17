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

def test_biregular(h):
    if not np.all(h == 0):
        for row in h:
            assert np.sum(row) == deg_check, f'row weight: {np.sum(row)}'
        for col in h.T:
            assert np.sum(col) == deg_bit, f'col weight: {np.sum(col)}'

def syndrome_weight_batch(h, error_weight, nsamples=1000):
    if np.all(h == 0):
        return np.nan
    n, m = h.shape
    assert error_weight <= n
    # Generate error bit strings
    errors = np.zeros((nsamples, n), dtype=int)
    

parser = argparse.ArgumentParser()
parser.add_argument('--size', dest='s', type=int, required=True, help='multiplier of deg_check (deg_bit) to get n (m)')
parser.add_argument('--radius', dest='r', type=float, default=1.0, help='distance threshold for RGG code')
parser.add_argument('--seed', dest='seed', type=int, default=0, help='rng seed for generating RGG code')
parser.add_argument('--error_weight', dest='ewt', type=int, default=0, help='error weight for bitstring samples')
parser.add_argument('--readdir', dest='readdir', type=str, default='/Users/yitan/Google Drive/My Drive/from_cannon/qmemory_simulation/data/rgg_code')
parser.add_argument('--savedir', dest='savedir', type=str, default='/Users/yitan/Google Drive/My Drive/from_cannon/qmemory_simulation/data/rgg_code')
args = parser.parse_args()
deg_bit = 4
deg_check = 5
size = args.s
r = args.r
seed = args.seed
n = deg_check*size
m = deg_bit*size
error_weight = args.ewt
readdir = args.readdir
savedir = args.savedir
readname = f'hclassical_nolocality_n={n}_m={m}_degbit={deg_bit}_degcheck={deg_check}_seed={seed}.txt'
savename = f'syndrome_weight_hclassical_nolocality_n={n}_m={m}_degbit={deg_bit}_degcheck={deg_check}_seed={seed}.npy'
readpath = os.path.join(readdir, readname)
h = read_pc(readpath)

# syndrome_weight(h, error_weight)