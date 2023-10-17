import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import chain, combinations, zip_longest
from operator import itemgetter
from ldpc.mod2 import *
from ldpc.code_util import *
from bposd.hgp import hgp
from bposd.css import *
import os
from sys import argv
import argparse
from timeit import default_timer as timer
import json
from itertools import chain

parser = argparse.ArgumentParser()
parser.add_argument('--n', dest='n', type=int, default=20) # number of nodes
parser.add_argument('--deglo', dest='deglo', type=int, default=3, help='lower bound of degree of each node')
parser.add_argument('--degup', dest='degup', type=int, default=5, help='upper bound of degree of each node')
parser.add_argument('--p', dest='p', type=float, default=0.5)
parser.add_argument('--d', dest='d', type=int, default=6)
parser.add_argument('--L', dest='L', type=int, default=20) # linear system size in SHS's example of using sqaure lattice base graph
parser.add_argument('--seed', dest='seed', type=int, default=0, help='random seed')
parser.add_argument('--savedir', dest='savedir', type=str, default='/Users/yitan/Google Drive/My Drive/from_cannon/qmemory_simulation/data/laplacian_code')
args = parser.parse_args()
n = args.n
deglo = args.deglo
degup = args.degup
p = args.p
d = args.d
L = args.L
seed = args.seed
savedir = args.savedir
if not os.path.exists(savedir):
    os.makedirs(savedir)

def configuration_model_noprledge_my(n, deglo, degup, seed=0):
    rng = np.random.default_rng(seed)
    deg_sequence = rng.integers(deglo, degup+1, n)
    if deg_sequence.sum() % 2 != 0:  # sum of degrees must be even
        selector = np.where(deg_sequence < degup)[0]
        idx = rng.choice(selector)
        deg_sequence[idx] += 1
    
    pc = np.zeros((n, n), dtype=int)
    stubs = list(chain.from_iterable([n] * d for d, n in enumerate(deg_sequence)))
    remain = stubs.copy()
    pairings = []
    while len(remain):
        # pull out 2 nodes
        trial = 0
        node1, node2 = rng.choice(remain, 2, replace=False)
        while [node1, node2] in pairings or [node2, node1] in pairings:
            trial += 1
            node1, node2 = rng.choice(remain, 2, replace=False)
            # if trial > 10000:
            #     print('Failed to find a pair not in exisiting pairings. Exiting...')
            #     break
        pairings.append([node1, node2])
        remain.remove(node1)
        remain.remove(node2)
    count = {} # count the number of times a pair appears
    for pair in pairings:
        count[tuple(pair)] = count.get(tuple(pair), 0) + 1
    assert not any([count[pair] > 1 for pair in count]), 'Error: repeated pair'
    
    for pair in pairings:
        pc[pair[0], pair[1]] = 1
        pc[pair[1], pair[0]] = 1
    return pc

def configuration_model_noprledge(n, deglo, degup, seed=0):
    rng = np.random.default_rng(seed)
    deg_sequence = rng.integers(deglo, degup+1, n)
    if deg_sequence.sum() % 2 != 0:  # sum of degrees must be even
        selector = np.where(deg_sequence < degup)[0]
        idx = rng.choice(selector)
        deg_sequence[idx] += 1
    
    G = nx.configuration_model(deg_sequence)
    # G = nx.Graph(G) # remove parallel edges
    # G.remove_edges_from(nx.selfloop_edges(G)) # remove self-loops
    laplacian = nx.laplacian_matrix(G).toarray()
    deg_sequence_actual = np.diag(laplacian)
    h = np.mod(laplacian, 2)
    return h, deg_sequence_actual

def erdos_renyi(n, p, seed=0):
    rng = np.random.default_rng(seed)
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    laplacian = nx.laplacian_matrix(G).toarray()
    deg_sequence_actual = np.diag(laplacian)
    h = np.mod(laplacian, 2)
    return h, deg_sequence_actual

def random_regular(n, d, seed=0):
    rng = np.random.default_rng(seed)
    G = nx.random_regular_graph(d, n, seed=seed)
    laplacian = nx.laplacian_matrix(G).toarray()
    deg_sequence_actual = np.diag(laplacian)
    h = np.mod(laplacian, 2)
    return h, deg_sequence_actual

def shs_squarelattic_basegraph(L):
    pc = np.zeros((L, L, L, L), dtype=int) # adjacency matrix
    for i in range(L):
        for j in range(L):
            pc[i, j, (i+1)%L, j] = 1
            pc[i, j, (i-1)%L, j] = 1
            pc[i, j, i, (j+1)%L] = 1
            pc[i, j, i, (j-1)%L] = 1
            pc[(i+1)%L, j, i, j] = 1
            pc[(i-1)%L, j, i, j] = 1
            pc[i, (j+1)%L, i, j] = 1
            pc[i, (j-1)%L, i, j] = 1
    pc = pc.reshape(L**2, L**2)
    deg_sequence_actual = np.ones(L**2, dtype=int) * 4
    return pc, deg_sequence_actual

# pc, deg_sequence_actual = configuration_model_noprledge(n, deglo, degup, seed)
# pc, deg_sequence_actual = erdos_renyi(n, p, seed)
# pc, deg_sequence_actual = random_regular(n, d, seed)
pc, deg_sequence_actual = shs_squarelattic_basegraph(L)
print(deg_sequence_actual)
# savename = f'hclassical_configurationmodel_n={n}_deglo={deglo}_degcheck={degup}_seed={seed}.txt'
# savename = f'hclassical_erdosrenyi_n={n}_p={p}_seed={seed}.txt'
# savename = f'hclassical_randomregular_n={n}_d={d}_seed={seed}.txt'
savename = f'hclassical_shs_squarelattice_L={L}.txt'
savepath = os.path.join(savedir, savename)
np.savetxt(savepath, pc, fmt='%d')


