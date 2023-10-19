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

def configuration_model_my(n, deglo, degup, seed=0):
    '''My modified implementation of configuration model, trying to avoid parallel edges.
    It turns out to be too slow.
    '''
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
    
def configuration_model(n, deglo, degup, noprledge=False, noselfloop=False, seed=0):
    rng = np.random.default_rng(seed)
    while True:
        deg_sequence = rng.integers(deglo, degup+1, n)
        if deg_sequence.sum() % 2 == 0:  # sum of degrees must be even
            break

    G = nx.configuration_model(deg_sequence, seed=seed)
    if noprledge:
        G = nx.Graph(G)
    if noselfloop:
        G.remove_edges_from(nx.selfloop_edges(G))
    pc = nx.laplacian_matrix(G).toarray()
    deg_sequence_actual = np.diag(pc)
    h = np.mod(pc, 2)
    return h, deg_sequence_actual

def erdos_renyi(n, p, seed=0):
    rng = np.random.default_rng(seed)
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    laplacian = nx.laplacian_matrix(G).toarray()
    deg_sequence_actual = np.diag(laplacian)
    h = np.mod(laplacian, 2)
    return h, deg_sequence_actual

def random_gnm(n, m, seed=0):
    rng = np.random.default_rng(seed)
    G = nx.gnm_random_graph(n, m, seed=seed)
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

def random_geometric(n, r, seed=0):
    rng = np.random.default_rng(seed)
    G = nx.random_geometric_graph(n, r, seed=seed)
    laplacian = nx.laplacian_matrix(G).toarray()
    deg_sequence_actual = np.diag(laplacian)
    h = np.mod(laplacian, 2)
    return h, deg_sequence_actual

def shs_squarelattice_basegraph(L):
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


parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest='mode')

# Configuration model G(n, deg_sequence)
config_parser = subparsers.add_parser('config')
config_parser.add_argument('--n', dest='n', type=int, default=20) # number of nodes
config_parser.add_argument('--deglo', dest='deglo', type=int, default=3, help='lower bound of degree of each node')
config_parser.add_argument('--degup', dest='degup', type=int, default=5, help='upper bound of degree of each node')
config_parser.add_argument('--noprledge', dest='noprledge', action='store_true', default=False, help='prune parallel edges in configuration model when set to True')
config_parser.add_argument('--noselfloop', dest='noselfloop', action='store_true', default=False, help='prune self-loops in configuration model when set to True')
config_parser.add_argument('--seed', dest='seed', type=int, default=0, help='random seed')

# Erdos-Renyi model G(n,p)
er_parser = subparsers.add_parser('er')
er_parser.add_argument('--n', dest='n', type=int, default=20) # number of nodes
er_parser.add_argument('--p', dest='p', type=float, default=0.5)
er_parser.add_argument('--seed', dest='seed', type=int, default=0, help='random seed')

# Gnm random graph G(n,m) with given number of nodes n and edges m
gnm_parser = subparsers.add_parser('gnm')
gnm_parser.add_argument('--n', dest='n', type=int, default=20) # number of nodes
gnm_parser.add_argument('--m', dest='m', type=int, default=10, help='number of edges')
gnm_parser.add_argument('--seed', dest='seed', type=int, default=0, help='random seed')

# Random regular graph G(n,d)
rrg_parser = subparsers.add_parser('rrg')
rrg_parser.add_argument('--n', dest='n', type=int, default=20) # number of nodes
rrg_parser.add_argument('--d', dest='d', type=int, default=6)
rrg_parser.add_argument('--seed', dest='seed', type=int, default=0, help='random seed')

# Local random geometric graph model G(n,r) in 2D
lrg_parser = subparsers.add_parser('lrgg')
lrg_parser.add_argument('--n', dest='n', type=int, default=20) # number of nodes
lrg_parser.add_argument('--r', dest='r', type=float, default=0.5)
lrg_parser.add_argument('--seed', dest='seed', type=int, default=0, help='random seed')

# SHS square lattice base graph in 2D
shs_parser = subparsers.add_parser('shs')
shs_parser.add_argument('--L', dest='L', type=int, default=20) # linear system size in SHS's example of using sqaure lattice base graph
shs_parser.add_argument('--seed', dest='seed', type=int, default=0, help='random seed')

# general arguments
parser.add_argument('--savedir', dest='savedir', type=str, default='/Users/yitan/Google Drive/My Drive/from_cannon/qmemory_simulation/data/laplacian_code')


args = parser.parse_args()
savedir = args.savedir
if not os.path.exists(savedir):
    os.makedirs(savedir)

if args.mode == 'config':
    'Configuration model G(n, deg_sequence)'
    n = args.n
    deglo = args.deglo
    degup = args.degup
    noprledge = args.noprledge
    noselfloop = args.noselfloop
    seed = args.seed
    savename = f'hclassical_configurationmodel_n={n}_deglo={deglo}_degup={degup}_noprledge={noprledge}_noselfloop={noselfloop}_seed={seed}.txt'
    pc, deg_sequence_actual = configuration_model(n, deglo, degup, noprledge=noprledge, noselfloop=noselfloop, seed=seed)
elif args.mode == 'er':
    'Erdos-Renyi model G(n,p)'
    n = args.n
    p = args.p
    seed = args.seed
    savename = f'hclassical_erdosrenyi_n={n}_p={p}_seed={seed}.txt'
    pc, deg_sequence_actual = erdos_renyi(n, p, seed=seed)
elif args.mode == 'gnm':
    'Gnm random graph G(n,m) with given number of nodes n and edges m'
    n = args.n
    m = args.m
    seed = args.seed
    savename = f'hclassical_gnm_n={n}_m={m}_seed={seed}.txt'
    pc, deg_sequence_actual = random_gnm(n, m, seed=seed)
elif args.mode == 'rrg':
    'Random regular graph G(n,d)'
    n = args.n
    d = args.d
    seed = args.seed
    savename = f'hclassical_randomregular_n={n}_d={d}_seed={seed}.txt'
    pc, deg_sequence_actual = random_regular(n, d, seed=seed)
elif args.mode == 'lrgg':
    'Local random geometric graph model G(n,r) in 2D'
    n = args.n
    r = args.r
    seed = args.seed
    savename = f'hclassical_localrandomgeometric_n={n}_r={r}_seed={seed}.txt'
    pc, deg_sequence_actual = random_geometric(n, r, seed=seed)
elif args.mode == 'shs':
    'SHS square lattice base graph in 2D'
    L = args.L
    savename = f'hclassical_shs_squarelattice_L={L}.txt'
    pc, deg_sequence_actual = shs_squarelattice_basegraph(L)


print(deg_sequence_actual)
savepath = os.path.join(savedir, savename)
np.savetxt(savepath, pc, fmt='%d')

