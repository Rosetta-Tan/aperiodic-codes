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
from sys import argv
import argparse


def gen_unipartite_rgg_code(n, d, p, seed=None):
    if seed is not None:
        np.random.seed(seed)
    pos = {i: np.random.uniform(size=d) for i in range(n)}
    G = nx.Graph()
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(pos[i]-pos[j]) < p:
                G.add_edge(i, j)
    return G, pos

def gen_bipartite_rgg_code(m, n, d, p, seed=None):
    if seed is not None:
        np.random.seed(seed)
    pos = {i: np.random.uniform(size=d) for i in range(n+m)}
    G = nx.Graph()
    for i in range(n):
        for j in range(n, n+m):
            if np.linalg.norm(pos[i]-pos[j]) < p:
                G.add_edge(i, j)
    return G, pos

def config_model_with_distance_bound(n, m, deg_bit, deg_check, r, seed=0):
    G=nx.empty_graph(n+m)
    rng = np.random.default_rng(seed)
    
    if not n*deg_bit==m*deg_check:
        raise nx.NetworkXError(\
              'invalid degree sequences, n*deg_bit!=m*deg_check,%s,%s'\
              %(n*deg_bit,m*deg_check))

    pos = {i: rng.uniform(size=2) for i in range(n+m)}
    G.add_nodes_from(range(0,n+m))
    'add bipartite attribute to nodes'
    b=dict(zip(range(0,n),[0]*n))
    shapes=dict(zip(range(0,n),['o']*n)) # circle
    b.update(dict(zip(range(n,n+m),[1]*m)))
    shapes.update(dict(zip(range(n,n+m),['s']*m))) # square
    nx.set_node_attributes(G,b,'bipartite') 
    'add position attribute to nodes'
    nx.set_node_attributes(G,pos,'pos')
    # print(G.nodes(data=True))

    # build lists of degree-repeated vertex numbers
    stubs = [[v]*deg_bit for v in range(0,n)]
    astubs = [x for subseq in stubs for x in subseq]
    # print('astubs: ', astubs)
    stubs = [[v]*deg_check for v in range(n,n+m)]
    bstubs = [x for subseq in stubs for x in subseq]
    # print('bstubs: ', bstubs)
    # shuffle lists
    rng.shuffle(astubs)
    rng.shuffle(bstubs)
    # print('astubs: ', astubs)
    # print('bstubs: ', bstubs)
    edge_list = list(zip(astubs,bstubs))
    for u,v in edge_list:
        if np.linalg.norm(pos[u]-pos[v]) < r:
            G.add_edge(u,v)
    return G

def plot_unipartite_graph(G, pos):
    nx.draw(G, pos, with_labels=True)
    plt.show()

def plot_bipartite_graph(G, n, m, pos):
    print('pos: ', pos)
    pos1 = {i: pos[i] for i in range(n)}
    print('pos1: ', pos1)
    pos2 = {i: pos[i] for i in range(n, n+m)}
    print('pos2: ', pos2)
    # nx.draw(G, pos1, with_labels=True)
    # nx.draw(G, pos2, with_labels=True)


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('s', type=int, nargs='?', help='multiplier of deg_check (deg_bit) to get n (m)')
# parser.add_argument('--seed', type=int, default=0, help='rng seed for generating RGG code')
args = parser.parse_args()
deg_bit = 4
deg_check = 5
s = args.s
n = 5*s
m = 4*s
r = 0.2
# rs = [0.1, 0.2, 0.3, 0.4]
seed = 42

#################################################################################################
# Generate classical RGG code
#################################################################################################
G = config_model_with_distance_bound(n, m, deg_bit=deg_bit, deg_check=deg_check, r=r, seed=seed)
pc = nx.bipartite.biadjacency_matrix(G, row_order=range(n,n+m), column_order=range(n))
pc = pc.toarray().astype(int)
np.savetxt(f'../data/rgg_code/hclassical_n{n}_m{m}_degbit{deg_bit}_degcheck{deg_check}_r{r}_seed{seed}.txt', pc, fmt='%d')

#################################################################################################
# Plot classical RGG code
#################################################################################################
# fig, ax = plt.subplots(2, 2)
# Gs = [config_model_with_distance_bound(n, m, deg_bit=deg_bit, deg_check=deg_check, r=rs[i]) for i in range(4)]
# for i in range(4):
#     pc = nx.bipartite.biadjacency_matrix(Gs[i], row_order=range(n,n+m), column_order=range(n))
#     pc = pc.toarray().astype(int)
#     np.savetxt(f'../data/rgg_code/hclassical_n{n}_m{m}_degbit{deg_bit}_degcheck{deg_check}_r{rs[i]}_seed{seed}.txt', pc, fmt='%d')
#     row = i // 2
#     col = i % 2
#     G = Gs[i]
#     pos = nx.get_node_attributes(G, 'pos')
#     pos_nodes = [pos[i] for i in range(n)]
#     pos_checks = [pos[i] for i in range(n, n+m)]
#     for edge in G.edges():
#         u, v = edge
#         ax[row, col].plot(*zip(pos[u], pos[v]), color='k', linestyle='-', linewidth=0.5)
#     ax[row, col].scatter(*zip(*pos_nodes), color='r', marker='o')
#     ax[row, col].scatter(*zip(*pos_checks), color='b', marker='s')
# plt.show()

#################################################################################################
# Generate HGP of two 2D RGG codes
#################################################################################################
# G1, pos1 = config_model_with_distance_bound(n, m, deg_bit=deg_bit, deg_check=deg_check, r=r, seed=seed1)
# G2, pos2 = config_model_with_distance_bound(n, m, deg_bit=deg_bit, deg_check=deg_check, r=r, seed=seed2)
# pc1 = pc1.toarray().astype(int)
# pc2 = pc2.toarray().astype(int)

