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

def config_model_with_distance_bound(n, m, deg_bit, deg_check, r, rescale_factor=1, seed=0):
    G=nx.empty_graph(n+m)
    rng = np.random.default_rng(seed)
    if not n*deg_bit==m*deg_check:
        raise nx.NetworkXError(\
              'invalid degree sequences, n*deg_bit!=m*deg_check,%s,%s'\
              %(n*deg_bit,m*deg_check))

    pos = {i: rng.uniform(low=0, high=rescale_factor, size=2) for i in range(n+m)}
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

parser = argparse.ArgumentParser()
parser.add_argument('--size', dest='s', type=int, required=True, help='multiplier of deg_check (deg_bit) to get n (m)')
parser.add_argument('--radius', dest='r', type=float, help='distance threshold for RGG code')
parser.add_argument('--seed', dest='seed', type=int, default=0, help='rng seed for generating RGG code')
args = parser.parse_args()
deg_bit = 4
deg_check = 5
size = args.s
r = args.r
seed = args.seed
n = deg_check*size
m = deg_bit*size
savedir = '/Users/yitan/Google Drive/My Drive/from_cannon/qmemory_simulation/data/rgg_code'

def gen_rgg_code(n, m, deg_bit, deg_check, r, seed):
    G = config_model_with_distance_bound(n, m, deg_bit=deg_bit, deg_check=deg_check, r=r, seed=seed)
    pc = nx.bipartite.biadjacency_matrix(G, row_order=range(n,n+m), column_order=range(n))
    pc = pc.toarray().astype(int)
    savepath = os.path.join(savedir, f'hclassical_n{n}_m{m}_degbit{deg_bit}_degcheck{deg_check}_r{r}_seed{seed}.txt')
    np.savetxt(savepath, pc, fmt='%d')

def gen_rgg_code_rescaled(n, m, deg_bit, deg_check, r, seed):
    G = config_model_with_distance_bound(n, m, deg_bit=deg_bit, deg_check=deg_check, r=r, rescale_factor=np.sqrt(size/10) ,seed=seed)
    pc = nx.bipartite.biadjacency_matrix(G, row_order=range(n,n+m), column_order=range(n))
    pc = pc.toarray().astype(int)
    savepath = os.path.join(savedir, f'hclassical_rescaled_n{n}_m{m}_degbit{deg_bit}_degcheck{deg_check}_r{r}_seed{seed}.txt')
    np.savetxt(savepath, pc, fmt='%d')

def min_dist(n, m, deg_bit, deg_check, r, seed):
    G = config_model_with_distance_bound(n, m, deg_bit=deg_bit, deg_check=deg_check, r=r, seed=seed)
    pos = nx.get_node_attributes(G, 'pos')
    edge_list = list(G.edges())
    dist_list = [np.linalg.norm(pos[u]-pos[v]) for u, v in edge_list]    
    min_dist = min(dist_list)
    savepath = os.path.join(savedir, f'mindist_n{n}_m{m}_degbit{deg_bit}_degcheck{deg_check}_r{r}_seed{seed}.npy')
    np.save(savepath, min_dist)

def min_dist_rescaled(n, m, deg_bit, deg_check, r, seed):
    G = config_model_with_distance_bound(n, m, deg_bit=deg_bit, deg_check=deg_check, r=r, rescale_factor=np.sqrt(size/10), seed=seed)
    pos = nx.get_node_attributes(G, 'pos')
    edge_list = list(G.edges())
    dist_list = [np.linalg.norm(pos[u]-pos[v]) for u, v in edge_list]    
    min_dist = min(dist_list)
    savepath = os.path.join(savedir, f'mindist_rescaled_n{n}_m{m}_degbit{deg_bit}_degcheck{deg_check}_r{r}_seed{seed}.npy')
    np.save(savepath, min_dist)

gen_rgg_code(n, m, deg_bit, deg_check, r, seed)
min_dist(n, m, deg_bit, deg_check, r, seed)
gen_rgg_code_rescaled(n, m, deg_bit, deg_check, r, seed)
min_dist_rescaled(n, m, deg_bit, deg_check, r, seed)