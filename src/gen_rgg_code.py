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
from timeit import default_timer as timer
import json

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

def config_model_noprledge(n, m, deg_bit, deg_check, r, rescale_factor=1, seed=0):
    """Generate LDPC code using configuration model.

    Args:
        n (int): number of bit nodes = size * deg_check
        m (int): number of check nodes = size * deg_bit
        deg_bit (int): degree of bit nodes (McKay: t)
        deg_check (int): degree of check nodes (McKay: s)
        seed (int): random seed
    """
    rng = np.random.default_rng(seed)
    pc = np.zeros((m, n), dtype=int)
    # build lists of degree-repeated vertex numbers
    stubs = [[v]*deg_bit for v in range(0,n)]
    bit_stubs = [x for subseq in stubs for x in subseq]
    stubs = [[v]*deg_check for v in range(n,n+m)]
    check_stubs = [x for subseq in stubs for x in subseq]
    # do a random pairing of stubs
    def draw_pairs(remaining_bit_stubs, remaining_check_stubs):
        pairings = []
        rng.shuffle(remaining_bit_stubs)
        for bit_stub in remaining_bit_stubs:
            # randomly choose a check_stub to connect
            trial = 0
            check_stub = rng.choice(remaining_check_stubs)
            # check this pair is not already in pairings
            while [bit_stub, check_stub] in pairings:
                trial += 1
                check_stub = rng.choice(remaining_check_stubs)
                if trial > 10000:
                    raise ValueError('Error: too many trials')
                    # exit(1)
            pairings.append([bit_stub, check_stub])
            # remove the chosen check_stub from remaining_check_stubs
            original_length = len(remaining_check_stubs)
            remaining_check_stubs.remove(check_stub)
            assert len(remaining_check_stubs) == original_length - 1, 'Error: check_stub not removed or too many check_stubs removed'
        return pairings
            
    remaining_bit_stubs = bit_stubs.copy()
    remaining_check_stubs = check_stubs.copy()
    # print('remaining_bit_stubs: ', remaining_bit_stubs)
    # print('remaining_check_stubs: ', remaining_check_stubs)
    pairing = draw_pairs(remaining_bit_stubs, remaining_check_stubs)
    count = {} # count the number of times a pair appears
    for pair in pairing:
        count[tuple(pair)] = count.get(tuple(pair), 0) + 1
    assert not any([count[pair] > 1 for pair in count]), 'Error: repeated pair'
    
    for pair in pairing:
        pc[pair[1]-n, pair[0]] = 1
    return pc

def config_model_noprledge_with_distance_bound(n, m, deg_bit, deg_check, r, rescale_factor=1, seed=0):
    """Generate LDPC code using configuration model with distance bound.

    Args:
        n (int): number of bit nodes = size * deg_check
        m (int): number of check nodes = size * deg_bit
        deg_bit (int): degree of bit nodes (McKay: t)
        deg_check (int): degree of check nodes (McKay: s)
        seed (int): random seed
    """
    rng = np.random.default_rng(seed)
    pc = np.zeros((m, n), dtype=int)
    pos = {i: rng.uniform(low=0, high=rescale_factor, size=2) for i in range(n+m)}
    stubs = [[v]*deg_bit for v in range(0,n)]
    bit_stubs = [x for subseq in stubs for x in subseq]
    stubs = [[v]*deg_check for v in range(n,n+m)]
    check_stubs = [x for subseq in stubs for x in subseq]
    # do a random pairing of stubs
    def draw_pairs(remaining_bit_stubs, remaining_check_stubs):
        pairings = []
        rng.shuffle(remaining_bit_stubs)
        for bit_stub in remaining_bit_stubs:
            # randomly choose a check_stub to connect
            trial = 0
            check_stub = rng.choice(remaining_check_stubs)
            # check this pair is not already in pairings
            while [bit_stub, check_stub] in pairings:
                trial += 1
                check_stub = rng.choice(remaining_check_stubs)
                if trial > 10000:
                    raise ValueError('Error: too many trials')
                    # exit(1)
            pairings.append([bit_stub, check_stub])
            # remove the chosen check_stub from remaining_check_stubs
            original_length = len(remaining_check_stubs)
            remaining_check_stubs.remove(check_stub)
            assert len(remaining_check_stubs) == original_length - 1, 'Error: check_stub not removed or too many check_stubs removed'
        return pairings
            
    remaining_bit_stubs = bit_stubs.copy()
    remaining_check_stubs = check_stubs.copy()
    # print('remaining_bit_stubs: ', remaining_bit_stubs)
    # print('remaining_check_stubs: ', remaining_check_stubs)
    pairing = draw_pairs(remaining_bit_stubs, remaining_check_stubs)
    count = {} # count the number of times a pair appears
    for pair in pairing:
        count[tuple(pair)] = count.get(tuple(pair), 0) + 1
    assert not any([count[pair] > 1 for pair in count]), 'Error: repeated pair'
    
    for pair in pairing:
        if np.linalg.norm(pos[pair[0]]-pos[pair[1]]) < r:
            pc[pair[1]-n, pair[0]] = 1
    
    return pc, pos

def balanced_ordering(n, m, deg_bit, deg_check, r, seed):
    pass

def preferential_attachment_with_distance_bound():
    pass


parser = argparse.ArgumentParser()
parser.add_argument('--size', dest='s', type=int, required=True, help='multiplier of deg_check (deg_bit) to get n (m)')
parser.add_argument('--radius', dest='r', type=float, help='distance threshold for RGG code')
parser.add_argument('--seed', dest='seed', type=int, default=0, help='rng seed for generating RGG code')
parser.add_argument('--savedir', dest='savedir', type=str, default='/Users/yitan/Google Drive/My Drive/from_cannon/qmemory_simulation/data/rgg_code', help='directory to save RGG code')
args = parser.parse_args()
deg_bit = 8
deg_check = 10
size = args.s
r = args.r
seed = args.seed
n = deg_check*size
m = deg_bit*size
savedir = args.savedir

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


# gen_rgg_code(n, m, deg_bit, deg_check, r, seed)
# min_dist(n, m, deg_bit, deg_check, r, seed)
# gen_rgg_code_rescaled(n, m, deg_bit, deg_check, r, seed)
# min_dist_rescaled(n, m, deg_bit, deg_check, r, seed)
pc, pos = config_model_noprledge_with_distance_bound(n, m, deg_bit, deg_check, r=r, rescale_factor=np.sqrt(size/10), seed=seed)
savename_pc = f'hclassical_noprledgelocal_n={n}_m={m}_degbit={deg_bit}_degcheck={deg_check}_r={r}_seed={seed}.txt'
savename_pos = f'pos_noprledgelocal_n={n}_m={m}_degbit={deg_bit}_degcheck={deg_check}_r={r}_seed={seed}.json'
savepath_pc = os.path.join(savedir, savename_pc)
savepath_pos = os.path.join(savedir, savename_pos)
pos = {k: v.tolist() for k, v in pos.items()}
np.savetxt(savepath_pc, pc, fmt='%d')
with open(savepath_pos, 'w') as f:
    json.dump(pos, f)