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

def config_model_with_distance_bound_my(n, m, deg_bit, deg_check, r, rescale_factor=1, seed=0):
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

def config_model_noprledge_my(n, m, deg_bit, deg_check, seed=0):
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

def config_model_noprledge_with_distance_bound_my(n, m, deg_bit, deg_check, r, rescale_factor=1, seed=0):
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

def config_model_noprledge_with_distand_bound_pruning_my(n, m, deg_bit, deg_check, r, rescale_factor=1, seed=0):
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

############################################################################################################
def config_model_nonlocal(n, m, deg_bit, deg_check, noprledge=False, seed=0):
    rng = np.random.default_rng(seed)
    deg_check_sequence = [deg_check]*m
    deg_bit_sequence = [deg_bit]*n
    G = nx.bipartite.configuration_model(deg_check_sequence, deg_bit_sequence, seed=seed)
    if noprledge:
        G = nx.Graph(G)
    pos = {i: rng.uniform(low=0, high=1, size=2) for i in range(m+n)}
    pc = nx.bipartite.biadjacency_matrix(G, row_order=range(m), column_order=range(m, m+n)).toarray()
    return pc, pos

def config_model_local(n, m, deg_bit, deg_check, r, rescaled=True, noprledge=False, prune=False, seed=0):
    rng = np.random.default_rng(seed) 
    deg_check_sequence = [deg_check]*m
    deg_bit_sequence = [deg_bit]*n
    G = nx.bipartite.configuration_model(deg_check_sequence, deg_bit_sequence, seed=seed)
    if noprledge:
        G = nx.Graph(G)
    pos = {i: rng.uniform(low=0, high=1, size=2) for i in range(m+n)}
    if rescaled:
        # the rescale factor is sqrt(size_factor/10), where size_factor=n/deg_check, so that the base graph size is (n=10*deg_check, m=10*deg_bit)
        size_factor = n/deg_check
        pos = {i: np.sqrt(size_factor/10)*pos[i] for i in range(m+n)}
    for i in range(m):
        for j in range(n):
            if np.linalg.norm(pos[i]-pos[j+m]) > r:
                G.remove_edge(i, j+m)
    if prune:
        # remove isolated nodes
        G.remove_nodes_from(list(nx.isolates(G)))
    # after pruning, the node indices are not consecutive
    # reindex the nodes
    mapping = dict(zip(G.nodes(), range(m+n))) 
    G = nx.relabel_nodes(G, mapping)
    pos = {mapping[k]: v for k, v in pos.items()}
    pc = nx.bipartite.biadjacency_matrix(G, row_order=range(m), column_order=range(m, m+n)).toarray()
    return pc, pos


parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest='mode')

# Nonlocal configuration model G(n, deg_check_sequence, deg_bit_sequence)
nonlocal_parser = subparsers.add_parser('nonlocal')
nonlocal_parser.add_argument('--noprledge', dest='noprledge', action='store_true', default=False)

# Local configuration model G(n, deg_check_sequence, deg_bit_sequence) with distance bound
local_parser = subparsers.add_parser('local')
local_parser.add_argument('--radius', dest='r', type=float, help='distance threshold for RGG code')
local_parser.add_argument('--noprledge', dest='noprledge', action='store_true', default=False)
local_parser.add_argument('--rescaled', dest='rescaled', action='store_true', default=True)

# general arguments
parser.add_argument('--size', dest='s', type=int, required=True, help='multiplier of deg_check (deg_bit) to get n (m)')
parser.add_argument('--deg_bit', dest='deg_bit', type=int, default=4, help='degree of bit nodes')
parser.add_argument('--deg_check', dest='deg_check', type=int, default=5, help='degree of check nodes')
parser.add_argument('--seed', dest='seed', type=int, default=0, help='random seed')
parser.add_argument('--savedir', dest='savedir', type=str, default='/Users/yitan/Google Drive/My Drive/from_cannon/qmemory_simulation/data/laplacian_code')


args = parser.parse_args()
savedir = args.savedir
if not os.path.exists(savedir):
    os.makedirs(savedir)

if args.mode == 'nonlocal':
    'Bipartite configuration model G(n, deg_sequence)'
    n = args.s*args.deg_check
    m = args.s*args.deg_bit
    deg_bit = args.deg_bit
    deg_check = args.deg_check
    noprledge = args.noprledge
    seed = args.seed
    savename = f'hclassical_config_model_nonlocal_n={n}_m={m}_deg_bit={deg_bit}_deg_check={deg_check}_noprledge={noprledge}_seed={seed}.txt'
    savepath = os.path.join(savedir, savename)
    pc, pos = config_model_nonlocal(n, m, deg_bit, deg_check, noprledge=noprledge, seed=seed)
    pc = pc.astype(int)
    np.savetxt(savepath, pc, fmt='%d')

elif args.mode == 'local':
    'Bipartite configuration model G(n, deg_sequence) with distance bound'
    n = args.s*args.deg_check
    m = args.s*args.deg_bit
    deg_bit = args.deg_bit
    deg_check = args.deg_check
    r = args.r
    rescaled = args.rescaled
    noprledge = args.noprledge
    seed = args.seed
    savename = f'hclassical_config_model_local_n={n}_m={m}_deg_bit={deg_bit}_deg_check={deg_check}_r={r}_noprledge={noprledge}_rescaled={rescaled}_seed={seed}.txt'
    savepath = os.path.join(savedir, savename)
    pc, pos = config_model_local(n, m, deg_bit, deg_check, r, rescaled=rescaled, noprledge=noprledge, seed=seed)
    pc = pc.astype(int)
    np.savetxt(savepath, pc, fmt='%d')
    posname = savename.replace('.txt', '_pos.json')
    pospath = os.path.join(savedir, posname)
    with open(pospath, 'w') as f:
        json.dump(pos, f)
