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
from scipy.spatial import KDTree
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

def get_classical_code_distance_time_limit(h):
    if rank(h) == h.shape[1]:
        print('Code is full rank, no codewords')
        return np.inf
    else:
        start = timer()
        ker = nullspace(h)
        end = timer()
        def find_min_weight_while_build(matrix):
            span = []
            min_hamming_weight = np.inf
            for ir, row in enumerate(matrix):
                row_hamming_weight = np.sum(row)
                if row_hamming_weight < min_hamming_weight:
                    min_hamming_weight = row_hamming_weight
                temp = [row]
                for element in span:
                    newvec = (row + element) % 2
                    temp.append(newvec)
                    newvec_hamming_weight = np.sum(newvec)
                    if newvec_hamming_weight < min_hamming_weight:
                        min_hamming_weight = newvec_hamming_weight
                    end = timer()
                    if end - start > 5:
                        return min_hamming_weight
                span = list(np.unique(temp + span, axis=0))
            assert len(span) == 2**len(matrix) - 1
            return min_hamming_weight
        min_hamming_weight = find_min_weight_while_build(ker)
        end = timer()
        return min_hamming_weight

def select_points_within_radius(points, r):
    """
    Select k points within radius r of each point in points.
    """
    tree = KDTree(points)
    indices = tree.query_ball_point(points, r, p=2)   
    return indices

def get_connectivity_one_nodes(connected_component, connected):
    """
    Given a list of nodes, return a list of nodes that are connected to only one node'
    """
    connectivity_one_nodes = []
    for i in connected_component:
        connected_nodes_in_component = []
        for j in connected_component:
            if j in connected[i]:
                connected_nodes_in_component.append(j)
        if len(connected_nodes_in_component) == 1:
            connectivity_one_nodes.append(i)
    return connectivity_one_nodes

def get_connectivity_two_nodes(connected_component, connected):
    """
    Given a list of nodes, return a list of nodes that are connected to only one node'
    """
    connectivity_two_nodes = []
    for i in connected_component:
        connected_nodes_in_component = []
        for j in connected_component:
            if j in connected[i]:
                connected_nodes_in_component.append(j)
        if len(connected_nodes_in_component) == 2:
            connectivity_two_nodes.append(i)
    return connectivity_two_nodes

def rgg(n, m, const=8, r=False, seed=0):
    '''Keep pi * r^2 * n = const'''
    if r == False:
        r = np.sqrt(const/(np.pi*(m+n)))

    rng = np.random.default_rng(seed=seed)
    points = rng.uniform(size=(m+n, 2))
    ctgs = np.array([0]*m + [1]*n)  # 0 for check node, 1 for bit node
    connected = [set() for _ in range(m+n)]  # connected[i] is the set of nodes connected to node i
    indices = select_points_within_radius(points, r)
    
    # create rgg connectivity
    for i in range(m+n):
        if ctgs[i] == 0:  # check node
            for pt in indices[i]:
                if ctgs[pt] == 1:
                    connected[i].add(pt)
        else:  # bit node
            for pt in indices[i]:
                if ctgs[pt] == 0:
                    connected[i].add(pt)
    
    adj_mat = np.zeros((m+n, m+n))
    for i in range(m+n):
        if len(connected[i]) > 0:
            for j in connected[i]:
                adj_mat[i,j] = 1
    assert np.all(adj_mat == adj_mat.T), 'the adjacency matrix is not symmetric'
    G = nx.from_numpy_array(adj_mat)
    connected_components = list(nx.connected_components(G))    
    G_largest = G.subgraph(connected_components[0])
    checks_eff = [node for node in G_largest.nodes if ctgs[node] == 0]
    bits_eff = [node for node in G_largest.nodes if ctgs[node] == 1]
    if len(checks_eff) == 0:
        raise ValueError('The largest connected component does not contain any check node')
    if len(bits_eff) == 0:
        raise ValueError('The largest connected component does not contain any bit node')
    pc = nx.bipartite.biadjacency_matrix(G, row_order=checks_eff, column_order=bits_eff).toarray()

    removed_checks = []
    removed_bits = []
    for comp in connected_components[1:]:
        comp = list(comp)
        for item in comp:
            if ctgs[item] == 0:
                removed_checks.append(item)
            else:
                removed_bits.append(item)

    connectivity_one_nodes = get_connectivity_one_nodes(connected_components[0], connected)
    connectivity_two_nodes = get_connectivity_two_nodes(connected_components[0], connected)
    connectivity_one_checks = [node for node in connectivity_one_nodes if ctgs[node] == 0]
    connectivity_one_bits = [node for node in connectivity_one_nodes if ctgs[node] == 1]
    connectivity_two_checks = [node for node in connectivity_two_nodes if ctgs[node] == 0]
    connectivity_two_bits = [node for node in connectivity_two_nodes if ctgs[node] == 1]
    
    return pc, connected_components, np.asarray(points), np.asarray(removed_checks), np.asarray(removed_bits), np.asarray(connectivity_one_checks), np.asarray(connectivity_one_bits), np.asarray(connectivity_two_checks), np.asarray(connectivity_two_bits)

parser = argparse.ArgumentParser()
parser.add_argument('--n', dest='n', type=int, default=100)
parser.add_argument('--m', dest='m', type=int, default=80)
parser.add_argument('--proportionality', dest='proportionality', type=float, default=0.1)
parser.add_argument('--const', dest='const', type=int, default=0)
parser.add_argument('--seed', dest='seed', type=int, default=0)
parser.add_argument('--savedir', dest='savedir', type=str, default='/Users/yitan/Google Drive/My Drive/from_cannon/qmemory_simulation/data/ldpc_code')
args = parser.parse_args()

n = args.n
m = args.m
proportionality = args.proportionality
const = args.const
savedir = args.savedir
seed = args.seed

# n, m = np.array([100, 100]) * 5
# n, m = int(n), int(m)
# proportionality = 0.1
# const = 8
# seed = 5

pc, connected_components, points, removed_checks, removed_bits, connectivity_one_checks, connectivity_one_bits, connectivity_two_checks, connectivity_two_bits = rgg(n, m, const=const, r=False, seed=seed)
savename = f'pc_rgg_benchmark_n={n}_m={m}_proportionality={proportionality}_const={const}_seed={seed}.txt'
np.savetxt(os.path.join(savedir, savename), pc, fmt='%d')

# print('m_eff, n_eff: ', pc.shape)
# print('len(connected_components): ', len(connected_components))
# print('k = ', pc.shape[1] - rank(pc))
# print('d = ', get_classical_code_distance_time_limit(pc))
# print('ldpc params', get_ldpc_params(pc))

data = {}
data['m_eff'] = pc.shape[0]
data['n_eff'] = pc.shape[1]
data['k'] = pc.shape[1] - rank(pc)
data['d'] = get_classical_code_distance_time_limit(pc)
data['ldpc_params'] = get_ldpc_params(pc)
data['removed_checks'] = removed_checks
data['removed_bits'] = removed_bits
data['connectivity_one_checks'] = connectivity_one_checks
data['connectivity_one_bits'] = connectivity_one_bits
data['connectivity_two_checks'] = connectivity_two_checks
data['connectivity_two_bits'] = connectivity_two_bits
data['points'] = points.tolist()

savedataname = f'removed_data_rgg_benchmark_n={n}_m={m}_proportionality={proportionality}_const={const}_seed={seed}.npz'
np.savez(os.path.join(savedir, savedataname), **data)