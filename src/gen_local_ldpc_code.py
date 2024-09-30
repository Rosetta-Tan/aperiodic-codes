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
from collections import deque


def get_classical_code_distance(h):
    if rank(h) == h.shape[1]:
        print('Code is full rank, no codewords')
        return np.inf
    else:
        start = timer()
        print('Code is not full rank, there are codewords')
        print('Computing codeword space basis ...')
        ker = nullspace(h)
        print('debug: ker = ', ker)
        end = timer()
        print(f'Elapsed time for computing codeword space basis: {end-start} seconds', flush=True)
        print('len of ker: ', len(ker))
        print('Start finding minimum Hamming weight while buiding codeword space ...')
        start = end
        # @jit
        def find_min_weight_while_build(matrix):
            span = []
            min_hamming_weight = np.inf
            for ir, row in enumerate(matrix):
                print('debug: ir = ', ir, 'current min_hamming_weight = ', min_hamming_weight, flush=True)  # debug
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
                span = list(np.unique(temp + span, axis=0))
            assert len(span) == 2**len(matrix) - 1
            return min_hamming_weight
        min_hamming_weight = find_min_weight_while_build(ker)
        end = timer()
        print(f'Elapsed time for finding minimum Hamming weight while buiding codeword space : {end-start} seconds', flush=True)
        
        return min_hamming_weight

def select_points_within_radius(points, r):
    """
    Select k points within radius r of each point in points.
    """
    tree = KDTree(points)
    indices = tree.query_ball_point(points, r, p=2)   
    return indices

def valid_point(n, m, deg_bit, deg_check, point, connected):
    if point < m:
        # check node
        if len(connected[point]) <= deg_check:
            return True
    else:
        # bit node
        if len(connected[point]) <= deg_bit:
            return True
        
def available_to_new_connection(n, m, deg_bit, deg_check, point, connected):
    if point < m:
        # check node
        if len(connected[point]) < deg_check:
            return True
    else:
        # bit node
        if len(connected[point]) < deg_bit:
            return True

def choose_next_points(n, m, point, indices, visited):
    next_points = [] 
    if point < m:
        # check node
        for i in indices[point]:
            if i not in visited and i >= m:
                next_points.append(i)
    else:
        for i in indices[point]:
            if i not in visited and i < m:
                next_points.append(i)
    return next_points

def search(n, m, kmax_bit, kmax_check, kmin_bit, kmin_check, r, density, seed=0):
    rng = np.random.default_rng(seed)
    linear_length = np.sqrt((n+m)/density)

    points = linear_length * rng.random((m+n, 2))
    checks = points[:m]
    bits = points[m:]
    queue = deque([]) # queue is for bfs search
    visited = set()
    connected = [] # connected[i] is the set of nodes connected to node i
    for i in range(m+n):
        connected.append(set())
    indices = select_points_within_radius(points, r)

    # '''Interleaving bits and checks'''
    # # randomly select a check node to start
    # start = rng.integers(m)
    # # randomly select a bit node to start
    # start = rng.integers(m, m+n)
    # queue.append(start)
    # while queue:
    #     node = queue.popleft()
    #     if not node in visited and available_to_new_connection(n, m, kmax_bit, kmax_bit, node, connected):
    #         visited.add(node)
    #         possible_next_points = choose_next_points(n, m, node, indices, visited)
    #         rng.shuffle(possible_next_points)
    #         for possible_next_point in possible_next_points:
    #             if available_to_new_connection(n, m, kmax_bit, kmax_check, possible_next_point, connected):
    #                 queue.append(possible_next_point)
    #                 connected[node].add(possible_next_point)
    #                 connected[possible_next_point].add(node)
    #             if node < m: # check node
    #                 if len(connected[node]) == kmax_check:
    #                     break
    #             elif node >= m: # bit node
    #                 if len(connected[node]) == kmax_bit:
    #                     break
    
    '''Check first'''
    checks = np.arange(m)
    rng.shuffle(checks)
    bits = np.arange(m, m+n)
    rng.shuffle(bits)
    for check in checks:
        possible_next_bits = choose_next_points(n, m, check, indices, visited)
        rng.shuffle(possible_next_bits)
        for possible_next_bit in possible_next_bits:
            if len(connected[check]) < kmax_check:
                if available_to_new_connection(n, m, kmax_bit, kmax_check, possible_next_bit, connected):
                    connected[check].add(possible_next_bit)
                    connected[possible_next_bit].add(check)
                # print(f'check {check}, connected to {len(connected[check])} other bits')
            elif len(connected[check]) == kmax_check:
                break
    for bit in bits:
        possible_next_checks = choose_next_points(n, m, bit, indices, visited)
        rng.shuffle(possible_next_checks)
        for possible_next_check in possible_next_checks:
            if len(connected[bit]) < kmax_bit:
                if available_to_new_connection(n, m, kmax_bit, kmax_check, possible_next_check, connected):
                    connected[bit].add(possible_next_check)
                    connected[possible_next_check].add(bit)
                # print(f'bit {bit}, connected to {len(connected[bit])} other checks')
            elif len(connected[bit]) == kmax_bit:
                break

    # '''Bit first'''
    # checks = np.arange(m)
    # rng.shuffle(checks)
    # bits = np.arange(m, m+n)
    # rng.shuffle(bits)
    # for bit in bits:
    #     possible_next_checks = choose_next_points(n, m, bit, indices, visited)
    #     rng.shuffle(possible_next_checks)
    #     for possible_next_check in possible_next_checks:
    #         if len(connected[bit]) < deg_bit:
    #             if available_to_new_connection(n, m, deg_bit, deg_check, possible_next_check, connected):
    #                 connected[bit].add(possible_next_check)
    #                 connected[possible_next_check].add(bit)
    #             # print(f'bit {bit}, connected to {len(connected[bit])} other checks')
    #         elif len(connected[bit]) == deg_bit:
    #             break
    # for check in checks:
    #     possible_next_bits = choose_next_points(n, m, check, indices, visited)
    #     rng.shuffle(possible_next_bits)
    #     for possible_next_bit in possible_next_bits:
    #         if len(connected[check]) < deg_check:
    #             if available_to_new_connection(n, m, deg_bit, deg_check, possible_next_bit, connected):
    #                 connected[check].add(possible_next_bit)
    #                 connected[possible_next_bit].add(check)
    #             # print(f'check {check}, connected to {len(connected[check])} other bits')
    #         elif len(connected[check]) == deg_check:
    #             break


    adj_mat = np.zeros((m+n, m+n))
    for i in range(m+n):
        if len(connected[i]) > 0:
            for j in connected[i]:
                adj_mat[i,j] = 1
    assert np.all(adj_mat == adj_mat.T), 'the adjacency matrix is not symmetric'
    G = nx.from_numpy_array(adj_mat)
    connected_components = list(nx.connected_components(G))
    # create subgraph for the largest connected component
    G_largest = G.subgraph(connected_components[0])
    checks_eff = [node for node in G_largest.nodes if node < m]
    bits_eff = [node for node in G_largest.nodes if node >= m]
    pc = nx.bipartite.biadjacency_matrix(G, row_order=checks_eff, column_order=bits_eff).toarray()

    removed_checks = []
    for comp in connected_components[1:]:
        comp = list(comp)
        for item in comp:
            if item < m:
                removed_checks.append(item)
    return pc, np.asarray(points), np.asarray(removed_checks)

parser = argparse.ArgumentParser()
parser.add_argument('--n', dest='n', type=int, default=100)
parser.add_argument('--m', dest='m', type=int, default=80)
parser.add_argument('--kmax_bit', dest='kmax_bit', type=int, default=8)
parser.add_argument('--kmax_check', dest='kmax_check', type=int, default=10)
parser.add_argument('--kmin_bit', dest='kmin_bit', type=int, default=3)
parser.add_argument('--kmin_check', dest='kmin_check', type=int, default=3)
parser.add_argument('--r', dest='r', type=float, default=0.8)
parser.add_argument('--density', dest='density', type=int, default=50)
parser.add_argument('--seed', dest='seed', type=int, default=0)
parser.add_argument('--savedir', dest='savedir', type=str, default='/Users/yitan/Google Drive/My Drive/from_cannon/qmemory_simulation/data/ldpc_code')
args = parser.parse_args()

n = args.n
m = args.m
kmax_bit = args.kmax_bit
kmax_check = args.kmax_check
kmin_bit = args.kmin_bit
kmin_check = args.kmin_check
r = args.r
density = args.density
savedir = args.savedir
seed = args.seed


# n, m = np.array([100, 100])
# n, m = int(n), int(m)
# kmax_bit = args.kmax_bit
# kmax_check = args.kmax_check
# kmin_bit = args.kmin_bit
# kmin_check = args.kmin_check
# r = args.r
# density = 50
# seed = 10


pc, points, removed_checks = search(n, m, kmax_bit, kmax_check, kmin_bit, kmin_check, r=r, density=density, seed=seed)
savename = f'hclassical_kdtreealgo_local_n={n}_m={m}_kmax_bit={kmax_bit}_kmax_check={kmax_check}_kmin_bit={kmin_bit}_kmin_check={kmin_check}_r={r}_density={density}_seed={seed}.txt'
np.savetxt(os.path.join(savedir, savename), pc, fmt='%d')
savepointsname = f'removed_check_data_local_n={n}_m={m}_kmax_bit={kmax_bit}_kmax_check={kmax_check}_kmin_bit={kmin_bit}_kmin_check={kmin_check}_r={r}_density={density}_seed={seed}.npz'
np.savez(os.path.join(savedir, savepointsname), points=points, removed_checks=removed_checks)

# print('m_eff, n_eff: ', pc.shape)
# print('k = ', len(nullspace(pc)))
# print(('d = ', get_classical_code_distance(pc)))