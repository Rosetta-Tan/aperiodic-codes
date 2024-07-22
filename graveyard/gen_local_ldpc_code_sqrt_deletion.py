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
from scipy.stats import qmc
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

def valid_point(n, m, deg_bit, deg_check, point, connected):
    if point < m:
        # check node
        if len(connected[point]) <= deg_check:
            return True
    else:
        # bit node
        if len(connected[point]) <= deg_bit:
            return True
        
def available_to_new_connection(n, m, kappa_max_b, kappa_max_c, point, connected):
    if point < m:
        # check node
        if len(connected[point]) < kappa_max_c:
            return True
    else:
        # bit node
        if len(connected[point]) < kappa_max_b:
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

def sanity_check_component_connected(connected_component, adj_mat):
    G_sub = nx.from_numpy_array(adj_mat[connected_component,:][:,connected_component])
    np.savetxt('adj_mat.txt', adj_mat[connected_component,:][:,connected_component], fmt='%d')
    assert nx.is_connected(G_sub), 'the alleged connected component is not connected'

def get_local_code(n, m, kappa_max_b, kappa_max_c, kappa_min_b, kappa_min_c, const=8, r=False, seed=0):
    '''Keep pi * r^2 * n = const'''
    if r == False:
        r = np.sqrt(const/(np.pi*(n)))

    # rng = np.random.default_rng(seed=seed)
    # points = rng.uniform(size=(m+n, 2))
    sampler_check = qmc.Sobol(d=2, seed=seed)
    checks = sampler_check.random(n=m)
    sampler_bit = qmc.Sobol(d=2, seed=seed+42)
    bits = sampler_bit.random(n=n)
    points = np.concatenate((checks, bits), axis=0)
    visited = set()
    ctgs = np.array([0]*m + [1]*n)  # 0 for check node, 1 for bit node
    connected = [set() for _ in range(m+n)]  # connected[i] is the set of nodes connected to node i
    adj_mat = np.zeros((m+n, m+n))
    indices = select_points_within_radius(points, r)
    
    ##########################################################
    ################### bound from above #####################
    ##########################################################
    checks = np.arange(m)
    # rng.shuffle(checks)
    bits = np.arange(m, m+n)
    # rng.shuffle(bits)
    for check in checks:
        possible_next_bits = choose_next_points(n, m, check, indices, visited)
        # rng.shuffle(possible_next_bits)
        for possible_next_bit in possible_next_bits:
            if len(connected[check]) < kappa_max_c:
                if available_to_new_connection(n, m, kappa_max_b, kappa_max_c, possible_next_bit, connected):
                    connected[check].add(possible_next_bit)
                    connected[possible_next_bit].add(check)
                # print(f'check {check}, connected to {len(connected[check])} other bits')
            elif len(connected[check]) == kappa_max_c:
                break
    for bit in bits:
        possible_next_checks = choose_next_points(n, m, bit, indices, visited)
        # rng.shuffle(possible_next_checks)
        for possible_next_check in possible_next_checks:
            if len(connected[bit]) < kappa_max_b:
                if available_to_new_connection(n, m, kappa_max_b, kappa_max_c, possible_next_check, connected):
                    connected[bit].add(possible_next_check)
                    connected[possible_next_check].add(bit)
                # print(f'bit {bit}, connected to {len(connected[bit])} other checks')
            elif len(connected[bit]) == kappa_max_b:
                break
    
    for i in range(m+n):
        if len(connected[i]) > 0:
            for j in connected[i]:
                adj_mat[i,j] = 1
    assert np.all(adj_mat == adj_mat.T), 'the adjacency matrix is not symmetric'
    G = nx.from_numpy_array(adj_mat)
    connected_components = list(nx.connected_components(G))    
    
    ##########################################################
    ################### prune from below #####################
    ##########################################################
    def judge_prune_success(connected_component, connected):
        connectivity_one_nodes = get_connectivity_one_nodes(connected_component, connected)
        connectivity_two_nodes = get_connectivity_two_nodes(connected_component, connected)
        return len(connectivity_one_nodes) == 0 and len(connectivity_two_nodes) == 0

    def prune_from_below(connected_component):        
        for node in connected_component:
            if ctgs[node] == 0 and len(connected[node]) < kappa_min_c:
                removed_checks.append(node)
                assert node < m
                # update connected and adj_mat
                for neighbor in connected[node]:
                    connected[neighbor].remove(node)
                    adj_mat[node, neighbor] = 0
                    adj_mat[neighbor, node] = 0
                connected[node] = set()
            elif ctgs[node] == 1 and len(connected[node]) < kappa_min_b:
                removed_bits.append(node)
                assert node >= m and node < m+n
                # update connected and adj_mat
                for neighbor in connected[node]:
                    connected[neighbor].remove(node)
                    adj_mat[node, neighbor] = 0
                    adj_mat[neighbor, node] = 0
                connected[node] = set()
        assert np.all(adj_mat == adj_mat.T), 'the adjacency matrix is not symmetric'

    connected_component = list(connected_components[0])
    sanity_check_component_connected(connected_component, adj_mat)
    removed_checks = [node for node in range(m) if not node in connected_component]
    removed_bits = [node for node in range(m, m+n) if not node in connected_component]
    prune_success = judge_prune_success(connected_component, connected)
    while not prune_success:  # not succeed
        prune_from_below(connected_component)
        checks_eff = [node for node in range(m) if not node in removed_checks]
        bits_eff = [node for node in range(m, m+n) if not node in removed_bits]
        connected_component = checks_eff + bits_eff
        sanity_check_component_connected(connected_component, adj_mat)
        prune_success = judge_prune_success(connected_component, connected)
    
    ##########################################################
    ################# delete sqrt(n) checks ##################
    ##########################################################    


    ##########################################################
    ####################### finalize #########################
    ##########################################################    
    checks_eff = [node for node in range(m) if not node in removed_checks]
    bits_eff = [node for node in range(m, m+n) if not node in removed_bits]
    connected_component = checks_eff + bits_eff
    sanity_check_component_connected(connected_component, adj_mat)

    if len(checks_eff) == 0:
        raise ValueError('The largest connected component does not contain any check node')
    if len(bits_eff) == 0:
        raise ValueError('The largest connected component does not contain any bit node')
    pc = np.zeros((len(checks_eff), len(bits_eff)), dtype=int)
    for i, check in enumerate(checks_eff):
        for j, bit in enumerate(bits_eff):
            assert check < m
            assert bit >= m and bit < m+n
            if adj_mat[check, bit] == 1:
                pc[i,j] = 1

    connectivity_one_nodes = get_connectivity_one_nodes(checks_eff + bits_eff, connected)
    connectivity_two_nodes = get_connectivity_two_nodes(checks_eff + bits_eff, connected)
    connectivity_one_checks = [node for node in connectivity_one_nodes if ctgs[node] == 0]
    connectivity_one_bits = [node for node in connectivity_one_nodes if ctgs[node] == 1]
    connectivity_two_checks = [node for node in connectivity_two_nodes if ctgs[node] == 0]
    connectivity_two_bits = [node for node in connectivity_two_nodes if ctgs[node] == 1]

    return pc, adj_mat, connected_components, np.asarray(points), np.asarray(removed_checks), np.asarray(removed_bits), np.asarray(connectivity_one_checks), np.asarray(connectivity_one_bits), np.asarray(connectivity_two_checks), np.asarray(connectivity_two_bits)

parser = argparse.ArgumentParser()
parser.add_argument('--n', dest='n', type=int, default=100)
parser.add_argument('--m', dest='m', type=int, default=100)
parser.add_argument('--kappa_max_b', dest='kappa_max_b', type=int, default=10)
parser.add_argument('--kappa_max_c', dest='kappa_max_c', type=int, default=10)
parser.add_argument('--kappa_min_b', dest='kappa_min_b', type=int, default=3)
parser.add_argument('--kappa_min_c', dest='kappa_min_c', type=int, default=3)
parser.add_argument('--proportionality', dest='proportionality', type=float, default=0.1)
parser.add_argument('--const', dest='const', type=int, default=8)
parser.add_argument('--seed', dest='seed', type=int, default=0)
parser.add_argument('--savedir', dest='savedir', type=str, default='/Users/yitan/Google Drive/My Drive/from_cannon/qmemory_simulation/data/ldpc_code')
args = parser.parse_args()

n = args.n
m = args.m
proportionality = args.proportionality
const = args.const
kappa_max_b = args.kappa_max_b
kappa_max_c = args.kappa_max_c
kappa_min_b = args.kappa_min_b
kappa_min_c = args.kappa_min_c
savedir = args.savedir
seed = args.seed

n, m = np.array([100, 100]) * 1
n, m = int(n), int(m)
proportionality = 0.1
const = 8
kappa_max_b = 6
kappa_max_c = 6
kappa_min_b = 3
kappa_min_c = 3
seed = 10

pc, adj_mat, connected_components, points, removed_checks, removed_bits, connectivity_one_checks, connectivity_one_bits, connectivity_two_checks, connectivity_two_bits = get_local_code(n, m, kappa_max_b, kappa_max_c, kappa_min_b, kappa_min_c, const=const, r=False, seed=seed)
savename = f'pc_local_ldpc_sqrt_deletion_n={n}_m={m}_kappa_max_b={kappa_max_b}_kappa_max_c={kappa_max_c}_kappa_min_b={kappa_min_b}_kappa_min_c={kappa_min_c}_proportionality={proportionality}_const={const}_seed={seed}.txt'
np.savetxt(os.path.join(savedir, savename), pc, fmt='%d')


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
data['connected_components'] = connected_components
data['points'] = points
data['adj_mat'] = adj_mat

print('m_eff, n_eff: ', pc.shape)
print('len(connected_components): ', len(connected_components))
print('k = ', data['k'])
print('d = ', data['d'])
print('ldpc params', get_ldpc_params(pc))
print('min row weight: ', np.min(np.sum(pc, axis=1)))
print('min col weight: ', np.min(np.sum(pc, axis=0)))
print('num removed checks: ', len(removed_checks))
print('num removed bits: ', len(removed_bits))
print('num connectivity one checks: ', len(connectivity_one_checks))
print('num connectivity two checks: ', len(connectivity_two_checks))
print('num connectivity one bits: ', len(connectivity_one_bits))
print('num connectivity two bits: ', len(connectivity_two_bits))

savedataname = f'data_local_ldpc_sqrt_deletion_n={n}_m={m}_kappa_max_b={kappa_max_b}_kappa_max_c={kappa_max_c}_kappa_min_b={kappa_min_b}_kappa_min_c={kappa_min_c}_proportionality={proportionality}_const={const}_seed={seed}.npz'
np.savez(os.path.join(savedir, savedataname), **data)