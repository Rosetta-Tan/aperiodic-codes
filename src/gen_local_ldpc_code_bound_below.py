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
from collections import deque
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

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

# def valid_point(n, m, deg_bit, deg_check, point, connected):
#     if point < m:
#         # check node
#         if len(connected[point]) <= deg_check:
#             return True
#     else:
#         # bit node
#         if len(connected[point]) <= deg_bit:
#             return True
        
# def available_to_new_connection(n, m, deg_bit, deg_check, point, connected):
#     if point < m:
#         # check node
#         if len(connected[point]) < deg_check:
#             return True
#     else:
#         # bit node
#         if len(connected[point]) < deg_bit:
#             return True

# def choose_next_points(n, m, point, indices, visited):
#     next_points = [] 
#     if point < m:
#         # check node
#         for i in indices[point]:
#             if i not in visited and i >= m:
#                 next_points.append(i)
#     else:
#         for i in indices[point]:
#             if i not in visited and i < m:
#                 next_points.append(i)
#     return next_points

def close_to_boundary(point, cutoff=0.1):
    x, y = point
    if x > 0.5:
        x = 1. - x
    if y > 0.5:
        y = 1. - y
    lookat = min(x, y)
    if lookat <= cutoff:
        return True
    else:
        return False


def search(n, m, kmin_bit, kmin_check, const=5, r=False, seed=0):
    '''Keep pi * r^2 * n = const'''
    if r == False:
        r = np.sqrt(const/(np.pi*(n)))

    sampler_check = qmc.Halton(d=2, scramble=True, seed=seed)
    checks_points = sampler_check.random(m)
    checks_ctg = [0] * m
    sampler_bit = qmc.Halton(d=2, scramble=True, seed=seed+42)
    bits_points = sampler_bit.random(n)
    bits_ctg = [1] * n
    points = np.concatenate((checks_points, bits_points), axis=0)
    ctgs = np.array(checks_ctg + bits_ctg)
    connected = []  # connected[i] is the set of nodes connected to node i
    for i in range(m+n):
        connected.append(set())
    indices = select_points_within_radius(points, r)
    
    # for icheck in range(m):
    #     for pt in indices[icheck]:
    #         if pt >= m
    #             connected[icheck].add(pt)

    # for ibit in range(m, m+n):
    #     for pt in indices[ibit]:
    #         if pt < m:
    #             connected[ibit].add(pt)
    for i in range(len(points)):
        if ctgs[i] == 0:  # check node
            for pt in indices[i]:
                if ctgs[pt] == 1:
                    connected[i].add(pt)
        else:  # bit node
            for pt in indices[i]:
                if ctgs[pt] == 0:
                    connected[i].add(pt) 

    removed_checks = []
    removed_bits = []
    for i in range(m):
        if close_to_boundary(points[i]) and len(connected[i]) < kmin_check:
        # if len(connected[i]) < kmin_check:
            removed_checks.append(i)
    for i in range(m, m+n):
        if len(connected[i]) < kmin_bit:
            removed_bits.append(i)

    ##############################################
    removed_checks_points = points[removed_checks]
    ##############################################

    points = np.delete(points, removed_checks + removed_bits, axis=0)
    ctgs = np.delete(ctgs, removed_checks + removed_bits)

    # refilll bits until all bits satisfy kmin_bit
    success = False
    iter = 0
    start = timer()
    while not success:
        iter += 1
        print('iter = ', iter) 

        n_refill = len(removed_bits)
        sampler_bit = qmc.Halton(d=2, scramble=True, seed=seed+42)
        bits_points = sampler_bit.random(n_refill)
        points = np.concatenate((points, bits_points), axis=0)
        ctgs = np.concatenate((ctgs, np.ones(n_refill)))
        connected = [set() for _ in range(len(points))]  # connected[i] is the set of nodes connected to node i
        indices = select_points_within_radius(points, r)  # indices[i] is the set of points within radius r of point i, without point i itself

        for i in range(len(points)):
            if ctgs[i] == 0:  # check node
                for pt in indices[i]:
                    if ctgs[pt] == 1:
                        connected[i].add(pt)
            else:  # bit node
                for pt in indices[i]:
                    if ctgs[pt] == 0:
                        connected[i].add(pt)

        removed_bits = []
        for i in range(len(points)):
            if ctgs[i] == 1 and len(connected[i]) < kmin_bit:
                removed_bits.append(i)
        
        points = np.delete(points, removed_bits, axis=0)
        ctgs = np.delete(ctgs, removed_bits)

        end = timer()
        if end -start > 10:
            break
        print(f'Elapsed time for refilling bits: {end-start} seconds; num of removed bits: {len(removed_bits)}',  flush=True)
        
        if len(removed_bits) == 0:
            success = True


    connected = []  # connected[i] is the set of nodes connected to node i
    for i in range(len(points)):
        connected.append(set())
    indices = select_points_within_radius(points, r)  # indices[i] is the set of points within radius r of point i, without point i itself

    for i in range(len(points)):
        if ctgs[i] == 0:  # check node
            for pt in indices[i]:
                if ctgs[pt] == 1:
                    connected[i].add(pt)
        else:  # bit node
            for pt in indices[i]:
                if ctgs[pt] == 0:
                    connected[i].add(pt)

    eff_checks = []
    eff_bits = []
    for i in range(len(points)):
        if ctgs[i] == 0:
            eff_checks.append(i)
        else:
            eff_bits.append(i)

    pc = np.zeros((len(eff_checks), len(eff_bits)), dtype=int)
    for i, check in enumerate(eff_checks):
        for j, bit in enumerate(eff_bits):
            if bit in connected[check]:
                pc[i,j] = 1
                

    return pc, connected, indices, np.asarray(removed_checks_points), np.asarray(removed_checks)


parser = argparse.ArgumentParser()
parser.add_argument('--n', dest='n', type=int, default=100)
parser.add_argument('--m', dest='m', type=int, default=80)
parser.add_argument('--kmin_bit', dest='kmin_bit', type=int, default=3)
parser.add_argument('--kmin_check', dest='kmin_check', type=int, default=3)
# parser.add_argument('--r', dest='r', type=float, default=0.8)
parser.add_argument('--const', dest='const', type=int, default=5)
parser.add_argument('--seed', dest='seed', type=int, default=0)
parser.add_argument('--savedir', dest='savedir', type=str, default='/Users/yitan/Google Drive/My Drive/from_cannon/qmemory_simulation/data/ldpc_code')
args = parser.parse_args()

n = args.n
m = args.m
kmin_bit = args.kmin_bit
kmin_check = args.kmin_check
const = args.const
savedir = args.savedir
seed = args.seed


# n, m = np.array([100, 100]) * 20.0
# n, m = int(n), int(m)
# kmin_bit = 3
# kmin_check = 3
# const = 5
# seed = 42



pc, connected, indices, removed_checks_points, removed_checks = search(n, m, kmin_bit, kmin_check, const=const, r=False, seed=seed)
savename = f'pc_local_ldpc_bound_below_n={n}_m={m}_kmin_bit={kmin_bit}_kmin_check={kmin_check}_const={const}_seed={seed}.txt'
np.savetxt(os.path.join(savedir, savename), pc, fmt='%d')
savepointsname = f'removed_check_data_local_ldpc_bound_below_n={n}_m={m}_kmin_bit={kmin_bit}_kmin_check={kmin_check}_const={const}_seed={seed}.npz'
np.savez(os.path.join(savedir, savepointsname), removed_checks_points=removed_checks_points, removed_checks=removed_checks)

# print('connectivity of removed checks: ', [indices[i] for i in removed_bits])
# print('num of remaining points:', len(points))
# print('number of removed checks:', len(removed_checks))
# print('number of removed bits:', m + n - len(points) - len(removed_checks))
# assert len(points) == n + m - len(removed_checks)
# assert len(indices) == n + m - len(removed_checks)

# print('connectivity of removed checks: ', removed_checks)
# print('connectivity of removed bits: ', removed_bits)
# print('len of removed_checks: ', len(removed_checks))
# print('len of removed_bits: ', len(removed_bits))
print('eff (m,n) = ', pc.shape)
print('k = ', pc.shape[1] - rank(pc))
# m + n - len(points) - len(removed_checks) = len(removed_bits)