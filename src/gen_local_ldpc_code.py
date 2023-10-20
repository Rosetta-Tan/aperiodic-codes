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
from scipy.spatial import KDTree
from collections import deque

def select_points_within_radius(points, r, k):
    """
    Select k points within radius r of each point in points.
    """
    tree = KDTree(points)
    indices = tree.query_ball_point(points, r, k)
    for i in range(len(indices)):
        indices[i].remove(i)
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

def search(n, m, deg_bit, deg_check, r, seed=0):
    rng = np.random.default_rng(seed)
    points = rng.random((m+n,2))
    queue = deque([]) # queue is for bfs search
    visited = set()
    connected = [] # connected[i] is the set of nodes connected to node i
    for i in range(m+n):
        connected.append(set())
    indices = select_points_within_radius(points, r, 4)
    
    # randomly select a check node to start
    start = rng.integers(m)
    queue.append(start)
    print('starting node: ', start)
    while queue:
        node = queue.popleft()
        if not node in visited and available_to_new_connection(n, m, deg_bit, deg_check, node, connected):
            visited.add(node)
            possible_next_points = choose_next_points(n, m, node, indices, visited)
            rng.shuffle(possible_next_points)
            for possible_next_point in possible_next_points:
                if available_to_new_connection(n, m, deg_bit, deg_check, possible_next_point, connected):
                    queue.append(possible_next_point)
                    connected[node].add(possible_next_point)
                    connected[possible_next_point].add(node)
                # print(f'node {node}, connected to {len(connected[node])} other nodes')
                if node < m: # check node
                    if len(connected[node]) == deg_check:
                        break
                elif node >= m: # bit node
                    if len(connected[node]) == deg_bit:
                        break

    # checks = np.arange(m)
    # rng.shuffle(checks)
    # bits = np.arange(m, m+n)
    # rng.shuffle(bits)
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

    return pc, checks_eff, bits_eff, points

