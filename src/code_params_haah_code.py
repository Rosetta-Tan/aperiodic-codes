import numpy as np
from ldpc.code_util import *
from ldpc.mod2sparse import *
from bposd.css import *
from bposd.hgp import hgp
from numba import jit, njit
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from sys import argv


"""
Monte Carlo search for lowest Hamming weight logical operators.
"""

L = int(argv[1])
nsamples = 1000
hx_readpath = f'/n/home01/ytan/scratch/qmemory_simulation/data/haah_code/hx_L{L}.txt'
hz_readpath = f'/n/home01/ytan/scratch/qmemory_simulation/data/haah_code/hz_L{L}.txt'
code_distance_logz_savepath = f'/n/home01/ytan/scratch/qmemory_simulation/data/haah_code/code_distance_logz_L{L}.npy'
min_weight_logz_savepath = \
    f'/n/home01/ytan/scratch/qmemory_simulation/data/haah_code/min_weight_logz_L{L}_nsamples{nsamples:.2e}.txt'

####################################################################################################
# Prepare model
####################################################################################################
def read_pc(filepath):
    """
    Read parity check matrix from file.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    pc = []
    for line in lines:
        row = [int(x) for x in line.split()]
        pc.append(row)
    return np.array(pc, dtype=np.uint8)

hx = read_pc(hx_readpath)  
hz = read_pc(hz_readpath)
qcode = css_code(hx=hx, hz=hz)  # all kinds of parameters are already obtained during init
# logx_basis = qcode.lx
logz_basis = qcode.lz
rng = np.random.default_rng(seed=0)

####################################################################################################
# Monte Carlo search for code distance
####################################################################################################

def monte_carlo(beta, burn_in=1000, nsweeps=10000, h=hx.astype(float), log_basis=logz_basis.astype(float)):
    """
    Monte Carlo search for lowest weight logical operators.
    """
    # state = np.zeros(len(log_basis), dtype=np.uint8)
    # synd_basis = log_basis@(h.T) % 2
    @njit
    def hamming_weight(state):
        # synd = state@synd_basis % 2  # combination of syndrome basis vectors
        log = state@log_basis % 2
        return np.sum(log)
    @njit
    def update(state, flip_idx:int, metropolis_rand, beta):
        """
        Metropolis algorithm for sampling.
        """
        # new_state = propose(state)
        new_state = state.copy()
        new_state[flip_idx]  = 1 - state[flip_idx]
        delta_E = hamming_weight(new_state) - hamming_weight(state)
        if delta_E <= 0:
            return new_state
        elif metropolis_rand < np.exp(-beta * delta_E):
            return new_state
        else:
            return state
    
    cur = rng.choice([0.,1.], size=len(log_basis))
    while np.allclose(cur, 0):
        cur = rng.choice([0.,1.], size=len(log_basis))
    min_hamming_weight = hamming_weight(cur)
    min_weight_state = cur.copy()
    print(f'start! min weight now: {min_hamming_weight}, current weight: {hamming_weight(cur)}')
    
    'simulated annealing'
    start = timer()
    for i in range(burn_in):
        flip_idx = rng.integers(len(log_basis))
        metropolis_rand = rng.random()
        cur = update(cur, flip_idx, metropolis_rand, beta)
    print(f'burn in time: {timer()-start:.2f}s')
    for i in range(nsweeps):
        if i % 100 == 0:
            print(f'{i} sweeps, current beta: {beta}, min weight now: {min_hamming_weight}, current weight: {hamming_weight(cur)}, time: {timer()-start:.2f}s')
            beta *= 1.0
        flip_idx = rng.integers(len(log_basis))
        metropolis_rand = rng.random()
        next = update(cur, flip_idx, metropolis_rand, beta)
        while np.allclose(next, 0):
            print('stuck in zero state, resample. flip_idx: ', flip_idx, 'metropolis_rand: ', metropolis_rand)
            flip_idx = rng.integers(len(log_basis))
            metropolis_rand = rng.random()
            next = update(cur, flip_idx, metropolis_rand, beta)
        if hamming_weight(next) < min_hamming_weight:
            min_hamming_weight = hamming_weight(next)
            min_weight_state = next.copy()
        cur = next
    return min_hamming_weight, min_weight_state, min_weight_state@log_basis % 2

beta = 1./(L)
min_hamming_weight, min_weight_state, min_weight_logz = monte_carlo(beta)
np.save(code_distance_logz_savepath, min_hamming_weight)
np.savetxt(min_weight_logz_savepath, min_weight_logz, fmt='%d')
print(f'Code distance of logical Z: {min_hamming_weight}')

####################################################################################################
# Debug
####################################################################################################
synd_basis = logz_basis@(hx.T) % 2
def hamming_weight(state):
    synd = state@synd_basis % 2  # combination of syndrome basis vectors
    return np.sum(synd)
# print(np.sum(logz_basis, axis=1))
# print(logz_basis.shape)
# print(hx.T.shape)
nsamples=1000
naive = np.loadtxt(f'/n/home01/ytan/scratch/qmemory_simulation/data/haah_code/min_weight_logz_L{L}_nsamples{nsamples:.2e}.txt')
print('Min weight in certain basis: ', np.sum(naive))

    