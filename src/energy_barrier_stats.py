import numpy as np
from ldpc.code_util import *
from ldpc.mod2sparse import *
from bposd.css import *
from bposd.hgp import hgp
from numba import njit
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from sys import argv

"""
Statistical analysis of energy barrier.
"""

d = int(argv[1])  # toric code
# log2L = int(argv[1])  # Haah's code
# L = 2**log2L
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

hx = read_pc(f'../data/toric_code/hx_d{d}.txt')  # toric code
hz = read_pc(f'../data/toric_code/hz_d{d}.txt')  # toric code
# hx = read_pc(f'../data/haah_code/hx_L{L}.txt')  # toric code
# hz = read_pc(f'../data/haah_code/hz_L{L}.txt')  # toric code
qcode = css_code(hx=hx, hz=hz)  # all kinds of parameters are already obtained during init
# logx_space = row_span(qcode.lx)
# logz_space = row_span(qcode.lz)

num_sample = 1000
rng = np.random.default_rng(seed=0)
####################################################################################################
# Energy cost stats for low weight logical operators
####################################################################################################
logz_space = qcode.lz
min_weight_logz = logz_space[np.argmin(np.sum(logz_space, axis=1))]
np.save(f'../data/toric_code/min_weight_logz_d{d}.npy', min_weight_logz)
# np.save(f'../data/haah_code/min_weight_logz_L{L}.npy', min_weight_logz)
selector = np.where(min_weight_logz == 1)[0]
print('len of selector: ', len(selector))
numones = np.arange(1, len(selector), 1)
synd_weights_list = []
# synd_weights
energy_barrier_list = []
for numone in numones:
    qvecs = np.zeros((num_sample, qcode.N), dtype=int)
    synds = np.zeros((num_sample, qcode.N), dtype=int)
    for i in range(num_sample):
        posones = rng.choice(range(len(selector)), size=numone, replace=False)
        bitstring = bin(sum(1<<i for i in posones))[2:].zfill(len(selector))
        # print(bitstring)
        qvecs[i, selector] = np.array([int(x) for x in bitstring])
        # print(qvecs[i])
        # qvecs[i, selector] = rng.choice([0, 1], size=len(selector), p=[p, 1-p])
    synds = np.mod(qvecs@(hx.T), 2)
    synd_weights = np.sum(synds, axis=1)
    synd_weights_list.append(synd_weights)
    energy_barrier_list.append(np.min(synd_weights))

print(energy_barrier_list)

####################################################################################################
# Plot
####################################################################################################
fig, ax = plt.subplots()
ax.scatter(numones, energy_barrier_list,label='energy barrier')
ax.scatter(numones, np.min(synd_weights_list, axis=1),label='min syndrome weight')
ax.scatter(numones, np.max(synd_weights_list, axis=1),label='max syndrome weight')
ax.fill_between(numones, np.min(synd_weights_list, axis=1), np.max(synd_weights_list, axis=1), alpha=0.2)
ax.set_xlabel('Number of ones')
ax.set_ylabel('Syndrome weight')
ax.set_title('Syndrome weight distribution')
fig.savefig(f'../figures/toric_code/synd_weight_dist_d{d}.png', dpi=300)
# fig.savefig(f'../figures/haah_code/synd_weight_dist_L{L}.png', dpi=300)
plt.show()


####################################################################################################
# Debug
####################################################################################################
'''Correct implementation of logical space?'''
# print('qcode.K: ', qcode.K)
# print('len of qcode.lx', np.sum(qcode.lz, axis=1))
# print('number of logical operators: ', len(logx_space))
# print('logical X operators: ', logx_space)
# print('logical X space basis: ', qcode.lx)
# print('test row span: ', row_span(np.array([[1,1,0],[0,1,1]])))