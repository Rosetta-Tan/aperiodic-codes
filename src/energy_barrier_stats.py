import numpy as np
from ldpc.code_util import *
from bposd.css import *
from bposd.hgp import hgp
from numba import njit
import matplotlib.pyplot as plt
from timeit import default_timer as timer

"""
Greedy algorithm to search for the logical operator
with the lowest energy barrier.
"""

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

hx = read_pc('../data/hx_outputBalancedProduct.txt')
hz = read_pc('../data/hz_outputBalancedProduct.txt')
qcode = css_code(hx, hz)  # all kinds of parameters are already obtained during init
logx_space = row_span(qcode.lx)
logz_space = row_span(qcode.lz)

num_sample = 10000
rng = np.random.default_rng(seed=0)
################################################################

min_weight_logz = logz_space[np.argmin(np.sum(logz_space, axis=1))]
select = np.where(min_weight_logz == 1)[0]
ps = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
synd_weights_list = []
energy_barrier_list = []
for p in ps:
    qvecs = np.zeros((num_sample, qcode.N), dtype=int)
    synds = np.zeros((num_sample, qcode.N), dtype=int)
    for i in range(num_sample):
        qvecs[i, select] = rng.choice([0, 1], size=len(select), p=[p, 1-p])
    synds = np.mod(qvecs@(hx.T), 2)
    synd_weights = np.sum(synds, axis=1)
    synd_weights_list.append(synd_weights)
    energy_barrier_list.append(np.min(synd_weights))
################################################################
# plt.figure()
# plt.hist(synd_weights, bins=np.linspace(np.min(synd_weights), np.max(synd_weights), 50), density=True)
# plt.xlabel('Syndrome weight')
# plt.ylabel('Probability')
# plt.title('Syndrome weight distribution')
# plt.savefig('../figures/synd_weight_dist.png', dpi=300)
# plt.show()
print(energy_barrier_list)    


