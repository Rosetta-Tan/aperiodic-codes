import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
# from ldpc.code_util import *
from ldpc.mod2 import *

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


# ns = [50,100,150,200,250,300,350,400,450,500]
# ms = [40,80,120,160,200,240,280,320,360,400]
ss = np.array([10,20,30,40,50,60,70,80,90,100])
deg_bit = 4
deg_check = 5
ns = deg_check*ss
ms = deg_bit*ss
r = 0.2

dimkers = []
dimcoimgs = []
for i, s in enumerate(ss):
    hclassical_readpath = f'../data/rgg_code/hclassical_n{ns[i]}_m{ms[i]}_degbit{deg_bit}_degcheck{deg_check}_r{r}.txt'
    h = read_pc(hclassical_readpath)
    dimkers.append(len(nullspace(h)))
    dimcoimgs.append(ms[i] - len(row_basis(h)))

fig, ax = plt.subplots(1, 2)
ax[0].plot(ns, dimkers, 'o-', label='dim ker')
ax[0].set_xlabel('n')
ax[0].set_ylabel('dim ker')
ax[0].set_title('dim ker vs s')
ax[1].plot(ns, dimcoimgs, 'o-', label='dim coimg')
ax[1].set_xlabel('n')
ax[1].set_ylabel('dim coimg')
ax[1].set_title('dim coimg vs s')
fig.legend()
plt.show()