import numpy as np
from ldpc import code_util
from bposd.hgp import hgp
from numba import njit
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


L = 4
assert np.log2(L).is_integer(), "L must be a power of 2."
n = L**3
sqrtL = int(np.sqrt(L))

####################################################################################################
# Generate Haah's code. (ref: https://arxiv.org/abs/1101.1962)
####################################################################################################

@njit
def h1():
    '''The first 3 numbers are positions of checks,
    (i, j, k) represents cube centered at (i+1/2, j+1/2, k+1/2).
    The last 3 are positions of bits,
    (i, j, k) represents bit at (i, j, k).
    Connection:
    c(i+1/2, j+1/2, k+1/2) -> v(i+1, j+1, k+1)
    c(i+1/2, j+1/2, k+1/2) -> v(i, j+1, k+1)
    c(i+1/2, j+1/2, k+1/2) -> v(i+1, j, k+1)
    c(i+1/2, j+1/2, k+1/2) -> v(i+1, j+1, k)
    '''
    h1 = np.zeros((sqrtL, sqrtL, sqrtL, sqrtL, sqrtL, sqrtL), dtype=np.uint8)
    for i in range(sqrtL):
        for j in range(sqrtL):
            for k in range(sqrtL):
                h1[i, j, k, (i+1)//sqrtL, (j+1)//sqrtL, (k+1)//sqrtL] = 1
                h1[i, j, k, i, (j+1)//sqrtL, (k+1)//sqrtL] = 1
                h1[i, j, k, (i+1)//sqrtL, j//L, (k+1)//sqrtL] = 1
                h1[i, j, k, (i+1)//sqrtL, (j+1)//sqrtL, k//sqrtL] = 1
    return h1.reshape((sqrtL**3, sqrtL**3))

@njit
def h2():
    '''Connection:
    c(i+1/2, j+1/2, k+1/2) -> v(i, j, k)
    c(i+1/2, j+1/2, k+1/2) -> v(i+1, j+1, k)
    c(i+1/2, j+1/2, k+1/2) -> v(i+1, j, k+1)
    c(i+1/2, j+1/2, k+1/2) -> v(i, j+1, k+1)
    '''
    h2 = np.zeros((sqrtL, sqrtL, sqrtL, sqrtL, sqrtL, sqrtL), dtype=np.uint8)
    for i in range(sqrtL):
        for j in range(sqrtL):
            for k in range(sqrtL):
                h2[i, j, k, i, j, k] = 1
                h2[i, j, k, (i+1)//sqrtL, (j+1)//sqrtL, k] = 1
                h2[i, j, k, (i+1)//sqrtL, j, (k+1)//sqrtL] = 1
                h2[i, j, k, i, (j+1)//sqrtL, (k+1)//sqrtL] = 1
    return h2.reshape((sqrtL**3, sqrtL**3))
h1, h2 = h1(), h2()
haah_code = hgp(h1, h2)

####################################################################################################
# Save data
####################################################################################################
# np.savetxt('../data/haah_code_hx_L{}.txt'.format(L), haah_code.hx, fmt='%d')
# np.savetxt('../data/haah_code_hz_L{}.txt'.format(L), haah_code.hz, fmt='%d')


####################################################################################################
# Visualization
####################################################################################################
checks = np.array([(i+0.5,j+0.5,k+0.5) for i in range(sqrtL) for j in range(sqrtL) for k in range(sqrtL)])
vars = np.array([(i,j,k) for i in range(sqrtL) for j in range(sqrtL) for k in range(sqrtL)])
vars_ghost = np.array([(sqrtL,j,k) for j in range(sqrtL) for k in range(sqrtL)] \
                    + [(i,sqrtL,k) for i in range(sqrtL) for k in range(sqrtL)] \
                    + [(i,j,sqrtL) for i in range(sqrtL) for j in range(sqrtL)] \
                    + [(sqrtL,sqrtL,k) for k in range(sqrtL)] \
                    + [(sqrtL,j,sqrtL) for j in range(sqrtL)] \
                    + [(i,sqrtL,sqrtL) for i in range(sqrtL)] \
                    + [(sqrtL,sqrtL,sqrtL)])

'Visualiing classical code 1'
# edges = []
# for i in range(sqrtL):
#         for j in range(sqrtL):
#             for k in range(sqrtL):
#                 edges.append(((i+0.5,j+0.5,k+0.5), (i+1,j+1,k+1)))
#                 edges.append(((i+0.5,j+0.5,k+0.5), (i,j+1,k+1)))
#                 edges.append(((i+0.5,j+0.5,k+0.5), (i+1,j,k+1)))
#                 edges.append(((i+0.5,j+0.5,k+0.5), (i+1,j+1,k)))
# edges = np.asarray(edges)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.scatter(*checks.T, marker="s", s=100, label="checks")
# ax.scatter(*vars.T, s=100, label="bits")
# ax.scatter(*vars_ghost.T, s=100, c='#ff7f0e', alpha=0.2)
# for edge in edges:
#     ax.plot(*edge.T, c='k')
# ax.set_xticks(range(sqrtL+1))
# ax.set_yticks(range(sqrtL+1))
# ax.set_zticks(range(sqrtL+1))
# ax.legend()
# fig.tight_layout()
# fig.savefig('../figures/haah_classical_code1_L{}.png'.format(L))
# plt.show()

'Visualiing classical code 2'
# edges = []
# for i in range(sqrtL):
#         for j in range(sqrtL):
#             for k in range(sqrtL):
#                 edges.append(((i+0.5,j+0.5,k+0.5), (i,j,k)))
#                 edges.append(((i+0.5,j+0.5,k+0.5), (i+1,j+1,k)))
#                 edges.append(((i+0.5,j+0.5,k+0.5), (i+1,j,k+1)))
#                 edges.append(((i+0.5,j+0.5,k+0.5), (i,j+1,k+1)))
# edges = np.asarray(edges)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.scatter(*checks.T, marker="s", s=100, label="checks")
# ax.scatter(*vars.T, s=100, label="bits")
# ax.scatter(*vars_ghost.T, s=100, c='#ff7f0e', alpha=0.2)
# for edge in edges:
#     ax.plot(*edge.T, c='k')
# ax.set_xticks(range(sqrtL+1))
# ax.set_yticks(range(sqrtL+1))
# ax.set_zticks(range(sqrtL+1))
# ax.legend()
# fig.tight_layout()
# fig.savefig('../figures/haah_classical_code2_L{}.png'.format(L))
# plt.show()

'Visualiing quantum code (X stabilizers)'
#TODO