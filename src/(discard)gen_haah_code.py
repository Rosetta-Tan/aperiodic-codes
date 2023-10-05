import numpy as np
from ldpc.code_util import get_code_parameters
from bposd.hgp import hgp
from bposd.css import *
from numba import njit
import matplotlib.pyplot as plt


L = 4
assert np.log2(L).is_integer(), "L must be a power of 2."
n = L**3
sqrtL = int(np.sqrt(L))

####################################################################################################
# Generate Haah's code. (ref: https://www.overleaf.com/project/64c9d73ea1c40344fe0756e7)
####################################################################################################
'''Caveat:
We cannot use two periodic boundary classical codes to obtain Haah's code.
Instead, we use two OBC classical codes to do HGP,
then take the PBC at the end.
'''

@njit
def h1(sqrtL):
    '''The first 3 numbers are positions of checks,
    (i, j, k) represents cube centered at (i+1/2, j+1/2, k+1/2).
    The last 3 are positions of bits,
    (i, j, k) represents bit at (i, j, k).
    Connection:
    c(i+1/2, j+1/2, k+1/2) -> v(i+1, j+1, k+1)
    c(i+1/2, j+1/2, k+1/2) -> v(i, j+1, k+1)
    c(i+1/2, j+1/2, k+1/2) -> v(i+1, j, k+1)
    c(i+1/2, j+1/2, k+1/2) -> v(i+1, j+1, k)
    Note:
    Ghost bits are added to implement OBC.
    '''
    h1 = np.zeros((sqrtL, sqrtL, sqrtL, sqrtL+1, sqrtL+1, sqrtL+1), dtype=np.uint8)
    for i in range(sqrtL):
        for j in range(sqrtL):
            for k in range(sqrtL):
                # h1[i, j, k, (i+1)//sqrtL, (j+1)//sqrtL, (k+1)//sqrtL] = 1
                # h1[i, j, k, i, (j+1)//sqrtL, (k+1)//sqrtL] = 1
                # h1[i, j, k, (i+1)//sqrtL, j//L, (k+1)//sqrtL] = 1
                # h1[i, j, k, (i+1)//sqrtL, (j+1)//sqrtL, k//sqrtL] = 1
                h1[i,j,k, i+1,j+1,k+1] = 1
                h1[i,j,k, i,j+1,k+1] = 1
                h1[i,j,k, i+1,j,k+1] = 1
                h1[i,j,k, i+1,j+1,k] = 1
    return h1.reshape((sqrtL**3, (sqrtL+1)**3))

@njit
def h2(sqrtL):
    '''Connection:
    c(i+1/2, j+1/2, k+1/2) -> v(i, j, k)
    c(i+1/2, j+1/2, k+1/2) -> v(i+1, j+1, k)
    c(i+1/2, j+1/2, k+1/2) -> v(i+1, j, k+1)
    c(i+1/2, j+1/2, k+1/2) -> v(i, j+1, k+1)
    '''
    h2 = np.zeros((sqrtL, sqrtL, sqrtL, sqrtL+1, sqrtL+1, sqrtL+1), dtype=np.uint8)
    for i in range(sqrtL):
        for j in range(sqrtL):
            for k in range(sqrtL):
                # h2[i, j, k, i, j, k] = 1
                # h2[i, j, k, (i+1)//sqrtL, (j+1)//sqrtL, k] = 1
                # h2[i, j, k, (i+1)//sqrtL, j, (k+1)//sqrtL] = 1
                # h2[i, j, k, i, (j+1)//sqrtL, (k+1)//sqrtL] = 1
                h2[i,j,k, i,j,k] = 1
                h2[i,j,k, i+1,j+1,k] = 1
                h2[i,j,k, i+1,j,k+1] = 1
                h2[i,j,k, i,j+1,k+1] = 1
    return h2.reshape((sqrtL**3, (sqrtL+1)**3))
h1, h2 = h1(sqrtL), h2(sqrtL)

'''
If we use BPOSD code base, the final shape after HGP will be
(sqrt(L)**3 * sqrt(L+1)**3, sqrt(L)**3 * sqrt(L)**3 + sqrt(L+1)**3 * sqrt(L+1)**3)
In stead, we implememt HGP by hand

The target shape is
(sqrt(L)**3 * sqrt(L)**3, sqrt(L)**3 * sqrt(L)**3 + sqrt(L)**3 * sqrt(L)**3)
'''
hx1 = np.kron(h1, np.eye(h2.shape[1]))  # shape(sqrtL**3 * (sqrtL+1)**3, (sqrtL+1)**3 * (sqrtL+1)**3)
# hx1.reshape(sqrtL*(sqrtL+1), sqrtL*(sqrtL+1), sqrtL*(sqrtL+1), L, L, L)
print(hx1.shape)
hx1.reshape(sqrtL, sqrtL+1, sqrtL, sqrtL+1, sqrtL, sqrtL+1, L**3)
'''Addressing the ghost bits of classical code 2'''
hx1[:, 0, :, 0:sqrtL, :, 0:sqrtL, :] += hx1[:, sqrtL, :, 0:sqrtL, :, 0:sqrtL, :]  # ghosts in yz-plane
hx1[:, 0:sqrtL, :, 0, :, 0:sqrtL, :] += hx1[:, 0:sqrtL, :, sqrtL, :, 0:sqrtL, :]  # ghosts in xz-plane
hx1[:, 0:sqrtL, :, 0:sqrtL, :, 0, :] += hx1[:, 0:sqrtL, :, 0:sqrtL, :, sqrtL, :]  # ghosts in xy-plane
hx1[:, 0, :, 0, :, 0:sqrtL, :] += hx1[:, sqrtL, :, sqrtL, :, 0:sqrtL, :]  # ghosts on z line
hx1[:, 0, :, 0:sqrtL, :, 0, :] += hx1[:, sqrtL, :, 0:sqrtL, :, sqrtL, :]  # ghosts on y line
hx1[:, 0:sqrtL, :, 0, :, 0, :] += hx1[:, 0:sqrtL, :, sqrtL, :, sqrtL, :]  # ghosts on x line
hx1[:, 0, :, 0, :, 0, :] += hx1[:, sqrtL, :, sqrtL, :, sqrtL, :]  # ghost at corner
hx1 = hx1[:, 0:sqrtL, :, 0:sqrtL, :, 0:sqrtL, :]  # discard ghosts
hx1 = hx1.reshape(L**3, L**3)
print(hx1.shape)


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

####################################################################################################
# Debug
####################################################################################################
'''Check that the construction is correct
by checking the ground state degeneracy
GSD of Haah's code:
2[1 - 2 q2 + 2^{r+1}(q2 + 12 q15 + 60 q63)],
for 2<=L<=200, qn(L) = 1 if n divides L, 0 otherwise.
'''
# Ls = [4, 8, 16]
Ls = np.array([8])

def q(n, L):
    if L > 200:
        raise NotImplementedError("L > 200 is not implemented.")
    return 1 if L % n == 0 else 0
for L in Ls:
    sqrtL = int(np.sqrt(L))
    h1, h2 = h1(sqrtL), h2(sqrtL)
    haah_code = hgp(h1, h2)
    # print(haah_code.hx.shape)
    # print(haah_code.k1)
    # print(haah_code.k1t)
    # print(haah_code.k2)
    # print(haah_code.k2t)
    # gsd = 2*(1 - 2*q(2, L) + 2*L*(q(2, L) + 12*q(15, L) + 60*q(63, L)))
    log2GSD = 4*L -2 
    assert haah_code.K == log2GSD, "GSD is not correct."