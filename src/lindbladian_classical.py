import numpy as np
import matplotlib.pyplot as plt
plt.style.use('../figures/norm2.mplstyle')
from ldpc.code_util import *
from bposd.css import *

'''
Monte Carlo simulation for coherence time of
stabilizer codes subject to thermal noise.

Jump operators:
- Jump: |psi(t+dt)> = V |psi(t)>
    - jump rate: gamma(V) = -dE/(1-exp(beta*dE))
    - |psi(t+dtr)> = V |psi(t)> / sqrt(1-dE)
- time interval: dt = -ln(rand)/R, R = np.sum(gamma)

'''
d = 5

####################################################################################################
# Load code
####################################################################################################
def read_pc(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    lines = [line.split(' ') for line in lines]
    lines = [[int(i) for i in line] for line in lines]
    return np.asarray(lines)

hx = read_pc('../data/toric_code/hx_d{}.txt'.format(d))
hz = read_pc('../data/toric_code/hz_d{}.txt'.format(d))


####################################################################################################
def energy(h, state):
    '''Compute energy of a state with respect to a parity-check matrix.'''
    return np.sum(np.abs(h @ state)**2)

ham = ham(hx)
quham = quham(hx, hz)
eigvals, eigvecs = np.linalg.eigh(quham)
np.savetxt('../data/toric_code/eigvals_d{}.txt'.format(d), eigvals)

def mc(t, nsamples=100):
    '''Monte Carlo simulation for coherence time.'''
    n, m = hx.shape
    for i in range(nsamples):
        state = np.ones(n)
        state /= np.sqrt(n)
        E = energy(hx, state)
        E0 = E
        E_list = []
        t_list = []
        R = np.sum(gamma)
        dt = -np.log(np.random.rand())/R
        t += dt
        t_list.append(t)
        V = np.random.choice(V_list, p=gamma/R)
        state = V @ state
        E = energy(hx, state)
        E_list.append(E)
    return t_list, E_list, E0


