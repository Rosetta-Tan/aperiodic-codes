import numpy as np
import matplotlib.pyplot as plt
plt.style.use('../figures/norm2.mplstyle')

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
def ham(pc):
    '''Generate Hamiltonian from parity check matrix.'''
    n, m = pc.shape
    h = np.zeros((n, n))
    for i in range(m):
        h -= np.outer(pc[:, i], pc[:, i])
    return h

def energy(pc, state):
    '''Compute energy of state.'''
    return state.T @ ham(pc) @ state

def mc(t, nsamples=100):
    '''Monte Carlo simulation for coherence time.'''
    n, m = hx.shape
    state = np.ones(n)
    state /= np.sqrt(n)
    E = energy(hx, state)
    E0 = E
    E_list = []
    t_list = []
    for i in range(nsamples):
        R = np.sum(gamma)
        dt = -np.log(np.random.rand())/R
        t += dt
        t_list.append(t)
        V = np.random.choice(V_list, p=gamma/R)
        state = V @ state
        E = energy(hx, state)
        E_list.append(E)
    return t_list, E_list, E0


