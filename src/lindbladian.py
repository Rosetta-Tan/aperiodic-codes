import numpy as np
import matplotlib.pyplot as plt
plt.style.use('../figures/norm2.mplstyle')
from ldpc.code_util import *
from bposd.css import *
from bposd import bposd_decoder
import stim
import pymatching as pm
from pymatching import Matching
from numba import jit, njit
from timeit import default_timer as timer

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

def repr_stab(hx, hz):
    xs = []
    zs = []
    for i in range(hx.shape[0]):
        stab = "+"
        for j in range(hx.shape[1]):
            if hx[i,j] == 1:
                stab += 'X'
            else:
                stab += '_'
        xs.append(stim.PauliString(stab))
    for i in range(hz.shape[0]):
        stab = "+"
        for j in range(hz.shape[1]):
            if hz[i,j] == 1:
                stab += 'Z'
            else:
                stab += '_'
        zs.append(stim.PauliString(stab))
    return xs, zs

hx = read_pc('../data/toric_code/hx_d{}.txt'.format(d))
hz = read_pc('../data/toric_code/hz_d{}.txt'.format(d))
qcode = css_code(hx=hx, hz=hz)

####################################################################################################
# Utility functions
####################################################################################################

def stab_tableau(hx, hz):
    '''Generate quantum Hamiltonian from parity check matrices.
    The Hamiltonian is represented using stabilizer tableau.
    '''
    xs, zs=  repr_stab(hx, hz)
    stabs = xs + zs
    tableau = stim.Tableau.from_stabilizers(stabs, allow_redundant=True, allow_underconstrained=True)
    return tableau

def synd(qcode, qvec, error_type='X'):
    if error_type == 'X':
        h = qcode.hz
    if error_type == 'Z':
        h = qcode.hx
    return h @ qvec % 2
    
def energy(qcode, qvec, error_type='X'):
    '''Compute energy of a state with respect to a parity-check matrix.'''
    if error_type == 'X':
        h = qcode.hz
    if error_type == 'Z':
        h = qcode.hx
    return np.sum(h @ qvec % 2)

def determine_system(qcode, error_type='X'):
    '''Determine the system to be simulated.
    - error_type: 'X' or 'Z'
    '''
    if error_type == 'X':
        h = qcode.hz
        log_op_space = row_span(qcode.lx)
        stab_weights = np.sum(qcode.hz, axis=0)
    elif error_type == 'Z':
        h = qcode.hx
        log_op_space = row_span(qcode.lz)
        stab_weights = np.sum(qcode.hx, axis=0)
    return h, log_op_space, stab_weights

h, log_op_space, stab_weights = determine_system(qcode, error_type='X')

@njit
def distance_to_codespace(qvec, log_op_space):
    """
    Compute the minimum Hamming distance between a vector and the codespace.
    """
    return np.bitwise_or(qvec, log_op_space).sum(axis=1).min()

def mc_activate(qcode, beta=10, tmax=10e12, nsamples=20, error_type='X'):
    '''Monte Carlo simulation for coherence time.
    Using activate error correction.
    Decoder: MWPM
    '''
    if error_type == 'X':
        h = qcode.hz
        log_op_space = row_span(qcode.lx)
        stab_weights = np.sum(qcode.hz, axis=0)
    elif error_type == 'Z':
        h = qcode.hx
        log_op_space = row_span(qcode.lz)
        stab_weights = np.sum(qcode.hx, axis=0)

    rng = np.random.default_rng(seed=0)
    taus = np.zeros(nsamples)

    @jit
    def update_single_step(qvec, t):
        '''Evolve the system by one time step.'''
        synd_cur = np.mod(h @ qvec, 2)
        synd_weight_cur = np.sum(synd_cur)
        synd_next = np.bitwise_or(synd_cur, h.T)
        synd_next_test = np.array([np.bitwise_or(synd_cur, h[:,i]) for i in range(h.shape[1])])
        assert np.allclose(synd_next, synd_next_test)
        # assert synd_next.shape == synd_next_test.shape
        assert synd_next.shape == (h.shape[1], h.shape[0])
        synd_weights_next = np.sum(synd_next, axis=1)
        assert len(synd_weights_next) == h.shape[1]
        dEs = synd_weights_next - synd_weight_cur
        assert dEs.__class__ == np.ndarray
        # print('dEs', dEs)
        rates = np.array([-dE/(1-np.exp(beta*dE)) if dE!=0 else 1/beta for dE in dEs])
        # rates_test = -dEs/(1-np.exp(beta*dEs))
                
        total_rates = np.sum(rates)
        # print('total_rates', total_rates)
        probs = rates/total_rates  # normalize probablity 
        # print('probs', probs)
        idx_flip = rng.choice(range(len(qvec)), p=probs)  # the index of the qubit to be flipped
        'Update qvec'
        qvec[idx_flip] ^= 1
        'Update time'
        dt = -np.log(rng.random())/total_rates
        return qvec, dt, t+dt

    start = timer()
    for i in range(nsamples):
        qvec = np.zeros(qcode.N, dtype=int)
        t = 0.
        iter = 0
        while(t <= tmax):
            print('iter', iter)
            iter += 1
            qvec, dt, t = update_single_step(qvec, t)
        end = timer()
        print('sample {}/{}, time: {}'.format(i, nsamples, end-start))
    return taus, np.average(taus)

def decoding(synd, p_physical, h, mode='mwpm'):
    '''Decoding error correction.
    mode: 'mwpm' or 'bposd'
    p_physical: phenominological physical error rate,
    obtained by taking stats of the Monte Carlo snapshots.
    '''
    if mode == 'mwpm':
        pass
    elif mode == 'bposd':
        bpd = bposd_decoder(h, channel_probs=p_physical*np.ones(h.shape[1]),
                            bp_method="product_sum", osd_method="osd_e", osd_order=3, max_iter=10)
        decoding = bpd.decode(synd)

def decoding_stats(cur_time, synd_batch, mode='mwpm'):
    pass

####################################################################################################
# Run
####################################################################################################
beta = 10
tmax = 1e8
nsamples = 20
# taus, tau = mc_activate(qcode, beta=beta, tmax=tmax, nsamples=nsamples, error_type='X')
# np.savetxt(f'../data/toric_code/taus_d{d}_tmax_{tmax:.2e}_beta_{beta}.txt', taus)
# np.save(f'../data/toric_code/tau_d{d}_tmax_{tmax:.2e}_beta_{beta}.npy', tau)

qvec = np.random.choice([0,1], size=qcode.N).astype(int)
synd = h@qvec % 2
rates = np.array([-stab_weight/(1-np.exp(beta*stab_weight)) 
                          if stab_weight!=0 else 1/beta for stab_weight in stab_weights])
# channel_probs = rates/np.sum(rates)
total_rate = np.sum(rates)


####################################################################################################
# Debug
####################################################################################################
def test_quham(hx, hz):
    css = css_code(hx=hx, hz=hz)
    css.test()

# test_quham(hx, hz)
'''For toric code and configuration model x rep,
need to see that the error channel probabilities is uniform.
'''
# assert np.allclose(channel_probs, 1/len(channel_probs))
print('total rate: ', total_rate)

