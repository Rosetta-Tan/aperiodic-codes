import numpy as np
from ldpc.code_util import *
from ldpc.codes import ring_code
from bposd.hgp import *
from bposd.css import *
from sys import argv

d = int(argv[1])

####################################################################################################
# Generate toric code
####################################################################################################
def gen_toric_code(d):
    rep1 = ring_code(d)
    rep2 = ring_code(d)
    tc = hgp(rep1, rep2)
    return tc

tc = gen_toric_code(d)
np.savetxt('../data/toric_code/hx_d{}.txt'.format(d), tc.hx, fmt='%d')
np.savetxt('../data/toric_code/hz_d{}.txt'.format(d), tc.hz, fmt='%d')

####################################################################################################
# Load toric code
####################################################################################################
def read_pc(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    lines = [line.split(' ') for line in lines]
    lines = [[int(i) for i in line] for line in lines]
    return np.asarray(lines)

# hx = read_pc('../data/toric_code/hx_d{}.txt'.format(d))
# hz = read_pc('../data/toric_code/hz_d{}.txt'.format(d))