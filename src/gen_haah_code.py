import numpy as np
from ldpc.code_util import get_code_parameters
from bposd.hgp import hgp
from bposd.css import *
from numba import njit
import matplotlib.pyplot as plt

L = 4
assert np.log2(L).is_integer(), "L must be a power of 2."
n = 2*L**3
