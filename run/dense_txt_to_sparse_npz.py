import numpy as np
from scipy.sparse import csr_matrix, save_npz
import os, sys
from timeit import default_timer as timer

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
    return np.array(pc, dtype=int)

def dense_txt_to_sparse_npz(filepath):
    """
    Read dense parity check matrix from file and save as sparse matrix in npz format.
    """
    pc = read_pc(filepath)
    pc_sparse = csr_matrix(pc)
    pc_sparse = pc_sparse.astype(bool)
    savepath = filepath[0:-4] + '.npz'
    save_npz(savepath, pc_sparse)


readdir = '/Users/yitan/Google Drive/My Drive/from_cannon/qmemory_simulation/data/rgg_code'
savedir = '/Users/yitan/Google Drive/My Drive/from_cannon/qmemory_simulation/data/rgg_code'
for filename in os.listdir(readdir):
    if filename.endswith('.txt'):
        if filename.startswith('hxhgp') or filename.startswith('hzhgp'):
            dense_txt_to_sparse_npz(os.path.join(savedir, filename))
            print(f'Converted {filename} to sparse format.')