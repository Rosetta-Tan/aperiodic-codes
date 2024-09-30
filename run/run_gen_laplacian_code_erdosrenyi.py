import numpy as np
import subprocess
import sys, os
from timeit import default_timer as timer

savedir = '/Users/yitan/Google Drive/My Drive/from_cannon/qmemory_simulation/data/laplacian_code'
command = ['python', '../src/gen_laplacian_code.py', '--savedir', savedir]

'Erdos-Renyi model G(n,p)'
ns = np.arange(10,31,1)
numerators = np.arange(5, 11, 1)
seeds = range(0, 51)
start = timer()
for n in ns:
    for numerator in numerators:
        for k, seed in enumerate(seeds):
            print('='*80, flush=True)
            subcommand = command + ['er', '--n', f'{n}', '--p', f'{numerator/n}', '--seed', f'{seed}']
            subprocess.run(subcommand)
            end = timer()
            print(f'Elapsed time for n {n}, expected edges {numerator} and seed {seed} (Erdos-Renyi): {end-start} seconds', flush=True)
            start = end