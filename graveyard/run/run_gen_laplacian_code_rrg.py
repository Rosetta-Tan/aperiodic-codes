import numpy as np
import subprocess
import sys, os
from timeit import default_timer as timer

savedir = '/Users/yitan/Google Drive/My Drive/from_cannon/qmemory_simulation/data/laplacian_code'
command = ['python', '../src/gen_laplacian_code.py', '--savedir', savedir]

'Random regular graph G(n,d)'
ns = np.arange(10,31,1)
ds = np.arange(5, 11, 1)
seeds = range(0, 51)
start = timer()
for n in ns:
    for d in ds:
        for k, seed in enumerate(seeds):
            print('='*80, flush=True)
            subcommand = command + ['rrg', '--n', f'{n}', '--d', f'{d}', '--seed', f'{seed}']
            subprocess.run(subcommand)
            end = timer()
            print(f'Elapsed time for n {n}, degree {d} and seed {seed} (rgg): {end-start} seconds', flush=True)
            start = end