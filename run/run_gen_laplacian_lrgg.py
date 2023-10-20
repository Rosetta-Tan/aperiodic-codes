import numpy as np
import subprocess
import sys, os
from timeit import default_timer as timer

savedir = '/Users/yitan/Google Drive/My Drive/from_cannon/qmemory_simulation/data/laplacian_code'
command = ['python', '../src/gen_laplacian_code.py', '--savedir', savedir]

'Local random geometric graph model G(n,r) in 2D'
ns = np.arange(10,31,1)
rs = [0.1, 0.2, 0.4, 0.6]
seeds = range(0, 51)
start = timer()
for n in ns:
    for r in rs:
        for k, seed in enumerate(seeds):
            print('='*80, flush=True)
            subcommand = command + ['lrgg', '--n', f'{n}', '--r', f'{r}', '--seed', f'{seed}']
            subprocess.run(subcommand)
            end = timer()
            print(f'Elapsed time for n {n}, radius {r} and seed {seed} (rgg): {end-start} seconds', flush=True)
            start = end