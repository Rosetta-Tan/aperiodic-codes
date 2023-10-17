import subprocess
import sys, os
from timeit import default_timer as timer

savedir = '/Users/yitan/Google Drive/My Drive/from_cannon/qmemory_simulation/data/laplacian_code'
ns = [16, 24, 32]  
seeds = range(0, 21)
start = timer()
'''Random d-regular graph on n nodes'''
for n in ns:
    ds = [n-4,n-3,n-2,n-1,n]
    for d in ds:
        for k, seed in enumerate(seeds):            
            subprocess.run(["python", "../src/gen_laplacian_code.py", "--n", f'{n}', "--d", f'{d}', "--seed", f'{seed}', '--savedir', savedir])
            end = timer()
            print(f'Elapsed time for n {n}, d {d} and seed {seed}: {end-start} seconds', flush=True)
            start = end