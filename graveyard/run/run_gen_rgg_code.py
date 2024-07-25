import subprocess
import sys, os
from timeit import default_timer as timer

savedir = '/Users/yitan/Google Drive/My Drive/from_cannon/qmemory_simulation/data/rgg_code'
# savedir = '../src'
# savedir = '/n/home01/ytan/scratch/qmemory_simulation/data/rgg_code'
# sizes = range(10,110,10)  # for scaling analysis of classical codes
# sizes = range(10,21,1)  # for scaling analysis of HGP quantum codes (large range)
sizes = range(30,36,1)  # for scaling analysis of HGP quantum codes (small range)
rs = [0.2, 0.4, 0.6, 0.8]
seeds = range(0, 21)
start = timer()
for i, s in enumerate(sizes):
    for r in rs[1:]:
        for k, seed in enumerate(seeds):            
            subprocess.run(["python", "../src/gen_rgg_code.py", "--size", f'{s}', "--radius", f'{r}', "--seed", f'{seed}', '--savedir', savedir])
            end = timer()
            print(f'Elapsed time for size {s}, radius {r} and seed {seed}: {end-start} seconds', flush=True)
            start = end