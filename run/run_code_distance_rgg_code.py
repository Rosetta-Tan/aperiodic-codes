import subprocess
import sys, os
from timeit import default_timer as timer
import argparse

# readdir = '/Users/yitan/Google Drive/My Drive/from_cannon/qmemory_simulation/data/rgg_code'
# savedir = '/Users/yitan/Google Drive/My Drive/from_cannon/qmemory_simulation/data/rgg_code'
readdir = '/n/home01/ytan/scratch/qmemory_simulation/data/rgg_code'
savedir = '/n/home01/ytan/scratch/qmemory_simulation/data/rgg_code'

sizes = range(7,8,1)  # for scaling analysis of HGP quantum codes
rs = [0.2]
seeds = [0, 20]
start = timer()
for i, s in enumerate(sizes):
    for j, r in enumerate(rs):
        for k, seed in enumerate(seeds):            
            subprocess.run(["python", "../src/code_distance_rgg_code.py", "--size", f'{s}', "--radius", f'{r}', "--seed", f'{seed}', "--readdir", readdir, "--savedir", savedir])
            end = timer()
            print(f'Elapsed time for size {s}, radius {r}, and seed {seed}: {end-start} seconds', flush=True)
            print('='*80, flush=True)
            start = end