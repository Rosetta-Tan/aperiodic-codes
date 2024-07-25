import subprocess
import sys, os
from timeit import default_timer as timer

# readdir = '/Users/yitan/Google Drive/My Drive/from_cannon/qmemory_simulation/data/rgg_code'
# savedir = '/Users/yitan/Google Drive/My Drive/from_cannon/qmemory_simulation/data/rgg_code'
readdir = '/n/home01/ytan/scratch/qmemory_simulation/data/rgg_code'
savedir = '/n/home01/ytan/scratch/qmemory_simulation/data/rgg_code'

sizes = range(10,11,1)
rs = [0.2]
seed1 = 0
seed2 = 1

start = timer()
for i, s in enumerate(sizes):
    for j, r in enumerate(rs):
        subprocess.run(["python", "../src/code_distance_rgg_code_hgp.py", "--size1", f'{s}', "--size2", f'{s}', "--radius1", f'{r}', "--radius2", f'{r}', "--seed1", f'{seed1}', "--seed2", f'{seed2}', "--readdir", readdir, "--savedir", savedir])
        end = timer()
        print(f'Elapsed time for size {s}, radius {r}: {end-start} seconds', flush=True)
        start = end