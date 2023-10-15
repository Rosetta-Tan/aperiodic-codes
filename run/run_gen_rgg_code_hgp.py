import subprocess
import sys, os
from timeit import default_timer as timer

sizes = range(11,21,1)  # for scaling analysis of HGP quantum codes
rs = [0.2, 0.4, 0.6, 0.8]
seed1 = 0
seed2 = 20
start = timer()
for i, s in enumerate(sizes):
    for j, r in enumerate(rs):
        subprocess.run(["python", "../src/gen_rgg_code_hgp.py", "--size1", f'{s}', "--size2", f'{s}', "--radius1", f'{r}', "--radius2", f'{r}', "--seed1", f'{seed1}', "--seed2", f'{seed2}'])
        end = timer()
        print(f'Elapsed time for size {s} and radius {r}: {end-start} seconds', flush=True)
        start = end