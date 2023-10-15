import subprocess
import sys, os
from timeit import default_timer as timer

# sizes = range(10,110,10)  # for scaling analysis of classical codes
# sizes = range(10,32,2)  # for scaling analysis of HGP quantum codes
# sizes = range(11,31,2)  # for scaling analysis of HGP quantum codes
# rs = [0.2, 0.4, 0.6, 0.8]
# seeds = range(0, 40)
sizes = range(10,11,1)  # for scaling analysis of HGP quantum codes
rs = [0.2]
seeds = range(0, 1)
start = timer()
for i, s in enumerate(sizes):
    for j, r in enumerate(rs):
        for k, seed in enumerate(seeds):            
            subprocess.run(["python", "../src/gen_rgg_code.py", "--size", f'{s}', "--radius", f'{r}', "--seed", f'{seed}'])
            end = timer()
            print(f'Elapsed time for size {s}, radius {r}, and seed {seed}: {end-start} seconds', flush=True)
            start = end