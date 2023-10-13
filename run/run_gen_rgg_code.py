import subprocess
import sys, os

sizes = range(10, 110, 10)
rs = [0.4, 0.6, 0.8]
seeds = range(0, 40)
for i, s in enumerate(sizes):
    for j, r in enumerate(rs):
        for k, seed in enumerate(seeds):
            subprocess.run(["python", "../src/gen_rgg_code.py", "--size", f'{s}', "--radius", f'{r}', "--seed", f'{seed}'])