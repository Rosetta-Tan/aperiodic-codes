import numpy as np
import subprocess
import sys, os
from timeit import default_timer as timer
import multiprocessing as mp

savedir = '/Users/yitan/Google Drive/My Drive/from_cannon/qmemory_simulation/data/ldpc_code'
command = ['python', '../src/gen_ldpc_code.py', '--savedir', savedir]

'Local random geometric graph model G(n,r) in 2D'
sizes = np.arange(10,101,10)
degs_bit = [4, 5, 8]
degs_check = [5, 6, 10]
seeds = range(0, 101)

def run_command(size, deg_bit, deg_check, seed):
    start = timer()
    n, m = size*deg_check, size*deg_bit
    print('='*80, flush=True)
    subcommand = command + ['--size', f'{size}', '--deg_bit', f'{deg_bit}', '--deg_check', f'{deg_check}', '--seed', f'{seed}', 'nonlocal']
    subprocess.run(subcommand)
    end = timer()
    print(f'Elapsed time for (n,m)=({n},{m}), (deg_check, deg_check)=({deg_bit},{deg_check}) and seed={seed} (nonlocal): {end-start} seconds', flush=True)
    start = end
    subcommand_noprledge = subcommand + ['--noprledge']
    subprocess.run(subcommand_noprledge)
    end = timer()
    print(f'Elapsed time for (n,m)=({n},{m}), (deg_check, deg_check)=({deg_bit},{deg_check}) and seed={seed} (nonlocal, no prledge): {end-start} seconds', flush=True)

if __name__ == '__main__':
    with mp.Pool(processes=mp.cpu_count()) as pool:
        for size in sizes:
            for deg_bit, deg_check in zip(degs_bit, degs_check):
                for seed in seeds:
                    pool.apply_async(run_command, args=(size, deg_bit, deg_check, seed))
        pool.close()
        pool.join()
