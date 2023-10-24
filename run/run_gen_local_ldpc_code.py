import numpy as np
import subprocess
import sys, os
from timeit import default_timer as timer
import multiprocessing as mp

savedir = '/Users/yitan/Google Drive/My Drive/from_cannon/qmemory_simulation/data/ldpc_code'
command = ['python', '../src/gen_local_ldpc_code.py', '--savedir', savedir]

'Local LDPC codes in 2D generated using KDTree algorithm'
degs_bit = [4, 5, 8]
degs_check = [5, 5, 10]
ns = np.arange(50, 501, 50)
ms = np.arange(40, 401, 40)
r = 0.8
density = 50
seeds = range(0, 101)

def run_command(n, m, deg_bit, deg_check, seed):
    start = timer()
    print('='*80, flush=True)
    subcommand = command + ['--n', f'{n}', '--m', f'{m}', '--deg_bit', f'{deg_bit}', '--deg_check', f'{deg_check}', '--r', f'{r}', '--density', f'{density}', '--seed', f'{seed}']
    subprocess.run(subcommand)
    end = timer()
    print(f'Elapsed time for (n,m)=({n},{m}), (deg_check, deg_check)=({deg_bit},{deg_check}), r={r}, density={density} and seed={seed} (KDTree algo, local): {end-start} seconds', flush=True)

if __name__ == '__main__':
    with mp.Pool(processes=mp.cpu_count()) as pool:
        for n, m in zip(ns, ms):
            for deg_bit, deg_check in zip(degs_bit, degs_check):
                for seed in seeds:
                    pool.apply_async(run_command, args=(n, m, deg_bit, deg_check, seed))
        pool.close()
        pool.join()
