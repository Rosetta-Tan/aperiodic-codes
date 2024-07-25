import numpy as np
import subprocess
import sys, os
from timeit import default_timer as timer
import multiprocessing as mp

savedir = '/Users/yitan/Google Drive/My Drive/from_cannon/qmemory_simulation/data/ldpc_code'
command = ['python', '../src/gen_local_ldpc_code.py', '--savedir', savedir]

'Local LDPC codes in 2D generated using KDTree algorithm'
ksmax_bit = [6]
ksmax_check = [10]
ksmin_bit = [3]
ksmin_check = [3]
ns = np.arange(100, 550, 50)
ms = np.arange(100, 550, 50)
r = 0.8
density = 50
seeds = range(0, 51)

def run_command(n, m, kmax_bit, kmax_check, seed):
    start = timer()
    print('='*80, flush=True)
    subcommand = command + ['--n', f'{n}', '--m', f'{m}', '--kmax_bit', f'{kmax_bit}', '--kmax_check', f'{kmax_check}', '--r', f'{r}', '--density', f'{density}', '--seed', f'{seed}']
    subprocess.run(subcommand)
    end = timer()
    print(f'Elapsed time for (n,m)=({n},{m}), (kmax_bit, kmax_check)=({kmax_bit},{kmax_check}), r={r}, density={density} and seed={seed} (KDTree algo, local): {end-start} seconds', flush=True)

if __name__ == '__main__':
    with mp.Pool(processes=mp.cpu_count()) as pool:
        for n, m in zip(ns, ms):
            for kmax_bit, kmax_check in zip(ksmax_bit, ksmax_check):
                for seed in seeds:
                    pool.apply_async(run_command, args=(n, m, kmax_bit, kmax_check, seed))
        pool.close()
        pool.join()
