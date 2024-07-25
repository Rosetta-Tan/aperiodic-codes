import numpy as np
import subprocess
import sys, os
from timeit import default_timer as timer
import multiprocessing as mp

savedir = '/Users/yitan/Google Drive/My Drive/from_cannon/qmemory_simulation/data/ldpc_code'
command = ['python', '../src/gen_local_ldpc_code_bound_below.py', '--savedir', savedir]

'Local LDPC codes in 2D generated using KDTree algorithm'
ksmin_bit = [3]
ksmin_check = [3]
ns = np.arange(1000, 2050, 100)
ms = np.arange(1000, 2050, 100)
consts = [5]
seeds = range(0, 51)

def run_command(n, m, kmin_bit, kmin_check, const, seed):
    start = timer()
    print('='*80, flush=True)
    subcommand = command + ['--n', f'{n}', '--m', f'{m}', '--kmin_bit', f'{kmin_bit}', '--kmin_check', f'{kmin_check}', '--const', f'{const}', '--seed', f'{seed}']
    subprocess.run(subcommand)
    end = timer()
    print(f'Elapsed time for (n,m)=({n},{m}), (kmin_bit, kmin_check)=({kmin_bit},{kmin_check}), const={const} and seed={seed} (local, bound below): {end-start} seconds', flush=True)

if __name__ == '__main__':
    with mp.Pool(processes=mp.cpu_count()) as pool:
        for n, m in zip(ns, ms):
            for kmax_bit, kmax_check in zip(ksmin_bit, ksmin_check):
                for const in consts:
                    for seed in seeds:
                        pool.apply_async(run_command, args=(n, m, kmax_bit, kmax_check, const, seed))
        pool.close()
        pool.join()
