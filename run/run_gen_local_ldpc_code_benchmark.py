import numpy as np
import subprocess
import sys, os
from timeit import default_timer as timer
import multiprocessing as mp

savedir = '/Users/yitan/Google Drive/My Drive/from_cannon/qmemory_simulation/data/ldpc_code'
command = ['python', '../src/gen_local_ldpc_code_benchmark.py', '--savedir', savedir]

'Local LDPC codes in 2D generated using KDTree algorithm'
proportionality = 0.1
ns = np.arange(100, 1050, 100)
ms = np.arange(100, 1050, 100)
consts = [6]
seeds = range(0, 101)

def run_command(n, m, proportionality, const, seed):
    start = timer()
    print('='*80, flush=True)
    subcommand = command + ['--n', f'{n}', '--m', f'{m}', '--proportionality', f'{proportionality}', '--const', f'{const}', '--seed', f'{seed}']
    subprocess.run(subcommand)
    end = timer()
    print(f'Elapsed time for (n,m)=({n},{m}), proportionality={proportionality}, const={const} and seed={seed} (local, bound below): {end-start} seconds', flush=True)

if __name__ == '__main__':
    with mp.Pool(processes=mp.cpu_count()) as pool:
        for n, m in zip(ns, ms):
            for const in consts:
                for seed in seeds:
                    pool.apply_async(run_command, args=(n, m, proportionality, const, seed))
        pool.close()
        pool.join()
