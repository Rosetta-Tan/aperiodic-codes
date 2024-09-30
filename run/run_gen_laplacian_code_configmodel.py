import subprocess
import sys, os
from timeit import default_timer as timer
import multiprocessing as mp

savedir = '/Users/yitan/Google Drive/My Drive/from_cannon/qmemory_simulation/data/laplacian_code'
command = ['python', '../src/gen_laplacian_code.py', '--savedir', savedir]

'configuration model G(n, deg_sequence)'
ns = range(50, 501, 50)
dslo = range(5,11,1)
seeds = range(0, 101)

def run_command(subcommand, n, dlo, seed):
    print('='*80, flush=True)
    start = timer()
    subprocess.run(subcommand)
    end = timer()
    print(f'Elapsed time for n {n}, deg lowerbound {dlo} and seed {seed} (config): {end-start} seconds', flush=True)
    start = end
    subcommand_noprledge = subcommand + ['--noprledge']
    subprocess.run(subcommand_noprledge)
    end = timer()
    print(f'Elapsed time for n {n}, deg lowerbound {dlo} and seed {seed} (config noprledge): {end-start} seconds', flush=True)
    start = end
    subcommand_noselfloop = subcommand + ['--noselfloop']
    subprocess.run(subcommand_noselfloop)
    end = timer()
    print(f'Elapsed time for n {n}, deg lowerbound {dlo} and seed {seed} (config no selfloop): {end-start} seconds', flush=True)
    start = end
    subcommand_noprledge_noselfloop = subcommand + ['--noprledge', '--noselfloop']
    subprocess.run(subcommand_noprledge_noselfloop)
    end = timer()
    print(f'Elapsed time for n {n}, deg lowerbound {dlo} and seed {seed} (config noprledge no selfloop): {end-start} seconds', flush=True)
    start = end


if __name__ == '__main__':
    pool = mp.Pool(mp.cpu_count())
    for n in ns:
        for dlo in dslo:
            for seed in seeds:
                subcommand = command + ['config', '--n', f'{n}', '--deglo', f'{dlo}', '--degup', f'{dlo+2}', '--seed', f'{seed}']
                pool.apply_async(run_command, args=(subcommand, n, dlo, seed))
    pool.close()
    pool.join()
