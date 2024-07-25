import os, sys
import subprocess

sizes = range(11,21,1)
rs = [0.2, 0.4, 0.6, 0.8]
seeds = [0, 20]
outdir = '/n/home01/ytan/scratch/qmemory_simulation/output/'
readdir = '/n/home01/ytan/scratch/qmemory_simulation/data/rgg_code/'
savedir = '/n/home01/ytan/scratch/qmemory_simulation/data/rgg_code/'
for s in sizes: 
  for r in rs: 
    for seed in seeds: 
      file_name = f'sbatch_scripts/code_distance_rgg_code_size={s}_r={r}_seed={seed}.sh'
      subprocess.run(["sbatch", file_name])