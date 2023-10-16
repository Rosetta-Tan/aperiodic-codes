import os, sys

sizes = range(11,21,1)
rs = [0.2, 0.4, 0.6, 0.8]
seeds = [0, 20]
outdir = '/n/home01/ytan/scratch/qmemory_simulation/output/'
readdir = '/n/home01/ytan/scratch/qmemory_simulation/data/rgg_code/'
savedir = '/n/home01/ytan/scratch/qmemory_simulation/data/rgg_code/'

for s in sizes:
  for r in rs:
    for seed in seeds:
      file_name = os.path.join(outdir, f'bash_scripts/code_distance_rgg_code_size={s}_r={r}_seed={seed}.sh')
        with open (file_name, 'w') as rsh:
          rsh.write('''\
#!/bin/bash -l
#SBATCH -n 1                # Number of cores
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH -t 0-2:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p cpu_requeue,shared	    # Partition to submit to
#SBATCH --mem=100G          # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o /n/home01/ytan/scratch/qmemory_simulation/output/myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e /n/home01/ytan/scratch/qmemory_simulation/output/myoutput_%j.err  # File to which STDERR will be written, %j inserts jobid

mamba activate qec_numerics
''')
          rsh.write(f"python /n/home01/ytan/qmemory_simulation/src/code_distance_rgg_code.py --size {s} --radius {r} --seed {seed} --readdir {readdir} --savedir {savedir}\n")