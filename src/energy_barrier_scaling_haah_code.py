import numpy as np
import matplotlib.pyplot as plt
plt.style.use('../figures/norm2.mplstyle')
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from sys import argv

"""
Data analysis of energy barrier.
"""

nsamples = 1000
Ls = [4, 5, 6, 7, 8, 9, 10]
energy_barrier_opts = []
for L in Ls:
    readpath = \
        f'/n/home01/ytan/scratch/qmemory_simulation/data/haah_code/energy_barrier_list_L{L}_nsamples{nsamples:.2e}.txt'
    data = np.loadtxt(readpath)
    peaks, _ = find_peaks(data)
    energy_barrier_opts.append(data[peaks[0]])

fig, ax = plt.subplots()
ax.plot(Ls, energy_barrier_opts, 'o-')
for i, txt in enumerate(energy_barrier_opts):
    ax.annotate(f'{txt}', (Ls[i], energy_barrier_opts[i]), textcoords="offset points", xytext=(0,10), ha='center')
ax.set_xlabel('L')
ax.set_ylabel('Energy barrier')
fig.tight_layout()
fig.savefig(f'/n/home01/ytan/qmemory_simulation/figures/haah_code/energy_barrier_scaling.png', dpi=300)
plt.show()