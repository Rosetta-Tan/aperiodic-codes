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
ds = [3, 4, 5, 6, 7, 8, 9]
energy_barrier_opts = []
for d in ds:
    readpath = \
        f'/n/home01/ytan/scratch/qmemory_simulation/data/toric_code/energy_barrier_list_d{d}_nsamples{nsamples:.2e}.txt'
    data = np.loadtxt(readpath)
    try:
        peaks, _ = find_peaks(data, height=0)
        energy_barrier_opts.append(data[peaks[0]])
    except IndexError:
        assert np.all(data == data[0])
        energy_barrier_opts.append(data[0])

fig, ax = plt.subplots()
ax.plot(ds, energy_barrier_opts, 'o-')
for i, txt in enumerate(energy_barrier_opts):
    ax.annotate(f'{txt}', (ds[i], energy_barrier_opts[i]), textcoords="offset points", xytext=(0,10), ha='center')
ax.set_xlabel('d')
ax.set_ylabel('Energy barrier')
fig.tight_layout()
fig.savefig(f'/n/home01/ytan/qmemory_simulation/figures/toric_code/energy_barrier_scaling.png', dpi=300)
plt.show()