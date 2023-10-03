import numpy as np
import matplotlib.pyplot as plt
plt.style.use('../figures/norm2.mplstyle')

'''
Monte Carlo simulation for coherence time of
stabilizer codes subject to thermal noise.

Jump operators:
- Jump: |psi(t+dt)> = V |psi(t)>
    - probability: p(V) = gamma(dE)dt = -dE*dt/(1-exp(beta*dE))
    - |psi(t+dtr) = V |psi(t)> / sqrt(1-dE)
- 
'''