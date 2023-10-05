import numpy as np
import matplotlib.pyplot as plt
from ldpc.code_util import *
from bposd.css import *
import networkx as nx
from numba import njit
from timeit import default_timer as timer

'''Generate Bipartite configuration model (for a given degree sequence)
1. Start with a list of node degrees. For instance, the sequence [3, 2, 2, 1] means 
   we want node 1 to have 3 edges, node 2 to have 2 edges, and so on.
2. Each node gets as many "stubs" or "half-edges" as its degree.
3. Pair up these "half-edges" randomly to form the edges of the graph.
'''

nvars = 500
nchecks = 300
degvars = 6
degchecks = 10
assert nvars*degvars == nchecks*degchecks
seed = 0

####################################################################################################
# Helia's implementation
####################################################################################################
def random_biregular_graph(n, m, col_deg, row_deg):
    assert n * col_deg == m * row_deg
    N = n * col_deg
    rng = np.random.default_rng()
    success = False
    while not success:
      row_links = rng.choice(N, N, replace=False)
      col_links = rng.choice(N, N, replace=False)
      H = np.zeros((m, n))
      for i in range(N):
        if H[row_links[i]//row_deg][col_links[i]//col_deg] == 0:
          H[row_links[i]//row_deg][col_links[i]//col_deg] = 1
        else:
          break
        if i == N-1:
          success = True
    return H

####################################################################################################
# Networkx implementation
####################################################################################################

config_model = nx.bipartite.configuration_model([degchecks]*nchecks, [degvars]*nvars, seed=seed)
pc = nx.bipartite.biadjacency_matrix(config_model, row_order=range(nchecks))
print(pc.shape)


####################################################################################################
# Plot
####################################################################################################
top = nx.bipartite.sets(config_model)[0]
pos = nx.bipartite_layout(config_model, top)
# nx.draw(config_model, pos=pos, with_labels=True)
plt.spy(pc, markersize=1)
plt.show()
