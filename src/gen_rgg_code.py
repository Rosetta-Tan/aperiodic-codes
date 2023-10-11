import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def gen_unipartite_rgg_code(n, d, p, seed=None):
    if seed is not None:
        np.random.seed(seed)
    pos = {i: np.random.uniform(size=d) for i in range(n)}
    G = nx.Graph()
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(pos[i]-pos[j]) < p:
                G.add_edge(i, j)
    return G, pos

def gen_bipartite_rgg_code(m, n, d, p, seed=None):
    if seed is not None:
        np.random.seed(seed)
    pos = {i: np.random.uniform(size=d) for i in range(n+m)}
    G = nx.Graph()
    for i in range(n):
        for j in range(n, n+m):
            if np.linalg.norm(pos[i]-pos[j]) < p:
                G.add_edge(i, j)
    return G, pos

def config_model_with_distance_bound(n, m, deg_bit, deg_check):
    G=nx.empty_graph(n+m)

    if not seed is None:
        random.seed(seed)    

    # length and sum of each sequence
    lena=len(aseq)
    lenb=len(bseq)
    suma=sum(aseq)
    sumb=sum(bseq)

    if not suma==sumb:
        raise networkx.NetworkXError(\
              'invalid degree sequences, sum(aseq)!=sum(bseq),%s,%s'\
              %(suma,sumb))

    G=_add_nodes_with_bipartite_label(G,lena,lenb)
                       
    if max(aseq)==0: return G  # done if no edges

    # build lists of degree-repeated vertex numbers
    stubs=[]
    stubs.extend([[v]*aseq[v] for v in range(0,lena)])  
    astubs=[]
    astubs=[x for subseq in stubs for x in subseq]

    stubs=[]
    stubs.extend([[v]*bseq[v-lena] for v in range(lena,lena+lenb)])  
    bstubs=[]
    bstubs=[x for subseq in stubs for x in subseq]

    # shuffle lists
    random.shuffle(astubs)
    random.shuffle(bstubs)

    G.add_edges_from([[astubs[i],bstubs[i]] for i in range(suma)])

    G.name="bipartite_configuration_model"
    return G

def plot_unipartite_graph(G, pos):
    nx.draw(G, pos, with_labels=True)
    plt.show()

def plot_bipartite_graph(G, n, m, pos):
    pass