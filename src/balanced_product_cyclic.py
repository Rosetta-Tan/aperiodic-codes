import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

cycorder = 15
####################################################################################################
# Load the parity check matrices
####################################################################################################
h1 = np.loadtxt('../data/h1_inputBalancedProduct.txt', dtype=int)
hx = np.loadtxt('../data/hx_outputBalancedProduct.txt', dtype=int)
hz = np.loadtxt('../data/hz_outputBalancedProduct.txt', dtype=int)
row_quotient = h1.shape[0]//cycorder
col_quotient = h1.shape[1]//cycorder

####################################################################################################
# check the symmetry of h1
####################################################################################################
'''The symmetry of h1
e.g., h1 is a 255x435 matrix, symmetry operations are translation by (17, 29)*k, k=0,1,2,...,14.
'''
for i in range(len(h1)):
    for j in range(len(h1[i])):
        # print(h1[i][j] == h1[(i+17)%255][(j+29)%435])
        assert h1[i][j] == h1[(i+17)%255][(j+29)%435]
h1_coarse_grained = h1.reshape((cycorder, h1.shape[0]//cycorder, cycorder, h1.shape[1]//cycorder))
for bi in range(cycorder):
    for bj in range(cycorder):
        assert np.all(h1_coarse_grained[bi, :, bj, :] == h1_coarse_grained[(bi-1)%cycorder, :, (bj-1)%cycorder, :])

'''Generate the quotient matrix of h1 by modding out the Zcycorder symmetry.
e.g., h1 is a 255x435 matrix, and we mod out the Z15 symmetry,
Originally there are (255+435)=690 nodes, now there are (255+435)//15=46 effective nodes.
The definition of quotient graph is following: https://doi.org/10.1063/1.5064375
'''
# effective_checks = [list(range(i, row_quotient*i)) for i in range(cycorder)]
effective_checks = 

####################################################################################################
# Reproduce hx
# (construct the basis of 
# double complex using anti-
# diagonal symmetry operations)
####################################################################################################
h1 = np.loadtxt('../data/h1_inputBalancedProduct.txt', dtype=int)
h2 = np.loadtxt('../data/h2_inputBalancedProduct.txt', dtype=int)
hx_my = np.zeros((255, 690), dtype=int)
hz_my = np.zeros((435, 690), dtype=int)

def find_orbit_inner(i1, j1, i2, j2, blockshape1):
    '''
    (i1, j1): coordinate of matrix h1, h1.T, I_m1 or I_n1, depending on the situation
    (i2, j2): coordinate of matrix h2, h2.T, I_m2 or I_n2, but all of them are cycorderxcycorder matrices
    '''
    bi1 = i1//blockshape1[0]
    bj1 = j1//blockshape1[1]
    ci1 = i1%blockshape1[0]
    cj1 = j1%blockshape1[1]
    orbital_i = (bi1 + i2)%cycorder
    orbital_j = (bj1 + j2)%cycorder
    return ci1, cj1, orbital_i, orbital_j

def build_hx_left():
    def find_qubits_involved_left(orbital_i, ci1):
        '''orbital_i: row index of the blocks of h1
           ci1: row index within each block of h1
           NOTE: the blocks of h1 are right-tilted-diagonal invariant
        '''
        orbits_inner_to_find = np.where(h1[0*17+ci1])[0]
        'The above line finds what columns indices in the ci1-th row within the first block of h1 are non-zero.'
        lst = []
        for col in orbits_inner_to_find:
            qubit = find_orbit_inner(0*17+ci1, col, orbital_i, orbital_i, blockshape1=(17, 29))
            if not (qubit in lst):
                lst.append(qubit)
        return lst
    hx_left = np.zeros((cycorder, 17, cycorder, 29), dtype=int)
    for orbital_i in range(cycorder):
        for ci1 in range(17):
            to_process = find_qubits_involved_left(orbital_i, ci1)
            for qubit in to_process:
                hx_left[orbital_i, ci1, qubit[3], qubit[1]] = 1
    return hx_left.reshape((255, 435))

def build_hx_right():
    '''hx_right is reduced from I_m1 \otimes h2.T by modding Zcycorder symmetry.
    '''
    def find_qubits_involved_right(orbital_i, ci1):
        '''orbital_i: row index of the blocks of h1
           ci1: row index within each block of h1
           NOTE: the blocks of h1 are right-tilted-diagonal invariant
        '''
        orbits_inner_to_find = np.where(h2.T[0])[0]
        'The above line finds what columns indices in the 0-th row of h2.T are non-zero.'
        lst = []
        for col in orbits_inner_to_find:
            qubit = find_orbit_inner(orbital_i*17+ci1, orbital_i*17+ci1, 0, col, blockshape1=(17, 17))
            if not (qubit in lst):
                lst.append(qubit)
        return lst
    hx_right = np.zeros((cycorder, 17, cycorder, 17), dtype=int)
    for orbital_i in range(cycorder):
        for ci1 in range(17):
            to_process = find_qubits_involved_right(orbital_i, ci1)
            for qubit in to_process:
                hx_right[orbital_i, ci1, qubit[3], qubit[1]] = 1
    return hx_right.reshape((255, 255))

hx_left = build_hx_left()
hx_right = build_hx_right()
print('hx_left correct? ', np.all(hx_left == hx[:,0:435]))
print('hx_right correct? ', np.all(hx_right == hx[:,435:690]))
    
####################################################################################################
# Reproduce hz
####################################################################################################

def build_hz_left():
    '''reducing from I_n1 \otimes h2'''
    def find_qubits_involved(orbital_i, ci1):
        '''orbital_i: row index of the blocks of h1
           ci1: row index within each block of h1
           NOTE: the blocks of h1 are right-tilted-diagonal invariant
        '''
        orbits_inner_to_find = np.where(h2[0])[0]
        'The above line finds what columns indices in the 0-th row of h2 are non-zero.'
        lst = []
        for col in orbits_inner_to_find:
            qubit = find_orbit_inner(orbital_i*29+ci1, orbital_i*29+ci1, 0, col, blockshape1=(29, 29))
            if not (qubit in lst):
                lst.append(qubit)
        return lst
    hz_left = np.zeros((cycorder, 29, cycorder, 29), dtype=int)
    for orbital_i in range(cycorder):
        for ci1 in range(29):
            to_process = find_qubits_involved(orbital_i, ci1)
            for qubit in to_process:
                hz_left[orbital_i, ci1, qubit[3], qubit[1]] = 1
    return hz_left.reshape((435, 435))

def build_hz_right():
    '''Reducing from h1.T \otimes I_m2'''
    def find_qubits_involved(orbital_i, ci1):
        '''orbital_i: row index of the blocks of h1
           ci1: row index within each block of h1
           NOTE: the blocks of h1 are right-tilted-diagonal invariant
        '''
        orbits_inner_to_find = np.where(h1.T[0*29+ci1])[0]
        'The above line finds what columns indices in the ci1-th row within the first block of h1.T are non-zero.'
        lst = []
        for col in orbits_inner_to_find:
            qubit = find_orbit_inner(0*29+ci1, col, orbital_i, orbital_i, blockshape1=(29, 17))
            if not (qubit in lst):
                lst.append(qubit)
        return lst
    hz_right = np.zeros((cycorder, 29, cycorder, 17), dtype=int)
    for orbital_i in range(cycorder):
        for ci1 in range(29):
            to_process = find_qubits_involved(orbital_i, ci1)
            for qubit in to_process:
                hz_right[orbital_i, ci1, qubit[3], qubit[1]] = 1
    return hz_right.reshape((435, 255))

hz_left = build_hz_left()
hz_right = build_hz_right()
print('hz_left correct? ', np.all(hz_left == hz[:,0:435]))
print('hz_right correct? ', np.all(hz_right == hz[:,435:690]))

####################################################################################################
# Plot
####################################################################################################

'''Plot the spy plot of h1.'''
fig, ax = plt.subplots()
ax.spy(h1)
ax.set_title('h1 spy plot')
fig.savefig('../figures/spy_h1_inputBalancedProduct.png')
plt.close(fig)

'''Plot the Tanner graph of h1'''
g1 = nx.Graph()
g1.add_nodes_from(range(h1.shape[0]+h1.shape[1]))
for i in range(len(h1)):
    for j in range(len(h1[i])):
        if h1[i][j] == 1:
            g1.add_edge(i, j+h1.shape[0])            
top = nx.bipartite.sets(g1)[0]
pos = nx.bipartite_layout(g1, top)
'''change the nodes range(0,h1.shape[0]), i.e., checks, to be red and square'''
nodes = range(h1.shape[0])
fig, ax = plt.subplots()
nx.draw(g1, pos=pos, node_size=10, width=0.1)
nx.draw_networkx_nodes(g1, pos, nodelist=nodes, node_color='r', node_shape='s', node_size=10)
ax.set_title('Tanner graph of h1')
fig.savefig('../figures/tanner_h1_inputBalancedProduct.png')
# plt.close(fig)

'''Visualize the symmetry of h1 (matrix way)'''
grid = np.zeros((cycorder, cycorder), dtype=int)
for shift in range(cycorder):
    start = grid[0, shift]
    for i in range(cycorder):
        grid[i, (i+shift)%cycorder] = shift
fig, ax = plt.subplots()
cmap = plt.get_cmap('tab20')
ax.imshow(grid, cmap=cmap)
ax.set_title('Symmetry of h1')
fig.savefig('../figures/symmetry_h1_inputBalancedProduct.png')
plt.close(fig)

plt.show()