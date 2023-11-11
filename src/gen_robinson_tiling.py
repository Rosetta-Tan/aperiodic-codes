import numpy as np
import matplotlib.pyplot as plt
import os
from timeit import default_timer as timer
from ldpc.mod2 import row_basis, nullspace, rank
from helpers_distance import get_classical_code_distance_time_limit, get_classical_code_transpose_distance_special_treatment

def subdivide(triangles):
    result = []
    for ctg, A, B, C in triangles:
        if ctg == 0:
            P1 = B + (C-B)*(np.sqrt(5)-1)/2
            
            F1 = (0, C, A, P1)
            F2 = (1, B, P1, A)

            result += [F1, F2]
        elif ctg == 1:
            P1 = C + (A-C)*(np.sqrt(5)-1)/2
            P2 = C + (B-C)*(np.sqrt(5)-1)/2

            F1 = (1, B, P1, A)
            F2 = (0, P2, P1, B)
            F3 = (1, P1, P2, C)

            result += [F1, F2, F3]
    return result

def close(a, b):
    return np.linalg.norm(a-b) < 1e-5

def get_vertices(faces):
    vertices = []
    for face in faces:
        vertices.append(face[1])
        vertices.append(face[2])
        vertices.append(face[3])
    vertices_new = []
    for v in vertices:
        if not any(close(v, v2) for v2 in vertices_new):
            vertices_new.append(v)
    return vertices_new

def get_edges(faces, vertices):
    def vertices_on_face(face, vertices):
        vs_on_f = [face[0]] # ctg
        for v in vertices:
            if close(face[1], v):
                vs_on_f.append(v)
        for v in vertices:
            if close(face[2], v):
                vs_on_f.append(v)
        for v in vertices:
            if close(face[3], v):
                vs_on_f.append(v)
        return vs_on_f

    edges = []
    for face in faces:
        vs_on_f = vertices_on_face(face, vertices)
        edges.append((vs_on_f[1], vs_on_f[2]))
        edges.append((vs_on_f[2], vs_on_f[3]))
        edges.append((vs_on_f[3], vs_on_f[1])) 
    return edges

def get_geometric_center(face):
    return (face[1]+face[2]+face[3])/3

def get_qc_code(faces, vertices):
    h = np.zeros((len(faces), len(vertices)))
    for i, face in enumerate(faces):
        if face[0] == 0:
            for j in range(len(vertices)):
                if close(face[1], vertices[j]) or close(face[2], vertices[j]) or close(face[3], vertices[j]):
                    h[i,j] = 1
        if face[0] == 1:
             for j in range(len(vertices)):
                if close(face[1], vertices[j]) or close(face[2], vertices[j]) or close(face[3], vertices[j]) or close(face[1]+(np.sqrt(5)-1)/2*(face[3]-face[1]), vertices[j]) or close(face[3]+(np.sqrt(5)-1)/2*(face[1]-face[3]), vertices[j]):
                    h[i,j] = 1
    return h

def draw(faces, vertices):
    vertices_pos = [np.array([vertex.real, vertex.imag]) for vertex in vertices]
    edges = get_edges(faces, vertices)
    fig, ax = plt.subplots()
    ax.scatter(np.array(vertices_pos)[:,0], np.array(vertices_pos)[:,1], marker='o', c='b')
    edges = get_edges(faces, vertices)
    for edge in edges:
        ax.plot([edge[0].real, edge[1].real], [edge[0].imag, edge[1].imag], color='k', linewidth=0.5)
    # faces_pos = [np.array([get_geometric_center(face).real, get_geometric_center(face).imag]) for face in faces]
    # for i in range(len(faces)):
    #     for j in range(len(vertices)):
    #         if h[i,j] == 1:
    #             ax.plot([faces_pos[i][0], vertices_pos[j][0]], [faces_pos[i][1], vertices_pos[j][1]], color='gray', linewidth=3, zorder=-1)
   
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    return fig, ax

def get_classical_code_distance(h):
    if rank(h) == h.shape[1]:
        print('Code is full rank, no codewords')
        return np.inf
    else:
        start = timer()
        print('Code is not full rank, there are codewords')
        print('Computing codeword space basis ...')
        ker = nullspace(h)
        end = timer()
        print(f'Elapsed time for computing codeword space basis: {end-start} seconds', flush=True)
        print('len of ker: ', len(ker))
        print('Start finding minimum Hamming weight while buiding codeword space ...')
        start = end
        # @jit
        def find_min_weight_while_build(matrix):
            span = []
            min_hamming_weight = np.inf
            for ir, row in enumerate(matrix):
                print('debug: ir = ', ir, 'current min_hamming_weight = ', min_hamming_weight, flush=True)  # debug
                row_hamming_weight = np.sum(row)
                if row_hamming_weight < min_hamming_weight:
                    min_hamming_weight = row_hamming_weight
                temp = [row]
                for element in span:
                    newvec = (row + element) % 2
                    temp.append(newvec)
                    newvec_hamming_weight = np.sum(newvec)
                    if newvec_hamming_weight < min_hamming_weight:
                        min_hamming_weight = newvec_hamming_weight
                span = list(np.unique(temp + span, axis=0))
            assert len(span) == 2**len(matrix) - 1
            return min_hamming_weight
        min_hamming_weight = find_min_weight_while_build(ker)
        end = timer()
        print(f'Elapsed time for finding minimum Hamming weight while buiding codeword space : {end-start} seconds', flush=True)
        
        return min_hamming_weight

def draw_qc_code_logical(faces, vertices, h, logical_op):
    faces_pos = [np.array([get_geometric_center(face).real, get_geometric_center(face).imag]) for face in faces]
    vertices = get_vertices(faces)
    vertices_pos = [np.array([vertex.real, vertex.imag]) for vertex in vertices]
    
    fig, ax = plt.subplots()
    ax.scatter(np.array(faces_pos)[:,0], np.array(faces_pos)[:,1], marker='s', c='r')
    ax.scatter(np.array(vertices_pos)[:,0], np.array(vertices_pos)[:,1], marker='o', c='b')
    edges = get_edges(faces, vertices)
    for edge in edges:
        ax.plot([edge[0].real, edge[1].real], [edge[0].imag, edge[1].imag], color='k', linewidth=0.5)
    for i in range(len(faces)):
        for j in range(len(vertices)):
            if h[i,j] == 1:
                ax.plot([faces_pos[i][0], vertices_pos[j][0]], [faces_pos[i][1], vertices_pos[j][1]], color='gray', linewidth=3, zorder=-1)
    
    ones = [i for i in range(len(logical_op)) if logical_op[i] == 1]
    x = [vertices_pos[i][0] for i in ones]
    y = [vertices_pos[i][1] for i in ones]
    ax.scatter(x, y, marker='*', c='g', s=200, zorder=100)

    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    return fig, ax

def draw_qc_transposecode_logical(faces, vertices, h, logical_op):
    assert len(logical_op) == len(faces)
    faces_pos = [np.array([get_geometric_center(face).real, get_geometric_center(face).imag]) for face in faces]
    vertices = get_vertices(faces)
    vertices_pos = [np.array([vertex.real, vertex.imag]) for vertex in vertices]
    
    fig, ax = plt.subplots()
    ax.scatter(np.array(faces_pos)[:,0], np.array(faces_pos)[:,1], marker='o', c='b')
    ax.scatter(np.array(vertices_pos)[:,0], np.array(vertices_pos)[:,1], marker='s', c='r')
    edges = get_edges(faces, vertices)
    for edge in edges:
        ax.plot([edge[0].real, edge[1].real], [edge[0].imag, edge[1].imag], color='k', linewidth=0.5)
    # for i in range(len(vertices)):
    #     for j in range(len(faces)):
    #         if h[i,j] == 1:
    #             ax.plot([faces_pos[j][0], vertices_pos[i][0]], [faces_pos[j][1], vertices_pos[i][1]], color='gray', linewidth=3, zorder=-1)
    
    ones = [i for i in range(len(logical_op)) if logical_op[i] == 1]
    x = [faces_pos[i][0] for i in ones]
    y = [faces_pos[i][1] for i in ones]
    ax.scatter(x, y, marker='*', c='g', s=200, zorder=100)

    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    return fig, ax


gen = 6
triangles = []
ctg = 0
# A = -np.cos(2*np.pi/5)+0.j
# B = 0.+np.sin(2*np.pi/5)*1j
# C = np.cos(2*np.pi/5)+0.j
B = 0.+0.j
for i in range(1):
    A = np.exp(1j*np.pi/5*(i-1/2))
    C = np.exp(1j*np.pi/5*(i+1/2))
    if i % 2 == 1:
        A, C = C, A
    triangles.append((ctg, A, B, C))
for _ in range(gen):
    triangles = subdivide(triangles)
vertices = get_vertices(triangles)
h = get_qc_code(triangles, vertices)
h = h.T
m, n = h.shape
logical_basis = row_basis(nullspace(h))
k = len(logical_basis)

print('shape of h = ', h.shape)
print('k = ', k)
# print('d = ', get_classical_code_distance(h))

fig, ax = draw(triangles, vertices)
plt.show()

# # logical_coeffs = np.array(list(itertools.product([0, 1], repeat=k)))
# savedir = '/Users/yitan/Google Drive/My Drive/from_cannon/qmemory_simulation/data/qc_code/goodman_strauss'
# subdir = f'gen={gen}_special_treatment'
# if not os.path.exists(os.path.join(savedir, subdir)):
#     os.makedirs(os.path.join(savedir, subdir))


# # # random sample 1000 length k binary vectors
# # logical_coeffs = np.random.randint(2, size=(1000, k))
# # for i in range(logical_coeffs.shape[0])[:]:
# #     logical_op = np.matmul(logical_coeffs[i], logical_basis) % 2
# #     logical_op = logical_op.reshape((n,))
# #     fig, ax = draw_qc_code_p3_acute_triangle_boundary_logical(rhombs, vertices, h, logical_op)
# #     ax.set_title(f'logical operator {i}')
# #     savename = f'goodman_strauss_tiling_3pipver7_rhomb_boundary_logical_{i}.pdf'
# #     fig.savefig(os.path.join(savedir, subdir, savename), bbox_inches='tight')
    
savedir = '/Users/yitan/Google Drive/My Drive/from_cannon/qmemory_simulation/data/qc_code/robinson_tiling'
subdir = f'gen={gen}'
if not os.path.exists(os.path.join(savedir, subdir)):
    os.makedirs(os.path.join(savedir, subdir))

# d_bound, logical_op = get_classical_code_transpose_distance_special_treatment(h, gen=4, target_weight=9)
# print('d_bound = ', d_bound)
# fig, ax = draw_qc_transposecode_logical(triangles, vertices, h, logical_op)
# savename = f'transpose_low_weight_logical.pdf'
# # fig, ax = draw_qc_code_logical(triangles, vertices, h, logical_op)
# # savename = f'low_weight_logical.pdf'
# ax.set_title(f'low weight logical operator')
# fig.set_size_inches(12, 12)

# fig.savefig(os.path.join(savedir, subdir, savename), bbox_inches='tight')
# plt.show()