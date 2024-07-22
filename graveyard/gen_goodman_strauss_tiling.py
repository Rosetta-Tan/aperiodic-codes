import numpy as np
import matplotlib.pyplot as plt
import os
from timeit import default_timer as timer
from ldpc.mod2 import row_basis, nullspace, rank

def subdivide(rhombs):
    result = []
    scale_factor = 0.5/(1+np.cos(np.pi/7))

    for ctg, A, B, C, D in rhombs:
        base_vector = B-A
        if ctg == 0:  # 5pi/7 rhomb
            l0 = scale_factor*base_vector*np.exp(-1j*np.pi/7)
            l1 = scale_factor*base_vector
            l2 = scale_factor*base_vector*np.exp(-1j*2*np.pi/7)
            l3 = scale_factor*base_vector*np.exp(1j*np.pi/7)
            l4 = scale_factor*base_vector*np.exp(-1j*3*np.pi/7)

            P1 = A + l1
            P2 = A + l2
            P3 = P1 + l3
            P4 = P1 + l2
            P5 = P2 + l0
            P6 = P3 + l0
            P7 = P3 + l2
            P8 = P4 + l0
            P9 = P5 + l4
            P10 = P7 + l0
            P11 = P8 + l4
            P12 = P10 + l1
            P13 = P10 + l0
            P14 = P10 + l4
            P15 = P11 + l2
            P16 = P13 + l1
            P17 = P13 + l4
            P18 = P14 + l2
            P19 = P17 + l1
            P20 = P17 + l2

            F1 = (0, A, P1 ,P4, P2)
            F2 = (1, P3, P7, P4, P1)
            F3 = (2, P5, P2, P4, P8)
            F4 = (2, P6, P10, P7, P3)
            F5 = (0, P10, P8, P4, P7)
            F6 = (1, P9, P5, P8, P11)
            F7 = (0, P6, B, P12, P10)
            F8 = (1, P14, P11, P8, P10)
            F9 = (0, P9, P11, P15, D)
            F10 = (2, P13, P10, P12, P16)
            F11 = (0, P17, P14, P10, P13)
            F12 = (1, P14, P18, P15, P11)
            F13 = (1, P17, P13, P16, P19)
            F14 = (2, P17, P20, P18, P14)
            F15 = (0, P17, P19, C, P20)

            result += [F1, F2, F3, F4, F5, F6, F7, F8, F9, F10,
                            F11, F12, F13, F14, F15]
        elif ctg == 1:  # 3pi/7 rhomb
            l1 = scale_factor*base_vector*np.exp(-1j*np.pi/7)
            l2 = scale_factor*base_vector*np.exp(-1j*3*np.pi/7)
            l3 = scale_factor*base_vector
            l4 = scale_factor*base_vector*np.exp(-1j*4*np.pi/7)
            l5 = scale_factor*base_vector*np.exp(1j*np.pi/7)
            l6 = scale_factor*base_vector*np.exp(-1j*5*np.pi/7)

            P1 = A + l3
            P2 = A + l4
            P3 = P1 + l5
            P4 = P1 + l4
            P5 = P2 + l2
            P6 = P5 + l6
            P7 = P3 + l1
            P8 = P3 + l4
            P9 = P4 + l2
            P10 = P7 + l4
            P11 = P8 + l2
            P12 = P9 + l6
            P13 = B + l4
            P14 = P10 + l2
            P15 = P11 + l6
            P16 = P13 + l2
            P17 = P14 + l6
            P18 = P15 + l4
            P19 = P12 + l4
            P20 = P16 + l6
            P21 = P17 + l4
            
            F1 = (1, A, P1, P4, P2)
            F2 = (0, P3, P8, P4, P1)
            F3 = (1, P5, P2, P4, P9)
            F4 = (1, P7, P10, P8, P3)
            F5 = (1, P11, P9, P4, P8)
            F6 = (0, P6, P5, P9, P12)
            F7 = (1, P7, B, P13, P10)
            F8 = (0, P14, P11, P8, P10)
            F9 = (2, P15, P12, P9, P11)
            F10 = (1, P6, P12, P19, D)
            F11 = (1, P14, P10, P13, P16)
            F12 = (1, P17, P15, P11, P14)
            F13 = (0, P15, P18, P19, P12)
            F14 = (0, P17, P14, P16, P20)
            F15 = (1, P17, P21, P18, P15)
            F16 = (1, P17, P20, C, P21)
            result += [F1, F2, F3, F4, F5, F6, F7, F8, F9, F10,
                          F11, F12, F13, F14, F15, F16]
        else:  # pi/7 rhomb
            l1 = scale_factor*base_vector*np.exp(-1j*np.pi/7)
            l2 = scale_factor*base_vector*np.exp(-1j*5*np.pi/7)
            l3 = scale_factor*base_vector
            l4 = scale_factor*base_vector*np.exp(-1j*6*np.pi/7)

            P3 = A + l3
            P2 = P3 - l4
            P1 = P2 + l1
            P4 = A + l4
            P5 = D - l4
            P6 = A + l2
            P7 = P1 + l4
            P8 = P3 + l2
            P9 = P6 + l4
            P10 = P9 + l3
            P11 = B + l4
            P12 = P11 + l2
            P13 = P12 - l3
            P14 = C - l3
            P15 = D + l3

            F1 = (0, P1, P7, P3, P2)
            F2 = (0, P6, A, P3, P8)
            F3 = (2, P6, P9, P4, A)
            F4 = (2, P1, B, P11, P7)
            F5 = (1, P13, P8, P3, P7)
            F6 = (2, P6, P8, P10, P9)
            F7 = (2, P5, P9, P15, D)
            F8 = (0, P13, P7, P11, P12)
            F9 = (0, P13, C, P10, P8)
            F10 = (2, P14, P9, P10, C)
            result += [F1, F2, F3, F4, F5, F6, F7, F8, F9, F10]
    return result

def close(a, b):
    return np.linalg.norm(a-b) < 1e-5

def get_vertices(rhombs):
    vertices = []
    for rhomb in rhombs:
        vertices.append(rhomb[1])
        vertices.append(rhomb[2])
        vertices.append(rhomb[3])
        vertices.append(rhomb[4])
    vertices_new = []
    for v in vertices:
        if not any(close(v, v2) for v2 in vertices_new):
            vertices_new.append(v)
    return vertices_new

def get_edges(faces, vertices):
    def vertices_on_face(face, vertices):
        vs_on_f = [face[0]] # color
        for v in vertices:
            if close(face[1], v):
                vs_on_f.append(v)
        for v in vertices:
            if close(face[2], v):
                vs_on_f.append(v)
        for v in vertices:
            if close(face[3], v):
                vs_on_f.append(v)
        for v in vertices:
            if close(face[4], v):
                vs_on_f.append(v)
        return vs_on_f

    edges = []
    for face in faces:
        vs_on_f = vertices_on_face(face, vertices)
        edges.append((vs_on_f[1], vs_on_f[2]))
        edges.append((vs_on_f[2], vs_on_f[3]))
        edges.append((vs_on_f[3], vs_on_f[4]))
        edges.append((vs_on_f[4], vs_on_f[1]))
    return edges

def get_geometric_center(face):
    return (face[1]+face[3])/2

def get_qc_code_goodmanstrauss(faces, vertices):
    h = np.zeros((len(faces), len(vertices)))
    for i, face in enumerate(faces):
        for j in range(len(vertices)):
            if close(face[1], vertices[j]) or close(face[2], vertices[j]) or close(face[3], vertices[j]) or close(face[4], vertices[j]):
                h[i,j] = 1
    return h

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

def draw_qc_code_p3_acute_triangle_boundary_logical(faces, vertices, h, logical_op):
    faces_pos = [np.array([get_geometric_center(face).real, get_geometric_center(face).imag]) for face in faces]
    vertices = get_vertices(faces)
    vertices_pos = [np.array([vertex.real, vertex.imag]) for vertex in vertices]
    h = np.zeros((len(faces), len(vertices)))
    for i, face in enumerate(faces):
        for j in range(len(vertices)):
            if len(face) == 4:
                if close(face[1], vertices[j]) or close(face[2], vertices[j]) or close(face[3], vertices[j]):
                    h[i,j] = 1
            elif len(face) == 5:
                if close(face[1], vertices[j]) or close(face[2], vertices[j]) or close(face[3], vertices[j]) or close(face[4], vertices[j]):
                    h[i,j] = 1
    
    fig, ax = plt.subplots()
    ax.scatter(np.array(faces_pos)[:,0], np.array(faces_pos)[:,1], marker='s', c='r')
    ax.scatter(np.array(vertices_pos)[:,0], np.array(vertices_pos)[:,1], marker='o', c='b')
    edges = get_edges(faces, vertices)
    for edge in edges:
        ax.plot([edge[0].real, edge[1].real], [edge[0].imag, edge[1].imag], color='k', linewidth=0.5)
    # for i in range(len(faces)):
        # for j in range(len(vertices)):
        #     if h[i,j] == 1:
        #         ax.plot([faces_pos[i][0], vertices_pos[j][0]], [faces_pos[i][1], vertices_pos[j][1]], color='gray', linewidth=3, zorder=-1)

    
    ones = [i for i in range(len(logical_op)) if logical_op[i] == 1]
    x = [vertices_pos[i][0] for i in ones]
    y = [vertices_pos[i][1] for i in ones]
    ax.scatter(x, y, marker='*', c='g', s=200, zorder=100)

    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    return fig, ax

gen = 3
rhombs = []

# ctg = 0
# A = 0.+0.j
# B = np.exp(1j*np.pi/7)
# C = 2*np.cos(np.pi/7)+0.j
# D = np.exp(-1j*np.pi/7)

ctg = 1
A = 0.+0.j
B = np.exp(1j*2*np.pi/7)
C = 2*np.cos(2*np.pi/7)+0.j
D = np.exp(-1j*2*np.pi/7)

# ctg = 2
# A = 0.+0.j
# B = np.exp(1j*3*np.pi/7)
# C = 2*np.cos(3*np.pi/7)+0.j
# D = np.exp(-1j*3*np.pi/7)

rhombs.append((ctg, A, B, C, D))
for _ in range(gen):
    rhombs = subdivide(rhombs)
# draw_goodmanstrauss(rhombs, show=True, store=False)
vertices = get_vertices(rhombs)
h = get_qc_code_goodmanstrauss(rhombs, vertices)
m, n = h.shape
logical_basis = row_basis(nullspace(h))
k = len(logical_basis)

print('shape of h = ', h.shape)
print('k = ', k)

# logical_coeffs = np.array(list(itertools.product([0, 1], repeat=k)))
savedir = '/Users/yitan/Google Drive/My Drive/from_cannon/qmemory_simulation/data/qc_code/goodman_strauss'
subdir = f'gen={gen}_special_treatment'
if not os.path.exists(os.path.join(savedir, subdir)):
    os.makedirs(os.path.join(savedir, subdir))


# # random sample 1000 length k binary vectors
# logical_coeffs = np.random.randint(2, size=(1000, k))
# for i in range(logical_coeffs.shape[0])[:]:
#     logical_op = np.matmul(logical_coeffs[i], logical_basis) % 2
#     logical_op = logical_op.reshape((n,))
#     fig, ax = draw_qc_code_p3_acute_triangle_boundary_logical(rhombs, vertices, h, logical_op)
#     ax.set_title(f'logical operator {i}')
#     savename = f'goodman_strauss_tiling_3pipver7_rhomb_boundary_logical_{i}.pdf'
#     fig.savefig(os.path.join(savedir, subdir, savename), bbox_inches='tight')

def get_classical_code_distance_special_treatment(h, gen, target_weight):
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
                    if min_hamming_weight <= target_weight:
                        return min_hamming_weight, newvec
                span = list(np.unique(temp + span, axis=0))
            assert len(span) == 2**len(matrix) - 1
            return min_hamming_weight
        min_hamming_weight, logical_op = find_min_weight_while_build(ker)
        end = timer()
        print(f'Elapsed time for finding minimum Hamming weight while buiding codeword space : {end-start} seconds', flush=True)
        return min_hamming_weight, logical_op

# print('d = ', get_classical_code_distance(h))
d_bound, logical_op = get_classical_code_distance_special_treatment(h, gen=3, target_weight=49)
print('d_bound = ', d_bound)
fig, ax = draw_qc_code_p3_acute_triangle_boundary_logical(rhombs, vertices, h, logical_op)
ax.set_title(f'low weight logical operator')
savename = f'goodman_strauss_tiling_3piover7_rhomb_boundary_low_weight_logical.pdf'
fig.savefig(os.path.join(savedir, subdir, savename), bbox_inches='tight')
plt.show()
