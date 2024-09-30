import numpy as np
import cmath
import matplotlib.pyplot as plt
import os
from timeit import default_timer as timer
from ldpc.mod2 import row_basis, nullspace, rank
from helpers_distance import get_classical_code_distance_time_limit, get_classical_code_distance_special_treatment
from helpers_qc import *

def subdivide(triangles):
    result = []
    for ctg, A, B, C in triangles:
        if ctg == 0:
            P1 = A + (C-A)*(np.sqrt(5)-1)/2
            
            F1 = (0, B, C, P1)
            F2 = (1, P1, A, B)

            result += [F1, F2]
        elif ctg == 1:
            P1 = C + (B-C)*(np.sqrt(5)-1)/2
            P2 = C + (A-C)*(np.sqrt(5)-1)/2

            F1 = (1, P1, A, B)
            F2 = (0, P1, P2, A)
            F3 = (1, P2, P1, C)

            result += [F1, F2, F3]
    return result

def close(a, b):
    return np.linalg.norm(a-b) < 1e-5
    
def get_geometric_center(face):
    if len(face) == 4:
        return (face[1]+face[2]+face[3])/3
    elif len(face) == 5:
        if face[0] == '1AC':
            return (face[1]+face[3])/2
        if face[0] == '1AB':
            return (face[1]+face[2])/2        

def merge(triangle_list):
    """
    Remove triangles giving rise to identical rhombuses from the
    ensemble.
    """
    def detect_merge_1AC(triangle_i, triangle_j):
        color_i, A_i, B_i, C_i = triangle_i
        color_j, A_j, B_j, C_j = triangle_j
        if color_i != 1 or color_j != 0:
            return False
        elif color_i == 1 and color_j == 0:
            D_i = A_i + (np.sqrt(5)-1)/2 * (C_i-B_i)
            if close(D_i, B_j) and close(A_i, A_j) and close(C_i, C_j):
                return True
            elif close(D_i, C_j) and close(A_i, A_j) and close(C_i, B_j):
                return True
            else:
                return False
                        
    def detect_merge_1AB(triangle_i, triangle_j):
        color_i, A_i, B_i, C_i = triangle_i
        color_j, A_j, B_j, C_j = triangle_j
        if color_i == color_j:
            return False
        elif color_i == 1 and color_j == 0:
            D_i = A_i + (np.sqrt(5)-1)/2 * (B_i-C_i)
            if close(D_i, C_j) and close(A_i, A_j) and close(B_i, B_j):
                return True
            elif close(D_i, B_j) and close(A_i, A_j) and close(B_i, C_j):
                return True
            else:
                return False     
    
    # def detect_merge_0AB_1AC(triangle_i, triangle_j):
    #     color_i, A_i, B_i, C_i = triangle_i
    #     color_j, A_j, B_j, C_j = triangle_j
    #     if color_i == color_j:
    #         return False
    #     elif color_i == 1 and color_j == 0:
    #         D_i = A_i + (np.sqrt(5)-1)/2 * (C_i-B_i)
    #         if close(D_i, C_j) and close(A_i, A_j) and close(C_i, B_j):
    #             return True
    #         else:
    #             return False
    #     elif color_i == 0 and color_j == 1:
    #         D_j = A_j + (np.sqrt(5)-1)/2 * (C_j-B_j)
    #         if close(C_i, D_j) and close(A_i, A_j) and close(B_i, C_j):
    #             return True
    #         else:
    #             return False
    
    # def detect_merge_0AC_1AB(triangle_i, triangle_j):
    #     color_i, A_i, B_i, C_i = triangle_i
    #     color_j, A_j, B_j, C_j = triangle_j
    #     if color_i == color_j:
    #         return False
    #     elif color_i == 1 and color_j == 0:
    #         D_i = A_i + (np.sqrt(5)-1)/2 * (B_i-C_i)
    #         if close(D_i, B_j) and close(A_i, A_j) and close(B_i, C_j):
    #             return True
    #         else:
    #             return False
    #     elif color_i == 0 and color_j == 1:
    #         D_j = A_j + (np.sqrt(5)-1)/2 * (C_j-B_j)
    #         if close(B_i, D_j) and close(A_i, A_j) and close(C_i, B_j):
    #             return True
    #         else:
    #             return False
        
    merged_faces = []
    visited = []
    for i, triangle in enumerate(triangle_list):
        if not i in visited and triangle[0] == 1:  # go over all type 1 (yellow) triangles first
            color, A, B, C = triangle            
            visited.append(i)
            found = False
            '''trying to find pair triangle of type 1AC'''
            for j in range(len(triangle_list)):
                if not j in visited and detect_merge_1AC(triangle, triangle_list[j]):
                    visited.append(j)
                    found = True
                    D = A + (np.sqrt(5)-1)/2 * (C-B)
                    merged_faces.append(('1AC', A, B, C, D))
                    break
            '''Trying to find pair triangle of type 1AB'''
            if not found:
                for j in range(len(triangle_list)):
                    if not j in visited and detect_merge_1AB(triangle, triangle_list[j]):
                        visited.append(j)
                        found = True
                        D = A + (np.sqrt(5)-1)/2 * (B-C)
                        merged_faces.append(('1AB', A, B, C, D))
                        break
            '''Not found any pairable green triangle'''
            if not found:
                merged_faces.append((color, A, B, C))
    for i, triangle in enumerate(triangle_list):
        if not i in visited and triangle[0] == 0:  # go over all type 0 triangles
            # all the mergable green triangles have been merged
            visited.append(i)
            merged_faces.append(triangle)
    assert len(visited) == len(triangle_list)
    return merged_faces

def get_edges(faces, vertices):
    edges = []
    for i, face in enumerate(faces):
        vs_on_f = vertices_on_face(face, vertices)
        if len(face) == 4:
            edges.append((vs_on_f[1], vs_on_f[2]))
            edges.append((vs_on_f[2], vs_on_f[3]))
            edges.append((vs_on_f[3], vs_on_f[1]))
        elif len(face) == 5:
            if face[0] == '1AC':
                edges.append((vs_on_f[1], vs_on_f[2]))
                edges.append((vs_on_f[2], vs_on_f[3]))
                edges.append((vs_on_f[3], vs_on_f[4]))
                edges.append((vs_on_f[4], vs_on_f[1]))
            if face[0] == '1AB':
                edges.append((vs_on_f[1], vs_on_f[3]))
                edges.append((vs_on_f[3], vs_on_f[2]))
                edges.append((vs_on_f[2], vs_on_f[4]))
                edges.append((vs_on_f[4], vs_on_f[1]))
    return edges

def get_qc_code(faces, vertices):
    h = np.zeros((len(faces), len(vertices)))
    for i, face in enumerate(faces):
        if len(face) == 4:
            if face[0] == 0:
                for j in range(len(vertices)):
                    if close(face[1], vertices[j]) or close(face[2], vertices[j]) or close(face[3], vertices[j]):
                        h[i,j] = 1
            if face[0] == 1:
                for j in range(len(vertices)):
                    if close(face[1], vertices[j]) or close(face[2], vertices[j]) or close(face[3], vertices[j]) or close(face[2]+(np.sqrt(5)-1)/2*(face[3]-face[2]), vertices[j]) or close(face[3]+(np.sqrt(5)-1)/2*(face[2]-face[3]), vertices[j]):
                        h[i,j] = 1
        elif len(face) == 5:
            for j in range(len(vertices)):
                    if close(face[1], vertices[j]) or close(face[2], vertices[j]) or close(face[3], vertices[j]) or close(face[4], vertices[j]) or close(face[2]+(np.sqrt(5)-1)/2*(face[3]-face[2]), vertices[j]) or close(face[3]+(np.sqrt(5)-1)/2*(face[2]-face[3]), vertices[j]):
                        h[i,j] = 1
    return h

##############################################################################################################
gen = 4
triangles = []
for i in range(10):
    B = cmath.rect(1, (2*i) * np.pi / 10)
    C = cmath.rect(1, (2*i - 2) * np.pi / 10)
    triangles.append((0, 0j, B, C))
for _ in range(gen):
    triangles = subdivide(triangles)

faces = merge(triangles)
faces_pos = [np.array([get_geometric_center(face).real, get_geometric_center(face).imag]) for face in faces]

vertices = get_vertices(faces)
h = get_qc_code(faces, vertices)
h = h.T
edges = get_edges(faces, vertices)
print('h.shape = ', h.shape)
# fig, ax = draw(triangles, vertices, edges)
logical_op = []
d_bound, logical_op = get_classical_code_distance_special_treatment(h, target_weight=get_classical_code_distance_time_limit(h))
# fig, ax = draw_qc_code_logical(faces, vertices, edges, faces_pos, h, logical_op)
fig, ax = draw_qc_transposecode_logical(faces, vertices, edges, faces_pos, h, logical_op)
plt.show()
