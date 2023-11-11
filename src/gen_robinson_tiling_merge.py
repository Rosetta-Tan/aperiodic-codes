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
        if face[0] == '0-BC' or face[0] == '1-BC':
            return (face[2]+face[3])/2
        if face[0] == '0-AC' or face[0] == '1-AC':
            return (face[1]+face[3])/2
        if face[0] == '0-AB' or face[0] == '1-AB':
            return (face[1]+face[2])/2

def merge(triangle_list):
    """
    Remove triangles giving rise to identical rhombuses from the
    ensemble.
    """
    def judge_reflection_BC(triangle_i, triangle_j):
            color_i, A_i, B_i, C_i = triangle_i
            D_i = B_i + C_i - A_i
            color_j, A_j, B_j, C_j = triangle_j
            if color_i == color_j and close(D_i, A_j) and close(B_i, B_j) and close(C_i, C_j):
                return True
            elif color_i == color_j and close(D_i, A_j) and close(B_i, C_j) and close(C_i, B_j):
                return True
            else:
                return False

    def judge_mirror_AC(triangle_i, triangle_j):
        color_i, A_i, B_i, C_i = triangle_i
        D_i = 2*(A_i + (A_i-C_i)*np.cos(2*np.pi/5)) - B_i
        color_j, A_j, B_j, C_j = triangle_j
        if color_i == color_j and close(D_i, B_j) and close(A_i, A_j) and close(C_i, C_j):
            return True
        elif color_i == color_j and close(D_i, C_j) and close(A_i, A_j) and close(C_i, B_j):
            return True
        else:
            return False
        
    def judge_mirror_AB(triangle_i, triangle_j):
        color_i, A_i, B_i, C_i = triangle_i
        D_i = 2*(A_i + (A_i-B_i)*np.cos(2*np.pi/5)) - C_i
        color_j, A_j, B_j, C_j = triangle_j
        if color_i == color_j and close(D_i, C_j) and close(A_i, A_j) and close(B_i, B_j):
            return True
        elif color_i == color_j and close(D_i, B_j) and close(A_i, A_j) and close(B_i, C_j):
            return True
        else:
            return False
        
    def judge_reflection_AC(triangle_i, triangle_j):
        color_i, A_i, B_i, C_i = triangle_i
        D_i = A_i + C_i - B_i
        color_j, A_j, B_j, C_j = triangle_j
        if color_i == color_j and close(D_i, B_j) and close(A_i, C_j) and close(C_i, A_j):
            return True
        elif color_i == color_j and close(D_i, C_j) and close(A_i, B_j) and close(C_i, A_j):
            return True
        else:
            return False

    def judge_reflection_AB(triangle_i, triangle_j):
        color_i, A_i, B_i, C_i = triangle_i
        D_i = A_i + B_i - C_i
        color_j, A_j, B_j, C_j = triangle_j
        if color_i == color_j and close(D_i, C_j) and close(A_i, B_j) and close(B_i, A_j):
            return True
        elif color_i == color_j and close(D_i, B_j) and close(A_i, C_j) and close(B_i, A_j):
            return True
        else:
            return False
        
    merged_faces = []
    visited = []
    for i, triangle in enumerate(triangle_list):
        if not i in visited:
            color, A, B, C = triangle
            visited.append(i)
            if color == 1:
                found = False
                '''trying to find pair triangle reflected along BC'''
                for j in range(len(triangle_list)):
                    if j!=i and not j in visited and judge_reflection_BC(triangle, triangle_list[j]):
                        visited.append(j)
                        found = True
                        D = B + C - A
                        merged_faces.append(('1-BC', A, B, C, D))
                        break
                '''trying to find triangle mirror along AC'''
                if found == False:
                    for j in range(len(triangle_list)):
                        if j!=i and not j in visited and judge_mirror_AC(triangle, triangle_list[j]):
                            visited.append(j)
                            found = True
                            D = 2*(A + (A-C)*np.cos(2*np.pi/5)) - B 
                            merged_faces.append(('1-AC', A, B, C, D))
                            break
                '''trying to find triangle mirror along AB'''
                if found == False:
                    for j in range(len(triangle_list)):
                        if j!=i and not j in visited and judge_mirror_AB(triangle, triangle_list[j]):
                            visited.append(j)
                            found = True
                            D = 2*(A + (A-B)*np.cos(2*np.pi/5)) - C
                            merged_faces.append(('1-AB', A, B, C, D))
                            break
                '''Not found any triangle reflected on BC, AC, AB'''
                if found == False:
                    merged_faces.append((color, A, B, C))
           
            elif color == 0:
                found = False
                '''trying to find pair triangle reflected along BC'''
                for j in range(len(triangle_list)):
                    if j!=i and not j in visited and judge_reflection_BC(triangle, triangle_list[j]):
                        visited.append(j)
                        found = True
                        D = B + C - A
                        merged_faces.append(('0-BC', A, B, C, D))
                        break
                '''trying to find pair triangle reflected along AC'''
                if found == False:
                    for j in range(len(triangle_list)):
                        if j!=i and not j in visited and judge_reflection_AC(triangle, triangle_list[j]):
                            visited.append(j)
                            found = True
                            D = A + C - B
                            merged_faces.append(('0-AC', A, B, C, D))
                            break
                '''trying to find pair triangle reflected along AB'''
                if found == False:
                    for j in range(len(triangle_list)):
                        if j!=i and not j in visited and judge_reflection_AB(triangle, triangle_list[j]):
                            visited.append(j)
                            found = True
                            D = A + B - C
                            merged_faces.append(('0-AB', A, B, C, D))
                            break
                '''Not found any triangle reflected on BC, AC, AB'''
                if found == False:
                    merged_faces.append((color, A, B, C))
            
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
            if face[0] == '0-BC' or '1-BC':
                edges.append((vs_on_f[1], vs_on_f[2]))
                edges.append((vs_on_f[1], vs_on_f[3]))
                edges.append((vs_on_f[4], vs_on_f[2]))
                edges.append((vs_on_f[4], vs_on_f[3]))
            if face[0] == '1-AC':
                edges.append((vs_on_f[1], vs_on_f[4]))
                edges.append((vs_on_f[4], vs_on_f[3]))
                edges.append((vs_on_f[3], vs_on_f[2]))
                edges.append((vs_on_f[2], vs_on_f[1]))
            if face[0] == '1-AB':
                edges.append((vs_on_f[1], vs_on_f[4]))
                edges.append((vs_on_f[4], vs_on_f[2]))
                edges.append((vs_on_f[2], vs_on_f[3]))
                edges.append((vs_on_f[3], vs_on_f[1]))
            if face[0] == '0-AC':
                edges.append((vs_on_f[1], vs_on_f[2]))
                edges.append((vs_on_f[2], vs_on_f[3]))
                edges.append((vs_on_f[3], vs_on_f[4]))
                edges.append((vs_on_f[4], vs_on_f[1]))
            if face[0] == '0-AB':
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
                    if close(face[1], vertices[j]) or close(face[2], vertices[j]) or close(face[3], vertices[j]) or close(face[4], vertices[j]):
                        h[i,j] = 1
    return h

##############################################################################################################
gen = 6
triangles = []
for i in range(1):
    B = cmath.rect(1, (2*i) * np.pi / 10)
    C = cmath.rect(1, (2*i - 2) * np.pi / 10)
    triangles.append((0, 0j, B, C))
for _ in range(gen):
    triangles = subdivide(triangles)

faces = merge(triangles)
faces_pos = [np.array([get_geometric_center(face).real, get_geometric_center(face).imag]) for face in faces]

vertices = get_vertices(faces)
h = get_qc_code(faces, vertices)
# h = h.T
edges = get_edges(faces, vertices)
print('h.shape = ', h.shape)
# fig, ax = draw(triangles, vertices, edges)
logical_op = []
# d_bound, logical_op = get_classical_code_distance_special_treatment(h, target_weight=get_classical_code_distance_time_limit(h))
fig, ax = draw_qc_code_logical(faces, vertices, edges, faces_pos, h, logical_op)
# fig, ax = draw_qc_transposecode_logical(faces, vertices, edges, faces_pos, h, logical_op)
plt.show()
