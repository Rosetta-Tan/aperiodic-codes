import numpy as np
import cmath
import matplotlib.pyplot as plt
import os
from timeit import default_timer as timer
from ldpc.mod2 import row_basis, nullspace, rank
from helpers_distance import get_classical_code_distance_time_limit, get_classical_code_distance_special_treatment
from helpers_qc import *

goldenRatio = (1 + np.sqrt(5)) / 2

def subdivide(triangles):
        result = []
        for ctg, A, B, C in triangles:
            if ctg == 0:
                # Subdivide red triangle
                P = A + (B - A) / goldenRatio
                result += [(0, C, P, B), (1, P, C, A)]
            else:
                # Subdivide blue triangle
                Q = B + (A - B) / goldenRatio
                R = B + (C - B) / goldenRatio
                result += [(1, R, C, A), (1, Q, R, B), (0, R, Q, A)]
        return result

def get_edges(faces, vertices):
    edges = []
    for face in faces:
        vs_on_f = vertices_on_face(face, vertices)
        if len(face) == 4:
            edges.append((vs_on_f[1], vs_on_f[2]))
            edges.append((vs_on_f[2], vs_on_f[3]))
            edges.append((vs_on_f[3], vs_on_f[1]))
        elif len(face) == 5:
            edges.append((vs_on_f[1], vs_on_f[2]))
            edges.append((vs_on_f[1], vs_on_f[3]))
            edges.append((vs_on_f[4], vs_on_f[2]))
            edges.append((vs_on_f[4], vs_on_f[3]))
    return edges

def get_geometric_center(face):
    if len(face) == 4:
        # the geometric shape is a triangle
        return (face[1]+face[2]+face[3])/3
    elif len(face) == 5:
        # the geometric shape is a rhombus
        return (face[1]+face[2]+face[3]+face[4])/4

def is_on_boundary(pt):
    line_segments = []
    for i in range(10):
        B = cmath.rect(1, (2*i - 1) * np.pi / 10)
        C = cmath.rect(1, (2*i + 1) * np.pi / 10)
        line_segments.append((B,C))
    for line_segment in line_segments:
        if close(pt, line_segment[0]) or close(pt, line_segment[1]):
            return True
        else:
            # check if pt is on the line segment
            line = line_segment[1] - line_segment[0]
            diff = pt - line_segment[0]
            vec_line = np.array([line.real, line.imag])
            vec_diff = np.array([diff.real, diff.imag])
            if np.linalg.norm(np.cross(vec_line, vec_diff)) < 1e-5:
                return True            
    return False

def get_reflection_center(triangle):
    'center is defined as (B+C)/2 for both the red and blue triangles'
    return (triangle[2]+triangle[3])/2

def merge(triangle_list):
    """
    Remove triangles giving rise to identical rhombuses from the
    ensemble.
    """

    # Triangles give rise to identical rhombuses if these rhombuses have the same centre.
    reflection_centers = [get_reflection_center(t) for t in triangle_list]
    merged_faces = []
    visited = []
    for i, triangle in enumerate(triangle_list):
        if not i in visited:
            if is_on_boundary(reflection_centers[i]):
                merged_faces.append(triangle)
                visited.append(i)
            else:
                color, A, B, C = triangle
                D = B + C - A
                merged_faces.append((color, A, B, C, D))
                visited.append(i)
                found = False
                for j in range(len(triangle_list)):
                    if j!=i and \
                    triangle_list[j][0]==color and close(triangle_list[j][1], D) \
                    and ( (close(triangle_list[j][2], C) and close(triangle_list[j][3], B)) or (close(triangle_list[j][2], B) and close(triangle_list[j][3], C)) ):
                        visited.append(j)
                        found = True
                        break
                if found == False:
                    raise Exception('cannot find the other triangle')
    return merged_faces

def get_qc_code(faces, vertices):
    h = np.zeros((len(faces), len(vertices)))
    for i, face in enumerate(faces):
        for j in range(len(vertices)):
            if len(face) == 4:
                if close(face[1], vertices[j]) or close(face[2], vertices[j]) or close(face[3], vertices[j]):
                    h[i,j] = 1
            elif len(face) == 5:
                if close(face[1], vertices[j]) or close(face[2], vertices[j]) or close(face[3], vertices[j]) or close(face[4], vertices[j]):
                    h[i,j] = 1
    return h

##############################################################################################################

gen = 4
triangles = []
for i in range(10):
    B = cmath.rect(1, (2*i - 1) * np.pi / 10)
    C = cmath.rect(1, (2*i + 1) * np.pi / 10)
    if i % 2 == 0:
        B, C = C, B  # Make sure to mirror every second triangle
    triangles.append((0, 0j, B, C))
for _ in range(gen):
    triangles = subdivide(triangles)
faces = merge(triangles)
vertices = get_vertices(faces)
edges = get_edges(faces, vertices)
h = get_qc_code(faces, vertices)
# h = h.T
m, n = h.shape
print('m, n', m, n)
logical_basis = row_basis(nullspace(h))
k = len(logical_basis)

print('shape of h = ', h.shape)
print('k = ', k)
print('d = ', get_classical_code_distance_time_limit(h))

# fig, ax = draw(faces, vertices, edges)
# plt.show()

d_bound, logical_op = get_classical_code_distance_special_treatment(h, target_weight=get_classical_code_distance_time_limit(h))
print('d_bound = ', d_bound)
# fig, ax = draw_qc_transposecode_logical(faces, vertices, edges, h, logical_op)
# savename = f'transpose_low_weight_logical.pdf'
fig, ax = draw_qc_code_logical(faces, vertices, edges, h, logical_op)
savename = f'low_weight_logical.pdf'
ax.set_title(f'low weight logical operator')
fig.set_size_inches(12, 12)

# fig.savefig(os.path.join(savedir, subdir, savename), bbox_inches='tight')
plt.show()