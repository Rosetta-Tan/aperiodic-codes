import numpy as np
import matplotlib.pyplot as plt
import os
from timeit import default_timer as timer
from ldpc.mod2 import row_basis, nullspace, rank

def close(a, b):
    return np.linalg.norm(a-b) < 1e-4

def get_vertices(faces):
    vertices = []
    for face in faces:
        if len(face) == 4:
            vertices.append(face[1])
            vertices.append(face[2])
            vertices.append(face[3])
        elif len(face) == 5:
            vertices.append(face[1])
            vertices.append(face[2])
            vertices.append(face[3])
            vertices.append(face[4])
    vertices_new = []
    for v in vertices:
        if not any(close(v, v2) for v2 in vertices_new):
            vertices_new.append(v)
    return vertices_new

def vertices_on_face(face, vertices):
        vs_on_f = [face[0]] # ctg
        if len(face) == 4:
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
        elif len(face) == 5:
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

def draw(faces, vertices, edges):
    vertices_pos = [np.array([vertex.real, vertex.imag]) for vertex in vertices]
    fig, ax = plt.subplots()
    ax.scatter(np.array(vertices_pos)[:,0], np.array(vertices_pos)[:,1], marker='o', c='b')
    for edge in edges:
        ax.plot([edge[0].real, edge[1].real], [edge[0].imag, edge[1].imag], color='k', linewidth=0.5)
   
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    return fig, ax

def draw_qc_code_logical(faces, vertices, edges, faces_pos, h, logical_op):
    vertices = get_vertices(faces)
    vertices_pos = [np.array([vertex.real, vertex.imag]) for vertex in vertices]
    
    fig, ax = plt.subplots()
    ax.scatter(np.array(faces_pos)[:,0], np.array(faces_pos)[:,1], marker='s', c='r')

    ## degub
    # annotate faces
    for i in range(len(faces)):
        ax.annotate(str(i), (faces_pos[i][0], faces_pos[i][1]), fontsize=10, color='k')

    ax.scatter(np.array(vertices_pos)[:,0], np.array(vertices_pos)[:,1], marker='o', c='b')

    ## degub
    # annotate vertices
    for i in range(len(vertices)):
        ax.annotate(str(i), (vertices_pos[i][0], vertices_pos[i][1]), fontsize=10, color='purple')

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

def draw_qc_transposecode_logical(faces, vertices, edges, faces_pos, h, logical_op):
    assert len(logical_op) == len(faces)
    vertices = get_vertices(faces)
    vertices_pos = [np.array([vertex.real, vertex.imag]) for vertex in vertices]
    
    fig, ax = plt.subplots()
    ax.scatter(np.array(faces_pos)[:,0], np.array(faces_pos)[:,1], marker='o', c='b')
    ax.scatter(np.array(vertices_pos)[:,0], np.array(vertices_pos)[:,1], marker='s', c='r')
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