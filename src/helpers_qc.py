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
        num_vertices = len(face) - 1
        for i in range(num_vertices):
            vertices.append(face[i+1])
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
        elif len(face) == 6:
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
            for v in vertices:
                if close(face[5], v):
                    vs_on_f.append(v)
            return vs_on_f
        elif len(face) == 7:
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
            for v in vertices:
                if close(face[5], v):
                    vs_on_f.append(v)
            for v in vertices:
                if close(face[6], v):
                    vs_on_f.append(v)        
            return vs_on_f
        
def vertices_on_face_repr_indices(face, vertices):
        vs_on_f_indices = [face[0]] # ctg
        if len(face) == 4:
            for iv, v in enumerate(vertices):
                if close(face[1], v):
                    vs_on_f_indices.append(iv)
            for iv, v in enumerate(vertices):
                if close(face[2], v):
                    vs_on_f_indices.append(iv)
            for iv, v in enumerate(vertices):
                if close(face[3], v):
                    vs_on_f_indices.append(iv)
            return vs_on_f_indices
        elif len(face) == 5:
            for iv, v in enumerate(vertices):
                if close(face[1], v):
                    vs_on_f_indices.append(iv)
            for iv, v in enumerate(vertices):
                if close(face[2], v):
                    vs_on_f_indices.append(iv)
            for iv, v in enumerate(vertices):
                if close(face[3], v):
                    vs_on_f_indices.append(iv)
            for iv, v in enumerate(vertices):
                if close(face[4], v):
                    vs_on_f_indices.append(iv)
            return vs_on_f_indices
        elif len(face) == 6:
            for iv, v in enumerate(vertices):
                if close(face[1], v):
                    vs_on_f_indices.append(iv)
            for iv, v in enumerate(vertices):
                if close(face[2], v):
                    vs_on_f_indices.append(iv)
            for iv, v in enumerate(vertices):
                if close(face[3], v):
                    vs_on_f_indices.append(iv)
            for iv, v in enumerate(vertices):
                if close(face[4], v):
                    vs_on_f_indices.append(iv)
            for iv, v in enumerate(vertices):
                if close(face[5], v):
                    vs_on_f_indices.append(iv)
            return vs_on_f_indices
        elif len(face) == 7:
            for iv, v in enumerate(vertices):
                if close(face[1], v):
                    vs_on_f_indices.append(iv)
            for iv, v in enumerate(vertices):
                if close(face[2], v):
                    vs_on_f_indices.append(iv)
            for iv, v in enumerate(vertices):
                if close(face[3], v):
                    vs_on_f_indices.append(iv)
            for iv, v in enumerate(vertices):
                if close(face[4], v):
                    vs_on_f_indices.append(iv)
            for iv, v in enumerate(vertices):
                if close(face[5], v):
                    vs_on_f_indices.append(iv)
            for iv, v in enumerate(vertices):
                if close(face[6], v):
                    vs_on_f_indices.append(iv)
            return vs_on_f_indices

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
    ax.scatter(np.array(vertices_pos)[:,0], np.array(vertices_pos)[:,1], marker='o', c='b')

    # # debug (use different colors for different ctgs of faces)
    # for i, face in enumerate(faces):
    #     if face[0] == 0:
    #         ax.scatter(np.array(faces_pos)[i,0], np.array(faces_pos)[i,1], marker='s', c='#8A2422')
    #     elif face[0] == 1:
    #         ax.scatter(np.array(faces_pos)[i,0], np.array(faces_pos)[i,1], marker='s', c='#FFCC66')
    #     elif face[0] == 2:
    #         ax.scatter(np.array(faces_pos)[i,0], np.array(faces_pos)[i,1], marker='s', c='#6C74A4')
    #     elif face[0] == 3:
    #         ax.scatter(np.array(faces_pos)[i,0], np.array(faces_pos)[i,1], marker='s', c='#BDCAE3')
    #     elif face[0] == 4:
    #         ax.scatter(np.array(faces_pos)[i,0], np.array(faces_pos)[i,1], marker='s', c='lightgray')
    #     else:
    #         ax.scatter(np.array(faces_pos)[i,0], np.array(faces_pos)[i,1], marker='s', c='pink')

    # # annotate faces
    # for i in range(len(faces)):
    #     ax.annotate(str(i), (faces_pos[i][0], faces_pos[i][1]), fontsize=10, color='k')

    # ## debug
    # # annotate vertices
    # for i in range(len(vertices)):
    #     ax.annotate(str(i), (vertices_pos[i][0], vertices_pos[i][1]), fontsize=10, color='purple')

    for edge in edges:
        ax.plot([edge[0].real, edge[1].real], [edge[0].imag, edge[1].imag], color='k', linewidth=0.5)
    for i in range(len(faces)):
        for j in range(len(vertices)):
            if h[i,j] == 1:
                ax.plot([faces_pos[i][0], vertices_pos[j][0]], [faces_pos[i][1], vertices_pos[j][1]], color='gray', linewidth=3, zorder=-1)
    
    ones = [i for i in range(len(logical_op)) if logical_op[i] == 1]
    x = [vertices_pos[i][0] for i in ones]
    y = [vertices_pos[i][1] for i in ones]
    ax.scatter(x, y, marker='s', c='g', zorder=100)

    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    return fig, ax

def draw_qc_transposecode_logical(faces, vertices, edges, faces_pos, h, logical_op):
    vertices = get_vertices(faces)
    vertices_pos = [np.array([vertex.real, vertex.imag]) for vertex in vertices]
    
    fig, ax = plt.subplots()
    ax.scatter(np.array(faces_pos)[:,0], np.array(faces_pos)[:,1], marker='o', c='b')
    ax.scatter(np.array(vertices_pos)[:,0], np.array(vertices_pos)[:,1], marker='s', c='r')
    for edge in edges:
        ax.plot([edge[0].real, edge[1].real], [edge[0].imag, edge[1].imag], color='k', linewidth=0.5)
    for i in range(len(vertices)):
        for j in range(len(faces)):
            if h[i,j] == 1:
                ax.plot([faces_pos[j][0], vertices_pos[i][0]], [faces_pos[j][1], vertices_pos[i][1]], color='gray', linewidth=3, zorder=-1)
    
    # # debug: use different colors for different ctgs of faces
    # for i, face in enumerate(faces):
    #     if face[0] == 0:
    #         ax.scatter(np.array(faces_pos)[i,0], np.array(faces_pos)[i,1], marker='o', c='#8A2422')
    #     elif face[0] == 1:
    #         ax.scatter(np.array(faces_pos)[i,0], np.array(faces_pos)[i,1], marker='o', c='#FFCC66')
    #     elif face[0] == 2:
    #         ax.scatter(np.array(faces_pos)[i,0], np.array(faces_pos)[i,1], marker='o', c='#6C74A4')
    #     elif face[0] == 3:
    #         ax.scatter(np.array(faces_pos)[i,0], np.array(faces_pos)[i,1], marker='o', c='#BDCAE3')
    #     elif face[0] == 4:
    #         ax.scatter(np.array(faces_pos)[i,0], np.array(faces_pos)[i,1], marker='o', c='lightgray')
    #     else:
    #         ax.scatter(np.array(faces_pos)[i,0], np.array(faces_pos)[i,1], marker='o', c='pink')

    ones = [i for i in range(len(logical_op)) if logical_op[i] == 1]
    x = [faces_pos[i][0] for i in ones]
    y = [faces_pos[i][1] for i in ones]
    ax.scatter(x, y, marker='*', c='g', s=200, zorder=100)

    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    return fig, ax