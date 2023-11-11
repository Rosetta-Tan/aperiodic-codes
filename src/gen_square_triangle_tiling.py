import numpy as np
import cmath
import matplotlib.pyplot as plt
import os
from timeit import default_timer as timer
from ldpc.mod2 import row_basis, nullspace, rank
from helpers_distance import get_classical_code_distance_time_limit, get_classical_code_distance_special_treatment
from helpers_qc import *

def subdivide(faces):
        result = []
        for face in faces:
            ctg = face[0]
            if ctg == 0:
                A, B, C = face[1], face[2], face[3]
                base_vec = A-B
                scale_factor = 2 - np.sqrt(3)
                l0 = base_vec * scale_factor * np.exp(-1j * np.pi/3)
                l1 = l0 * np.exp(1j * np.pi/6)
                l2 = l0 * np.exp(-1j * np.pi/6)
                l3 = l0 * np.exp(1j * np.pi/3)
                l4 = l0 * np.exp(-1j * np.pi/3)
                l5 = l0 * np.exp(1j * np.pi/2)

                P2 = A - l3
                P3 = A + l4
                P4 = B + l3
                P6 = B + l0
                P7 = C - l4
                P8 = C - l0
                P10 = P2 - l1
                P11 = P10 + l2
                P12 = P11 + l0
                P13 = P3 + l2
                P14 = P11 + l4
                P15 = P14 - l5
                
                F1 = (0, A, P2, P3)
                F2 = (0, B, P6, P4)
                F3 = (0, C, P7, P8)
                F4 = (2, P7, P12, P13)
                F5 = (2, P3, P12, P13)
                F6 = (2, P2, P10, P11)
                F7 = (2, P4, P10, P11)
                F8 = (2, P6, P14, P15)
                F9 = (2, P8, P14, P15)
                F10 = (4, P12, P11, P3, P2)
                F11 = (4, P11, P14, P4, P6)
                F12 = (4, P12, P14, P7, P8)
                F13 = (1, P14, P12, P11)
                
                result += [F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12, F13]
            if ctg == 1:
                A, B, C = face[1], face[2], face[3]
                base_vec = A-B
                scale_factor = 2 - np.sqrt(3)
                l0 = base_vec * scale_factor * np.exp(-1j * np.pi/3)
                l1 = l0 * np.exp(1j * np.pi/6)
                l2 = l0 * np.exp(-1j * np.pi/6)
                l3 = l0 * np.exp(1j * np.pi/3)
                l4 = l0 * np.exp(-1j * np.pi/3)
                l5 = l0 * np.exp(1j * np.pi/2)

                P2 = A - l1
                P3 = P2 + l2
                P4 = A + l2
                P6 = B + l2
                P7 = B + l1
                P8 = B + l5
                P10 = C + l5
                P11 = C - l2
                P12 = C - l1
                P13 = P3 - l3
                P14 = P7 + l0
                P15 = P3 + l4
                P16 = P2 - l3
                P17 = P6 + l0
                P18 = P4 + l4
                
                F1 = (0, A, P3, P4)
                F2 = (0, A, P2, P3)
                F3 = (0, B, P7, P8)
                F4 = (0, B, P6, P7)
                F5 = (0, C, P11, P12)
                F6 = (0, C, P10, P11)
                F7 = (0, P13, P14, P15)
                F8 = (2, P3, P13, P15)
                F9 = (2, P7, P13, P14)
                F10 = (2, P11, P14, P15)
                F11 = (4, P14, P17, P7, P6)
                F12 = (4, P14, P17, P11, P12)
                F13 = (4, P15, P18, P11, P10)
                F14 = (4, P15, P18, P3, P4)
                F15 = (4, P13, P16, P3, P2)
                F16 = (4, P13, P16, P7, P8)

                result += [F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, 
                           F11, F12, F13, F14, F15, F16]
            if ctg == 2:
                A, B, C = face[1], face[2], face[3]
                base_vec = A-B
                scale_factor = 2 - np.sqrt(3)
                l0 = base_vec * scale_factor * np.exp(-1j * np.pi/3)
                l1 = l0 * np.exp(1j * np.pi/6)
                l2 = l0 * np.exp(-1j * np.pi/6)
                l3 = l0 * np.exp(1j * np.pi/3)
                l4 = l0 * np.exp(-1j * np.pi/3)
                l5 = l0 * np.exp(1j * np.pi/2)

                P2 = A - l1
                P3 = P2 + l2
                P4 = A + l2
                P6 = B + l0
                P7 = B + l3
                P10 = C - l4
                P11 = C - l0
                P12 = P3 - l3
                P13 = P2 - l3
                P14 = P4 + l4
                P15 = P3 + l4
                P16 = P6 + l2
                P17 = P12 + l4
                
                F1 = (0, A, P2, P3)
                F2 = (0, A, P3, P4)
                F3 = (0, B, P6, P7)
                F4 = (0, C, P10, P11)
                F5 = (1, P7, P12, P13)
                F6 = (1, P10, P15, P14)
                F7 = (2, P3, P12, P15)
                F8 = (2, P17, P12, P15)
                F9 = (2, P6, P16, P17)
                F10 = (2, P11, P16, P17)
                F11 = (4, P12, P13, P3, P2)
                F12 = (4, P15, P14, P3, P4)
                F13 = (3, P17, P12, P6, P7)
                F14 = (3, P17, P15, P11, P10)

                result += [F1, F2, F3, F4, F5, F6, F7, F8, F9, 
                           F10, F11, F12, F13, F14]

            if ctg == 3:
                A, B, C, D = face[1], face[2], face[3], face[4]
                base_vec = B-A
                scale_factor = 2 - np.sqrt(3)
                l0 = base_vec * scale_factor
                l1 = l0 * np.exp(1j * np.pi/6)
                l2 = l0 * np.exp(-1j * np.pi/6)
                l3 = l0 * np.exp(1j * np.pi/3)
                l4 = l0 * np.exp(-1j * np.pi/3)
                l5 = l0 * np.exp(1j * np.pi/2)
                
                P2 = A - l5
                P3 = A + l2
                P4 = A + l1
                P5 = C - l4
                P7 = C + l0
                P8 = C + l3
                P10 = D + l3
                P11 = D - l4
                P12 = D - l0
                P14 = B - l0
                P15 = B - l3
                P16 = B + l4
                P17 = P15 - l5
                P18 = P17 - l2
                P19 = P17 - l1
                P20 = P14 - l2
                P21 = P20 - l5
                P22 = P2 - l3
                P23 = P22 + l0
                P24 = P23 + l2
                P25 = P24 + l4
                P26 = P24 + l5
                P27 = P7 + l2
                P28 = P10 + l5
                
                F1 = (0, A, P3, P4)
                F2 = (0, A, P2, P3)
                F3 = (0, C, P5, P8)
                F4 = (0, C, P7, P8)
                F5 = (0, D, P11, P12)
                F6 = (0, D, P11, P10)
                F7 = (0, B, P15, P16)
                F8 = (0, B, P14, P15)
                F9 = (0, P17, P18, P19)
                F10 = (1, P14, P20, P21)
                F11 = (1, P2, P22, P23)
                F12 = (1, P24, P19, P25)
                F13 = (2, P18, P21, P26)
                F14 = (2, P3, P21, P26)
                F15 = (2, P26, P23, P24)
                F16 = (2, P8, P23, P24)
                F17 = (2, P7, P25, P27)
                F18 = (2, P12, P25, P27)
                F19 = (2, P11, P19, P17)
                F20 = (2, P15, P18, P17)
                F21 = (3, P18, P21, P15, P14)
                F22 = (3, P26, P23, P3, P2)
                F23 = (3, P26, P24, P18, P19)
                F24 = (4, P20, P21, P4, P3)
                F25 = (4, P22, P23, P5, P8)
                F26 = (4, P24, P25, P8, P7)
                F27 = (4, P25, P19, P12, P11)
                F28 = (4, P17, P28, P11, P10)
                F29 = (4, P17, P28, P15, P16)
                
                result += [F1, F2, F3, F4, F5, F6, F7, F8, F9,
                            F10, F11, F12, F13, F14, F15, F16, F17,
                            F18, F19, F20, F21, F22, F23, F24, F25,
                            F26, F27, F28, F29]

            if ctg == 4:
                A, B, C, D = face[1], face[2], face[3], face[4]
                base_vec = B-A
                scale_factor = 2 - np.sqrt(3)
                l0 = base_vec * scale_factor
                l1 = l0 * np.exp(1j * np.pi/6)
                l2 = l0 * np.exp(-1j * np.pi/6)
                l3 = l0 * np.exp(1j * np.pi/3)
                l4 = l0 * np.exp(-1j * np.pi/3)
                l5 = l0 * np.exp(1j * np.pi/2)

                P2 = A - l5
                P3 = A + l2
                P4 = A + l1
                P5 = C - l4
                P7 = C + l0
                P8 = C + l3
                P10 = D + l3
                P11 = D - l4
                P12 = D - l0
                P14 = B - l2
                P15 = B - l1
                P16 = B - l5
                P17 = P3 + l0
                P18 = P17 - l3
                P19 = P17 + l4
                P20 = P2 - l3
                P21 = P20 + l0
                P22 = P16 - l3
                P23 = P16 + l4
                P24 = P7 + l1
                P25 = P24 + l3
                P26 = P24 - l4
                P27 = P4 + l0
                P28 = P7 + l2

                F1 = (0, A, P3, P4)
                F2 = (0, A, P2, P3)
                F3 = (0, C, P5, P8)
                F4 = (0, C, P7, P8)
                F5 = (0, D, P11, P12)
                F6 = (0, D, P10, P11)
                F7 = (0, B, P15, P16)
                F8 = (0, B, P14, P15)
                F9 = (1, P2, P20, P21)
                F10 = (1, P24, P25, P26)
                F11 = (1, P16, P22, P23)
                F12 = (2, P3, P18, P17)
                F13 = (2, P15, P17, P19)
                F14 = (2, P18, P21, P26)
                F15 = (2, P8, P21, P26)
                F16 = (2, P19, P22, P25)
                F17 = (2, P11, P22, P25)
                F18 = (2, P7, P24, P28)
                F19 = (2, P12, P24, P28)
                F20 = (3, P18, P21, P3, P2)
                F21 = (3, P19, P22, P15, P16)
                F22 = (4, P27, P17, P4, P3)
                F23 = (4, P17, P27, P15, P14)
                F24 = (4, P20, P21, P5, P8)
                F25 = (4, P22, P23, P11, P10)
                F26 = (4, P25, P26, P19, P18)
                F27 = (4, P26, P24, P8, P7)
                F28 = (4, P24, P25, P12, P11)

                result += [F1, F2, F3, F4, F5, F6, F7, F8, F9,
                            F10, F11, F12, F13, F14, F15, F16, F17,
                            F18, F19, F20, F21, F22, F23, F24, F25,
                            F26, F27, F28]
                
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
####################################################################################################
gen = 1
faces = []
faces.append((2, cmath.rect(1, np.pi/2), cmath.rect(1, 7*np.pi/6), cmath.rect(1, 11*np.pi/6)))
# faces.append((3, cmath.rect(1, 3*np.pi/4), cmath.rect(1, np.pi/4), cmath.rect(1, 5*np.pi/4), cmath.rect(1, 7*np.pi/4)))
for _ in range(gen):
     faces = subdivide(faces)
vertices = get_vertices(faces)
edges = get_edges(faces, vertices)
fig, ax = draw(faces, vertices, edges)
plt.show()

