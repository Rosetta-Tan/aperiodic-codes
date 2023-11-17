import numpy as np
import cmath
import matplotlib.pyplot as plt
import os
from timeit import default_timer as timer
from ldpc.mod2 import row_basis, nullspace, rank
from helpers_distance import get_classical_code_distance_time_limit, get_classical_code_distance_special_treatment
from helpers_qc import *
from timeit import default_timer as timer

def subdivide(faces):
        result = []
        for face in faces:
            ctg = face[0]
            if ctg == 0:
                A, B, C = face[1], face[2], face[3]
                scale_factor = 2 - np.sqrt(3)
                l0 = scale_factor * (C-B)
                l1 = scale_factor * (A+C-2*B)/np.sqrt(3)
                l2 = scale_factor * (2*C-A-B)/np.sqrt(3)
                l3 = scale_factor * (A-B)
                l4 = scale_factor * (C-A)
                l5 = scale_factor * (2*A-B-C)/np.sqrt(3)

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
                F4 = (2, P7, P13, P12)
                F5 = (2, P3, P12, P13)
                F6 = (2, P2, P10, P11)
                F7 = (2, P4, P11, P10)
                F8 = (2, P6, P15, P14)
                F9 = (2, P8, P14, P15)
                F10 = (4, P12, P11, P3, P2)
                F11 = (4, P11, P14, P4, P6)
                F12 = (4, P14, P12, P8, P7)
                F13 = (1, P14, P12, P11)
                
                result += [F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12, F13]
            if ctg == 1:
                A, B, C = face[1], face[2], face[3]
                scale_factor = 2 - np.sqrt(3)
                l0 = scale_factor * (C-B)
                l1 = scale_factor * (A+C-2*B)/np.sqrt(3)
                l2 = scale_factor * (2*C-A-B)/np.sqrt(3)
                l3 = scale_factor * (A-B)
                l4 = scale_factor * (C-A)
                l5 = scale_factor * (2*A-B-C)/np.sqrt(3)

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
                F9 = (2, P7, P14, P13)
                F10 = (2, P11, P15, P14)
                F11 = (4, P14, P17, P7, P6)
                F12 = (4, P17, P14, P12, P11)
                F13 = (4, P15, P18, P11, P10)
                F14 = (4, P18, P15, P4, P3)
                F15 = (4, P13, P16, P3, P2)
                F16 = (4, P16, P13, P8, P7)

                result += [F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, 
                           F11, F12, F13, F14, F15, F16]
            if ctg == 2:
                A, B, C = face[1], face[2], face[3]
                scale_factor = 2 - np.sqrt(3)
                l0 = scale_factor * (C-B)
                l1 = scale_factor * (A+C-2*B)/np.sqrt(3)
                l2 = scale_factor * (2*C-A-B)/np.sqrt(3)
                l3 = scale_factor * (A-B)
                l4 = scale_factor * (C-A)
                l5 = scale_factor * (2*A-B-C)/np.sqrt(3)

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
                F6 = (1, P10, P14, P15)
                F7 = (2, P3, P12, P15)
                F8 = (2, P17, P15, P12)
                F9 = (2, P6, P16, P17)
                F10 = (2, P11, P17, P16)
                F11 = (4, P12, P13, P3, P2)
                F12 = (4, P14, P15, P4, P3)
                F13 = (3, P17, P12, P6, P7)
                F14 = (3, P17, P15, P11, P10)

                result += [F1, F2, F3, F4, F5, F6, F7, F8, F9, 
                           F10, F11, F12, F13, F14]

            if ctg == 3:
                A, B, C, D = face[1], face[2], face[3], face[4]
                scale_factor = 2 - np.sqrt(3)
                l0 = scale_factor * (B-A)
                l1 = scale_factor * 0.5*((A-C)+np.sqrt(3)*(B-A))
                l2 = scale_factor * 0.5*((C-A)+np.sqrt(3)*(B-A))
                l3 = scale_factor * 0.5*(np.sqrt(3)*(A-C)+(B-A))
                l4 = scale_factor * 0.5*(np.sqrt(3)*(C-A)+(B-A))
                l5 = scale_factor * (A-C)
                
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
                F3 = (0, C, P8, P5)
                F4 = (0, C, P7, P8)
                F5 = (0, D, P11, P12)
                F6 = (0, D, P10, P11)
                F7 = (0, B, P15, P16)
                F8 = (0, B, P14, P15)
                F9 = (0, P17, P18, P19)
                F10 = (1, P14, P20, P21)
                F11 = (1, P2, P22, P23)
                F12 = (1, P25, P19, P24)
                F13 = (2, P18, P21, P26)
                F14 = (2, P3, P26, P21)
                F15 = (2, P26, P23, P24)
                F16 = (2, P8, P24, P23)
                F17 = (2, P7, P27, P25)
                F18 = (2, P12, P25, P27)
                F19 = (2, P11, P17, P19)
                F20 = (2, P15, P18, P17)
                F21 = (3, P18, P21, P15, P14)
                F22 = (3, P26, P23, P3, P2)
                F23 = (3, P26, P24, P18, P19)
                F24 = (4, P20, P21, P4, P3)
                F25 = (4, P22, P23, P5, P8)
                F26 = (4, P24, P25, P8, P7)
                F27 = (4, P25, P19, P12, P11)
                F28 = (4, P17, P28, P11, P10)
                F29 = (4, P28, P17, P16, P15)
                
                result += [F1, F2, F3, F4, F5, F6, F7, F8, F9,
                            F10, F11, F12, F13, F14, F15, F16, F17,
                            F18, F19, F20, F21, F22, F23, F24, F25,
                            F26, F27, F28, F29]

            if ctg == 4:
                A, B, C, D = face[1], face[2], face[3], face[4]
                scale_factor = 2 - np.sqrt(3)
                l0 = scale_factor * (B-A)
                l1 = scale_factor * 0.5*((A-C)+np.sqrt(3)*(B-A))
                l2 = scale_factor * 0.5*((C-A)+np.sqrt(3)*(B-A))
                l3 = scale_factor * 0.5*(np.sqrt(3)*(A-C)+(B-A))
                l4 = scale_factor * 0.5*(np.sqrt(3)*(C-A)+(B-A))
                l5 = scale_factor * (A-C)

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
                F3 = (0, C, P8, P5)
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
                F15 = (2, P8, P26, P21)
                F16 = (2, P19, P25, P22)
                F17 = (2, P11, P22, P25)
                F18 = (2, P7, P28, P24)
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
                F29 = (0, P17, P18, P19)

                result += [F1, F2, F3, F4, F5, F6, F7, F8, F9,
                            F10, F11, F12, F13, F14, F15, F16, F17,
                            F18, F19, F20, F21, F22, F23, F24, F25,
                            F26, F27, F28, F29]
                
        return result

##############################################################################################################
# specify boundaries here
def remove_double_faces(faces):
    faces_new = []
    geometric_centers = []
    for face in faces:
        face_pos = get_geometric_center(face)
        if not any(close(face_pos, center) for center in geometric_centers):
            geometric_centers.append(face_pos)
            faces_new.append(face)
    return faces_new

def on_boundary(vertex, eps=1e-6):
    '''triangle boundary'''
    distance_to_boundary = [
        np.abs(vertex.imag + 0.5),
        np.abs(np.sqrt(3)*vertex.real + vertex.imag - 1),
        np.abs(-np.sqrt(3)*vertex.real + vertex.imag - 1)
    ]
    # 'square boundry'
    # distance_to_boundary = [
    #     np.abs(np.abs(vertex.real) - 0.5*np.sqrt(2)),
    #     np.abs(np.abs(vertex.imag) - 0.5*np.sqrt(2))
    # ]
    return any(distance < eps for distance in distance_to_boundary)

def within_boundary(vertex, eps=1e-6):
    '''triangle boundary'''
    return vertex.imag > -0.5 - eps and (np.sqrt(3)*vertex.real + vertex.imag) < 1 + eps and (-np.sqrt(3)*vertex.real + vertex.imag) < 1 + eps
    # '''square boundary'''
    # return np.abs(vertex.real) < (0.5*np.sqrt(2) + eps) and np.abs(vertex.imag) < (0.5*np.sqrt(2) + eps)

def distance_to_boundary(vertex):
    '''triangle boundary'''
    distance_to_boundary = [
        np.abs(vertex.imag + 0.5),
        np.abs(np.sqrt(3)*vertex.real + vertex.imag - 1),
        np.abs(-np.sqrt(3)*vertex.real + vertex.imag - 1)
    ]
    # 'square boundry'
    # distance_to_boundary = [
    #     np.abs(np.abs(vertex.real) - 0.5*np.sqrt(2)),
    #     np.abs(np.abs(vertex.imag) - 0.5*np.sqrt(2))
    # ]
    return min(distance_to_boundary)

def boundary_cut(faces):
    faces_center_cplx = [get_geometric_center(face) for face in faces]
    remaining_faces = []
    for i, face in enumerate(faces):
        if all(within_boundary(vertex) for vertex in face[1:]):
            remaining_faces.append(face)
        else:
            if on_boundary(faces_center_cplx[i]):
                remaining_faces.append(face)
    return remaining_faces

def boundary_cut_special_treatment(faces):
    faces_center_cplx = [get_geometric_center(face) for face in faces]
    remaining_faces = []
    for i, face in enumerate(faces):
        if all(within_boundary(vertex) for vertex in face[1:]):
            remaining_faces.append(face)
        elif all((not within_boundary(vertex)) for vertex in face[1:]):
            continue
        elif face[0] != 'hex' and face[0] != 'rhomb':
            if on_boundary(faces_center_cplx[i]):
                remaining_faces.append(face)
        elif face[0] == 'hex':
            edge_length = np.abs(face[1] - face[2])
            vertices_on_boundary = []
            vertices_within_boundary = []
            for vertex in face[1:]:
                if on_boundary(vertex):
                    vertices_on_boundary.append(vertex)
                elif within_boundary(vertex):
                    vertices_within_boundary.append(vertex)
            
            if len(vertices_within_boundary) == 3:
                assert len(vertices_on_boundary) == 0, f'face {i}, len(vertices_within_boundary) = {len(vertices_within_boundary)}'
                remaining_faces.append(face)
                continue
            elif len(vertices_within_boundary) == 2:
                assert len(vertices_on_boundary) == 2, f'face {i}, len(vertices_within_boundary) = {len(vertices_within_boundary)}'
                C, D = vertices_on_boundary
                for vertex in vertices_within_boundary:
                    if np.abs(np.abs(vertex - C) - edge_length) < 1e-5:
                        A = vertex
                    elif np.abs(np.abs(vertex - D) - edge_length) < 1e-5:
                        B = vertex
                remaining_faces.append(('hex-boundary-cut', A, B, C, D))
            else:
                continue

        elif face[0] == 'rhomb':
            # vertices_out_boundary = []
            for iv, vertex in enumerate(face[1:], start=1):
                if not within_boundary(vertex) and (iv == 1 or iv == 4):
                    # A or D is out of boundary, do append this rhomb
                    remaining_faces.append(face)
                    break
                elif not within_boundary(vertex) and (iv == 2):
                    # B is out of boundary, append half of this rhomb
                    remaining_faces.append(('rhomb-boundary-cut', face[1], face[3], face[4]))
                    break
                elif not within_boundary(vertex) and (iv == 3):
                    # C is out of boundary, append half of this rhomb
                    remaining_faces.append(('rhomb-boundary-cut', face[1], face[2], face[4]))
                    break

            #         vertices_out_boundary.append(vertex)
            # remaining_faces.append(('rhomb-boundary-cut', vertices_within_boundary[0], vertices_within_boundary[1], vertices_within_boundary[2]))
    return remaining_faces

##############################################################################################################

def detect_mergable_ctg0(input_triangle, faces_pos):
    def transform_input_triangle_around_A(input_triangle):
        ctg, A, B, C = input_triangle
        center = (A + B + C)/3
        faces_pos_new = [
            center,
            A + C - center,
            A + B - center,
            2 * A - center,
            A - C + center,
            A + B - center
        ]
        return faces_pos_new
    def transform_input_triangle_around_B(input_triangle):
        ctg, A, B, C = input_triangle
        center = (A + B + C)/3
        faces_pos_new = [
            center,
            B + A - center,
            B - C + center,
            2 * B - center,
            B - A + center,
            B + C - center
        ]
        return faces_pos_new
    def transform_input_triangle_around_C(input_triangle):
        ctg, A, B, C = input_triangle
        center = (A + B + C)/3
        faces_pos_new = [
            center,
            C + B - center,
            C - A + center,
            2 * C - center,
            C - B + center,
            C + A - center
        ]
        return faces_pos_new
    
    def point_in_face_pos(point, faces_pos):
        for face_pos in faces_pos:
            if close(point, face_pos):
                return True
        return False
    
    transformed_around_A_pos = transform_input_triangle_around_A(input_triangle)
    transformed_around_A_pos_in_face_pos = [point_in_face_pos(pos, faces_pos) for pos in transformed_around_A_pos]
    if all(transformed_around_A_pos_in_face_pos):
        return 'A', transformed_around_A_pos
    
    transformed_around_B_pos = transform_input_triangle_around_B(input_triangle)
    transformed_around_B_pos_in_face_pos = [point_in_face_pos(pos, faces_pos) for pos in transformed_around_B_pos]
    if all(transformed_around_B_pos_in_face_pos):
        return 'B', transformed_around_B_pos
    
    transformed_around_C_pos = transform_input_triangle_around_C(input_triangle)
    transformed_around_C_pos_in_face_pos = [point_in_face_pos(pos, faces_pos) for pos in transformed_around_C_pos]
    if all(transformed_around_C_pos_in_face_pos):
        return 'C', transformed_around_C_pos
    
    return None, []

def detect_mergable_ctg2(input_triangle, faces_pos):
    ctg, A, B, C = input_triangle
    center = (A + B + C)/3
    mirroed_along_BC_pos = B+C-center
    for face_pos in faces_pos:
        if close(mirroed_along_BC_pos, face_pos):
            return mirroed_along_BC_pos
    return None

def orientation_ctg2(input_triangle, eps=1e-6):
    ctg, A, B, C = input_triangle
    valid_orientation_expressions = [
        np.abs(B.imag-C.imag),  # horizontal
        np.abs(B.real-C.real),  # vertical
        np.abs((B.imag-C.imag)-np.sqrt(3)*(B.real-C.real)),  # (1/2, sqrt(3)/2) mirror axis
        np.abs(np.sqrt(3)*(B.imag-C.imag)-(B.real-C.real)),  # (sqrt(3)/2, 1/2) mirror axis
        np.abs((B.imag-C.imag)+np.sqrt(3)*(B.real-C.real)),  # (-1/2, sqrt(3)/2) mirror axis
        np.abs(np.sqrt(3)*(B.imag-C.imag)+(B.real-C.real))   # (-sqrt(3)/2, 1/2) mirror axis
    ]
    if any(valid_orientation_expressions[i] < eps for i in range(len(valid_orientation_expressions))):
        return 'mergable'
    else:
        return 'notmergable'

def detect_mergable_ctg4(input_square, faces_pos):
    ctg, A, B, C, D = input_square
    center = (A + B + C + D)/4
    mirroed_along_AB_pos = A+B-center
    for face_pos in faces_pos:
        if close(mirroed_along_AB_pos, face_pos):
            return mirroed_along_AB_pos
    return None

def merge(faces, faces_pos):
    merged_faces = []
    visited = []
    # loop over all boundary ctg 0 faces to detect mergable
    for i, face in enumerate(faces):
        if face[0] == 0 and (i not in visited):
            ctg, A, B, C, = face
            on_boundary_vertices = []
            off_boundary_vertices = []
            for vertex in face[1:]:
                if on_boundary(vertex):
                    on_boundary_vertices.append(vertex)
                else:
                    off_boundary_vertices.append(vertex)
            if len(on_boundary_vertices) == 0 or len(on_boundary_vertices) == 3:
                continue
            elif len(on_boundary_vertices) == 1:
                touch_point = on_boundary_vertices[0]
                rem_point1 = off_boundary_vertices[0]
                rem_point2 = off_boundary_vertices[1]
                center = (touch_point + rem_point1 + rem_point2)/3
                transformed = [touch_point + rem_point1 - center, touch_point + rem_point2 - center]
                local_match1 = None
                local_match2 = None
                for j, face_j in enumerate(faces):
                    if face_j[0] == 0 and all(within_boundary(vertex_j) for vertex_j in face_j[1:]) and close((face_j[1]+face_j[2]+face_j[3])/3, transformed[0]):
                        local_match1 = j
                    if face_j[0] == 0 and all(within_boundary(vertex_j) for vertex_j in face_j[1:]) and close((face_j[1]+face_j[2]+face_j[3])/3, transformed[1]):
                        local_match2 = j
                if local_match1 != None and local_match2 != None:  # not complete local match
                    visited.append(i)
                    visited.append(local_match1)
                    visited.append(local_match2)
                    merged_faces.append(('hex-boundary-cut', rem_point1, rem_point2, (touch_point+rem_point1-rem_point2), (touch_point+rem_point2-rem_point1)))
                else:
                    continue
                
            elif len(on_boundary_vertices) == 2:
                touch_point1 = on_boundary_vertices[0]
                touch_point2 = on_boundary_vertices[1]
                rem_point = off_boundary_vertices[0]
                center = (touch_point1 + touch_point2 + rem_point)/3
                transformed_set1 = [touch_point1 + rem_point - center, touch_point1 + center - touch_point2]
                transformed_set2 = [touch_point2 + rem_point - center, touch_point2 + center - touch_point1]
                local_match1 = None
                local_match2 = None
                for j, face_j in enumerate(faces):
                    if face_j[0] == 0 and all(within_boundary(vertex_j) for vertex_j in face_j[1:]) and close((face_j[1]+face_j[2]+face_j[3])/3, transformed_set1[0]):
                        local_match1 = j
                    if face_j[0] == 0 and all(within_boundary(vertex_j) for vertex_j in face_j[1:]) and close((face_j[1]+face_j[2]+face_j[3])/3, transformed_set1[1]):
                        local_match2 = j
                if local_match1 != None and local_match2 != None:
                    visited.append(i)
                    visited.append(local_match1)
                    visited.append(local_match2)
                    merged_faces.append(('hex-boundary-cut', rem_point, touch_point1+rem_point-touch_point2, touch_point2, 2*touch_point1-touch_point2))
                else:
                    local_match1 = None
                    local_match2 = None
                    for j, face_j in enumerate(faces):
                        if face_j[0] == 0 and all(within_boundary(vertex_j) for vertex_j in face_j[1:]) and close((face_j[1]+face_j[2]+face_j[3])/3, transformed_set2[0]):
                            local_match1 = j
                        if face_j[0] == 0 and all(within_boundary(vertex_j) for vertex_j in face_j[1:]) and close((face_j[1]+face_j[2]+face_j[3])/3, transformed_set2[1]):
                            local_match2 = j
                    if i == 13:
                        print(f'here, local_match1={local_match1}, local_match2={local_match2}')
                    if local_match1 != None and local_match2 != None:
                        visited.append(i)
                        visited.append(local_match1)
                        visited.append(local_match2)
                        merged_faces.append(('hex-boundary-cut', rem_point, touch_point2+rem_point-touch_point1, touch_point1, 2*touch_point2-touch_point1))
                    else:
                        continue

    # loop over all ctg 0 faces to detect mergable
    for i, face in enumerate(faces):
        if face[0] == 0 and (i not in visited):
            visited.append(i)
            ctg, A, B, C = face
            merge_around, transformed_pos = detect_mergable_ctg0((ctg, A, B, C), faces_pos)
            if merge_around == None:
                merged_faces.append(face)
            elif merge_around == 'A':
                for pos in transformed_pos[1:]:  # skip the first one, which is the starting face
                    for j, face_j in enumerate(faces):
                        if j!=i and (j not in visited) and face_j[0] == 0 and close((face_j[1]+face_j[2]+face_j[3])/3, pos):
                            visited.append(j)
                            break
                merged_faces.append(('hex', B, C, (A+C-B), 2*A-B, 2*A-C, A+B-C))
            elif merge_around == 'B':
                for pos in transformed_pos[1:]:
                    for j, face_j in enumerate(faces):
                        if j!=i and (j not in visited) and face_j[0] == 0 and close((face_j[1]+face_j[2]+face_j[3])/3, pos):
                            visited.append(j)
                            break
                merged_faces.append(('hex', C, A, (B+A-C), 2*B-C, 2*B-A, B+C-A))
            elif merge_around == 'C':
                for pos in transformed_pos[1:]:
                    for j, face_j in enumerate(faces):
                        if j!=i and (j not in visited) and face_j[0] == 0 and close((face_j[1]+face_j[2]+face_j[3])/3, pos):
                            visited.append(j)
                            break
                merged_faces.append(('hex', A, B, (C+B-A), 2*C-A, 2*C-B, C+A-B))

    # # loop over ctg 1 triangles to detect mergable (2blue1white)
    # for i, face in enumerate(faces):
    #     if face[0] == 1 and (i not in visited):
    #         ctg, A, B, C = face
    #         center = (A + B + C)/3
    #         transformed_pos = [
    #             ((A+B)/2 - center)*np.sqrt(3) + (A+B)/2,
    #             ((B+C)/2 - center)*np.sqrt(3) + (B+C)/2,
    #             ((C+A)/2 - center)*np.sqrt(3) + (C+A)/2
    #         ]
    #         local_match = []
    #         num_ctg4_match = 0
    #         for pos in transformed_pos:
    #             for j, face_j in enumerate(faces):
    #                 if j!=i and (j not in visited) and face_j[0] == 3 and all(within_boundary(vertex_j) for vertex_j in face_j[1:]):
    #                     if close((face_j[1]+face_j[2]+face_j[3]+face_j[4])/4, pos):
    #                         local_match.append(face_j)
    #                         break
    #                 if j!=i and (j not in visited) and face_j[0] == 4 and all(within_boundary(vertex_j) for vertex_j in face_j[1:]):
    #                     if close((face_j[1]+face_j[2]+face_j[3]+face_j[4])/4, pos):
    #                         local_match.append(face_j)
    #                         num_ctg4_match += 1
    #                         break
    #         if len(local_match) == 3 and num_ctg4_match == 1:
    #             for which_local_match, face_j in enumerate(local_match):
    #                 if face_j[0] == 4:
    #                     visited.append(i)
    #                     visited.append(faces.index(face_j))
    #                     if which_local_match == 0:  # mirrored along AB
    #                         D = A + (center - C)*np.sqrt(3)
    #                         E = B + (center - C)*np.sqrt(3)
    #                         merged_faces.append(('five-sided', C, A, D, E, B))
    #                     elif which_local_match == 1:  # mirrored along BC
    #                         D = B + (center - A)*np.sqrt(3)
    #                         E = C + (center - A)*np.sqrt(3)
    #                         merged_faces.append(('five-sided', A, B, D, E, C))
    #                     elif which_local_match == 2:  # mirrored along CA
    #                         D = C + (center - B)*np.sqrt(3)
    #                         E = A + (center - B)*np.sqrt(3)
    #                         merged_faces.append(('five-sided', B, C, D, E, A))
    #                     break
    #         else:
    #             continue

    # loop over ctg 1 triangles to detect mergable (3white)
    for i, face in enumerate(faces):
        if face[0] == 1 and (i not in visited):
            ctg, A, B, C = face
            center = (A + B + C)/3
            transformed_pos = [
                ((A+B)/2 - center)*np.sqrt(3) + (A+B)/2,
                ((B+C)/2 - center)*np.sqrt(3) + (B+C)/2,
                ((C+A)/2 - center)*np.sqrt(3) + (C+A)/2
            ]
            local_match = []
            for pos in transformed_pos:
                for j, face_j in enumerate(faces):
                    if j!=i and (j not in visited) and face_j[0] == 4 and all(within_boundary(vertex_j) for vertex_j in face_j[1:]):
                        if close((face_j[1]+face_j[2]+face_j[3]+face_j[4])/4, pos):
                            local_match.append(face_j)
                            break
            if len(local_match) == 3:
                visited.append(i)
                edges = [[A,B], [B,C], [C,A]]
                for edge in edges:
                    if np.abs(edge[0].imag - edge[1].imag) < 1e-5:  # horizontal edge
                        if edge[0].real > edge[1].real:  # make sure edge[0] is on the left
                            edge[0], edge[1] = edge[1], edge[0]
                        rem_point = 3*center - edge[0] - edge[1]
                        if rem_point.imag < edge[0].imag:  # downward pointing triangle
                            for j, face_j in enumerate(faces):
                                if j!=i and (j not in visited) and face_j[0] == 4 and all(within_boundary(vertex_j) for vertex_j in face_j[1:]):
                                    if close((face_j[1]+face_j[2]+face_j[3]+face_j[4])/4, ((edge[1]+rem_point)/2 - center)*np.sqrt(3) + (edge[1]+rem_point)/2):   
                                        visited.append(j)
                                        D = edge[1] + (center - edge[0])*np.sqrt(3)
                                        E = rem_point + (center - edge[0])*np.sqrt(3)
                                        merged_faces.append(('five-sided', edge[0], edge[1], D, E, rem_point))
                                        break
                        elif rem_point.imag > edge[0].imag:  # upward pointing triangle
                            for j, face_j in enumerate(faces):
                                if j!=i and (j not in visited) and face_j[0] == 4 and all(within_boundary(vertex_j) for vertex_j in face_j[1:]):
                                    if close((face_j[1]+face_j[2]+face_j[3]+face_j[4])/4, ((edge[0]+edge[1])/2 - center)*np.sqrt(3) + (edge[0]+edge[1])/2):
                                        visited.append(j)
                                        D = edge[0] + np.sqrt(3)*(center-rem_point)
                                        E = edge[1] + np.sqrt(3)*(center-rem_point)
                                        merged_faces.append(('five-sided', rem_point, edge[0], D, E, edge[1]))
                                        break
                    elif np.abs(edge[0].real - edge[1].real) < 1e-5:  # vertical edge
                        if edge[0].imag < edge[1].imag:
                            edge[0], edge[1] = edge[1], edge[0]  # make sure edge[0] is at the top
                        rem_point = 3*center - edge[0] - edge[1]
                        if rem_point.real < edge[0].real:  # left pointing triangle
                            for j, face_j in enumerate(faces):
                                if j!=i and (j not in visited) and face_j[0] == 4 and all(within_boundary(vertex_j) for vertex_j in face_j[1:]):
                                    if close((face_j[1]+face_j[2]+face_j[3]+face_j[4])/4, ((edge[0]+rem_point)/2 - center)*np.sqrt(3) + (edge[0]+rem_point)/2):
                                        visited.append(j)
                                        D = edge[0] + np.sqrt(3)*(center-edge[1])
                                        E = rem_point + np.sqrt(3)*(center-edge[1])
                                        merged_faces.append(('five-sided', edge[1], edge[0], D, E, rem_point))
                                        break
                        elif rem_point.real > edge[0].real:  # right pointing triangle
                            for j, face_j in enumerate(faces):
                                if j!=i and (j not in visited) and face_j[0] == 4 and all(within_boundary(vertex_j) for vertex_j in face_j[1:]):
                                    if close((face_j[1]+face_j[2]+face_j[3]+face_j[4])/4, ((edge[0]+edge[1])/2 - center)*np.sqrt(3) + (edge[0]+edge[1])/2):
                                        visited.append(j)
                                        D = edge[0] + np.sqrt(3)*(center-rem_point)
                                        E = edge[1] + np.sqrt(3)*(center-rem_point)
                                        merged_faces.append(('five-sided', rem_point, edge[0], D, E, edge[1]))
                                        break
            else:
                continue

    # loop over all ctg 2 trianges to detect mergable
    for i, face in enumerate(faces):
        if face[0] == 2 and (i not in visited):
            visited.append(i)
            ctg, A, B, C = face
        # consider the orientation of the triangle, only three possible orientations are allowed
            if orientation_ctg2(face) == 'notmergable':
                merged_faces.append(face)
            else:
                mirrored_pos = detect_mergable_ctg2(face, faces_pos)
                if mirrored_pos == None:
                    merged_faces.append(face)
                else:
                    found_pair = False
                    for j, face_j in enumerate(faces):
                        if j!= i and (j not in visited) and face_j[0] == 2 and close((face_j[1]+face_j[2]+face_j[3])/3, mirrored_pos):
                            found_pair = True
                            visited.append(j)
                            break
                    if found_pair:
                        merged_faces.append(('rhomb', A, B, C, B+C-A))
                    else:
                        merged_faces.append(face)

    # loop over all ctg 4 squares to detect mergable
    for i, face in enumerate(faces):
        if face[0] == 4 and (i not in visited):
            visited.append(i)
            ctg, A, B, C, D = face
            mirrored_pos = detect_mergable_ctg4(face, faces_pos)
            if mirrored_pos == None:
                merged_faces.append(face)
            else:
                found_pair = False
                for j, face_j in enumerate(faces):
                    if j!= i and (j not in visited) and face_j[0] == 4 and close((face_j[1]+face_j[2]+face_j[3]+face_j[4])/4, mirrored_pos):
                        found_pair = True
                        visited.append(j)
                        break
                if found_pair:
                    merged_faces.append(('long-square', 2*A-C, 2*B-D, C, D))
                else:
                    merged_faces.append(face)

    # loop over remaining faces
    for i, face in enumerate(faces):
        if i not in visited:
            visited.append(i)
            merged_faces.append(face)
    
    return merged_faces

def get_geometric_center(face):
    if len(face) == 4:
        return (face[1]+face[2]+face[3])/3
    elif len(face) == 5:
        return (face[1]+face[2]+face[3]+face[4])/4
    elif len(face) == 6:
        return 0.5*((face[2]+face[3]+face[4]+face[5])/4+(face[1]+face[2]+face[5])/3)
    elif len(face) == 7:
        return (face[1]+face[2]+face[3]+face[4]+face[5]+face[6])/6       

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
        elif len(face) == 6:
            edges.append((vs_on_f[1], vs_on_f[2]))
            edges.append((vs_on_f[2], vs_on_f[3]))
            edges.append((vs_on_f[3], vs_on_f[4]))
            edges.append((vs_on_f[4], vs_on_f[5]))
            edges.append((vs_on_f[5], vs_on_f[1]))
        elif len(face) == 7:  # hexagons
            edges.append((vs_on_f[1], vs_on_f[2]))
            edges.append((vs_on_f[2], vs_on_f[3]))
            edges.append((vs_on_f[3], vs_on_f[4]))
            edges.append((vs_on_f[4], vs_on_f[5]))
            edges.append((vs_on_f[5], vs_on_f[6]))
            edges.append((vs_on_f[6], vs_on_f[1]))
    return edges

def get_qc_code(faces, vertices):
    h = np.zeros((len(faces), len(vertices)))
    for i, face in enumerate(faces):
        if face[0] != 'long-square':
            for j in range(len(vertices)):
                for k in range(1, len(face)):
                    if close(face[k], vertices[j]):
                        h[i,j] = 1
        elif face[0] == 'long-square':
            for j in range(len(vertices)):
                for k in range(1, len(face)):
                    if close(face[k], vertices[j]):
                        h[i,j] = 1
                if close((face[1]+face[3])/2, vertices[j]):
                    h[i,j] = 1
                if close((face[2]+face[4])/2, vertices[j]):
                    h[i, j] = 1
    return h
####################################################################################################
start = timer()
gen = 3
faces = []
# ctg = 1
# faces.append((ctg, cmath.rect(1, np.pi/2), cmath.rect(1, 7*np.pi/6), cmath.rect(1, 11*np.pi/6)))
ctg = 4
faces.append((ctg, cmath.rect(1, 3*np.pi/4), cmath.rect(1, np.pi/4), cmath.rect(1, 5*np.pi/4), cmath.rect(1, 7*np.pi/4)))
for _ in range(gen):
     faces = subdivide(faces)
faces_pos_cplx = [get_geometric_center(face) for face in faces]
faces = remove_double_faces(faces)
# faces = boundary_cut(faces)
faces = merge(faces, faces_pos_cplx)
faces = boundary_cut_special_treatment(faces)
faces = remove_double_faces(faces)

vertices = get_vertices(faces)

edges = get_edges(faces, vertices)
faces_pos = [np.array([get_geometric_center(face).real, get_geometric_center(face).imag]) for face in faces]
# fig, ax = draw(faces, vertices, edges)
# plt.show()

h = get_qc_code(faces, vertices)
h = h.T
print('h.shape = ', h.shape)
# print('k = ', h.shape[1]-rank(h))
logical_op = []
# fig, ax = draw_qc_code_logical(faces, vertices, edges, faces_pos, h, logical_op)
fig, ax = draw_qc_transposecode_logical(faces, vertices, edges, faces_pos, h, logical_op)
d_bound, logical_op = get_classical_code_distance_special_treatment(h, target_weight=get_classical_code_distance_time_limit(h))
print('d_bound = ', d_bound)
# fig, ax = draw_qc_code_logical(faces, vertices, edges, faces_pos, h, logical_op)
fig, ax = draw_qc_transposecode_logical(faces, vertices, edges, faces_pos, h, logical_op)


# for i, face in enumerate(faces):
#     ax.text(get_geometric_center(face).real, get_geometric_center(face).imag, str(i))   
#     if i == 13:
#         ctg, A, B, C, = face
#         on_boundary_vertices = []
#         off_boundary_vertices = []
#         for vertex in face[1:]:
#             if on_boundary(vertex):
#                 on_boundary_vertices.append(vertex)
#             else:
#                 off_boundary_vertices.append(vertex)
#         touch_point1 = on_boundary_vertices[0]
#         touch_point2 = on_boundary_vertices[1]
#         rem_point = off_boundary_vertices[0]
#         center = (touch_point1 + touch_point2 + rem_point)/3
#         transformed_set1 = [touch_point1 + rem_point - center, touch_point1 + center - touch_point2]
#         transformed_set2 = [touch_point2 + rem_point - center, touch_point2 + center - touch_point1]
#         for pos in transformed_set1:
#             ax.scatter(pos.real, pos.imag, color='purple')
#         for pos in transformed_set2:
#             ax.scatter(pos.real, pos.imag, color='green')

# plot triangle boundaries
x1 = np.linspace(-np.sqrt(3)/2, np.sqrt(3)/2, 100)
y1 = -0.5 * np.ones(100)
x2 = np.linspace(0, np.sqrt(3)/2, 100)
y2 = 1 - np.sqrt(3) * x2
x3 = np.linspace(-np.sqrt(3)/2, 0, 100)
y3 = 1 + np.sqrt(3) * x3
ax.plot(x1, y1, color='black')
ax.plot(x2, y2, color='black')
ax.plot(x3, y3, color='black')

# # # plot square boundaries
# ax.hlines(0.5*np.sqrt(2), -0.5*np.sqrt(2), 0.5*np.sqrt(2), color='black')
# ax.hlines(-0.5*np.sqrt(2), -0.5*np.sqrt(2), 0.5*np.sqrt(2), color='black')
# ax.vlines(-0.5*np.sqrt(2), -0.5*np.sqrt(2), 0.5*np.sqrt(2), color='black')
# ax.vlines(0.5*np.sqrt(2), -0.5*np.sqrt(2), 0.5*np.sqrt(2), color='black')

end = timer()
print('time = ', end-start)
plt.show()