from typing import List, Tuple, Literal, TypeAlias  # TypeAlias requires Python 3.10
import struct
import numpy as np
from matplotlib import pyplot as plt
from dataclasses import dataclass
Face: TypeAlias = Tuple[int, complex, complex, complex]
Edge: TypeAlias = Tuple[complex, complex]

##### Functions for the tiling units #####
@dataclass
class Triangle:
    nop: int
    pairing: Literal['domino', 'kite', None]
    

class GeometryUtils:
    '''
    Class with static methods for geometric operations
    vtx: complex
    edge: Tuple[complex, complex]
    trig: Tuple[int, complex, complex, complex]
    '''
    @staticmethod
    def same_vtx(a: complex, b: complex) -> bool:
        # function to check if two complex numbers are close
        return abs(a-b) < 1e-5
    
    @staticmethod
    def same_edge(edge1: Edge, edge2: Edge) -> bool:
        return GeometryUtils.same_vtx(edge1[0], edge2[0]) and GeometryUtils.same_vtx(edge1[1], edge2[1]) or GeometryUtils.same_vtx(edge1[0], edge2[1]) and GeometryUtils.same_vtx(edge1[1], edge2[0])
    
    @staticmethod
    def get_trig_center(trig: Face) -> complex:
        return (trig[1]+trig[2]+trig[3])/3
    
    @staticmethod
    def get_trig_edges(trig: Face) -> List[Edge]:
        return [(trig[1], trig[2]), (trig[2], trig[3]), (trig[3], trig[1])]
    
    @staticmethod
    def subdivide(triangles: List[Face]) -> List[Face]:
        result: List[Face] = []
        for ctg, A, B, C in triangles:
            if ctg == 0:
                P1 = A + 2/5*(B-A)
                P2 = A + 4/5*(B-A)
                P3 = 0.5*(A+C)
                P4 = 0.5*(P2+C)

                F1 = (0, A, P3, P1)
                F2 = (0, P2, P3, P1)
                F3 = (0, P3, P2, P4)
                F4 = (0, P3, C, P4)
                F5 = (0, C, B, P2)
                result += [F1, F2, F3, F4, F5] 
        return result

##### Functions for the the global tiling pattern after substitution #####

class Tiling:
    @staticmethod
    def uniq_vertices(faces: List[Face]) -> List[complex]:
        '''function to get all the unique vertices of the list of faces'''
        vtxs: List[complex] = []
        for face in faces:
            for va in face[1:]:  # enumerate the vertices of the face
                if not any(GeometryUtils.same_vtx(va, v) for v in vtxs):
                    vtxs.append(va)
        return vtxs
    
    @staticmethod
    def uniq_edges(faces: List[Face]) -> List[Edge]:
        '''function to get all the unique edges of the list of faces'''
        edges: List[Edge] = []
        for face in faces:
            edges_to_add = GeometryUtils.get_trig_edges(face)
            for ea in edges_to_add:
                if not any(GeometryUtils.same_edge(ea, e) for e in edges):
                    edges.append(ea)
        return edges

    @staticmethod
    def draw(faces: List[Face], vertices: List[complex]):
        vertices_pos = np.array([[vertex.real, vertex.imag] for vertex in vertices])
        edges = Tiling.get_edges(faces, vertices)
        fig, ax = plt.subplots()
        ax.scatter(np.array(vertices_pos)[:,0], np.array(vertices_pos)[:,1], marker='o', c='b')
        edges = Tiling.get_edges(faces, vertices)
        for edge in edges:
            ax.plot([edge[0].real, edge[1].real], [edge[0].imag, edge[1].imag], color='k', linewidth=0.5)
    
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

        return fig, ax