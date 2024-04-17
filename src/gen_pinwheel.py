import numpy as np
import matplotlib.pyplot as plt
from ldpc.mod2 import row_basis, nullspace, rank
import os
from timeit import default_timer as timer
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class Triangle:
    def __init__(self, ctg: int, A, B, C):
        self.ctg = ctg  # ctg: category of the triangle
        self.A = A
        self.B = B
        self.C = C

class GeometryUtils:
    @staticmethod
    def subdivide(triangles):
        # Your implementation of subdivide function
        result: List[Triangle] = []
        for triangle in triangles:
            pt1 = triangle.A + 0.4 * (triangle.B - triangle.A)
            pt2 = triangle.A + 0.8 * (triangle.B - triangle.A)
            pt3 = 0.5 * (triangle.A + triangle.C)
            pt4 = 0.5 * (pt2 + triangle.C)

            trig1 = Triangle(triangle.ctg, triangle.A, pt3, pt1)
            trig2 = Triangle(triangle.ctg, pt2, pt3, pt1)
            trig3 = Triangle(triangle.ctg, pt2, triangle.B, pt4)
            trig4 = Triangle(triangle.ctg, pt3, triangle.C, pt4)
            trig5 = Triangle(triangle.ctg, triangle.C, triangle.B, pt2)
            result.extend([trig1, trig2, trig3, trig4, trig5])
        return result

    @staticmethod
    def close(pt1, pt2):
        return np.linalg.norm(pt1 - pt2) < 1e-5

    @staticmethod
    def get_vertices(faces):
        vertices = []
        for face in faces:
            vertices.append(face[1])
            vertices.append(face[2])
            vertices.append(face[3])
        vertices_new = []
        for v in vertices:
            if not any(GeometryUtils().close(v, v2) for v2 in vertices_new):
                vertices_new.append(v)
        return vertices_new

    @staticmethod
    def get_edges(faces, vertices):
        # Your implementation of get_edges function
        pass

    @staticmethod
    def get_geometric_center(face):
        # Your implementation of get_geometric_center function
        pass

class QCCode:
    def __init__(self, faces, vertices):
        self.faces = faces
        self.vertices = vertices
        self.h = self.get_qc_code()

    def get_qc_code(self):
        # Your implementation of get_qc_code function
        pass

    def draw(self):
        # Your implementation of draw function
        pass

    def get_logical_basis(self):
        # Your implementation of get_logical_basis function
        pass

class CodeDistanceCalculator:
    @staticmethod
    def get_classical_code_distance(h):
        # Your implementation of get_classical_code_distance function
        pass

    @staticmethod
    def get_classical_code_distance_special_treatment(h, gen, target_weight):
        # Your implementation of get_classical_code_distance_special_treatment function
        pass

    @staticmethod
    def get_classical_code_transpose_distance_special_treatment(h, gen, target_weight):
        # Your implementation of get_classical_code_transpose_distance_special_treatment function
        pass

# Usage example:
gen = 4
triangles = [Triangle(0, 0.+0.j, 2.+1.j, 2.+0.j)]
for _ in range(gen):
    triangles = GeometryUtils.subdivide(triangles)
vertices = GeometryUtils.get_vertices(triangles)
qc_code = QCCode(triangles, vertices)
logical_basis = qc_code.get_logical_basis()

d_bound, logical_op = CodeDistanceCalculator.get_classical_code_transpose_distance_special_treatment(
    qc_code.h, gen=4, target_weight=4)
print('d_bound = ', d_bound)
fig, ax = qc_code.draw_logical(logical_op)
savename = f'transpose_low_weight_logical.pdf'
ax.set_title(f'low weight logical operator')
fig.set_size_inches(12, 12)
fig.savefig(os.path.join(savedir, subdir, savename), bbox_inches='tight')
plt.show()