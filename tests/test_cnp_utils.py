import unittest
import numpy as np
from scipy.spatial import ConvexHull
from aperiodic_codes.cut_and_project.cnp_utils import *

class TestCNPUtils(unittest.TestCase):
    def test_proj_orthogonality(self):
        proj_pos = gen_proj_pos()
        proj_neg = gen_proj_neg()
        for i in range(3):
            assert np.isclose(np.dot(proj_pos[i], proj_neg[i]), 0), f'{i}-th vector in positive and negative projections are not orthogonal.'

    def test_voronoi_projection(self):
        voronoi = gen_voronoi(dim=6)
        proj_neg = gen_proj_neg()
        # Projecting into negative eigenspace
        triacontahedron = ConvexHull((proj_neg @ voronoi).T)
        assert len(triacontahedron.simplices) == 60
