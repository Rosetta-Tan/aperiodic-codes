import unittest
from aperiodic_codes.cut_and_project.three_dim_self_dual import *

class TestThreeDimSelfDual(unittest.TestCase):
    def test_adj_mat(self):
        low = -2
        high = 3
        lat_pts = gen_lat(low, high, dim=6)
        proj_neg = gen_proj_neg()
        cut_pts, _ = cut(lat_pts, gen_voronoi(dim=6), proj_neg)
        adj_mat = gen_adj_mat(cut_pts)
        self.assertEqual(adj_mat.shape[0], cut_pts.shape[1])
        self.assertTrue(np.all(adj_mat == adj_mat.T))

    def test_laplacian_code(self):
        low = -2
        high = 3
        lat_pts = gen_lat(low, high, dim=6)
        proj_neg = gen_proj_neg()
        cut_pts, _ = cut(lat_pts, gen_voronoi(dim=6), proj_neg)
        adj_mat = gen_adj_mat(cut_pts)
        lap_code = get_laplacian_code(adj_mat)
        self.assertEqual(lap_code.shape[0], adj_mat.shape[0])
        self.assertEqual(lap_code.shape[1], adj_mat.shape[1])
        self.assertTrue(np.all(lap_code == lap_code.T))