import unittest
import numpy as np
from aperiodic_codes.cut_and_project.three_dim_from_hgp import *

class TestSixDimConnectivity(unittest.TestCase):
    def test_hx_vv(self):
        n = 3
        hx_vv = np.load('../data/three_dim_commuting/hx_vv.npy')
        pt1 = np.array([0, 0, 0, 1, 1, 1])
        pt2 = np.array([1, 0, 0, 1, 1, 1])
        self.assertTrue(are_connected(pt1, pt2, hx_vv, n))

        pt1 = np.array([0, 0, 0, 2, 2, 2])
        pt2 = np.array([1, 0, 0, 2, 2, 2])
        self.assertTrue(are_connected(pt1, pt2, hx_vv, n))
        