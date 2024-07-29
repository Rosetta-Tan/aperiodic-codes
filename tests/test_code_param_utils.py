import unittest
import numpy as np
from aperiodic_codes.cut_and_project.code_param_utils import *

class TestCodeParamUtils(unittest.TestCase):
    def test_gen_lat(self):
        low = -2
        high = 3
        dim = 6
        # TODO: 
        