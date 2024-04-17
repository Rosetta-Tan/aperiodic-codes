import sys, os, unittest
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.tus.helpers import GeometryUtils, Tiling

class TestTusHelpers(unittest.TestCase):
    # at generation 1, the number of faces is 1
    def test_uniq_vertices(self):
        faces = [(0, 0, 2+1j, 2)]
        self.assertEqual(Tiling.uniq_vertices(faces), [0, 2, 2 + 1j])

    def test_substitution(self):
        faces = [(0, 0, 2+1j, 2)]
        faces = GeometryUtils.subdivide(faces)
        self.assertEqual(len(faces), 5)