# 3D cut and project tiling
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
import matplotlib.pyplot as plt

def gen_lat(low, high, dim):
    '''
    Generate the dim-dimensional hypercubic lattice
    Return:
        lat_pts: np.array, shape=(dim, n**dim)
        - lattice pts are column vectors
        - lat_pts[0, :]: x0 coordinates for all pts
        - lat_pts[1, :]: x1 coordinates
        - ...
        - lat_pts[dim-1, :]: x_{dim-1} coordinates
    '''
    lat_pts = np.array(
            np.meshgrid(*([np.arange(low, high)] * dim), indexing='ij')
        ).reshape(dim, -1)
    return lat_pts

def gen_voronoi(dim):
    '''
    Compute the dim-dimensional Voronoi unit cell centered around origin
    Return:
        voronoi: np.array, shape=(6, 2**6)
        - lattice pts are column vectors
        - voronoi[:, 0]: the first pt [-0.5, -0.5, ..., -0.5, -0.5].T
        - voronoi[:, 1]: the second pt [-0.5, -0.5, ..., -0.5, 0.5].T
        - ...
        - voronoi[:, 2**dim-1]: the last pt [0.5, 0.5, ..., 0.5, 0.5].T
    '''
    voronoi = np.array(
        np.meshgrid(*([[-0.5, 0.5]] * dim), indexing='ij')
    ).reshape(dim, -1)
    return voronoi