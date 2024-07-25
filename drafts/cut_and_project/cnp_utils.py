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

def is_point_in_hull(point, hull, offset_vec=None):
    # Function to check if a point is inside the convex hull
    if offset_vec:
        raise NotImplementedError('Offset vector not implemented')
    # We add the homogeneous coordinate for the point
    point = np.append(point, 1)
    # Loop over each facet of the hull
    for equation in hull.equations:
        if offset_vec is None:
            if np.dot(equation, point) > 0:
                return False
        else:
            offset_3D = np.append(offset_3D, 0)
            if np.dot(equation, (point-offset_3D)) > 0:
                return False
    return True

def gen_proj_pos():
    '''
    Generate the projection matrix into the positive eigenvalue 3D subspace
    shape: (3, 6)
    Notation follows R. M. K. Dietl and J.-H. Eschenburg,
    see also Yi's notes page 4.
    '''
    costheta = 1/np.sqrt(5)
    sintheta = 2/np.sqrt(5)
    # Each vector is a row vector. shape: (1, 3)
    v6 = np.array([0, 0, np.sqrt(2)])
    v1 = np.array([np.sqrt(2)*sintheta, 0, np.sqrt(2)*costheta])
    v2 = np.array([
        np.sqrt(2)*sintheta*np.cos(2*np.pi/5),
        np.sqrt(2)*sintheta*np.sin(2*np.pi/5),
        np.sqrt(2)*costheta
        ])
    v3 = np.array([
        np.sqrt(2)*sintheta*np.cos(4*np.pi/5),
        np.sqrt(2)*sintheta*np.sin(4*np.pi/5),
        np.sqrt(2)*costheta
        ])
    v4 = np.array([
        np.sqrt(2)*sintheta*np.cos(6*np.pi/5),
        np.sqrt(2)*sintheta*np.sin(6*np.pi/5),
        np.sqrt(2)*costheta
        ])
    v5 = np.array([
        np.sqrt(2)*sintheta*np.cos(8*np.pi/5),
        np.sqrt(2)*sintheta*np.sin(8*np.pi/5),
        np.sqrt(2)*costheta
        ])
    proj_pos = np.vstack([v1, v2, v3, v4, v5, v6]).T  # shape: (3, 6)
    
    return proj_pos

def gen_proj_neg():
    '''
    Generate the projection matrix into the negative eigenvalue 3D subspace
    shape: (3, 6)
    Notation follows R. M. K. Dietl and J.-H. Eschenburg,
    see also Yi's notes page 4.
    '''
    costheta = 1/np.sqrt(5)
    sintheta = 2/np.sqrt(5)
    w6 = np.array([0, 0, -np.sqrt(2)])
    w1 = np.array([np.sqrt(2)*sintheta, 0, np.sqrt(2)*costheta])
    w2 = np.array([
        np.sqrt(2)*sintheta*np.cos(4*np.pi/5),
        np.sqrt(2)*sintheta*np.sin(4*np.pi/5),
        np.sqrt(2)*costheta
        ])
    w3 = np.array([
        np.sqrt(2)*sintheta*np.cos(8*np.pi/5),
        np.sqrt(2)*sintheta*np.sin(8*np.pi/5),
        np.sqrt(2)*costheta
        ])
    w4 = np.array([
        np.sqrt(2)*sintheta*np.cos(2*np.pi/5),
        np.sqrt(2)*sintheta*np.sin(2*np.pi/5),
        np.sqrt(2)*costheta
        ])
    w5 = np.array([
        np.sqrt(2)*sintheta*np.cos(6*np.pi/5),
        np.sqrt(2)*sintheta*np.sin(6*np.pi/5),
        np.sqrt(2)*costheta
        ])
    proj_neg = np.vstack([w1, w2, w3, w4, w5, w6]).T  # shape: (3, 6)

    return proj_neg