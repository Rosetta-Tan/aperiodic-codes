# 3D cut and project tiling
import numpy as np
from scipy.optimize import linprog
#from scipy.spatial import ConvexHull

def gen_lat(low, high, dim):
    '''
    Generate the dim-dimensional hypercubic lattice
    Returns:
        lat_pts: np.array, shape=(dim, n**dim)
        - lattice pts are column vectors
        - lat_pts[0, :]: x0 coordinates for all pts
        - lat_pts[1, :]: x1 coordinates
        - ...
        - lat_pts[dim-1, :]: x_{dim-1} coordinates
    '''
    lat_pts = np.array(
            np.meshgrid(*([np.arange(low, high+1)] * dim), indexing='ij')
        ).reshape(dim, -1)
    return lat_pts

def gen_voronoi(dim):
    '''
    Compute the dim-dimensional Voronoi unit cell centered around origin
    Returns:
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

def is_point_in_hull(points, x):
    n_points = len(points)
    n_dim = len(x)
    c = np.zeros(n_points)
    A = np.r_[points.T,np.ones((1,n_points))]
    b = np.r_[x, np.ones(1)]
    lp = linprog(c, A_eq=A, b_eq=b)
    return lp.success
'''
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
'''
def gen_proj_pos():
    '''
    Generate the projection matrix into the positive eigenvalue 3D subspace
    shape: (3, 6)
    Notation follows R. M. K. Dietl and J.-H. Eschenburg
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

def cut(lat_pts, voronoi, proj, offset_vec=None):
    '''
    Select lattice points in the 6D lattice.
    There are two qubits per vertex in the 6D lattice.
    A stabilizer will involve both VV and CC type qubits.
    We keep the feature that there are two qubits per vertex in 3D.

    VV type qubits are the first n**6 qubits.
    CC type qubits are the last n**6 qubits.
        - condition: the lattice point is inside the ocnvex hull of
        the Voronoi unit cell projected to the perpendiculr 3D space.
    Args:
        lat_pts: np.array, shape=(6, n**6)
        - lattice points in 6D
        voronoi: np.array, shape=(6, 2**6)
        - voronoi cell around origin in 6D
        proj: np.array, shape=(3, 6)
        - projection isometry matrix into the corresponding eigval 3D subspace
        (default: negative eigenvalue)
    Returns:
        cut_pts: np.array, shape=(6, n_cut)
        - selected points in 6D
        - cut_pts[0]: x0 coordinates
        - cut_pts[1]: x1 coordinates
        - ...
        - cut_pts[5]: x5 coordinates
    '''
    if offset_vec is not None:
        voronoi_shifted = voronoi + np.tile([offset_vec],(voronoi.shape[1],1)).T;
    else:
        voronoi_shifted = voronoi;

    window = (proj @ voronoi_shifted).T; 
 
    # Select lattice points inside the convex hull
    full_to_cut_ind_map = {}
    cut_pts = []
    for i in range(lat_pts.shape[1]):
        pt_proj = proj @ lat_pts[:, i]
        if is_point_in_hull(window,pt_proj):
            full_to_cut_ind_map.update({i: len(cut_pts)})
            cut_pts.append(lat_pts[:, i])
    cut_pts = np.asarray(cut_pts).T # shape=(6, n_cut)
 
    return cut_pts, full_to_cut_ind_map

def project(cut_pts, proj):
    '''
    Project the selected points into the selected eigenvalue's 3D subspace.
    (default: positive eigenvalue)
    Args:
        proj: np.array, shape=(3, 6)
    Returns:
        projected points: np.array, shape=(3, n_cut)
    '''
    return proj @ cut_pts
