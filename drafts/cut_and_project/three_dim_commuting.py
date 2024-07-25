'''
1. Construct two classical codes in 3D, H1 and H2
2. HGP of H1 and H2 -> Hx and Hz
3. Cut and project
4. Check commutation
'''
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from bposd.hgp import hgp
from cnp_utils import *

def coord_to_idx(x, y, z, n):
    return x * n**2 + y * n + z

def idx_to_coord(idx, n):
    x = idx // n**2
    y = (idx % n**2) // n
    z = idx % n
    return x, y, z

def gen_h1(n):
    '''
    Generate the first classical code in 3D.
    Polynomial: f(x, y, z) = 1 + x + y + z
    Coordinate to index: (x, y, z) = (x * n**2 + y * n + z)
    Parity-check relation:
    (x, y, z) -> {
        x + 1, y, z
        x, y + 1, z
        x, y, z + 1
    }
    Returns:
        np.array, shape=(3, n**3)
    '''
    h = np.zeros((n**3, n**3), dtype=int)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                idx = coord_to_idx(i, j, k, n)
                h[idx, coord_to_idx((i + 1) % n, j, k, n)] = 1
                h[idx, coord_to_idx(i, (j + 1) % n, k, n)] = 1
                h[idx, coord_to_idx(i, j, (k + 1) % n, n)] = 1            
    return h

def gen_h2(n):
    '''
    Generate the second classical code in 3D.
    Polynomial: f(x, y, z) = 1 + xy + yz + zx
    Coordinate to index: (x, y, z) = (x * n**2 + y * n + z)
    Parity-check relation:
    (x, y, z) -> {
        x + 1, y + 1, z
        x, y + 1, z + 1
        x + 1, y, z + 1
    }
    '''
    h = np.zeros((n**3, n**3), dtype=int)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                idx = coord_to_idx(i, j, k, n)
                h[idx, coord_to_idx((i + 1) % n, (j + 1) % n, k, n)] = 1
                h[idx, coord_to_idx(i, (j + 1) % n, (k + 1) % n, n)] = 1
                h[idx, coord_to_idx((i + 1) % n, j, (k + 1) % n, n)] = 1
    return h

def gen_hgp(h1, h2):
    '''
    Generate the HGP of H1 and H2.
    Hx = [H_1 x I_n2, I_m1 x H_2^T]
    Hz = [H_1^T x I_m2, I_n1 x H_2]
    Returns:
        Hx and Hz
    '''
    hgp_code = hgp(h1, h2)
    hx, hz = hgp_code.hx, hgp_code.hz
    return hx, hz

def six_dim_coord_to_ind(coords, n):
    '''
    (x0, x1, x2) -> coordinates of code1
    (x3, x4, x5) -> coordinates of code2
    6D ind: (x0* n**2 + x1 * n + x2) * n**3 + (x3 * n**2 + x4 * n + x5)
    '''
    x0, x1, x2, x3, x4, x5 = coords
    return (x0 * n**2 + x1 * n + x2) * n**3 + (x3 * n**2 + x4 * n + x5)

def six_dim_ind_to_coord(ind, n):
    '''
    ind -> (x0, x1, x2, x3, x4, x5)
    '''
    x0 = ind // (n**3 * n**2)
    x1 = (ind % (n**3 * n**2)) // (n**3)
    x2 = (ind % (n**3 * n**2)) % n**3
    x3 = x2 // (n**2)
    x4 = (x2 % (n**2)) // n
    x5 = x2 % n
    return x0, x1, x2, x3, x4, x5

def get_hx_vv_cc(hx, n):
    '''
    Generate the Hx matrix for VV and CC type qubits.
    '''
    hx_vv, hx_cc = hx[:, 0:hx.shape[1]//2], hx[:, hx.shape[1]//2:]
    assert hx_vv.shape == hx_cc.shape == (n**6, n**6), \
    print(f'hx_vv.shape: {hx_vv.shape}, hx_cc.shape: {hx_cc.shape}')
    return hx_vv, hx_cc

def get_hz_vv_cc(hz, n):
    '''
    Generate the Hz matrix for VV and CC type qubits.
    '''
    hz_vv, hz_cc = hz[:, 0:hz.shape[1]//2], hz[:, hz.shape[1]//2:]
    assert hz_vv.shape == hz_cc.shape == (n**6, n**6), \
    print(f'hz_vv.shape: {hz_vv.shape}, hz_cc.shape: {hz_cc.shape}')
    return hz_vv, hz_cc

def are_connected(pt1, pt2, parity_check_matrix, n):  
    '''
    Check if two points are connected by the parity-check in the 6D lattice.
    '''
    assert len(pt1) == len(pt2) == 6, 'Points should be 6-element vector'
    assert parity_check_matrix.shape == (n**6, n**6), \
    print(f'parity_check_matrix.shape: {parity_check_matrix.shape}')
    ind1 = six_dim_coord_to_ind(pt1, n)
    ind2 = six_dim_coord_to_ind(pt2, n)
    return parity_check_matrix[ind1, ind2] == 1

def cut(lat_pts, voronoi, original_parity_check_matrix,
        proj, n, offset_vec=None):
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
    Return:
        cut_pts: np.array, shape=(6, n_cut)
        - selected points in 6D
        - cut_pts[0]: x0 coordinates
        - cut_pts[1]: x1 coordinates
        - ...
        - cut_pts[5]: x5 coordinates
    '''
    # convex hull of projected Voronoi cell in 3D
    # scipy requires pts to be row vectors
    triacontahedron = ConvexHull((proj @ voronoi).T)
    
    # Select lattice points inside the convex hull
    cut_pts = []
    for i in range(lat_pts.shape[1]):
        pt_proj = proj @ lat_pts[:, i]
        if offset_vec is not None:
            offset_3D = proj @ offset_vec.reshape(-1, 1)
            if is_point_in_hull(pt_proj, triacontahedron, offset_3D):
                cut_pts.append(lat_pts[:, i])
        else:
            if is_point_in_hull(pt_proj, triacontahedron):
                cut_pts.append(lat_pts[:, i])

    '''
    connect neighboring points in cut_pts 
    according to connectivity in the 6D lattice:
    if two points are neighbors in the 6D lattice,
    then they are connected by an edge in the 3D tiling
    '''
    # Connect neighboring points in cut_pts
    cut_pts = np.asarray(cut_pts).T  # shape: (6, n_cut)
    n_cut = cut_pts.shape[1]
    new_parity_check_matrix = np.zeros((n_cut, n_cut), dtype=int)
    for i in range(n_cut):
        for j in range(i+1, n_cut):
            if are_connected(cut_pts[:, i], cut_pts[:, j],
                             original_parity_check_matrix, n):
                new_parity_check_matrix[i, j] = 1

    return cut_pts, new_parity_check_matrix

def project(cut_pts, proj):
    '''
    Project the selected points into the selected eigenvalue's 3D subspace.
    (default: positive eigenvalue)
    Args:
        proj: np.array, shape=(3, 6)
    Return:
        np.array, shape=(3, n_cut)
        - projected points
    '''
    return proj @ cut_pts

if __name__ == '__main__':
    n = 3
    assert n <=3, 'n should be less than or equal to 3'
    lat_pts = gen_lat(low=0, high=n, dim=6)
    assert lat_pts.shape[1] == n**6, 'Number of lattice points should be n**6'
    voronoi = gen_voronoi(dim=6)
    proj_pos = gen_proj_pos()
    proj_neg = gen_proj_neg()
    h1 = gen_h1(n)
    h2 = gen_h2(n)
    hx, hz = gen_hgp(h1, h2)
    hx_vv, hx_cc = get_hx_vv_cc(hx, n)
    hz_vv, hz_cc = get_hz_vv_cc(hz, n)
    np.save('../../data/three_dim_commuting/hx_vv.npy', hx_vv)
    np.save('../../data/three_dim_commuting/hx_cc.npy', hx_cc)
    np.save('../../data/three_dim_commuting/hz_vv.npy', hz_vv)
    np.save('../../data/three_dim_commuting/hz_cc.npy', hz_cc)

    cut_pts, new_hx_vv = cut(lat_pts, voronoi, hx_vv, proj_neg, n)
    _, new_hx_cc = cut(lat_pts, voronoi, hx_cc, proj_neg, n)
    _, new_hz_vv = cut(lat_pts, voronoi, hz_vv, proj_neg, n)
    _, new_hz_cc = cut(lat_pts, voronoi, hz_cc, proj_neg, n)
    proj_pts = project(cut_pts, proj_pos)

    np.save('../../data/three_dim_commuting/new_hx_vv.npy', new_hx_vv)
    np.save('../../data/three_dim_commuting/new_hx_cc.npy', new_hx_cc)
    np.save('../../data/three_dim_commuting/new_hz_vv.npy', new_hz_vv)
    np.save('../../data/three_dim_commuting/new_hz_cc.npy', new_hz_cc)

    new_hx = np.hstack((new_hx_vv, new_hx_cc))
    new_hz = np.hstack((new_hz_vv, new_hz_cc))

    # Check commutation
    for i in range(new_hx.shape[0]):
        for j in range(new_hz.shape[0]):
            if new_hx[:, i].dot(new_hz[:, j]) % 2 != 0:
                print(f'Commutation check failed: {i}, {j}')
                break



