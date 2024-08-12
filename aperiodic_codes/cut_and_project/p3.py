'''
Construct a pair of X and Z parity-check matrices on 3D cut-and-project tiling
from HGP of two classical codes on the 3D cubic lattice.
H1, H2: polynomial -> HGP -> 6D Hx, Hz -> cut & project -> 3D new Hx, Hz
'''
import numpy as np
from numpy import array,sqrt,cos,sin,pi
from bposd.hgp import hgp
from scipy.stats import special_ortho_group
from aperiodic_codes.cut_and_project.cnp_utils import *

def symmod(x,n):
    return (x+n)%(2*n+1)-n;

def coord2_to_idx(x, y, n):
    return (symmod(x,n)+n) * (2*n+1) + (symmod(y,n)+n)

def idx_to_coord2(idx, n):
    x = idx // (2*n+1) - n
    y = idx % (2*n+1) - n
    return x, y

def coord3_to_idx(x, y, z, n):
    return (symmod(x,n)+n) * (2*n+1)**2 + (symmod(y,n)+n) * (2*n+1) + (symmod(z,n)+n)

def idx_to_coord3(idx, n):
    x = idx // (2*n+1)**2 - n;
    y = (idx % (2*n+1)**2) // (2*n+1) - n;
    z = idx % (2*n+1) - n;
    return x, y, z

def proj_mat():
    return array([[   cos(0), cos(2*pi/5), cos(4*pi/5),  cos(6*pi/5),  cos(8*pi/5)],
                  [   sin(0), sin(2*pi/5), sin(4*pi/5),  sin(6*pi/5),  sin(8*pi/5)],
                  [   cos(0), cos(4*pi/5), cos(8*pi/5), cos(12*pi/5), cos(16*pi/5)],
                  [   sin(0), sin(4*pi/5), sin(8*pi/5), sin(12*pi/5), sin(16*pi/5)],
                  [1/sqrt(2),   1/sqrt(2),   1/sqrt(2),    1/sqrt(2),    1/sqrt(2)]])*sqrt(2/5);

def gen_h1(n):
    '''
    Generate the first classical code in 2D.
    Polynomial: f(x, y) = 1 + x + y
    Coordinate to index: (x, y) = (x * n + y)
    Parity-check relation:
    (x, y) -> {
        x + 1, y
        x, y + 1
    }
    Returns:
        np.array, shape=(2, n**2)
    '''
    h = np.zeros(((2*n+1)**2, (2*n+1)**2), dtype=int)
    for i in range(-n,n+1):
        for j in range(-n,n+1):
            idx = coord2_to_idx(i, j, n)
            h[idx, idx] = 1
            h[idx, coord2_to_idx(i-1, j, n)] = 1
            h[idx, coord2_to_idx(i, j+1, n)] = 1
            #h[idx, coord2_to_idx(i+1, j+1, n)] = 1
    return h

def gen_h2(n):
    '''
    Generate the second classical code in 3D.
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
    h = np.zeros(((2*n+1)**3, (2*n+1)**3), dtype=int)
    for i in range(-n,n+1):
        for j in range(-n,n+1):
            for k in range(-n,n+1):
                idx = coord3_to_idx(i, j, k, n)
                h[idx, idx] = 1
                h[idx, coord3_to_idx(i+1, j, k, n)] = 1
                h[idx, coord3_to_idx(i, j-1, k, n)] = 1
                h[idx, coord3_to_idx(i, j, k-1, n)] = 1
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

def coord5_to_ind(coords, n):
    '''
    (x0, x1) -> coordinates of code1
    (x2, x3, x4) -> coordinates of code2
    6D ind: (x0* n**2 + x1 * n + x2) * n**3 + (x3 * n**2 + x4 * n + x5)
    Args:
        coords: np.array, shape=(5,)
    '''
    x0, x1, x2, x3, x4 = coords
    return ((x0+n) * (2*n+1) + (x1+n)) * (2*n+1)**3 + ((x2+n) * (2*n+1)**2 + (x3+n) * (2*n+1) + (x4+n))

def ind_to_coord5(ind, n):
    '''
    ind -> (x0, x1, x2, x3, x4)
    '''
    x0 = (ind // (2*n+1)**3) // (2*n+1) - n;
    x1 = (ind // (2*n+1)**3) % (2*n+1) - n;
    x2 = (ind % (2*n+1)**3) // ((2*n+1)**2) - n;
    x3 = ((ind % (2*n+1)**3) % (2*n+1)**2) // (2*n+1) - n;
    x4 = ((ind % (2*n+1)**3) % (2*n+1)**2) % (2*n+1) - n;
    return np.array([x0, x1, x2, x3, x4])

def get_hx_vv_cc(hx, n):
    '''
    Generate the Hx matrix for VV and CC type qubits.
    '''
    hx_vv, hx_cc = hx[:, 0:hx.shape[1]//2], hx[:, hx.shape[1]//2:]
    assert hx_vv.shape == hx_cc.shape == ((2*n+1)**5, (2*n+1)**5), \
    print(f'hx_vv.shape: {hx_vv.shape}, hx_cc.shape: {hx_cc.shape}')
    return hx_vv, hx_cc

def get_hz_vv_cc(hz, n):
    '''
    Generate the Hz matrix for VV and CC type qubits.
    '''
    hz_vv, hz_cc = hz[:, 0:hz.shape[1]//2], hz[:, hz.shape[1]//2:]
    assert hz_vv.shape == hz_cc.shape == ((2*n+1)**5, (2*n+1)**5), \
    print(f'hz_vv.shape: {hz_vv.shape}, hz_cc.shape: {hz_cc.shape}')
    return hz_vv, hz_cc

def are_connected(pt1, pt2, parity_check_matrix, n):  
    '''
    Check if two points are connected by the parity-check in the 5D lattice.
    '''
    assert len(pt1) == len(pt2) == 5, 'Points should be 5-element vector'
    assert parity_check_matrix.shape == ((2*n+1)**5, (2*n+1)**5), \
    print(f'parity_check_matrix.shape: {parity_check_matrix.shape}')
    ind1 = coord5_to_ind(pt1, n)
    ind2 = coord5_to_ind(pt2, n)
    return parity_check_matrix[ind1, ind2] == 1

def get_neighbors(pt, parity_check_matrix, n):
    '''
    Get neighbors in 5D.
    '''
    ind = coord5_to_ind(pt, n)
    neighbor_inds = np.where(parity_check_matrix[ind] == 1)[0]
    neighbors = [ind_to_coord5(neighbor_ind, n) 
                 for neighbor_ind in neighbor_inds]
    return neighbor_inds, neighbors

def gen_new_pc_matrix(cut_pts,
                     full_to_cut_ind_map,
                     original_parity_check_matrix, n):
    '''
    Generate the new parity-check matrix after cutting and projecting.
    new_parity_check_matrix will contain all-zero rows,
    purge after combining CC and VV type
    '''
    n_cut = cut_pts.shape[1]
    new_parity_check_matrix = np.zeros((n_cut, n_cut), dtype=int)
    # Connect neighboring points in cut_pts
    for i_cut in range(n_cut):
        cut_pt = cut_pts[:, i_cut]
        neighbor_inds, _ = get_neighbors(cut_pt, 
                                        original_parity_check_matrix, n)
        # Check if all neighbors are inside the convex hull
        all_neighbors_in_hull = True
        #for i_full_neighbor in neighbor_inds:
        #    if i_full_neighbor not in full_to_cut_ind_map:
        #        all_neighbors_in_hull = False
    
        if all_neighbors_in_hull:
            for i_full_neighbor in neighbor_inds:
                if i_full_neighbor in full_to_cut_ind_map:
                    i_cut_neighbor = full_to_cut_ind_map[i_full_neighbor]
                    new_parity_check_matrix[i_cut, i_cut_neighbor] = 1
    return new_parity_check_matrix

def check_comm_after_proj(hx_vv, hx_cc, hz_vv, hz_cc):
    '''
    Check commutation of all pairs of stabilizers.
    '''
    assert hx_vv.shape == hx_cc.shape == hz_vv.shape == hz_cc.shape
    hx = np.hstack((hx_vv, hx_cc))
    hz = np.hstack((hz_vv, hz_cc))
    #return np.all((hx @ hz.T) % 2 == 0) and np.all((hz @ hx.T) % 2 == 0)
    return np.sum((hx @ hz.T) % 2)

if __name__ == '__main__':
    prefix = "/data/apc"
    n = 3;
    lat_pts = gen_lat(low=-n, high=n, dim=5)
    assert lat_pts.shape[1] == (2*n+1)**5, 'Number of lattice points should be n**5'
    voronoi = gen_voronoi(dim=5)
    P = proj_mat();
    proj_pos = P[:2,:];
    proj_neg = P[2:,:];
    
    R = special_ortho_group.rvs(5);
    proj_pos = proj_pos @ R.T;
    proj_neg = proj_neg @ R.T;

    h1 = gen_h1(n)
    h2 = gen_h2(n)
    hx, hz = gen_hgp(h1, h2)
    np.save(f'{prefix}/penrose_p3/hx_n={n}.npy', hx)
    np.save(f'{prefix}/penrose_p3/hz_n={n}.npy', hz)

    hx_vv, hx_cc = get_hx_vv_cc(hx, n)
    hz_vv, hz_cc = get_hz_vv_cc(hz, n)
    np.save(f'{prefix}/penrose_p3/hx_vv_n={n}.npy', hx_vv)
    np.save(f'{prefix}/penrose_p3/hx_cc_n={n}.npy', hx_cc)
    np.save(f'{prefix}/penrose_p3/hz_vv_n={n}.npy', hz_vv)
    np.save(f'{prefix}/penrose_p3/hz_cc_n={n}.npy', hz_cc)

    cut_pts, full_to_cut_ind_map, cut_to_full_ind_map = cut(lat_pts, voronoi, proj_neg)
    print(cut_pts.shape);
    proj_pts = project(cut_pts, proj_pos)
    new_hx_vv = gen_new_pc_matrix(cut_pts, full_to_cut_ind_map, hx_vv, n)
    new_hx_cc = gen_new_pc_matrix(cut_pts, full_to_cut_ind_map, hx_cc, n)
    new_hz_vv = gen_new_pc_matrix(cut_pts, full_to_cut_ind_map, hz_vv, n)
    new_hz_cc = gen_new_pc_matrix(cut_pts, full_to_cut_ind_map, hz_cc, n)

    """
    new_hx = np.hstack((new_hx_vv, new_hx_cc))
    new_hz = np.hstack((new_hz_vv, new_hz_cc))
    h_ac = (new_hx @ new_hz.T) % 2;
    print(h_ac);
    for i in range(h_ac.shape[0]):
        for j in range(h_ac.shape[1]):
            if h_ac[i,j] != 0:
                conflicts = np.argwhere(new_hx[i,:]+new_hz[j,:]==2)[:,0];
                print(str(i)+' '+str(j)+' '+str(conflicts));
                print(new_hx[i,:]);
                print(new_hz[j,:]);
                if(np.sum(new_hx[i,:]) > np.sum(new_hz[j,:])):
                    #print(new_hx[i,conflicts]);
                    new_hx[i,conflicts] = 0;
                    #print(new_hx[i,conflicts]);
                else:
                    new_hz[j,conflicts] = 0;

    new_hx_vv = new_hx[:,:new_hx_vv.shape[1]];
    new_hx_cc = new_hx[:,new_hx_vv.shape[1]:];
    new_hz_vv = new_hz[:,:new_hz_vv.shape[1]];
    new_hz_cc = new_hz[:,new_hz_vv.shape[1]:];
    """

    print(f'shape of proj_pts: {proj_pts.shape}')
    np.save(f'{prefix}/penrose_p3/proj_pts_n={n}.npy', proj_pts)
    np.save(f'{prefix}/penrose_p3/new_hx_vv_n={n}.npy', new_hx_vv)
    np.save(f'{prefix}/penrose_p3/new_hx_cc_n={n}.npy', new_hx_cc)
    np.save(f'{prefix}/penrose_p3/new_hz_vv_n={n}.npy', new_hz_vv)
    np.save(f'{prefix}/penrose_p3/new_hz_cc_n={n}.npy', new_hz_cc)

    # Check commutation
    print(check_comm_after_proj(new_hx_vv, new_hx_cc, new_hz_vv, new_hz_cc))
