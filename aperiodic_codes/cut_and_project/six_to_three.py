'''
Construct a pair of X and Z parity-check matrices on 3D cut-and-project tiling
from HGP of two classical codes on the 3D cubic lattice.
H1, H2: polynomial -> HGP -> 6D Hx, Hz -> cut & project -> 3D new Hx, Hz
'''
import logging.config
from os import getpid
from subprocess import run
import logging
import numpy as np
from numpy import array,sqrt,cos,sin,pi
from aperiodic_codes.cut_and_project.cnp_utils import *
from aperiodic_codes.cut_and_project.code_param_utils import *
logging.basicConfig(level=logging.INFO)

# def symmod(x,n):
#     return (x+n)%(2*n+1)-n;
# 
# def coord3_to_idx(x, y, z, n):
#     return (symmod(x,n)+n) * (2*n+1)**2 + (symmod(y,n)+n) * (2*n+1) + (symmod(z,n)+n)

def coord6_to_ind(coords, n):
    '''
    (x0, x1) -> coordinates of code1
    (x2, x3, x4) -> coordinates of code2
    6D ind: (x0* n**2 + x1 * n + x2) * n**3 + (x3 * n**2 + x4 * n + x5)
    Args:
        coords: np.array, shape=(5,)
    '''
    N = 2*n+1
    x0, x1, x2, x3, x4, x5 = coords
    return ((x0+n) * N**2 + (x1+n) * N + (x2+n)) * N**3 + ((x3+n) * N**2 + (x4+n) * N + (x5+n))

def ind_to_coord6(ind, n):
    '''
    ind -> (x0, x1, x2, x3, x4)
    '''
    N = 2*n+1
    x0 = (ind // N**3) // N**2 - n;
    x1 = ((ind // N**3) % N**2) // N - n;
    x2 = ((ind // N**3) % N**2) % N - n;
    x3 = (ind % N**3) // N**2 - n;
    x4 = ((ind % N**3) % N**2) // N - n;
    x5 = ((ind % N**3) % N**2) % N - n;
    return array([x0, x1, x2, x3, x4, x5])

def proj_mat():
    c = 1/sqrt(5);
    s = 2/sqrt(5);
    return array([[ s*cos(0*pi/5), s*sin(0*pi/5), c,  s*cos(0*pi/5),  s*sin(0*pi/5),  c],
                  [ s*cos(2*pi/5), s*sin(2*pi/5), c,  s*cos(4*pi/5),  s*sin(4*pi/5),  c],
                  [ s*cos(4*pi/5), s*sin(4*pi/5), c,  s*cos(8*pi/5),  s*sin(8*pi/5),  c],
                  [ s*cos(6*pi/5), s*sin(6*pi/5), c, s*cos(12*pi/5), s*sin(12*pi/5),  c],
                  [ s*cos(8*pi/5), s*sin(8*pi/5), c, s*cos(16*pi/5), s*sin(16*pi/5),  c],
                  [             0,             0, 1,              0,              0, -1]])/sqrt(2);

def get_hx_vv_cc(hx, n):
    '''
    Generate the Hx matrix for VV and CC type qubits.
    '''
    hx_vv, hx_cc = hx[:, 0:hx.shape[1]//2], hx[:, hx.shape[1]//2:]
    assert hx_vv.shape == hx_cc.shape == ((2*n+1)**6, (2*n+1)**6), \
    print(f'hx_vv.shape: {hx_vv.shape}, hx_cc.shape: {hx_cc.shape}')
    return hx_vv, hx_cc

def get_hz_vv_cc(hz, n):
    '''
    Generate the Hz matrix for VV and CC type qubits.
    '''
    hz_vv, hz_cc = hz[:, 0:hz.shape[1]//2], hz[:, hz.shape[1]//2:]
    assert hz_vv.shape == hz_cc.shape == ((2*n+1)**6, (2*n+1)**6), \
    print(f'hz_vv.shape: {hz_vv.shape}, hz_cc.shape: {hz_cc.shape}')
    return hz_vv, hz_cc

def are_connected(pt1, pt2, parity_check_matrix, n):  
    '''
    Check if two points are connected by the parity-check in the 5D lattice.
    '''
    assert len(pt1) == len(pt2) == 6, 'Points should be 5-element vector'
    assert parity_check_matrix.shape == ((2*n+1)**6, (2*n+1)**6), \
    print(f'parity_check_matrix.shape: {parity_check_matrix.shape}')
    ind1 = coord6_to_ind(pt1, n)
    ind2 = coord6_to_ind(pt2, n)
    return parity_check_matrix[ind1, ind2] == 1

def get_neighbors(pt, parity_check_matrix, n):
    '''
    Get neighbors in 5D.
    '''
    ind = coord6_to_ind(pt, n)
    neighbor_inds = np.where(parity_check_matrix[ind].todense().A1 == 1)[0]
    neighbors = [ind_to_coord6(neighbor_ind, n) for neighbor_ind in neighbor_inds]
    return neighbor_inds, neighbors

def gen_new_pc_matrix(cut_pts,
                     full_to_cut_ind_map,
                     original_parity_check_matrix, n):
    '''
    Generate the new parity-check matrix after cutting and projecting.
    new_parity_check_matrix will contain all-zero rows,
    purge after combining CC and VV type
    '''
    n_cut = cut_pts.shape[0]
    new_parity_check_matrix = np.zeros((n_cut, n_cut), dtype=int)
    
    # Connect neighboring points in cut_pts
    for i_cut in range(n_cut):
        cut_pt = cut_pts[i_cut,:]
        neighbor_inds, _ = get_neighbors(cut_pt, original_parity_check_matrix, n)
        for i_full_neighbor in neighbor_inds:
            if i_full_neighbor in full_to_cut_ind_map:
                i_cut_neighbor = full_to_cut_ind_map[i_full_neighbor]
                new_parity_check_matrix[i_cut, i_cut_neighbor] = 1
            
    return new_parity_check_matrix

if __name__ == '__main__':
    from config import prefix, tests
    # pid = "20240920_n=3_DIRS27_1"
    pid = "20240926_n=3_DIRS27_2"
    f_base = f'{prefix}/6d_to_3d/{pid}';
    nTh = 8;
    n = 3;

    # Generate 6d lattice objects
    lat_pts = gen_lat(low=-n, high=n, dim=6)
    assert lat_pts.shape[0] == (2*n+1)**6, 'Number of lattice points should be N**6'
    voronoi = gen_voronoi(dim=6)
    offset = array(tests[str(pid)]["offset"])
    # offset = np.zeros(6)
    bulk = np.all(abs(lat_pts) != n,axis=1);
    P = proj_mat();
    proj_pos = P[:,:3];
    proj_neg = P[:,3:];
    
    #R = gen_rotation((-3*pi/30,pi/30,2*pi/30,7*pi/30,3*pi/30,5*pi/30,3*pi/30,0.0,6*pi/30,4*pi/30),5);
    # R = special_ortho_group.rvs(6);
    thetas = tests[str(pid)]["thetas"]
    # thetas = np.zeros(15)
    code1 = array(tests[str(pid)]["code1"])
    code2 = array(tests[str(pid)]["code2"])
    # code1 = [1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,1,1,1]
    # code2 = [1,1,0,0,0,0,1,1,0,0,0,1,0,0,0,0,0,0,1,0,1,0,1,0,0,0,0]
    R = gen_rotation(thetas, 6)
    h1 = gen_code_3d(code1 ,n);
    h2 = gen_code_3d(code2 ,n);
    hx, hz = gen_hgp(h1, h2);
    hx_vv, hx_cc = get_hx_vv_cc(hx, n);
    hz_vv, hz_cc = get_hz_vv_cc(hz, n);

    proj_pos = R @ proj_pos;
    proj_neg = R @ proj_neg;

    cut_ind, full_to_cut_ind_map = cut_ext(lat_pts, voronoi, proj_neg, offset, f_base, nTh);
    cut_pts = lat_pts[cut_ind,:];
    proj_pts = project(cut_pts, proj_pos)
    cut_bulk = [i for i in range(len(cut_ind)) if bulk[cut_ind[i]]]
    logging.debug(f'cut_bulk: {cut_bulk}')


    new_hx_vv = gen_new_pc_matrix(cut_pts, full_to_cut_ind_map, hx_vv, n)
    new_hx_cc = gen_new_pc_matrix(cut_pts, full_to_cut_ind_map, hx_cc, n)
    new_hz_vv = gen_new_pc_matrix(cut_pts, full_to_cut_ind_map, hz_vv, n)
    new_hz_cc = gen_new_pc_matrix(cut_pts, full_to_cut_ind_map, hz_cc, n)

    print(f'shape of proj_pts: {proj_pts.shape}')
    np.savez(f'{f_base}.npz', proj_pts=proj_pts, cut_bulk=cut_bulk, offset=offset,
             hx_vv=new_hx_vv,hx_cc=new_hx_cc,hz_vv=new_hz_vv,hz_cc=new_hz_cc);
    np.save(f'{f_base}_cut_ind.npy', cut_ind)
        
    # Check commutation
    print(f'n_bulk: {len(cut_bulk)}')
    print(f'n_anti: {check_comm_after_proj(new_hx_vv, new_hx_cc, new_hz_vv, new_hz_cc, cut_bulk)}')
    #print(get_classical_code_distance_time_limit(np.hstack((new_hx_cc,new_hx_vv)),10));
    #print(get_classical_code_distance_time_limit(np.hstack((new_hz_cc,new_hz_vv)),10)); 
    # print(check_comm_after_proj(new_hx_vv, new_hx_cc, new_hz_vv, new_hz_cc))
