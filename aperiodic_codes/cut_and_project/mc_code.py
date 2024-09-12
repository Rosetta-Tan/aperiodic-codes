'''
Construct a pair of X and Z parity-check matrices on 3D cut-and-project tiling
from HGP of two classical codes on the 3D cubic lattice.
H1, H2: polynomial -> HGP -> 6D Hx, Hz -> cut & project -> 3D new Hx, Hz
'''
from os import getpid
import numpy as np
from numpy import array,exp,sqrt,cos,sin,pi
from aperiodic_codes.cut_and_project.cnp_utils import *

def coord6_to_ind(coords, n):
    '''
    (x0, x1, x2) -> coordinates of code1
    (x2, x3, x4) -> coordinates of code2
    6D ind: (x0* n**2 + x1 * n + x2) * n**3 + (x3 * n**2 + x4 * n + x5)
    Args:
        coords: np.array, shape=(6,)
    '''
    N = 2*n+1
    x0, x1, x2, x3, x4, x5 = coords
    return ((x0+n) * N**2 + (x1+n) * N + (x2+n)) * N**3 + ((x3+n) * N**2 + (x4+n) * N + (x5+n))

def ind_to_coord6(ind, n):
    '''
    ind -> (x0, x1, x2, x3, x4, x5)
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
    Check if two points are connected by the parity-check in the 6D lattice.
    '''
    assert len(pt1) == len(pt2) == 6, 'Points should be 6-element vector'
    assert parity_check_matrix.shape == ((2*n+1)**6, (2*n+1)**6), \
    print(f'parity_check_matrix.shape: {parity_check_matrix.shape}')
    ind1 = coord6_to_ind(pt1, n)
    ind2 = coord6_to_ind(pt2, n)
    return parity_check_matrix[ind1, ind2] == 1

def get_neighbors(pt, parity_check_matrix, n):
    '''
    Get neighbors in 6D.
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
    prefix = "/data/apc"
    pid = getpid();
    f_base = f'{prefix}/6d_to_3d/{pid}';
    DIRS = 27;
    nTh = 8;
    n = 3;

    # Generate 6d lattice objects
    lat_pts = gen_lat(low=-n, high=n, dim=6);
    assert lat_pts.shape[0] == (2*n+1)**6, 'Number of lattice points should be N**6'
    voronoi = gen_voronoi(dim=6);
    bulk = np.all(abs(lat_pts) != n,axis=1);
    P = proj_mat();
    proj_pos = P[:,:3];
    proj_neg = P[:,3:];

    # Setup RNG and MC params
    rng = np.random.default_rng(pid);
    offset = rng.uniform(0.0,1.0,6);
    beta = 15.0;
    cur_energy = np.inf;

    cut_ind, full_to_cut_ind_map = cut_ext(lat_pts, voronoi, proj_neg, offset, f_base, nTh);
    cut_pts = lat_pts[cut_ind,:];
    proj_pts = project(cut_pts, proj_pos);
    n_points = len(cut_ind);
    
    # Initial codes are generated randomly
    cur_code_1 = np.zeros(DIRS,dtype=int);
    cur_code_2 = np.zeros(DIRS,dtype=int);
    while np.sum(cur_code_1) < 7: cur_code_1 = rng.integers(0,1,DIRS,endpoint=True);
    while np.sum(cur_code_2) < 7: cur_code_2 = rng.integers(0,1,DIRS,endpoint=True);
    prop_code_1 = cur_code_1.copy();
    prop_code_2 = cur_code_2.copy();

    while True:
        # Try proposed codes
        h1 = gen_code_3d(prop_code_1,n);
        h2 = gen_code_3d(prop_code_2,n);
        hx, hz = gen_hgp(h1, h2);
        hx_vv, hx_cc = get_hx_vv_cc(hx, n);
        hz_vv, hz_cc = get_hz_vv_cc(hz, n);

        new_hx_vv = gen_new_pc_matrix(cut_pts, full_to_cut_ind_map, hx_vv, n);
        new_hx_cc = gen_new_pc_matrix(cut_pts, full_to_cut_ind_map, hx_cc, n);
        new_hz_vv = gen_new_pc_matrix(cut_pts, full_to_cut_ind_map, hz_vv, n);
        new_hz_cc = gen_new_pc_matrix(cut_pts, full_to_cut_ind_map, hz_cc, n);

        n_anti = check_comm_after_proj(new_hx_vv, new_hx_cc, new_hz_vv, new_hz_cc);
        prop_energy = n_anti/n_points;
        acc_prob = min(1.0,exp(-beta*(prop_energy-cur_energy)));

        # Accept with Boltzmann probability if projected code is sufficiently connected
        if np.sum(new_hx_vv)/n_points >= 3.0 and np.sum(new_hx_cc)/n_points >= 3.0 and rng.random() < acc_prob:
            if prop_energy < cur_energy:
                np.savez(f'{f_base}_opt.npz', proj_pts=proj_pts,code_1=prop_code_1,code_2=prop_code_2,
                         hx_vv=new_hx_vv,hx_cc=new_hx_cc,hz_vv=new_hz_vv,hz_cc=new_hz_cc);
                
            cur_code_1 = prop_code_1.copy();
            cur_code_2 = prop_code_2.copy();
            cur_energy = prop_energy;
            f = open(f'{f_base}.log','a');
            f.write(','.join(map(str,offset))+','+','.join(map(str,prop_code_1))+','+','.join(map(str,prop_code_2))+f',{n_anti},{n_points},True\n');
            f.close();
        else:
            f = open(f'{f_base}.log','a');
            f.write(','.join(map(str,offset))+','+','.join(map(str,prop_code_1))+','+','.join(map(str,prop_code_2))+f',{n_anti},{n_points},False\n');
            f.close();
        
        np.savez(f'{f_base}_cur.npz', proj_pts=proj_pts,code_1=prop_code_1,code_2=prop_code_2,
                 hx_vv=new_hx_vv,hx_cc=new_hx_cc,hz_vv=new_hz_vv,hz_cc=new_hz_cc);

        if(n_anti == 0):
            break;

        # Generate proposed cut
        count = 0;
        while count == 0 or np.sum(prop_code_1[1:]) < 6:
            prop_code_1 = cur_code_1.copy();
            flip = rng.integers(0,DIRS,1)[0];
            prop_code_1[flip] = 1-prop_code_1[flip];
            count += 1;

        count = 0;
        while count == 0 or np.sum(prop_code_2[1:]) < 6:
            prop_code_2 = cur_code_2.copy();
            flip = rng.integers(0,DIRS,1)[0];
            prop_code_2[flip] = 1-prop_code_2[flip];
            count += 1;
