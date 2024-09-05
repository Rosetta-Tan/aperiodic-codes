'''
Construct a pair of X and Z parity-check matrices on 3D cut-and-project tiling
from HGP of two classical codes on the 3D cubic lattice.
H1, H2: polynomial -> HGP -> 6D Hx, Hz -> cut & project -> 3D new Hx, Hz
'''
from os import getpid
from subprocess import run
import numpy as np
from numpy import array,exp,sqrt,cos,sin,pi
from scipy.linalg import expm
import scipy.sparse as sp
from scipy.stats import special_ortho_group
from aperiodic_codes.cut_and_project.cnp_utils import *
from aperiodic_codes.cut_and_project.code_param_utils import *

def coord3_to_idx(x, y, z, n):
    return np.nan if abs(x) > n or abs(y) > n or abs(z) > n else (x+n) * (2*n+1)**2 + (y+n) * (2*n+1) + (z+n);

def idx_to_coord3(idx, n):
    x = idx // (2*n+1)**2 - n;
    y = (idx % (2*n+1)**2) // (2*n+1) - n;
    z = idx % (2*n+1) - n;
    return x, y, z

def proj_mat():
    c = 1/sqrt(5);
    s = 2/sqrt(5);
    return array([[ s*cos(0*pi/5), s*sin(0*pi/5), c,  s*cos(0*pi/5),  s*sin(0*pi/5),  c],
                  [ s*cos(2*pi/5), s*sin(2*pi/5), c,  s*cos(4*pi/5),  s*sin(4*pi/5),  c],
                  [ s*cos(4*pi/5), s*sin(4*pi/5), c,  s*cos(8*pi/5),  s*sin(8*pi/5),  c],
                  [ s*cos(6*pi/5), s*sin(6*pi/5), c, s*cos(12*pi/5), s*sin(12*pi/5),  c],
                  [ s*cos(8*pi/5), s*sin(8*pi/5), c, s*cos(16*pi/5), s*sin(16*pi/5),  c],
                  [             0,             0, 1,              0,              0, -1]])*sqrt(2);

dirs = np.array([[ 0, 0, 0],
                 [ 1, 0, 0],
                 [-1, 0, 0],
                 [ 0, 1, 0],
                 [ 0,-1, 0],
                 [ 0, 0, 1],
                 [ 0, 0,-1],
                 [ 1, 1, 0],
                 [ 1, 0, 1],
                 [ 0, 1, 1],
                 [ 1,-1, 0],
                 [ 1, 0,-1],
                 [ 0, 1,-1],
                 [-1, 1, 0],
                 [-1, 0, 1],
                 [ 0,-1, 1],
                 [-1,-1, 0],
                 [-1, 0,-1],
                 [ 0,-1,-1],
                 [ 1, 1, 1],
                 [ 1, 1,-1],
                 [ 1,-1, 1],
                 [ 1,-1,-1],
                 [-1, 1, 1],
                 [-1, 1,-1],
                 [-1,-1, 1],
                 [-1,-1,-1]]);

def gen_code(spec,n):
    row = [];
    col = [];
    for i in range(-n,n+1):
        for j in range(-n,n+1):
            for k in range(-n,n+1):
                idx = coord3_to_idx(i, j, k, n);
                cur_col = np.array([coord3_to_idx(i+dirs[d,0], j+dirs[d,1], k+dirs[d,2], n) for d in np.where(spec == 1)[0]]);
                cur_col = cur_col[~np.isnan(cur_col)];
                row = row + [idx]*len(cur_col);
                col = col + cur_col.tolist();
                #for d in np.where(spec == 1)[0]:
                #    if(not np.isnan(loc := coord3_to_idx(i+dirs[d,0], j+dirs[d,1], k+dirs[d,2], n))):
                #        row.append(idx); col.append(loc);
    return sp.coo_matrix((np.ones_like(row,dtype=int),(row,col)) , shape=((2*n+1)**3,(2*n+1)**3)).tocsc();

def gen_hgp(h1, h2):
    '''
    Generate the HGP of H1 and H2.
    Hx = [H_1 x I_n2, I_m1 x H_2^T]
    Hz = [H_1^T x I_m2, I_n1 x H_2]
    Returns:
        Hx and Hz
    '''
    hx = sp.hstack((sp.kron(h1,sp.eye(h2.shape[1])),sp.kron(sp.eye(h1.shape[0]),h2.T.tocsc()))).tocsc();
    hz = sp.hstack((sp.kron(sp.eye(h1.shape[1]),h2),sp.kron(h1.T.tocsc(),sp.eye(h2.shape[0])))).tocsc();
    return hx, hz

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

def check_comm_after_proj(hx_vv, hx_cc, hz_vv, hz_cc,cut_bulk = None):
    '''
    Check commutation of all pairs of stabilizers.
    '''
    assert hx_vv.shape == hx_cc.shape == hz_vv.shape == hz_cc.shape
    hx = np.hstack((hx_vv, hx_cc))
    hz = np.hstack((hz_vv, hz_cc))
    return np.sum((hx @ hz.T) % 2) if cut_bulk == None else np.sum((hx @ hz.T)[np.ix_(cut_bulk,cut_bulk)] % 2);

def gen_rotation(thetas,d):
    assert len(thetas) == (d*(d-1))//2, "Must provide d*(d-1)/2 angles";
    T = np.zeros((d,d),dtype=float);
    a = 0;
    for i in range(d):
        for j in range(i+1,d):
            T[i,j] = thetas[a];
            T[j,i] = -thetas[a];
            a += 1;
    return expm(T);

def cut_ext(lat_pts , voronoi , proj_neg , offset, f_base, nTh):
    orth_pts = lat_pts @ proj_neg;
    orth_window = proj_neg.T @ (voronoi + np.tile([offset],(voronoi.shape[0],1))).T;
    np.savez(f'{f_base}_cut.npz',orth_pts=orth_pts,orth_window=orth_window);
    run(f'cut_multi {f_base} {nTh}',shell=True); 
    cut_inds = np.load(f'{f_base}_ind.npy');
    run(f'rm {f_base}_cut.npz',shell=True);
    run(f'rm {f_base}_ind.npy',shell=True);
    
    return cut_inds , {cut_inds[i]:i for i in range(len(cut_inds))};

if __name__ == '__main__':
    prefix = "/data/apc"
    pid = getpid();
    f_base = f'{prefix}/code/{pid}';
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
    beta = 30.0;
    cur_energy = np.inf;

    cut_ind, full_to_cut_ind_map = cut_ext(lat_pts, voronoi, proj_neg, offset, f_base, nTh);
    cut_pts = lat_pts[cut_ind,:];
    proj_pts = project(cut_pts, proj_pos);
    n_points = len(cut_ind);
    
    # Initial codes are generated randomly
    cur_code_1 = rng.integers(0,1,27,endpoint=True);
    cur_code_2 = rng.integers(0,1,27,endpoint=True);
    prop_code_1 = cur_code_1.copy();
    prop_code_2 = cur_code_2.copy();

    while(True):
        # Try proposed codes
        h1 = gen_code(prop_code_1,n);
        h2 = gen_code(prop_code_2,n);
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

        if(rng.random() < acc_prob):
            if(prop_energy < cur_energy):
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
        while count == 0 or np.sum(prop_code_1[1:]) < 3:
            prop_code_1 = cur_code_1.copy();
            flip = rng.integers(0,27,1)[0];
            prop_code_1[flip] = 1-prop_code_1[flip];
            count += 1;

        count = 0;
        while count == 0 or np.sum(prop_code_2[1:]) < 3:
            prop_code_2 = cur_code_2.copy();
            flip = rng.integers(0,27,1)[0];
            prop_code_2[flip] = 1-prop_code_2[flip];
            count += 1;
