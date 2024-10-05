'''
Construct a pair of X and Z parity-check matrices on 3D cut-and-project tiling
from HGP of two classical codes on the 3D cubic lattice.
H1, H2: polynomial -> HGP -> 6D Hx, Hz -> cut & project -> 3D new Hx, Hz
'''
import os,sys
import numpy as np
from concurrent import futures
from numpy import array,exp,sqrt,cos,sin,pi
from scipy.linalg import norm
from aperiodic_codes.cut_and_project.cnp_utils import *
import nevergrad as ng

def coord6_to_ind(coords, n):
    '''
    (x0, x1, x2) -> coordinates of code1
    (x3, x4, x5) -> coordinates of code2
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
    prefix = "/data/apc"
    spec_file = sys.argv[1];
    code_name = sys.argv[2];
    pid = os.getpid();
    np.random.seed(pid);
    f_base = f'{prefix}/ng/{code_name}/{pid}';
    os.makedirs(os.path.dirname(f_base), exist_ok=True);
    nA = 6*5//2;
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

    code_spec_1 = [];
    code_spec_2 = [];
    with open(spec_file) as sfile:
        for line in sfile:
            spec = line.split();
            if len(spec) > 2 and spec[0] == code_name:
                code_spec_1 = [int(i) for i in spec[1].split(",")];
                code_spec_2 = [int(i) for i in spec[2].split(",")];
    assert len(code_spec_1) == 27 and len(code_spec_2) == 27, \
        f'Code {code_name} not read from {spec_file}!';

    h1 = gen_code_3d(code_spec_1,n);
    h2 = gen_code_3d(code_spec_2,n);
    hx, hz = gen_hgp(h1, h2);
    hx_vv, hx_cc = get_hx_vv_cc(hx, n);
    hz_vv, hz_cc = get_hz_vv_cc(hz, n);
    offset = np.random.rand(6);

    def eval_cut(angles):
        R = gen_rotation(angles,6);
        P_plus = R @ proj_pos;
        P_minus = R @ proj_neg;

        pid2 = os.getpid();
        cut_ind, full_to_cut_ind_map = cut_ext(lat_pts, voronoi, P_minus, offset, f'{f_base}_{pid2}', nTh);
        cut_pts = lat_pts[cut_ind,:];
        proj_pts = project(cut_pts, P_plus);
        cut_bulk = [i for i in range(len(cut_ind)) if bulk[cut_ind[i]]];
        n_points = len(cut_ind);
        n_bulk = len(cut_bulk);

        new_hx_vv = gen_new_pc_matrix(cut_pts, full_to_cut_ind_map, hx_vv, n);
        new_hx_cc = gen_new_pc_matrix(cut_pts, full_to_cut_ind_map, hx_cc, n);
        new_hz_vv = gen_new_pc_matrix(cut_pts, full_to_cut_ind_map, hz_vv, n);
        new_hz_cc = gen_new_pc_matrix(cut_pts, full_to_cut_ind_map, hz_cc, n);

        n_anti = len(check_comm_after_proj(new_hx_vv, new_hx_cc, new_hz_vv, new_hz_cc)[0]);
        n_ones = np.count_nonzero(np.sum(new_hz_cc[np.ix_(cut_bulk,cut_bulk)],axis=0) == 1) + \
                np.count_nonzero(np.sum(new_hz_vv[np.ix_(cut_bulk,cut_bulk)],axis=0) == 1);

        f = open(f'{f_base}.log','a');
        f.write(','.join(map(str,offset))+','+','.join(map(str,angles))+ \
                f',{n_ones},{n_bulk},{n_anti},{n_points}\n');
        f.close();

        return [n_ones/n_bulk,n_anti/n_points];

    def min_const(angles):
        R = gen_rotation(angles[0][0],6);
        P_plus = R @ proj_pos;
        norms = norm(P_plus,axis=1);
        ovs = (P_plus@P_plus.T/np.outer(norms, norms))[np.triu_indices(6,k=1)];
        return 10*(2e-2 - np.min(abs(ovs)));

    def var_const(angles):
        R = gen_rotation(angles[0][0],6);
        P_plus = R @ proj_pos;
        norms = norm(P_plus,axis=1);
        ovs = (P_plus@P_plus.T/np.outer(norms, norms))[np.triu_indices(6,k=1)];
        return 10*(np.var(abs(ovs)) - 1/18);
   
    instrum = ng.p.Instrumentation(ng.p.Angles(init=np.zeros(nA,dtype=float)));
    optimizer = ng.optimizers.registry['TwoPointsDE'](parametrization=instrum, budget=40000, num_workers=16);
    optimizer.tell(ng.p.MultiobjectiveReference(), [2.0, 10.0])

    with futures.ProcessPoolExecutor(max_workers=optimizer.num_workers) as executor:
        recommendation = optimizer.minimize(eval_cut, constraint_violation=[min_const,var_const],
                                            executor=executor, batch_mode=False);
        print(recommendation.value)
