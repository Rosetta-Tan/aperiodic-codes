# 3D cut and project tiling
import numpy as np
from subprocess import run
import scipy.sparse as sp
from scipy.linalg import expm
from scipy.optimize import linprog

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
    lat_pts = np.array(np.meshgrid(*([np.arange(low, high+1)]*dim),indexing='ij')).reshape(dim, -1)
    return lat_pts.T

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
    voronoi = np.array(np.meshgrid(*([[-0.5, 0.5]]*dim),indexing='ij')).reshape(dim, -1)
    return voronoi.T

def coord3_to_idx(x, y, z, n):
    return None if abs(x) > n or abs(y) > n or abs(z) > n else (x+n) * (2*n+1)**2 + (y+n) * (2*n+1) + (z+n);

def idx_to_coord3(idx, n):
    x = idx // (2*n+1)**2 - n;
    y = (idx % (2*n+1)**2) // (2*n+1) - n;
    z = idx % (2*n+1) - n;
    return x, y, z

def gen_code_3d(spec,n):
    d3 = np.array([[ 0, 0, 0],
                   [ 1, 0, 0],[-1, 0, 0],[ 0, 1, 0],[ 0,-1, 0],[ 0, 0, 1],[ 0, 0,-1],
                   [ 1, 1, 0],[ 1, 0, 1],[ 0, 1, 1],[ 1,-1, 0],[ 1, 0,-1],[ 0, 1,-1],[-1, 1, 0],[-1, 0, 1],[ 0,-1, 1],[-1,-1, 0],[-1, 0,-1],[ 0,-1,-1],
                   [ 1, 1, 1],[ 1, 1,-1],[ 1,-1, 1],[ 1,-1,-1],[-1, 1, 1],[-1, 1,-1],[-1,-1, 1],[-1,-1,-1]]);

    row = [];
    col = [];
    for i in range(-n,n+1):
        for j in range(-n,n+1):
            for k in range(-n,n+1):
                idx = coord3_to_idx(i, j, k, n);

                cur_col = [y for x in np.flatnonzero(spec) if (y := coord3_to_idx(*(d3[x]+[i,j,k]),n)) is not None];
                row = row + [idx]*len(cur_col);
                col = col + cur_col;
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

def cut_ext(lat_pts , voronoi , proj_neg , offset, f_base, nTh):
    orth_pts = lat_pts @ proj_neg;
    orth_window = proj_neg.T @ (voronoi + np.tile([offset],(voronoi.shape[0],1))).T;
    np.savez(f'{f_base}_cut.npz',orth_pts=orth_pts,orth_window=orth_window);
    run(f'cut_multi {f_base} {nTh}',shell=True);
    cut_inds = np.load(f'{f_base}_ind.npy');
    run(f'rm {f_base}_cut.npz',shell=True);
    run(f'rm {f_base}_ind.npy',shell=True);

    return cut_inds , {cut_inds[i]:i for i in range(len(cut_inds))};

def is_point_in_hull(points, x):
    n_points = len(points)
    n_dim = len(x)
    c = np.zeros(n_points)
    A = np.r_[points.T,np.ones((1,n_points))]
    b = np.r_[x, np.ones(1)]
    lp = linprog(c, A_eq=A, b_eq=b)
    return lp.success

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
        voronoi_shifted = voronoi + np.tile([offset_vec],(voronoi.shape[0],1));
    else:
        voronoi_shifted = voronoi;

    window = voronoi_shifted @ proj;
 
    # Select lattice points inside the convex hull
    full_to_cut_ind_map = {}
    cut_pts = []
    proj_pts = lat_pts @ proj
    for i in range(proj_pts.shape[0]):
        pt_proj = proj_pts[i,:];
        if is_point_in_hull(window,pt_proj):
            full_to_cut_ind_map.update({i: len(cut_pts)})
            cut_pts.append(lat_pts[i,:])
    cut_pts = np.asarray(cut_pts) # shape=(6, n_cut)
 
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
    return cut_pts @ proj

def check_comm_after_proj(hx_vv, hx_cc, hz_vv, hz_cc,cut_bulk = None):
    '''
    Check commutation of all pairs of stabilizers.
    '''
    assert hx_vv.shape == hx_cc.shape == hz_vv.shape == hz_cc.shape
    hx = np.hstack((hx_vv, hx_cc))
    hz = np.hstack((hz_vv, hz_cc))
    return np.nonzero((hx @ hz.T) % 2) if cut_bulk == None else np.nonzero((hx @ hz.T)[np.ix_(cut_bulk,cut_bulk)] % 2);

def get_stabilizer_overlap(x_idx,z_idx,hx_vv,hx_cc,hz_vv,hz_cc):
    x_vv = hx_vv[x_idx,:].nonzero()[1];
    x_cc = hx_cc[x_idx,:].nonzero()[1];
    z_vv = hz_vv[z_idx,:].nonzero()[1];
    z_cc = hz_cc[z_idx,:].nonzero()[1];
    return np.intersect1d(x_vv,z_vv),np.intersect1d(x_cc,z_cc);
