# 3D cut and project tiling
import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
import scipy.sparse as sp
from scipy.linalg import expm
from scipy.optimize import linprog

def coord3_to_idx(x, y, z, n):
    return None if abs(x) > n or abs(y) > n or abs(z) > n else (x+n) * (2*n+1)**2 + (y+n) * (2*n+1) + (z+n);

def idx_to_coord3(idx, n):
    x = idx // (2*n+1)**2 - n;
    y = (idx % (2*n+1)**2) // (2*n+1) - n;
    z = idx % (2*n+1) - n;
    return x, y, z

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

def H_vv_cc(H):
    '''
    Generate the parity check matrix for VV and CC type qubits.
    '''
    return H[:, 0:H.shape[1]//2], H[:, H.shape[1]//2:];

cpp_cut = ctypes.CDLL("./libcnp.so").cut;
cpp_cut.argtypes = [ndpointer(ctypes.c_double), ndpointer(ctypes.c_double), ndpointer(ctypes.c_int),
                    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int];

def cut(lat_pts , voronoi , proj_neg , offset, nTh):
    orth_pts = lat_pts @ proj_neg;
    orth_window = proj_neg.T @ (voronoi + np.tile([offset],(voronoi.shape[0],1))).T;
    cut_mask = np.zeros(len(orth_pts),dtype=np.int32);
    cpp_cut(orth_pts,orth_window,cut_mask,
            orth_pts.shape[0],orth_window.shape[0],orth_window.shape[1],nTh);
    cut_inds = np.argwhere(cut_mask).flatten();

    return cut_inds , {cut_inds[i]:i for i in range(len(cut_inds))};

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
    return np.nonzero((hx @ hz.T) % 2) if cut_bulk is None else np.nonzero((hx @ hz.T)[np.ix_(cut_bulk,cut_bulk)] % 2);

def get_stabilizer_overlap(x_idx,z_idx,hx_vv,hx_cc,hz_vv,hz_cc):
    x_vv = hx_vv[x_idx,:].nonzero()[1];
    x_cc = hx_cc[x_idx,:].nonzero()[1];
    z_vv = hz_vv[z_idx,:].nonzero()[1];
    z_cc = hz_cc[z_idx,:].nonzero()[1];
    return np.intersect1d(x_vv,z_vv),np.intersect1d(x_cc,z_cc);
