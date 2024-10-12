'''
Construct a pair of X and Z parity-check matrices on 3D cut-and-project tiling
from HGP of two classical codes on the 3D cubic lattice.
H1, H2: polynomial -> HGP -> 6D Hx, Hz -> cut & project -> 3D new Hx, Hz
'''
from os import getpid
import numpy as np
from numpy import array,sqrt,cos,sin,pi
from aperiodic_codes.cut_and_project.cnp_utils import *

def proj_mat():
    c = 1/sqrt(5);
    s = 2/sqrt(5);
    return array([[ s*cos(0*pi/5), s*sin(0*pi/5), c,  s*cos(0*pi/5),  s*sin(0*pi/5),  c],
                  [ s*cos(2*pi/5), s*sin(2*pi/5), c,  s*cos(4*pi/5),  s*sin(4*pi/5),  c],
                  [ s*cos(4*pi/5), s*sin(4*pi/5), c,  s*cos(8*pi/5),  s*sin(8*pi/5),  c],
                  [ s*cos(6*pi/5), s*sin(6*pi/5), c, s*cos(12*pi/5), s*sin(12*pi/5),  c],
                  [ s*cos(8*pi/5), s*sin(8*pi/5), c, s*cos(16*pi/5), s*sin(16*pi/5),  c],
                  [             0,             0, 1,              0,              0, -1]])/sqrt(2);

if __name__ == '__main__':
    prefix = "/data/apc"
    pid = getpid();
    print(pid);
    f_base = f'{prefix}/6d_to_3d/{pid}';
    nTh = 8;
    n = 3;

    # Generate 6d lattice objects
    lat_pts = gen_lat(low=-n, high=n, dim=6)
    assert lat_pts.shape[0] == (2*n+1)**6, 'Number of lattice points should be N**6'
    voronoi = gen_voronoi(dim=6);
    bulk = np.all(abs(lat_pts) != n,axis=1);
    P = proj_mat();
    proj_pos = P[:,:3];
    proj_neg = P[:,3:];
    
    h1 = gen_code_3d([1,0,1,1,1,1,1,0,0,0,1,0,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0],n);
    h2 = gen_code_3d([1,1,1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0],n);
    hx, hz = gen_hgp(h1, h2);
    hx_vv, hx_cc = H_vv_cc(hx);
    hz_vv, hz_cc = H_vv_cc(hz);

    nA = 6*5//2;
    offset = np.zeros(6,dtype=float);#rng.uniform(0.0,1.0,6);
    angles = np.zeros(nA,dtype=float);#rng.uniform(0.0,2*pi,nA).tolist();
    R = gen_rotation(angles,6);
    proj_pos = R @ proj_pos;
    proj_neg = R @ proj_neg;

    cut_ind, full_to_cut_ind_map = cut(lat_pts, voronoi, proj_neg, offset, nTh);
    cut_pts = lat_pts[cut_ind,:];
    proj_pts = project(cut_pts, proj_pos)
    cut_bulk = [i for i in range(len(cut_ind)) if bulk[cut_ind[i]]];

    new_hx_vv = hx_vv[np.ix_(cut_ind,cut_ind)].A;
    new_hx_cc = hx_cc[np.ix_(cut_ind,cut_ind)].A;
    new_hz_vv = hz_vv[np.ix_(cut_ind,cut_ind)].A;
    new_hz_cc = hz_cc[np.ix_(cut_ind,cut_ind)].A;

    print(f'shape of proj_pts: {proj_pts.shape}')
    np.savez(f'{f_base}.npz', proj_pts=proj_pts,cut_bulk=cut_bulk,
             hx_vv=new_hx_vv,hx_cc=new_hx_cc,hz_vv=new_hz_vv,hz_cc=new_hz_cc);

    # Check commutation
    print(check_comm_after_proj(new_hx_vv, new_hx_cc, new_hz_vv, new_hz_cc))
