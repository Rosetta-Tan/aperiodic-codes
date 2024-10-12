'''
Construct a pair of X and Z parity-check matrices on 3D cut-and-project tiling
from HGP of two classical codes on the 3D cubic lattice.
H1, H2: polynomial -> HGP -> 6D Hx, Hz -> cut & project -> 3D new Hx, Hz
'''
from os import getpid
import numpy as np
from numpy import array,exp,sqrt,cos,sin,pi
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
    f_base = f'{prefix}/code/{pid}';
    DIRS = 27;
    nTh = 4;
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

    cut_ind, full_to_cut_ind_map = cut(lat_pts, voronoi, proj_neg, offset, nTh);
    cut_pts = lat_pts[cut_ind,:];
    proj_pts = project(cut_pts, proj_pos);
    cut_bulk = [i for i in range(len(cut_ind)) if bulk[cut_ind[i]]];
    n_points = len(cut_ind);
    n_bulk = len(cut_bulk);

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
        hx_vv, hx_cc = H_vv_cc(hx);
        hz_vv, hz_cc = H_vv_cc(hz);

        new_hx_vv = hx_vv[np.ix_(cut_ind,cut_ind)].A;
        new_hx_cc = hx_cc[np.ix_(cut_ind,cut_ind)].A;
        new_hz_vv = hz_vv[np.ix_(cut_ind,cut_ind)].A;
        new_hz_cc = hz_cc[np.ix_(cut_ind,cut_ind)].A;

        n_anti = check_comm_after_proj(new_hx_vv, new_hx_cc, new_hz_vv, new_hz_cc);
        n_low = np.count_nonzero(np.sum(new_hz_cc[np.ix_(cut_bulk,cut_bulk)],axis=0) < 3) + \
                np.count_nonzero(np.sum(new_hz_vv[np.ix_(cut_bulk,cut_bulk)],axis=0) < 3);
        prop_energy = n_anti/n_points/10 + 10*n_low/n_bulk;
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
            f.write(','.join(map(str,offset))+','+','.join(map(str,prop_code_1))+','+ \
                    ','.join(map(str,prop_code_2))+f',{n_low},{n_bulk},{n_anti},{n_points},True\n');
            f.close();
        else:
            f = open(f'{f_base}.log','a');
            f.write(','.join(map(str,offset))+','+','.join(map(str,prop_code_1))+','+ \
                    ','.join(map(str,prop_code_2))+f',{n_low},{n_bulk},{n_anti},{n_points},False\n');
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
