'''
Construct a pair of X and Z parity-check matrices on 3D cut-and-project tiling
from HGP of two classical codes on the 3D cubic lattice.
H1, H2: polynomial -> HGP -> 6D Hx, Hz -> cut & project -> 3D new Hx, Hz
'''
import os,sys
import numpy as np
from numpy import array,exp,sqrt,cos,sin,pi
from aperiodic_codes.cut_and_project.cnp_utils import *
from scipy.linalg import norm

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
    spec_file = sys.argv[1];
    code_name = sys.argv[2];
    pid = os.getpid();
    f_base = f'{prefix}/cold_R0/{code_name}/{pid}';
    os.makedirs(os.path.dirname(f_base), exist_ok=True);
    nTh = 8;
    n = 3;

    # Generate 6d lattice objects
    lat_pts = gen_lat(low=-n, high=n, dim=6);
    assert lat_pts.shape[0] == (2*n+1)**6, 'Number of lattice points should be N**6'
    voronoi = gen_voronoi(dim=6);
    bulk = np.all(abs(lat_pts) != n,axis=1);
    P = proj_mat();
    proj_pos_base = P[:,:3];
    proj_neg_base = P[:,3:];
    proj_pos = proj_pos_base.copy();
    proj_neg = proj_neg_base.copy();

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
    hx_vv, hx_cc = H_vv_cc(hx);
    hz_vv, hz_cc = H_vv_cc(hz);

    # Setup RNG and MC params
    rng = np.random.default_rng(pid);
    nA = 6*5//2;
    beta = 60.0;
    cur_energy = np.inf;

    # Start from trivial rotation, random offset
    cur_angles = np.zeros(nA,dtype=float);#rng.uniform(0.0,2*pi,nA).tolist();
    prop_angles = cur_angles.copy();
    R = gen_rotation(cur_angles,6);
    offset = rng.uniform(0.0,1.0,6);

    while(True):
        # Try proposed cut
        cut_ind, full_to_cut_ind_map = cut(lat_pts, voronoi, proj_neg, offset, nTh);
        cut_pts = lat_pts[cut_ind,:];
        proj_pts = project(cut_pts, proj_pos);
        cut_bulk = [i for i in range(len(cut_ind)) if bulk[cut_ind[i]]];
        n_points = len(cut_ind);
        n_bulk = len(cut_bulk);

        if n_bulk != 0:
            new_hx_vv = hx_vv[np.ix_(cut_ind,cut_ind)].A;
            new_hx_cc = hx_cc[np.ix_(cut_ind,cut_ind)].A;
            new_hz_vv = hz_vv[np.ix_(cut_ind,cut_ind)].A;
            new_hz_cc = hz_cc[np.ix_(cut_ind,cut_ind)].A;

            n_anti = len(check_comm_after_proj(new_hx_vv, new_hx_cc, new_hz_vv, new_hz_cc)[0]);
            n_low = np.count_nonzero(np.sum(new_hz_cc[np.ix_(cut_bulk,cut_bulk)],axis=0) < 3) + \
                    np.count_nonzero(np.sum(new_hz_vv[np.ix_(cut_bulk,cut_bulk)],axis=0) < 3);
            prop_energy = n_anti/n_points + 2*n_low/n_bulk;
            acc_prob = min(1.0,exp(-beta*(prop_energy-cur_energy)));

            if np.sum(new_hx_vv)/n_points >= 3.0 and np.sum(new_hx_cc)/n_points >= 3.0 and rng.random() < acc_prob:
                if prop_energy < cur_energy:
                    np.savez(f'{f_base}_opt.npz', proj_pts=proj_pts,
                             hx_vv=new_hx_vv,hx_cc=new_hx_cc,hz_vv=new_hz_vv,hz_cc=new_hz_cc);

                cur_angles = prop_angles.copy();
                cur_energy = prop_energy;
                f = open(f'{f_base}.log','a');
                f.write(','.join(map(str,offset))+','+','.join(map(str,prop_angles))+ \
                        f',{n_low},{n_bulk},{n_anti},{n_points},True\n');
                f.close();
            else:
                f = open(f'{f_base}.log','a');
                f.write(','.join(map(str,offset))+','+','.join(map(str,prop_angles))+ \
                        f',{n_low},{n_bulk},{n_anti},{n_points},False\n');
                f.close();

            np.savez(f'{f_base}_cur.npz', proj_pts=proj_pts,
                     hx_vv=new_hx_vv,hx_cc=new_hx_cc,hz_vv=new_hz_vv,hz_cc=new_hz_cc);

            if n_anti == 0 and n_low/n_bulk < 0.15:
                break;

        # Generate proposed cut and test uniformity
        ov_min = 0.0;
        ov_var = np.inf;
        while ov_min < 2e-2 or ov_var > 1/18:
            prop_angles = cur_angles.copy();
            prop_angles[rng.integers(0,nA,1)[0]] += rng.normal(0.0,0.02);
            R = gen_rotation(prop_angles,6);
            proj_pos = R @ proj_pos_base;
            proj_neg = R @ proj_neg_base;
            norms = norm(proj_pos,axis=1);
            ovs = (proj_pos@proj_pos.T/np.outer(norms, norms))[np.triu_indices(6,k=1)];
            ov_min = np.min(abs(ovs));
            ov_var = np.var(abs(ovs));
