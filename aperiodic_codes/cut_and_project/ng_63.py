'''
Construct a pair of X and Z parity-check matrices on 3D cut-and-project tiling
from HGP of two classical codes on the 3D cubic lattice.
H1, H2: polynomial -> HGP -> 6D Hx, Hz -> cut & project -> 3D new Hx, Hz
'''
import os,sys
import numpy as np
from concurrent import futures
from numpy import array,sqrt,cos,sin,pi
from aperiodic_codes.cut_and_project.cnp_utils import *
from scipy.linalg import norm
import nevergrad as ng

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
    np.random.seed(pid);
    f_base = f'{prefix}/ng/{code_name}/{pid}';
    os.makedirs(os.path.dirname(f_base), exist_ok=True);
    nA = 6*5//2;
    nTh = 16;
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
    assert len(code_spec_1) == 27 and len(code_spec_2) == 27, f'Code {code_name} not read from {spec_file}!';

    h1 = gen_code_3d(code_spec_1,n);
    h2 = gen_code_3d(code_spec_2,n);
    hx, hz = gen_hgp(h1, h2);
    hx_vv, hx_cc = H_vv_cc(hx);
    hz_vv, hz_cc = H_vv_cc(hz);
    offset = np.random.rand(6);

    MAGIC_MIN = 0.025;
    MAGIC_VAR = 1/18;
    MAGIC_PTS = 3231; 
    mask_it = np.arange(6);
    gram_mask = mask_it[:,None] < mask_it;

    def eval_cut(angles):
        R = gen_rotation(angles,6);
        P_plus = R @ proj_pos;
        P_minus = R @ proj_neg;

        cut_ind, full_to_cut_ind_map = cut(lat_pts, voronoi, P_minus, offset, nTh);
        cut_pts = lat_pts[cut_ind,:];
        proj_pts = project(cut_pts, P_plus);
        cut_bulk = [i for i in range(len(cut_ind)) if bulk[cut_ind[i]]];
        n_points = len(cut_ind);
        n_bulk = len(cut_bulk);

        new_hx_vv = hx_vv[np.ix_(cut_ind,cut_ind)].A;
        new_hx_cc = hx_cc[np.ix_(cut_ind,cut_ind)].A;
        new_hz_vv = hz_vv[np.ix_(cut_ind,cut_ind)].A;
        new_hz_cc = hz_cc[np.ix_(cut_ind,cut_ind)].A;

        n_anti = len(check_comm_after_proj(new_hx_vv, new_hx_cc, new_hz_vv, new_hz_cc)[0]);
        n_ones = np.count_nonzero(np.sum(new_hz_cc[np.ix_(cut_bulk,cut_bulk)],axis=0) == 1) + \
                np.count_nonzero(np.sum(new_hz_vv[np.ix_(cut_bulk,cut_bulk)],axis=0) == 1);

        f = open(f'{f_base}.log','a');
        f.write(','.join(map(str,offset))+','+','.join(map(str,angles))+f',{n_ones},{n_bulk},{n_anti},{n_points}\n');
        f.close();

        return [n_anti/n_points,n_ones/n_bulk,(MAGIC_PTS-n_points)/MAGIC_PTS];

    def gram_min(angles):
        P_plus = gen_rotation(angles,6) @ proj_pos;
        P_plus = P_plus / norm(P_plus,axis=1,keepdims=True);
        ov = abs(P_plus@P_plus.T);
        return np.min(ov) - MAGIC_MIN;

    def gram_var(angles):
        P_plus = gen_rotation(angles,6) @ proj_pos;
        P_plus = P_plus / norm(P_plus,axis=1,keepdims=True);
        ov = abs(P_plus@P_plus.T);
        return MAGIC_VAR - np.var(ov,where=gram_mask);

    DE_opt = ng.families.DifferentialEvolution(scale=0.005,initialization='gaussian',crossover='rotated_twopoints',popsize='large');
    optimizer = DE_opt(parametrization=ng.p.Angles(init=np.zeros(nA,dtype=float)), budget=60000, num_workers=32);
    optimizer.parametrization.register_cheap_constraint(gram_min);
    optimizer.parametrization.register_cheap_constraint(gram_var);
    optimizer.tell(ng.p.MultiobjectiveReference(), [8.0, 2.0, 1.0]);

    with futures.ThreadPoolExecutor(max_workers=optimizer.num_workers) as executor:
        optimizer.minimize(eval_cut , executor=executor, verbosity=0 , batch_mode=False);

    f = open(f'{f_base}.sol','w');
    for param in sorted(optimizer.pareto_front(), key=lambda p: p.losses[0]):
        f.write(','.join(map(str,param.value))+','+','.join(map(str,param.losses))+'\n');
    f.close();
