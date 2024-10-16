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
    pid = os.getpid();
    np.random.seed(pid);
    f_base = f'{prefix}/ng_code/{pid}';
    os.makedirs(os.path.dirname(f_base), exist_ok=True);
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

    offset = np.random.rand(6);
    cut_ind, full_to_cut_ind_map = cut(lat_pts, voronoi, proj_neg, offset, nTh);
    cut_pts = lat_pts[cut_ind,:];
    proj_pts = project(cut_pts, proj_pos);
    cut_bulk = [i for i in range(len(cut_ind)) if bulk[cut_ind[i]]];
    n_points = len(cut_ind);
    n_bulk = len(cut_bulk);

    MAGIC_WEIGHT = 6;

    def eval_code(vals):
        # Try proposed codes
        code_1 = (1,)+vals[:DIRS-1];
        code_2 = (1,)+vals[DIRS-1:];
        h1 = gen_code_3d(code_1,n);
        h2 = gen_code_3d(code_2,n);
        hx, hz = gen_hgp(h1, h2);
        hx_vv, hx_cc = H_vv_cc(hx);
        hz_vv, hz_cc = H_vv_cc(hz);

        new_hx_vv = hx_vv[np.ix_(cut_ind,cut_ind)].A;
        new_hx_cc = hx_cc[np.ix_(cut_ind,cut_ind)].A;
        new_hz_vv = hz_vv[np.ix_(cut_ind,cut_ind)].A;
        new_hz_cc = hz_cc[np.ix_(cut_ind,cut_ind)].A;

        n_anti = len(check_comm_after_proj(new_hx_vv, new_hx_cc, new_hz_vv, new_hz_cc)[0]);
        n_ones = np.count_nonzero(np.sum(new_hz_cc[np.ix_(cut_bulk,cut_bulk)],axis=0) == 1) + \
                 np.count_nonzero(np.sum(new_hz_vv[np.ix_(cut_bulk,cut_bulk)],axis=0) == 1);
        
        f = open(f'{f_base}.log','a');
        f.write(','.join(map(str,offset))+','+','.join(map(str,code_1))+','+ \
                ','.join(map(str,code_2))+f',{n_ones},{n_bulk},{n_anti},{n_points}\n');
        f.close();

        return [n_anti/n_points,n_ones/n_bulk];

    def code1_weight(vals):
        return float(np.sum(vals[:DIRS-1]) - MAGIC_WEIGHT);

    def code2_weight(vals):
        return float(np.sum(vals[DIRS-1:]) - MAGIC_WEIGHT);

    DE_opt = ng.families.DifferentialEvolution(crossover='rotated_twopoints',popsize='large');
    optimizer = DE_opt(parametrization=ng.p.Choice([0,1],repetitions=52), budget=60000, num_workers=32);
    optimizer.parametrization.register_cheap_constraint(code1_weight);
    optimizer.parametrization.register_cheap_constraint(code2_weight);
    optimizer.tell(ng.p.MultiobjectiveReference(), [27.0, 2.0]);

    with futures.ThreadPoolExecutor(max_workers=optimizer.num_workers) as executor:
        optimizer.minimize(eval_code , executor=executor, verbosity=0 , batch_mode=False);

    f = open(f'{f_base}.sol','w');
    for param in sorted(optimizer.pareto_front(), key=lambda p: p.losses[0]):
        f.write(','.join(map(str,(1,)+param.value[:DIRS-1]))+','+','.join(map(str,(1,)+param.value[DIRS-1:]))+\
                ','+','.join(map(str,param.losses))+'\n');
    f.close();
