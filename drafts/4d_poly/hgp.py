import numpy as np
from aperiodic_codes.cut_and_project.cnp_utils import *
from aperiodic_codes.cut_and_project.code_param_utils import compute_lz, get_classical_code_distance_time_limit

if __name__ == '__main__':
    n = 2
    code1 = np.array([1,1,0,1,0,0,0,0,0])
    code2 = np.array([1,0,1,0,1,0,0,0,0])
    
    h1 = gen_code_2d(code1, n)
    h2 = gen_code_2d(code2, n)
    h1 = h1.todense()
    h2 = h2.todense()

    dat = np.zeros((len(h1), len(h2)))
    for i in range(len(h1)):
        for j in range(len(h2)):
            hx, hz = gen_hgp(h1, h2)
            hx = hx.todense()
            hz = hz.todense()
            lz = compute_lz(hx, hz)
            k = len(lz)
            dat[i,j] = k
    
    

    # d1 = get_classical_code_distance_time_limit(h1.todense(), 10)
    # print(f'd1: {d1}')

    # n_points = hx.shape[0]
    
    # print(f'k: {k}, n_points: {n_points}')
                        # f',{n_low},{n_bulk},{n_anti},{n_points},True\n');