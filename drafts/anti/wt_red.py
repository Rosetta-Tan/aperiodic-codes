import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from aperiodic_codes.cut_and_project.cnp_utils import check_comm_after_proj
from aperiodic_codes.cut_and_project.z2 import row_echelon, nullspace, row_basis, rank
from aperiodic_codes.cut_and_project.code_param_utils import compute_lz
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

if __name__ == "__main__":
    data_path = "../../data/apc/6d_to_3d"
    pid = "20240920_n=3_DIRS27_1"  # high n_low, 0 n_anti
    filepath = os.path.join(data_path, f'{pid}.npz')
    data = np.load(filepath)
    
    proj_pts = data['proj_pts']
    cut_bulk = data['cut_bulk']
    n_points = proj_pts.shape[0]
    logging.info(f'number of points = {n_points}')

    new_hx_cc = data['hx_cc']
    new_hx_vv = data['hx_vv']
    new_hz_cc = data['hz_cc']
    new_hz_vv = data['hz_vv']
    new_hx = np.hstack((new_hx_cc, new_hx_vv)).astype(np.int64)
    new_hz = np.hstack((new_hz_cc, new_hz_vv)).astype(np.int64)
    # ns_x = np.load('ns_x.npy')
    ns_x = nullspace(new_hx)
    logging.info(f'ns_x.shape: {ns_x.shape}')

    new_new_hz = ns_x.copy()
    ns_new_new_z = nullspace(new_new_hz)
    max_wt_ind = np.argmax(np.sum(ns_new_new_z, axis=1))
    new_new_hz[max_wt_ind] = 0
    second_max_wt_ind = np.argmax(np.sum(new_new_hz, axis=1))
    vec = new_new_hz[second_max_wt_ind]
    wt = np.sum(vec)
    print(f'wt: {wt}')
    for i in range(len(ns_x)):
        if i == max_wt_ind:
            continue
        vec_tmp = (ns_x[i] + vec) % 2
        if np.sum(vec_tmp) < wt:
            vec = vec_tmp
            wt = np.sum(vec_tmp)
    for col in range(len(vec)):
        new_new_hz[second_max_wt_ind][col] = vec[col]
    print(f'wt: {wt}')
    
    lz = compute_lz(new_hx, new_new_hz)
    lx = compute_lz(new_new_hz, new_hx)
    k = len(lz)
    logging.info(f'weight of lz = {np.sum(lz[0])}')
    logging.info(f'weight of lx = {np.sum(lx[0])}')
