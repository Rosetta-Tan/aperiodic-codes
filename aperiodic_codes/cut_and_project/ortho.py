import os
import logging
import numpy as np
from aperiodic_codes.cut_and_project.cnp_utils import check_comm_after_proj
from aperiodic_codes.cut_and_project.z2 import row_echelon, nullspace, row_basis, rank
from aperiodic_codes.cut_and_project.code_param_utils import compute_lz
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

if __name__ == "__main__":
    data_path = "../../data/apc/6d_to_3d"
    pid = "20240914_n=3_DIRS27_1"  # high n_low, 0 n_anti
    filepath = os.path.join(data_path, f'{pid}.npz')
    data = np.load(filepath)
    
    proj_pts = data['proj_pts'];
    cut_bulk = data['cut_bulk'];
    n_points = proj_pts.shape[0];

    new_hx_cc = data['hx_cc']
    new_hx_vv = data['hx_vv']
    new_hz_cc = data['hz_cc']
    new_hz_vv = data['hz_vv']
    
    print(check_comm_after_proj(new_hx_vv, new_hx_cc, new_hz_vv, new_hz_cc, cut_bulk))