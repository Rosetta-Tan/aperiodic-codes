import numpy as np
import matplotlib.pyplot as plt
from aperiodic_codes.cut_and_project.config import tests


if __name__ == "__main__":
    pid = "20240914_n=3_DIRS27_1"
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
    logging.info(f'min weight of stab(Hx) = {np.min(np.sum(new_hx, axis=1))}')
    logging.info(f'min weight of stab(Hz) = {np.min(np.sum(new_hz, axis=1))}')
    logging.info(f'rank(Hx) = {rank(new_hx)}')
    logging.info(f'rank(Hz) = {rank(new_hz)}')
    ns_x = nullspace(new_hx)
    ns_z = nullspace(new_hz)
    logging.info(f'min weight of ker(Hx) = {np.min(np.sum(ns_x, axis=1))}')
    logging.info(f'min weight of ker(Hz) = {np.min(np.sum(ns_z, axis=1))}')
    np.save('ns_x.npy', ns_x)
    np.save('ns_z.npy', ns_z)
    