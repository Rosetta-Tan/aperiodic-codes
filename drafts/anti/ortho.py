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

def orthogonalize_mat_vec(mat, vec):
    for iv, v in enumerate(mat):
        if np.sum(np.dot(v, vec)) % 2 == 1:
            vec = (vec + v) % 2  # symmetri difference
    return vec

def gen_weight_series_ns(ns):
    fig, ax = plt.subplots()
    wt_series = np.sum(ns, axis=1)
    ax.plot(wt_series)
    ax.set_xlabel('index')
    ax.set_ylabel('weight')
    return fig, ax

def gen_hist_ns(ns):
    fig, ax = plt.subplots()
    maxwt = np.max(np.sum(ns, axis=1))
    ax.hist(np.sum(ns, axis=1), bins=range(0, maxwt+1), alpha=0.5)
    ax.set_xlabel('weight')
    ax.set_ylabel('frequency')
    return fig, ax

def gauging(hx, hz, ns_x, transform_idx):
    """
    Performs gauging without recursion.

    Args:
        hx: Matrix Hx.
        hz: Matrix Hz.
        ns_x: Nullspace of Hx.
        ns_x_gauge: Gauge transformation nullspace.

    Returns:
        The gauged Hz matrix, or None if gauging fails.
    """

    hz_new = hz.copy()
    deficiency = len(hx) - len(hz)
    iter = 0
    v = ns_x[0]

    while iter < len(ns_x_transform):
        if iter % 10 == 0:
            logging.info(f'[transform_idx: {transform_idx}; iter: {iter}]')
        hz_new_tmp = np.vstack([hz_new, v])
        if rank(hz_new_tmp) == len(hz_new_tmp):
            hz_new = hz_new_tmp
        iter += 1
        if iter < len(ns_x_transform):
            v = ns_x_transform[iter]
        else:
            break #Handle cases where ns_x_transform is shorter than deficiency

    if (rk:=rank(hz_new)) == (fl:=len(hx)):
        if transform_idx == 0:
            np.save('hz_gauging_0.npy', hz_new)
        return hz_new
    else:
        logging.info(f'Hz rank is not full: [{rk}/{fl}]. Trying next transformation ...')
        transform_idx += 1

    logging.error('Cannot do gauging after exhausting all transformations.')
    return None

def visualize_logical(lx, lz, proj_pts):
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})    
    # z_stab_legend_set = False
    x_vv_legend_set = False
    x_cc_legend_set = False
    z_vv_legend_set = False
    z_cc_legend_set = False

    for i in range(lx.shape[1]//2):
            if lx[0,i]:
                if not x_vv_legend_set:
                    ax.scatter([proj_pts[i, 0]], [proj_pts[i, 1]], [proj_pts[i, 2]], color='C3', s=30, alpha=1, label='L_X (VV)', zorder=10)
                    x_vv_legend_set = True
                else:
                    ax.scatter([proj_pts[i, 0]], [proj_pts[i, 1]], [proj_pts[i, 2]], color='C3', alpha=1, s=30)
            if lx[0,i+lx.shape[1]//2]:
                if not x_cc_legend_set:
                    ax.scatter([proj_pts[i, 0]], [proj_pts[i, 1]], [proj_pts[i, 2]], color='blue', s=30, alpha=1, label='L_X (CC)')
                    x_cc_legend_set = True
                else:
                    ax.scatter([proj_pts[i, 0]], [proj_pts[i, 1]], [proj_pts[i, 2]], color='blue', s=30, alpha=1)
    for i in range(lz.shape[1]//2):
        if lz[0,i]:
            if not z_vv_legend_set:
                ax.scatter([proj_pts[i, 0]], [proj_pts[i, 1]], [proj_pts[i, 2]], color='C2', s=30, alpha=0.5, label='L_Z (VV)')
                z_vv_legend_set = True
            else:
                ax.scatter([proj_pts[i, 0]], [proj_pts[i, 1]], [proj_pts[i, 2]], color='C2', s=30, alpha=0.5)
        if lz[0,i+lz.shape[1]//2]:
            if not z_cc_legend_set:
                ax.scatter([proj_pts[i, 0]], [proj_pts[i, 1]], [proj_pts[i, 2]], color='cyan', s=30, alpha=0.5, label='L_Z (CC)')
                z_cc_legend_set = True
            else:
                ax.scatter([proj_pts[i, 0]], [proj_pts[i, 1]], [proj_pts[i, 2]], color='cyan', s=30, alpha=0.5)
    ax.scatter(proj_pts[:,0], proj_pts[:,1],proj_pts[:,2],color='k', s=4, alpha=0.5)
    ax.legend()

    return fig, ax


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

    new_new_hz = ns_x.copy()
    logging.info(f'min weight of ker(new_new_hz) = {np.min(np.sum(new_new_hz, axis=1))}')
    logging.info(f'rank(new_new_hz) = {rank(new_new_hz)}')
    ns_new_new_z = nullspace(new_new_hz)
    logging.info(f'min weight of ker(ns_new_new_z) = {np.min(np.sum(ns_new_new_z, axis=1))}')

    fig_series_x, ax_series_x = gen_weight_series_ns(ns_x)
    ax_series_x.set_title('weight series of ker(Hx)')
    fig_series_x.savefig(f'wt_series_ker_Hx.png')
    fig_series_z, ax_series_z = gen_weight_series_ns(ns_z)
    ax_series_z.set_title('weight series of ker(Hz)')
    fig_series_z.savefig(f'wt_series_ker_Hz.png')

    fig_ns_x, ax_ns_x = gen_hist_ns(ns_x)
    ax_ns_x.set_title('weight distribution of ker(Hx)')
    fig_ns_x.savefig(f'wt_dist_ker_Hx.png')
    fig_ns_z, ax_ns_z = gen_hist_ns(ns_z)
    ax_ns_z.set_title('weight distribution of ker(Hz)')
    fig_ns_z.savefig(f'wt_dist_ker_Hz.png')
    fig_ns_new_new_z, ax_ns_new_new_z = gen_hist_ns(ns_new_new_z)
    ax_ns_new_new_z.set_title('weight distribution of ker(new_new_hz)')

    max_wt_i = np.argmax(np.sum(ns_new_new_z, axis=1))
    logging.info(f'max weight index = {max_wt_i}; max weight = {np.sum(ns_new_new_z[max_wt_i])}')
    new_new_hz[max_wt_i] = 0
    
    lz = compute_lz(new_hx, new_new_hz)
    lx = compute_lz(new_new_hz, new_hx)
    np.save('lz.npy', lz)
    np.save('lx.npy', lx)
    k = len(lz)
    logging.info(f'k = {k}')
    logging.info(f'weight of lz = {np.sum(lz[0])}')
    logging.info(f'weight of lx = {np.sum(lx[0])}')

    fig_logical, ax_logical = visualize_logical(lx, lz, proj_pts)
    ax_logical.set_title('Logical operators')
    fig_logical.savefig(f'logical_ops.png')
    plt.show()