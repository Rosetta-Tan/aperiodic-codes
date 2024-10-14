import os
import logging
from collections import Counter
from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt
from aperiodic_codes.cut_and_project.cnp_utils import check_comm_after_proj
from aperiodic_codes.cut_and_project.z2 import row_echelon, nullspace, row_basis, rank
from aperiodic_codes.cut_and_project.code_param_utils import compute_lz
from aperiodic_codes.viewer.py_viewer import visualize_logical

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def get_mode_leaf(inds):
    c = Counter(inds)
    max_freq = max(c.values())
    mode_ind = [k for k, v in c.items() if v == max_freq]
    leaf_ind = [k for k, v in c.items() if v == 1]
    return mode_ind, leaf_ind

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

def visualize_logical(lx, lz, proj_pts):
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})    
    # z_stab_legend_set = False
    x_vv_legend_set = False
    x_cc_legend_set = False
    z_vv_legend_set = False
    z_cc_legend_set = False

    for i in range(len(lx)//2):
            if lx[i]:
                if not x_vv_legend_set:
                    ax.scatter([proj_pts[i, 0]], [proj_pts[i, 1]], [proj_pts[i, 2]], color='C3', s=30, alpha=1, label='L_X (VV)', zorder=10)
                    x_vv_legend_set = True
                else:
                    ax.scatter([proj_pts[i, 0]], [proj_pts[i, 1]], [proj_pts[i, 2]], color='C3', alpha=1, s=30)
            if lx[i+len(lx)//2]:
                if not x_cc_legend_set:
                    ax.scatter([proj_pts[i, 0]], [proj_pts[i, 1]], [proj_pts[i, 2]], color='blue', s=30, alpha=1, label='L_X (CC)')
                    x_cc_legend_set = True
                else:
                    ax.scatter([proj_pts[i, 0]], [proj_pts[i, 1]], [proj_pts[i, 2]], color='blue', s=30, alpha=1)
    for i in range(len(lz)//2):
        if lz[i]:
            if not z_vv_legend_set:
                ax.scatter([proj_pts[i, 0]], [proj_pts[i, 1]], [proj_pts[i, 2]], color='C2', s=30, alpha=0.5, label='L_Z (VV)')
                z_vv_legend_set = True
            else:
                ax.scatter([proj_pts[i, 0]], [proj_pts[i, 1]], [proj_pts[i, 2]], color='C2', s=30, alpha=0.5)
        if lz[i+len(lz)//2]:
            if not z_cc_legend_set:
                ax.scatter([proj_pts[i, 0]], [proj_pts[i, 1]], [proj_pts[i, 2]], color='cyan', s=30, alpha=0.5, label='L_Z (CC)')
                z_cc_legend_set = True
            else:
                ax.scatter([proj_pts[i, 0]], [proj_pts[i, 1]], [proj_pts[i, 2]], color='cyan', s=30, alpha=0.5)
    ax.scatter(proj_pts[:,0], proj_pts[:,1],proj_pts[:,2],color='k', s=4, alpha=0.5)
    ax.legend()

    return fig, ax

def to_txt(filepath, data):
    if isinstance(data, np.ndarray) and data.ndim == 1:
        data = data.reshape(1, -1)
    with open(filepath, 'w') as f:
        for i in range(len(data)):
            line = ','.join(map(str, data[i]))
            f.write(line + '\n')

def get_min_wt_time_limit(h, v, time_limit=10):
    """
    Calculate the code distance of the classical code within the time limit.

    Returns:
        k: int, the dimension of the code
        d: int, the minimum Hamming distance found within the time limit
    """
    def _find_min_weight_while_build(matrix):
            span = []
            min_hamming_weight = np.inf
            for ir, row in enumerate(matrix):
                print('debug: ir = ', ir, 
                      'current min_hamming_weight = ', 
                      min_hamming_weight, flush=True)  # debug
                row_hamming_weight = np.sum((row + v) % 2)
                if row_hamming_weight < min_hamming_weight:
                    print(np.squeeze(np.argwhere(row == 1)));
                    min_hamming_weight = row_hamming_weight
                    end = timer()
                    if (end - start) > time_limit:
                        print('Time limit reached, aborting ...')
                        return min_hamming_weight, (row + v) % 2
                temp = [row]
                for element in span:
                    newvec = (row + element) % 2
                    temp.append(newvec)
                    newvec_hamming_weight = np.sum((newvec + v) % 2)
                    if newvec_hamming_weight < min_hamming_weight:
                        print(np.squeeze(np.argwhere(newvec == 1)));
                        min_hamming_weight = newvec_hamming_weight
                    end = timer()
                    if (end - start) > time_limit:
                        print('Time limit reached, aborting ...')
                        return min_hamming_weight, (newvec + v) % 2
                span = list(np.unique(temp + span, axis=0))
            assert len(span) == 2**len(matrix) - 1
            return min_hamming_weight
    
    start = timer()
    d, op = _find_min_weight_while_build(h)
    return d, op


if __name__ == "__main__":
    data_path = "../../data/apc/6d_to_3d"
    pid = "20241007_n=3_DIRS27_ng_8"  # n_anti = 6
    filepath = os.path.join(data_path, f'{pid}.npz')
    data = np.load(filepath)
    
    proj_pts = data['proj_pts']
    cut_bulk = data['cut_bulk']
    n_points = proj_pts.shape[0]
    logging.info(f'number of points = {n_points}')

    hx_cc = data['hx_cc']
    hx_vv = data['hx_vv']
    hz_cc = data['hz_cc']
    hz_vv = data['hz_vv']
    hx = np.hstack((hx_cc, hx_vv)).astype(np.int64)
    hz = np.hstack((hz_cc, hz_vv)).astype(np.int64)
    logging.info(f'{((hx @ hz.T) % 2).nonzero()}')

    n_anti = check_comm_after_proj(hx_vv, hx_cc, hz_vv, hz_cc)
    logging.info(f'n_anti = {n_anti}')

    ind_x, ind_z = ((hx @ hz.T) % 2).nonzero()
    # find the element in ind_x that has the highest frequency
    mode_x, leaf_x = get_mode_leaf(ind_x)
    mode_z, leaf_z = get_mode_leaf(ind_z)
    # logging.info(f'mode_x = {mode_x}')

    new_hx = hx.copy()
    new_hz = hz.copy()
    new_hx[mode_x] = 0
    new_hx[leaf_x] = 0
    for i in range(len(leaf_x)-1):
        new_vec = (hx[leaf_x[i]] + hx[leaf_x[i+1]]) % 2
        new_hx = np.vstack((new_hx, new_vec))
    # new_hz[mode_z] = 0
    new_vec = np.zeros(hx.shape[1], dtype=np.int64)
    # new_vec[[71,   72,   88,  249,  250,  266,  338,  339,  355,  516,  517,
    #     533,  605,  606,  622, 1229, 1781, 1853]] = 1
    # new_hx = np.vstack((new_hx, new_vec))

    logging.info(f'{((new_hx @ new_hz.T) % 2).nonzero()}')

    lz = compute_lz(new_hx, new_hz)
    lx = compute_lz(new_hz, new_hx)
    assert len(lz) == len(lx)
    k = len(lz)
    logging.info(f'k = {k}')

    fig0, ax0 = visualize_logical(lx[0], lz[0], proj_pts)
    ax0.set_title(f'L0, n_anti={n_anti}, k={k}')
    logging.info(f'weight of L_X[0] = {np.sum(lx[0])}')
    fig0.savefig(f'{pid}_L0.png')

    fig1, ax1 = visualize_logical(lx[1], lz[1], proj_pts)
    ax1.set_title(f'L1, n_anti={n_anti}, k={k}')
    logging.info(f'weight of L_X[1] = {np.sum(lx[1])}')
    fig1.savefig(f'{pid}_L1.png')

    lx_symm, lz_symm = (lx[0] + lx[1]) % 2, (lz[0] + lz[1]) % 2
    fig2, ax2 = visualize_logical(lx_symm, lz_symm, proj_pts)
    ax2.set_title(f'L_symm, n_anti={n_anti}, k={k}')
    fig2.savefig(f'{pid}_L_symm.png')
    logging.info(f'Lx_symm = {lx_symm.nonzero()}')
    logging.info(f'Lz_symm = {lz_symm.nonzero()}')

    to_txt('proj_pts.txt', proj_pts)
    lil_hx = [new_hx[i].nonzero()[0] for i in range(len(new_hx))]
    to_txt(f'hx.txt', lil_hx)
    lil_hz = [new_hz[i].nonzero()[0] for i in range(len(new_hz))]
    to_txt(f'hz.txt', lil_hz)
    to_txt('L_symm.txt', lz_symm)

    stab_z = hx.copy()
    # for i in range(len(lz)):
    #     stab_z = np.vstack((lz[i], stab_z))
    
    init_vec = lz_symm
    # d, op = get_min_wt_time_limit(stab_z, init_vec, time_limit=30)
    # logging.info(f'd = {d}')
    # logging.info(f'op = {op.nonzero()}')

    # plt.show()