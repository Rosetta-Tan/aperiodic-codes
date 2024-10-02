import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from ldpc.mod2 import nullspace
from aperiodic_codes.cut_and_project.z2 import rank, row_basis
from aperiodic_codes.cut_and_project.cnp_utils import check_comm_after_proj
from aperiodic_codes.cut_and_project.code_param_utils import compute_lz
from aperiodic_codes.cut_and_project.config import tests
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def check_comm_mat_vec(h, v):
    return np.sum((h @ v) % 2) == 0

def check_comm_mat_mat(h1, h2):
    return np.sum((h1 @ h2.T) % 2) == 0

def del_stab_compute_lxlz(hx,
                          hz,
                          ind_x,
                          ind_z,
                          delete_mode='X'):
    """
    Returns:
        lx: Lx basis
        lz: Lz basis
    """
    assert delete_mode in ['X', 'Z', 'XZ'], 'delete_mode must be X, Z or XZ'
    if delete_mode == 'X':
        del_hx = np.delete(hx, ind_x, axis=0)
        lz = compute_lz(del_hx, hz)
        lx = compute_lz(hz, del_hx)
    elif delete_mode == 'Z':
        del_hz = np.delete(hz, ind_z, axis=0)
        lz = compute_lz(hx, del_hz)
        lx = compute_lz(del_hz, hx)
    elif delete_mode == 'XZ':
        del_hx = np.delete(hx, ind_x, axis=0)
        del_hz = np.delete(hz, ind_z, axis=0)
        lz = compute_lz(del_hx, del_hz)
        lx = compute_lz(del_hz, del_hx)
    assert len(lx) == len(lz)
    return lx, lz

def examine_logical(proj_pts,
                    new_hx,
                    new_hz):
    # find the point closest to the origin
    distances = np.linalg.norm(proj_pts, axis=1)
    # distances = [d for d in distances if d > 3]
    # closest_ind = np.argwhere(distances == np.random.choice(distances)).flatten()
    # closest_ind = [np.argmin(distances)]
    x_stab_wt = np.sum(new_hx, axis=1)
    z_stab_wt = np.sum(new_hz, axis=1)
    q_to_x_wt = np.sum(new_hx, axis=0)
    q_to_z_wt = np.sum(new_hz, axis=0)
    choice = []
    for i in range(len(x_stab_wt)):
        condition1 = q_to_x_wt[i] > 2 and q_to_x_wt[i + len(q_to_x_wt)//2] >= 2 and q_to_z_wt[i] > 2 and q_to_z_wt[i + len(q_to_z_wt)//2] >= 2
        x_2_q = np.where(new_hx[i] != 0)[0]
        z_2_q = np.where(new_hz[i] != 0)[0]
        condition2 = True
        for j in x_2_q:
            if q_to_x_wt[j] == 1:
                condition2 = False
                break
        for j in z_2_q:
            if q_to_z_wt[j] == 1:
                condition2 = False
                break
        condition3 = distances[i] < 2
        condition4 = new_hx[i, i] == 1 and new_hz[i, i] == 1
        # if condition1 and condition2 and condition3 and condition4:
        choice.append(i)
    print(f'choice: {choice}')
    closest_ind = [choice[1]]
    print(f'closest point index: {closest_ind}')
    print(f'X stabilizer weight: {x_stab_wt[closest_ind]}, Z stabilizer weight: {z_stab_wt[closest_ind]}')
    delete_mode = 'Z'
    h_plot = new_hz
    ind_x = closest_ind[0]
    ind_z = closest_ind[0]
    lx, lz = del_stab_compute_lxlz(new_hx, new_hz, ind_x, ind_z, delete_mode=delete_mode)
    print(f'k = {len(lx)}')
    print(f'Lx weight = {np.sum(lx[0])}, Lz weight = {np.sum(lz[0])}')

    del_hx = new_hx.copy()
    del_hz = new_hz.copy()
    if delete_mode == 'X':
        for i in closest_ind:
            del_hx[i] = 0
    elif delete_mode == 'Z':
        for i in closest_ind:
            del_hz[i] = 0
    elif delete_mode == 'XZ':
        for i in closest_ind:
            del_hx[i] = 0
            del_hz[i] = 0
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter([proj_pts[closest_ind, 0]], [proj_pts[closest_ind, 1]], [proj_pts[closest_ind, 2]], color='C4', s=100, label=f'Deleted {delete_mode}')
    z_stab_legend_set = False
    x_vv_legend_set = False
    x_cc_legend_set = False
    z_vv_legend_set = False
    z_cc_legend_set = False

    for i in range(lx.shape[1]//2):
        if lx[0,i] or lx[0,i+lx.shape[1]//2]:
            # find all the points that are connected to i
            print(f'{i}-th qubit to Z stabilizer wt: {np.sum(h_plot[:,i])}; connected to: {np.where(h_plot[:,i] != 0)[0]}')
            connected_stab_inds = np.where(h_plot[:, i] != 0)[0]
            for i_stab in connected_stab_inds:
                if not z_stab_legend_set:
                    ax.scatter([proj_pts[i_stab, 0]], [proj_pts[i_stab, 1]], [proj_pts[i_stab, 2]], marker='s', color='cyan',s=30, alpha=0.2, label='Z stabilizer')
                    z_stab_legend_set = True
                else:
                    ax.scatter([proj_pts[i_stab, 0]], [proj_pts[i_stab, 1]], [proj_pts[i_stab, 2]], marker='s', color='cyan',s=30, alpha=0.2)
                for j in np.where(h_plot[i_stab, :] != 0)[0]:
                    # j = j % (lx.shape[1]//2)
                    if j < lx.shape[1]//2:
                        ax.quiver(proj_pts[i_stab, 0], proj_pts[i_stab, 1], proj_pts[i_stab, 2], proj_pts[j, 0]-proj_pts[i_stab, 0], proj_pts[j, 1]-proj_pts[i_stab, 1], proj_pts[j, 2]-proj_pts[i_stab, 2], color='gray', lw=1, alpha=0.5)
                    else:
                        j_shift = j - lx.shape[1]//2
                        ax.quiver(proj_pts[i_stab, 0], proj_pts[i_stab, 1], proj_pts[i_stab, 2], proj_pts[j_shift, 0]-proj_pts[i_stab, 0], proj_pts[j_shift, 1]-proj_pts[i_stab, 1], proj_pts[j_shift, 2]-proj_pts[i_stab, 2], color='gray', lw=1, alpha=0.5)
            if lx[0,i]:
                if not x_vv_legend_set:
                    ax.scatter([proj_pts[i, 0]], [proj_pts[i, 1]], [proj_pts[i, 2]], color='C3', s=30, alpha=1, label='L_X (VV)', zorder=10)
                    x_vv_legend_set = True
                else:
                    ax.scatter([proj_pts[i, 0]], [proj_pts[i, 1]], [proj_pts[i, 2]], color='C3', alpha=1, s=30)
            if lx[0,i+lx.shape[1]//2]:
                print('found CC logical, i:', i)
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

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    return fig, ax


def examine_hx(hx):
    """
    Hx: VV qubits from H1, CC qubits from H2.T
    VV --- S_Z
     |      |
    S_X --- CC
    There are weight-1 classical codewords in the kernel of Hx.
    That means these qubits are not connected to any stabilizer.

    The dimension of the kernel is the same as the difference between the number of checks and the number of bits,
    this means that there's no redundancy in the checks.
    Let's try removing some qubits form hx

    """
    # find the kernel of hx
    kernel = nullspace(hx)
    logging.info(f'length of ker(Hx): {len(kernel)}')
    minwt = np.min(np.sum(kernel, axis=1))
    logging.info(f'minimum weight of kernel: {minwt}')

    # plot the weight distribution of the kernel
    plt.hist(np.sum(kernel, axis=1), bins=range(0, 10))
    plt.xlabel('Weight')
    plt.ylabel('Frequency')
    plt.title('Weight distribution of kernel of Hx')
    
    # find the qubits that are not connected to any stabilizer
    isolated_qubits_from_hx = np.where(np.sum(hx, axis=0) == 0)[0]
    logging.info(f'from hx: {isolated_qubits_from_hx}')    
    
    plt.show()

def add_symm_from_hz(hx, hz):
    hz_new = []
    for i, v in enumerate(hz):
        if check_comm_mat_vec(hx, v):
            hz_new.append(v)
    return np.array(hz_new)

def _transform_idx_to_bin(transform_idx, n_bits):
    return np.array([int(bit) for bit in bin(transform_idx)[2:].zfill(n_bits)]).reshape(1, -1)

def gauging_old(hx, hz, ns_x, ns_x_gauge, transform_idx=False):
    """
    Params:
    ns_x: nullspace of Hx
    transform_idx: index of the group transformation Z stabilizer in nullspace of Hx
    -----------------------
    result of gauging:
    Can gauge, k=243. Initially 343 H_z are from Hz.
    For this case, without enforcing linear independence, k is also 243
    So the problem lies within the original Hz.
    
    Turns out that is a bug of mine.
    I wasn't enforcing the linear independence of the new Hz.
    Now, the rank of new Hz hasn't been completed.


    """
    if transform_idx == 2**len(ns_x_gauge) - 1:
        logging.ERROR('Cannot do gauging')
        return None
    
    bin_vec = _transform_idx_to_bin(transform_idx, len(ns_x_gauge))
    transform_element = (bin_vec @ ns_x_gauge) % 2
    ns_x_transform = (ns_x + transform_element) % 2
    
    hz_new = hz.copy()
    deficiency = len(hx) - len(hz)
    iter = 0
    v = ns_x_transform[0]

    while iter < deficiency:
        if iter % 10 == 0:
            logging.info(f'[transform_idx: {transform_idx}; iter: {iter}')
        hz_new_tmp = np.vstack([hz_new, v])
        if rank(hz_new_tmp) == len(hz_new_tmp):
            hz_new = hz_new_tmp
        iter += 1
        v = ns_x_transform[iter]

    if transform_idx == 0:
        np.save('hz_gauging_0.npy', hz_new)
    
    if (rk:=rank(hz_new)) < len(hx):
        logging.info(f'Rank of new Hz is not full: {rk}; expected: {len(hx)}')
        hz_new = gauging(hx, hz_new, ns_x, ns_x_gauge, transform_idx=transform_idx+1)

    return hz_new

def gauging(hx, hz, ns_x, ns_x_gauge, transform_idx):
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
    num_transforms = 2**len(ns_x_gauge)

    while transform_idx < num_transforms:
        bin_vec = _transform_idx_to_bin(transform_idx, len(ns_x_gauge))
        transform_element = (bin_vec @ ns_x_gauge) % 2
        ns_x_transform = (ns_x + transform_element) % 2

        hz_new = hz.copy()
        deficiency = len(hx) - len(hz)
        iter = 0
        v = ns_x_transform[0]

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

def gen_hist_ns(ns):
    fig, ax = plt.subplots()
    maxwt = np.max(np.sum(ns, axis=1))
    ax.hist(np.sum(ns, axis=1), bins=range(0, maxwt+1), alpha=0.5)
    ax.set_xlabel('weight')
    ax.set_ylabel('frequency')
    return fig, ax

if __name__ == "__main__":
    data_path = "../../data/apc/6d_to_3d"
    pid = "20240914_n=3_DIRS27_1"  # high n_low, 0 n_anti
    filepath = os.path.join(data_path, f'{pid}.npz')
    data = np.load(filepath)
    
    proj_pts = data['proj_pts'];
    n_points = proj_pts.shape[0];

    new_hx_cc = data['hx_cc']
    new_hx_vv = data['hx_vv']
    new_hz_cc = data['hz_cc']
    new_hz_vv = data['hz_vv']
    new_hx = np.hstack([new_hx_cc, new_hx_vv]).astype(np.int64)
    new_hz = np.hstack([new_hz_cc, new_hz_vv]).astype(np.int64)
    logging.info(f'min weight of stab(Hx) = {np.min(np.sum(new_hx, axis=1))}')
    logging.info(f'min weight of stab(Hz) = {np.min(np.sum(new_hz, axis=1))}')
    logging.info(f'rank of Hx: {rank(new_hx)}')
    logging.info(f'rank of Hz: {rank(new_hz)}')
    ns_x = nullspace(new_hx)
    ns_z = nullspace(new_hz)
    logging.info(f'min weight of ker(Hx) = {np.min(np.sum(ns_x, axis=1))}')
    logging.info(f'min weight of ker(Hz) = {np.min(np.sum(ns_z, axis=1))}')

    fig_ns_x, ax_ns_x = gen_hist_ns(ns_x)
    ax_ns_x.set_title('weight distribution of ker(Hx)')
    fig_ns_x.savefig(f'wt_dist_ker_Hx.png')
    fig_ns_z, ax_ns_z = gen_hist_ns(ns_z)
    ax_ns_z.set_title('weight distribution of ker(Hz)')
    fig_ns_z.savefig(f'wt_dist_ker_Hz.png')
    plt.show()

    del_ind = 788
    logging.info(f'weight of deleted X stabilizer: {np.sum(new_hx[del_ind])}')
    logging.info(f'weight of deleted Z stabilizer: {np.sum(new_hz[del_ind])}')
    lx, lz = del_stab_compute_lxlz(new_hx, new_hz, del_ind, del_ind, delete_mode='X')
    k = len(lx)
    logging.info(f'k = {k}')
    logging.info(f'weight of first element of nullspace(Hx): {np.sum(ns_x[0])}')
    logging.info(f'weight of first element of nullspace(Hz): {np.sum(ns_z[0])}')
    logging.info(f'weight of Lx: {np.sum(lx[0])}')
    logging.info(f'weight of Lz: {np.sum(lz[0])}')