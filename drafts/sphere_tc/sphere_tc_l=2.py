import numpy as np
import matplotlib.pyplot as plt
from aperiodic_codes.cut_and_project.z2 import nullspace, rank
from aperiodic_codes.cut_and_project.code_param_utils import compute_lz

if __name__ == '__main__':
    n_faces = 24
    n_vertices = 26
    n_edges = 48
    v_coords = {
        0: [0, 2, 2],
        1: [1, 2, 2],
        2: [2, 2, 2],
        3: [0, 1, 2],
        4: [1, 1, 2],
        5: [2, 1, 2],
        6: [0, 0, 2],
        7: [1, 0, 2],
        8: [2, 0, 2],
        9: [0, 2, 1],
        10: [1, 2, 1],
        11: [2, 2, 1],
        12: [0, 1, 1],
        13: [2, 1, 1],
        14: [0, 0, 1],
        15: [1, 0, 1],
        16: [2, 0, 1],
        17: [0, 2, 0],
        18: [1, 2, 0],
        19: [2, 2, 0],
        20: [0, 1, 0],
        21: [1, 1, 0],
        22: [2, 1, 0],
        23: [0, 0, 0],
        24: [1, 0, 0],
        25: [2, 0, 0]
    }

    f2e = {
        0: [0, 2, 17, 19],
        1: [1, 3, 19, 21],
        2: [2, 4, 16, 18],
        3: [3, 5, 18, 20],
        4: [10, 12, 27, 29],
        5: [11, 13, 29, 31],
        6: [12, 14, 26, 28],
        7: [13, 15, 28, 30],
        8: [16, 22, 32, 34],
        9: [17, 23, 34, 36],
        10: [22, 26, 33, 35],
        11: [23, 27, 35, 37],
        12: [20, 24, 42, 44],
        13: [21, 25, 44, 46],
        14: [24, 30, 43, 45],
        15: [25, 31, 45, 47],
        16: [4, 8, 32, 38],
        17: [5, 9, 38, 42],
        18: [8, 14, 33, 39],
        19: [9, 15, 39, 43],
        20: [0, 6, 36, 40],
        21: [1, 7, 40, 46],
        22: [6, 10, 37, 41],
        23: [7, 11, 41, 47]
    }

    v2e = {
        0: [0, 17, 36],
        1: [0, 1, 19, 40],
        2: [1, 21, 46],
        3: [16, 17, 2, 34],
        4: [2, 3, 18, 19],
        5: [3, 20, 21, 44],
        6: [16, 32, 4],
        7: [4, 5, 18, 38],
        8: [5, 20, 42],
        9: [36, 6, 23, 37],
        10: [6, 7, 40, 41],
        11: [25, 7, 46, 47],
        12: [22, 23, 34, 35],
        13: [24, 25, 44, 45],
        14: [8, 22, 32, 33],
        15: [8, 9, 38, 39],
        16: [9, 24, 42, 43],
        17: [10, 27, 37],
        18: [29, 10, 11, 41],
        19: [31, 11, 47],
        20: [12, 26, 27, 35],
        21: [12, 13, 28, 29],
        22: [13, 30, 31, 45],
        23: [14, 26, 33],
        24: [14, 15, 28, 39],
        25: [15, 30, 43]
    }

    v2e_mat = np.zeros((n_vertices, n_edges), dtype=np.int64)
    for i in range(n_vertices):
        v2e_mat[i, v2e[i]] = 1
    e2v_mat = v2e_mat.T
    assert np.all(np.sum(e2v_mat, axis=1) == 2)

    hz = np.zeros((n_faces, n_edges), dtype=np.int64)
    for i in range(n_faces):
        hz[i, f2e[i]] = 1
    # assert rank(hz) == n_faces - 1

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.hist(np.sum(hz, axis=1), bins=range(10))
    # plt.savefig('hist.png')

    hx = np.zeros((n_vertices, n_edges), dtype=np.int64)
    for i in range(n_vertices):
        hx[i, v2e[i]] = 1

    ns_x = nullspace(hx)
    ns_z = nullspace(hz)
    print('===== original matrices =====')
    print(f'dim ker(hx) = {len(ns_x)}')
    print(f'dim ker(hz) = {len(ns_z)}')
    print('=============================')
    for i in range(len(ns_x)):
        print(f'ns_x[{i}] = {ns_x[i]}')

    z_del = [0]
    # x_del = [15, 17]
    # x_del = [12, 20]
    x_del = [6, 22]
    for i in z_del:
        hz[i] = 0
    for i in x_del:
        hx[i] = 0
    
    assert np.all((hx @ hz.T) % 2 == 0)

    ns_x = nullspace(hx)
    ns_z = nullspace(hz)
    lz = compute_lz(hx, hz)
    lx = compute_lz(hz, hx)
    k = len(lz)
    print(f'k = {k}')
    print(f'd_x = {np.sum(lx[0])}')
    print(f'lx = {np.where(lx[0] == 1)[0]}')
    print(f'd_z = {np.sum(lz[0])}')
    print(f'lz = {np.where(lz[0] == 1)[0]}')

    print(f'first element of ns_x: {np.sum(ns_x[0])}')
    print(f'first element of ns_z: {np.sum(ns_z[0])}')
    
    print(f'len(ns_x) = {len(ns_x)}')
    print(f'len(ns_z) = {len(ns_z)}')
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), subplot_kw={'projection': '3d'})
    ax[0].set_box_aspect([1, 1, 1])
    for i in range(n_edges):
        v1, v2 = np.nonzero(e2v_mat[i])[0]
        ax[0].plot([v_coords[v1][0], v_coords[v2][0]], [v_coords[v1][1], v_coords[v2][1]], [v_coords[v1][2], v_coords[v2][2]], color='k')
        if lx[0][i] == 1:
            ax[0].plot([v_coords[v1][0], v_coords[v2][0]], [v_coords[v1][1], v_coords[v2][1]], [v_coords[v1][2], v_coords[v2][2]], color='r')
    if len(x_del) > 0:
        for cnt, i in enumerate(x_del):
            if cnt == 0:
                ax[0].scatter(v_coords[i][0], v_coords[i][1], v_coords[i][2], color='b', s=30, alpha=0.4, label='deleted $g_X$')
            else:
                ax[0].scatter(v_coords[i][0], v_coords[i][1], v_coords[i][2], color='b', s=30, alpha=0.4)

    if len(z_del) > 0:
        for cnt_i, i in enumerate(z_del):
            edges = f2e[i]
            for cnt_j, j in enumerate(edges):
                v1, v2 = np.nonzero(e2v_mat[j])[0]
                if cnt_i == 0 and cnt_j == 0:
                    ax[0].plot([v_coords[v1][0], v_coords[v2][0]], [v_coords[v1][1], v_coords[v2][1]], [v_coords[v1][2], v_coords[v2][2]], color='g', lw=5, alpha=0.2, label='deleted $g_Z$')
                else:
                    ax[0].plot([v_coords[v1][0], v_coords[v2][0]], [v_coords[v1][1], v_coords[v2][1]], [v_coords[v1][2], v_coords[v2][2]], color='g', lw=5, alpha=0.2)

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_zticks([])
    ax[0].set_title('logical X')

    ax[1].set_box_aspect([1, 1, 1])
    for i in range(n_edges):
        v1, v2 = np.nonzero(e2v_mat[i])[0]
        ax[1].plot([v_coords[v1][0], v_coords[v2][0]], [v_coords[v1][1], v_coords[v2][1]], [v_coords[v1][2], v_coords[v2][2]], color='k')
        if lz[0][i] == 1:
            ax[1].plot([v_coords[v1][0], v_coords[v2][0]], [v_coords[v1][1], v_coords[v2][1]], [v_coords[v1][2], v_coords[v2][2]], color='r')
    if len(x_del) > 0:
        for cnt, i in enumerate(x_del):
            ax[1].scatter(v_coords[i][0], v_coords[i][1], v_coords[i][2], color='b', s=30, alpha=0.4)

    if len(z_del) > 0:
        for cnt_i, i in enumerate(z_del):
            edges = f2e[i]
            for cnt_j, j in enumerate(edges):
                v1, v2 = np.nonzero(e2v_mat[j])[0]
                ax[1].plot([v_coords[v1][0], v_coords[v2][0]], [v_coords[v1][1], v_coords[v2][1]], [v_coords[v1][2], v_coords[v2][2]], color='g', lw=5, alpha=0.2)
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_zticks([])
    ax[1].set_title('logical Z')

    fig.legend()
    fig.savefig('logical_operators.png')
    plt.show()
