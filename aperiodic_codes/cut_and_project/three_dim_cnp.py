# 3D cut and project tiling
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from cnp_utils import *

def voronoi_project(voronoi, proj_neg, visualize=False):
    '''
    Project the Voronoi unit cell into the negative eigenvalue 3D subspace.
    Return:
        np.array, shape=(3, 2**6)
        - projected Voronoi pts in 3D
    '''
    pts = proj_neg @ voronoi
    triacontahedron = ConvexHull((proj_neg @ voronoi).T)
    assert len(triacontahedron.simplices) == 60, \
        f'len(triacontahedron.simplices): {len(triacontahedron.simplices)}'

    if visualize:    
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # make sure the three axes have the same scale
        ax.set_box_aspect([1,1,1])
        
        # plot the Voronoi cell projection
        ax.plot_trisurf(
            (proj_neg @ voronoi)[0],
            (proj_neg @ voronoi)[1],
            (proj_neg @ voronoi)[2],
            triangles=triacontahedron.simplices,
            alpha=0.8)
        
        # plot the Delaunay triangulation
        # ax.plot_trisurf(
        #     hull_pts[:, 0],
        #     hull_pts[:, 1],
        #     hull_pts[:, 2],
        #     triangles=del_obj.simplices,
        #     alpha=0.2)
        
        plt.show()
    
    return pts

def project(cut_pts, proj):
    '''
    Project the selected points into the selected eigenvalue's 3D subspace.
    (default: positive eigenvalue)
    Return:
        np.array, shape=(3, n_cut)
        - projected points
    '''
    return proj @ cut_pts

def are_connected(pt1, pt2):  
    return np.sum(np.abs(pt1 - pt2)) == 1

def gen_adj_mat(cut_pts):
    '''
    connect neighboring points in cut_pts 
    according to connectivity in the 6D lattice:
    if two points are neighbors in the 6D lattice,
    then they are connected by an edge in the 3D tiling
    '''
    # Connect neighboring points in cut_pts
    cut_pts = np.asarray(cut_pts).T  # shape: (6, n_cut)
    n_cut = cut_pts.shape[1]
    adjacency_matrix = np.zeros((n_cut, n_cut), dtype=bool)
    for i in range(n_cut):
        for j in range(i+1, n_cut):
            if are_connected(cut_pts[:, i], cut_pts[:, j]):
                adjacency_matrix[i, j] = True
                adjacency_matrix[j, i] = True

def tiling(low, high, visualize=False):
    lat_pts = gen_lat(low, high, dim=6)
    voronoi = gen_voronoi(dim=6)
    # projection isometry matrix into the positive eigenvalue 3D subspace
    proj_pos = gen_proj_pos()
    # projection isometry matrix into the negative eigenvalue 3D subspace
    proj_neg = gen_proj_neg()
    # cut and project
    cut_pts, adjacency_matrix = cut(lat_pts, voronoi, proj_neg)
    proj_pts = project(cut_pts, proj_pos)

    if visualize:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect([1,1,1])
        ax.scatter(proj_pts[0], proj_pts[1], proj_pts[2])
        # plot the edges
        for i in range(proj_pts.shape[1]):
            for j in range(i+1, proj_pts.shape[1]):
                if adjacency_matrix[i, j]:
                    ax.plot([proj_pts[0, i], proj_pts[0, j]],
                            [proj_pts[1, i], proj_pts[1, j]],
                            [proj_pts[2, i], proj_pts[2, j]],
                            color='black')
        plt.show()

    return proj_pts, adjacency_matrix

def tiling_neg(low, high, visualize=False):
    lat_pts = gen_lat(low, high, dim=6)
    voronoi = gen_voronoi(dim=6)
    # projection isometry matrix into the positive eigenvalue 3D subspace
    proj_pos = gen_proj_pos()
    # projection isometry matrix into the negative eigenvalue 3D subspace
    proj_neg = gen_proj_neg()
    # cut and project
    # offset_vec = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    offset_vec = np.zeros(6)
    cut_pts, adjacency_matrix = cut(lat_pts, voronoi, proj_pos,
                                     offset_vec=offset_vec)
    print(f'number of points in the cut: {cut_pts.shape[1]}')
    proj_pts = project(cut_pts, proj_neg)

    if visualize:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect([1,1,1])
        ax.scatter(proj_pts[0], proj_pts[1], proj_pts[2])
        # plot the edges
        for i in range(proj_pts.shape[1]):
            for j in range(i+1, proj_pts.shape[1]):
                if adjacency_matrix[i, j]:
                    ax.plot([proj_pts[0, i], proj_pts[0, j]],
                            [proj_pts[1, i], proj_pts[1, j]],
                            [proj_pts[2, i], proj_pts[2, j]],
                            color='black')
        plt.show()

    return proj_pts, adjacency_matrix


def test():
    high = 4
    low = -3
    voronoi = gen_voronoi(dim=6)
    proj_pos = gen_proj_pos()
    proj_neg = gen_proj_neg()
    for i in range(3):
        # check the orthogonality of proj_pos[i] and proj_neg[i]
        dot_prod = np.dot(proj_pos[i], proj_neg[i])
        # print(f'dot product: {dot_prod}')
        assert np.isclose(dot_prod, 0), f'dot product: {dot_prod}'
    
    '''
    Projecting into negative eigenspace
    '''
    triacontahedron = ConvexHull((proj_neg @ voronoi).T)
    print(len(triacontahedron.simplices))
    fig, ax = plt.subplots(1, 2, subplot_kw={'projection': '3d'})
    # ax = fig.add_subplot(111, projection='3d')
    ax[0].set_box_aspect([1,1,1])
    ax[0].plot_trisurf(
        (proj_neg @ voronoi)[0],
        (proj_neg @ voronoi)[1],
        (proj_neg @ voronoi)[2],
        triangles=triacontahedron.simplices, alpha=0.8)

    '''
    Projecting into positive eigenspace
    '''
    tria_pos = ConvexHull((proj_pos @ voronoi).T)
    print(len(tria_pos.simplices))
    ax[1].set_box_aspect([1,1,1])
    ax[1].plot_trisurf(
        (proj_pos @ voronoi)[0],
        (proj_pos @ voronoi)[1],
        (proj_pos @ voronoi)[2],
        triangles=tria_pos.simplices, alpha=0.8)

    '''
    Results:
    - The two convex hulls are the same. This is expected since the two
    projections both span the bases for an icosahedron.
    '''
    
    # make tiling
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    proj_pts, adjacency_matrix = tiling(low, high, visualize=False)
    ax.scatter(proj_pts[0], proj_pts[1], proj_pts[2])
    for i in range(proj_pts.shape[1]):
        for j in range(i+1, proj_pts.shape[1]):
            if adjacency_matrix[i, j]:
                ax.plot([proj_pts[0, i], proj_pts[0, j]],
                        [proj_pts[1, i], proj_pts[1, j]],
                        [proj_pts[2, i], proj_pts[2, j]],
                        color='gray', alpha=0.5)
    
    plt.show()

if __name__ == '__main__':
    # pipeline(6, visualize=True)
    test()