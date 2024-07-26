'''
Cut and project from 5D to 2D to generate a 2D Penrose tiling
'''
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
import matplotlib.pyplot as plt
from cnp_utils import gen_lat, gen_voronoi

def gen_proj_pos():
    '''
    Generate the projection matrix into the positive eigenvalue 2D subspace
    shape: (2, 5)
    see Phil's Mathematica notebook.
    '''
    # Each basis vector is a row vector. shape: (1, 2)
    v1 = np.array([1, 0])
    v2 = np.array([np.cos(2*np.pi/5), np.sin(2*np.pi/5)])
    v3 = np.array([np.cos(4*np.pi/5), np.sin(4*np.pi/5)])
    v4 = np.array([np.cos(6*np.pi/5), np.sin(6*np.pi/5)])
    v5 = np.array([np.cos(8*np.pi/5), np.sin(8*np.pi/5)])
    proj_pos = np.sqrt(2/5) * np.vstack([v1, v2, v3, v4, v5]).T  # shape: (2, 5)
    
    return proj_pos

def gen_proj_neg():
    '''
    Generate the projection matrix into the negative eigenvalue 3D subspace
    shape: (3, 5)
    see Phil's Mathematica notebook.
    '''
    # Each basis vector is a row vector. shape: (1, 3)
    v1 = np.array([1, 0, 1/np.sqrt(2)])
    v2 = np.array([np.cos(4*np.pi/5), np.sin(4*np.pi/5), 1/np.sqrt(2)])
    v3 = np.array([np.cos(8*np.pi/5), np.sin(8*np.pi/5), 1/np.sqrt(2)])
    v4 = np.array([np.cos(2*np.pi/5), np.sin(2*np.pi/5), 1/np.sqrt(2)])
    v5 = np.array([np.cos(6*np.pi/5), np.sin(6*np.pi/5), 1/np.sqrt(2)])
    proj_neg = np.sqrt(2/5) * np.vstack([v1, v2, v3, v4, v5]).T  # shape: (3, 5)

    return proj_neg

def voronoi_project(voronoi, proj_neg, visualize=False):
    '''
    Project the Voronoi unit cell into the negative eigenvalue 3D subspace.
    Return:
        np.array, shape=(3, 2**5)
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

def cut(lat_pts, voronoi, proj, offset_vec=None):
    '''
    Select lattice points in the 6D lattice.
        - condition: the lattice point is inside the ocnvex hull of
        the Voronoi unit cell projected to the perpendiculr 3D space.
    Args:
        lat_pts: np.array, shape=(6, n**6)
        - lattice points in 6D
        voronoi: np.array, shape=(6, 2**6)
        - voronoi cell around origin in 6D
        proj: np.array, shape=(3, 6)
        - projection isometry matrix into the negative eigenvalue 3D subspace
        (default: negative eigenvalue)
    Return:
        cut_pts: np.array, shape=(6, n_cut)
        - selected points in 5D
        - cut_pts[0]: x0 coordinates
        - cut_pts[1]: x1 coordinates
        - ...
        - cut_pts[4]: x4 coordinates
    '''
    # convex hull of projected Voronoi cell in 3D
    # scipy requires pts to be row vectors
    triacontahedron = ConvexHull((proj @ voronoi).T)
    
    # Function to check if a point is inside the convex hull
    def _is_point_in_hull(point, hull, offset_3D=None):
        # We add the homogeneous coordinate for the point
        point = np.append(point, 1)
        # Loop over each facet of the hull
        for equation in hull.equations:
            if offset_vec is None:
                if np.dot(equation, point) > 0:
                    return False
            else:
                offset_3D = np.append(offset_3D, 0)
                if np.dot(equation, (point-offset_3D)) > 0:
                    return False
        return True
    
    # Select lattice points inside the convex hull
    cut_pts = []
    for i in range(lat_pts.shape[1]):
        pt_proj = proj @ lat_pts[:, i]
        if offset_vec is not None:
            offset_3D = proj @ offset_vec.reshape(-1, 1)
            if _is_point_in_hull(pt_proj, triacontahedron, offset_3D):
                cut_pts.append(lat_pts[:, i])
        else:
            if _is_point_in_hull(pt_proj, triacontahedron):
                cut_pts.append(lat_pts[:, i])
    print(f'number of points in the cut: {len(cut_pts)}')

    '''
    connect neighboring points in cut_pts 
    according to connectivity in the 6D lattice:
    if two points are neighbors in the 6D lattice,
    then they are connected by an edge in the 3D tiling
    '''
    # Check if two points are neighbors in 6D
    def _are_neighbors(pt1, pt2):  
        return np.sum(np.abs(pt1 - pt2)) == 1
    # Connect neighboring points in cut_pts
    cut_pts = np.asarray(cut_pts).T  # shape: (6, n_cut)
    n_cut = cut_pts.shape[1]
    adjacency_matrix = np.zeros((n_cut, n_cut), dtype=bool)
    for i in range(n_cut):
        for j in range(i+1, n_cut):
            if _are_neighbors(cut_pts[:, i], cut_pts[:, j]):
                adjacency_matrix[i, j] = True
                adjacency_matrix[j, i] = True

    return cut_pts, adjacency_matrix

def project(cut_pts, proj):
    '''
    Project the cut points into the 2D subspace.
    Args:
        cut_pts: np.array, shape=(5, n_cut)
        - selected points in 6D
        - cut_pts[0]: x0 coordinates
        - cut_pts[1]: x1 coordinates
        - ...
        - cut_pts[4]: x4 coordinates
        proj: np.array, shape=(2, 5) or (3, 5)
        - projection isometry matrix into the corresponding eigenvalue subspace
    Return:
        proj_pts: np.array, shape=(2, n_cut) or (3, n_cut)
        - projected points
        - proj_pts[0]: x0 coordinates
        - proj_pts[1]: x1 coordinates
        - ...
    '''
    proj_pts = proj @ cut_pts
    return proj_pts

def tiling(n, visualize=False):
    lat_pts = gen_lat(n, 5)
    print(f'lat_pts shape: {lat_pts.shape}')
    voronoi = gen_voronoi(5)
    print(f'voronoi shape: {voronoi.shape}')
    # projection isometry matrix into the positive eigenvalue 3D subspace
    proj_pos = gen_proj_pos()
    # projection isometry matrix into the negative eigenvalue 3D subspace
    proj_neg = gen_proj_neg()
    print(f'proj_pos shape: {proj_pos.shape}')
    print(f'proj_neg shape: {proj_neg.shape}')
    # cut and project
    cut_pts, adjacency_matrix = cut(lat_pts, voronoi, proj_neg)
    print(f'cut_pts shape: {cut_pts.shape}')
    proj_pts = project(cut_pts, proj_pos)
    print(f'proj_pts shape: {proj_pts.shape}')

    if visualize:
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.set_aspect('equal')
        ax.scatter(proj_pts[0], proj_pts[1])
        # plot the edges
        for i in range(proj_pts.shape[1]):
            for j in range(i+1, proj_pts.shape[1]):
                if adjacency_matrix[i, j]:
                    ax.plot([proj_pts[0, i], proj_pts[0, j]],
                            [proj_pts[1, i], proj_pts[1, j]],
                            color='black')
        plt.show()

    return proj_pts, adjacency_matrix

def test():
    high = 3
    low = -2
    lat_pts = gen_lat(low=low, high=high, dim=5)
    print(f'lat_pts shape: {lat_pts.shape}')
    voronoi = gen_voronoi(dim=5)
    print(f'voronoi shape: {voronoi.shape}')
    # projection isometry matrix into the positive eigenvalue 3D subspace
    proj_pos = gen_proj_pos()
    # projection isometry matrix into the negative eigenvalue 3D subspace
    proj_neg = gen_proj_neg()
    print(f'proj_pos shape: {proj_pos.shape}')
    print(f'proj_neg shape: {proj_neg.shape}')
    # cut and project
    cut_pts, adjacency_matrix = cut(lat_pts, voronoi, proj_neg)
    print(f'cut_pts shape: {cut_pts.shape}')
    proj_pts = project(cut_pts, proj_pos)
    print(f'proj_pts shape: {proj_pts.shape}')

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_aspect('equal')
    ax.scatter(proj_pts[0], proj_pts[1])
    # plot the edges
    for i in range(proj_pts.shape[1]):
        for j in range(i+1, proj_pts.shape[1]):
            if adjacency_matrix[i, j]:
                ax.plot([proj_pts[0, i], proj_pts[0, j]],
                        [proj_pts[1, i], proj_pts[1, j]],
                        color='black')
    plt.show()

    return proj_pts, adjacency_matrix

if __name__ == '__main__':
    test()