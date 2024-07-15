# 3D cut and project tiling
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import shapely

def gen_lat(n):
    '''
    Generate the 6-dimensional hypercubic lattice
    Return:
        lat_pts: np.array, shape=(6, n**6)
        - lattice pts are column vectors
        - lat_pts[0, :]: x0 coordinates for all pts
        - lat_pts[1, :]: x1 coordinates
        - ...
        - lat_pts[5, :]: x5 coordinates
    '''
    lat_pts = np.array(
            np.meshgrid(*([np.arange(0, n)] * 6), indexing='ij')
        ).reshape(6, -1)
    return lat_pts

def gen_voronoi():
    '''
    Compute the 6-dimensional Voronoi unit cell centered around origin
    Return:
        voronoi: np.array, shape=(6, 2**6)
        - lattice pts are column vectors
        - voronoi[:, 0]: the first pt [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5].T
        - voronoi[:, 1]: the second pt [-0.5, -0.5, -0.5, -0.5, -0.5, 0.5].T
        - ...
        - voronoi[:, 2**6-1]: the last pt [0.5, 0.5, 0.5, 0.5, 0.5, 0.5].T
    '''
    voronoi = np.array(
        np.meshgrid(*([[-0.5, 0.5]] * 6), indexing='ij')
    ).reshape(6, -1)
    return voronoi

def gen_proj_pos():
    '''
    Generate the projection matrix into the positive eigenvalue 3D subspace
    shape: (3, 6)
    Notation follows R. M. K. Dietl and J.-H. Eschenburg,
    see also Yi's notes page 4.
    '''
    costheta = 1/np.sqrt(5)
    sintheta = 2/np.sqrt(5)
    v6 = np.array([0, 0, np.sqrt(2)])
    v1 = np.array([np.sqrt(2)*sintheta, 0, np.sqrt(2)*costheta])
    v2 = np.array([
        np.sqrt(2)*sintheta*np.cos(2*np.pi/5),
        np.sqrt(2)*sintheta*np.sin(2*np.pi/5),
        np.sqrt(2)*costheta
        ])
    v3 = np.array([
        np.sqrt(2)*sintheta*np.cos(4*np.pi/5),
        np.sqrt(2)*sintheta*np.sin(4*np.pi/5),
        np.sqrt(2)*costheta
        ])
    v4 = np.array([
        np.sqrt(2)*sintheta*np.cos(6*np.pi/5),
        np.sqrt(2)*sintheta*np.sin(6*np.pi/5),
        np.sqrt(2)*costheta
        ])
    v5 = np.array([
        np.sqrt(2)*sintheta*np.cos(8*np.pi/5),
        np.sqrt(2)*sintheta*np.sin(8*np.pi/5),
        np.sqrt(2)*costheta
        ])
    proj_pos = np.vstack([v1, v2, v3, v4, v5, v6]).T  # shape: (3, 6)
    
    return proj_pos

def gen_proj_neg():
    '''
    Generate the projection matrix into the negative eigenvalue 3D subspace
    shape: (3, 6)
    Notation follows R. M. K. Dietl and J.-H. Eschenburg,
    see also Yi's notes page 4.
    '''
    costheta = 1/np.sqrt(5)
    sintheta = 2/np.sqrt(5)
    w6 = np.array([0, 0, -np.sqrt(2)])
    w1 = np.array([np.sqrt(2)*sintheta, 0, np.sqrt(2)*costheta])
    w2 = np.array([
        np.sqrt(2)*sintheta*np.cos(4*np.pi/5),
        np.sqrt(2)*sintheta*np.sin(4*np.pi/5),
        np.sqrt(2)*costheta
        ])
    w3 = np.array([
        np.sqrt(2)*sintheta*np.cos(8*np.pi/5),
        np.sqrt(2)*sintheta*np.sin(8*np.pi/5),
        np.sqrt(2)*costheta
        ])
    w4 = np.array([
        np.sqrt(2)*sintheta*np.cos(2*np.pi/5),
        np.sqrt(2)*sintheta*np.sin(2*np.pi/5),
        np.sqrt(2)*costheta
        ])
    w5 = np.array([
        np.sqrt(2)*sintheta*np.cos(6*np.pi/5),
        np.sqrt(2)*sintheta*np.sin(6*np.pi/5),
        np.sqrt(2)*costheta
        ])
    proj_neg = np.vstack([w1, w2, w3, w4, w5, w6]).T  # shape: (3, 6)

    return proj_neg

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
    
    hull_pts = (proj_neg @ voronoi).T[triacontahedron.vertices]
    # Create a Delaunay triangulation object
    del_obj = Delaunay(hull_pts)
    
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

def cut(lat_pts, voronoi, proj_neg):
    '''
    Select lattice points in the 6D lattice.
        - condition: the lattice point is inside the ocnvex hull of
        the Voronoi unit cell projected to the perpendiculr 3D space.
    Args:
        lat_pts: np.array, shape=(6, n**6)
        - lattice points in 6D
        voronoi: np.array, shape=(6, 2**6)
        - voronoi cell around origin in 6D
        proj_neg: np.array, shape=(3, 6)
        - projection isometry matrix into the negative eigenvalue 3D subspace
    Return:
        cut_pts: np.array, shape=(6, n_cut)
        - selected points in 6D
        - cut_pts[0]: x0 coordinates
        - cut_pts[1]: x1 coordinates
        - ...
        - cut_pts[5]: x5 coordinates
    '''
    # convex hull of projected Voronoi cell in 3D
    # scipy requires pts to be row vectors
    triacontahedron = ConvexHull((proj_neg @ voronoi).T)
    
    # Function to check if a point is inside the convex hull
    def _is_point_in_hull(point, hull):
        # We add the homogeneous coordinate for the point
        point = np.append(point, 1)
        # Loop over each facet of the hull
        for equation in hull.equations:
            if np.dot(equation, point) > 0:
                return False
        return True
    
    # Select lattice points inside the convex hull
    cut_pts = []
    for i in range(lat_pts.shape[1]):
        pt_proj = proj_neg @ lat_pts[:, i]
        if _is_point_in_hull(pt_proj, triacontahedron):
            cut_pts.append(lat_pts[:, i])

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

def project(cut_pts, proj_pos):
    '''
    Project the selected points into the postive eigenvalue 3D subspace.
    Return:
        np.array, shape=(3, n_cut)
        - projected points in
    '''
    return proj_pos @ cut_pts

def tiling(n, visualize=False):
    lat_pts = gen_lat(n)
    voronoi = gen_voronoi()
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

def test():
    n = 6
    lat_pts = gen_lat(n)
    voronoi = gen_voronoi()
    proj_pos = gen_proj_pos()
    proj_neg = gen_proj_neg()
    for i in range(3):
        # check the orthogonality of proj_pos[i] and proj_neg[i]
        dot_prod = np.dot(proj_pos[i], proj_neg[i])
        # print(f'dot product: {dot_prod}')
        assert np.isclose(dot_prod, 0), f'dot product: {dot_prod}'
    
    triacontahedron = ConvexHull((proj_neg @ voronoi).T)
    print(len(triacontahedron.simplices))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # make sure the three axes have the same scale
    # ax.set_box_aspect([1,1,1])
    # ax.plot_trisurf((proj_neg @ voronoi)[0], (proj_neg @ voronoi)[1], (proj_neg @ voronoi)[2], triangles=triacontahedron.simplices, alpha=0.8)
    
    # make tiling
    proj_pts, adjacency_matrix = tiling(n, visualize=False)
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