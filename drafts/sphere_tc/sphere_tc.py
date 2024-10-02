import numpy as np


def vertices(L):
    xs, ys, zs = np.meshgrid(np.arange(L+1), np.arange(L+1), np.arange(L+1), indexing='ij')
    full = np.transpose([[xs.flatten(), ys.flatten(), zs.flatten()]])
    mask = full[:, 0]  > 0 & full[:, 0] < L & \
            full[:, 1]  > 0 & full[:, 1] < L & \
            full[:, 2]  > 0 & full[:, 2] < L
    mask = np.logical_xor(mask, np.ones(mask.shape, dtype=bool))
    return full[mask]

def faces(L):
    xs, ys, zs = np.meshgrid(np.arange(L), np.arange(L), np.arange(L), indexing='ij')
    z_faces = np.transpose([[(xs+0.5).flatten(), (ys+0.5).flatten(), zs.flatten()]])
    y_faces = np.transpose([[(xs+0.5).flatten(), ys.flatten(), (zs+0.5).flatten()]])
    x_faces = np.transpose([[xs.flatten(), (ys+0.5).flatten(), (zs+0.5).flatten()]])
    full = np.concatenate([x_faces, y_faces, z_faces], axis=0)
    mask = full[:, 0]  > 0 & full[:, 0] < L & \
            full[:, 1]  > 0 & full[:, 1] < L & \
            full[:, 2]  > 0 & full[:, 2] < L
    mask = np.logical_xor(mask, np.ones(mask.shape, dtype=bool))
    return full[mask]

def edges(L):
    xs, ys, zs = np.meshgrid(np.arange(L), np.arange(L), np.arange(L), indexing='ij')
    z_edges = np.transpose([[xs.flatten(), ys.flatten(), (zs+0.5).flatten()]])
    y_edges = np.transpose([[xs.flatten(), (ys+0.5).flatten(), zs.flatten()]])
    x_edges = np.transpose([[(xs+0.5).flatten(), ys.flatten(), zs.flatten()]])
    full = np.concatenate([x_edges, y_edges, z_edges], axis=0)
    mask = full[:, 0]  > 0 & full[:, 0] < L & \
            full[:, 1]  > 0 & full[:, 1] < L & \
            full[:, 2]  > 0 & full[:, 2] < L
    mask = np.logical_xor(mask, np.ones(mask.shape, dtype=bool))
    return full[mask]

def edges_to_ind(edges):
    return np.array({e: i for i, e in enumerate(edges)})

def z_stabilizers(L, q2i):
    xs, ys, zs = np.meshgrid(np.arange(L), np.arange(L), np.arange(L), indexing='ij')
    stabs = []

    x_faces = np.vstack(
        np.transpose([[np.zeros(xs.shape).flatten(), (ys+0.5).flatten(), (zs+0.5).flatten()]]),
        np.transpose([[L*np.ones(xs.shape).flatten(), (ys+0.5).flatten(), (zs+0.5).flatten()]])
    )
    x_face_to_edge_deltas = np.array([
        [0, 0.5, 0.5],
        [0, -0.5, 0.5],
        [0, 0.5, -0.5],
        [0, -0.5, -0.5]
    ])
    for x_face in x_faces:
        qubits = [x_face + delta for delta in x_face_to_edge_deltas]
        qubit_inds = [q2i[q] for q in qubits]
        stabs.append(qubit_inds)
    
    y_faces = np.vstack(
        np.transpose([[(xs+0.5).flatten(), np.zeros(xs.shape).flatten(), (zs+0.5).flatten()]]),
        np.transpose([[(xs+0.5).flatten(), L*np.ones(xs.shape).flatten(), (zs+0.5).flatten()]])
    )
    y_face_to_edge_deltas = np.array([
        [0.5, 0, 0.5],
        [-0.5, 0, 0.5],
        [0.5, 0, -0.5],
        [-0.5, 0, -0.5]
    ])
    for y_face in y_faces:
        qubits = [y_face + delta for delta in y_face_to_edge_deltas]
        qubit_inds = [q2i[q] for q in qubits]
        stabs.append(qubit_inds)

    z_faces = np.vstack(
        np.transpose([[(xs+0.5).flatten(), (ys+0.5).flatten(), np.zeros(xs.shape).flatten()]]),
        np.transpose([[(xs+0.5).flatten(), (ys+0.5).flatten(), L*np.ones(xs.shape).flatten()]])
    )
    z_face_to_edge_deltas = np.array([
        [0.5, 0.5, 0],
        [-0.5, 0.5, 0],
        [0.5, -0.5, 0],
        [-0.5, -0.5, 0]
    ])
    for z_face in z_faces:
        qubits = [z_face + delta for delta in z_face_to_edge_deltas]
        qubit_inds = [q2i[q] for q in qubits]
        stabs.append(qubit_inds)
    
    return np.array(stabs)

def x_stabilizers(L, q2i):
    pass