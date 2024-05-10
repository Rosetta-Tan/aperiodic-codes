export const goldenRatio = (1 + Math.sqrt(5)) / 2;

export class Vtx {
    constructor(x, y) {
        this.x = x;
        this.y = y;
    }

    add(other) {
        return new Vtx(this.x + other.x, this.y + other.y);
    }

    scale(s) {
        return new Vtx(this.x * s, this.y * s);
    }
}

export class Edge {
    constructor(v1, v2) {
        this.v1 = v1;
        this.v2 = v2;
    }
}
export class Face {
    constructor(ctg, v1, v2, v3) {
        this.ctg = ctg;
        this.v1 = v1;
        this.v2 = v2;
        this.v3 = v3;
    }
}

export class GeometryUtils {
    static sameVtx(v1, v2) {
        return Math.abs(v1.x - v2.x) < 0.00001 && Math.abs(v1.y - v2.y) < 0.00001;
    }

    static sameEdge(e1, e2) {
        return GeometryUtils.sameVtx(e1.v1, e2.v1) && GeometryUtils.sameVtx(e1.v2, e2.v2) || GeometryUtils.sameVtx(e1.v1, e2.v2) && GeometryUtils.sameVtx(e1.v2, e2.v1);
    }

    static getFaceCenter(face) {
        return new Vtx((face.v1.x + face.v2.x + face.v3.x) / 3, (face.v1.y + face.v2.y + face.v3.y) / 3);
    }

    static getFaceEdges(face) {
        // return [new Edge(face.v1, face.v2), new Edge(face.v2, face.v3), new Edge(face.v3, face.v1)];
        return [new Edge(face.v1, face.v2), new Edge(face.v1, face.v3)];
    }

    static subdivide(faces) {
        let newfaces = [];
        for (let i = 0; i < faces.length; i++) {
            let face = faces[i];
            let ctg = face.ctg;
            let v1 = face.v1;
            let v2 = face.v2;
            let v3 = face.v3;

            if (ctg == 1) {
                let P = v1.add((v2.add(v1.scale(-1))).scale(1/goldenRatio));
                newfaces.push(new Face(1, v3, P, v2));
                newfaces.push(new Face(2, P, v3, v1));
            } else {
                let Q = v2.add((v1.add(v2.scale(-1))).scale(1/goldenRatio));
                let R = v2.add((v3.add(v2.scale(-1))).scale(1/goldenRatio));
                newfaces.push(new Face(2, R, v3, v1));
                newfaces.push(new Face(2, Q, R, v2));
                newfaces.push(new Face(1, R, Q, v1));
            }
        }
        return newfaces;
    }
}

export class Tiling {
    static uniqVtxs(faces) {
        let vtxs = [];
        for (let i = 0; i < faces.length; i++) {
            let face = faces[i];
            let faceVtxs = [face.v1, face.v2, face.v3];
            for (let j = 0; j < faceVtxs.length; j++) {
                let vtx = faceVtxs[j];
                let found = false;
                for (let k = 0; k < vtxs.length; k++) {
                    if (GeometryUtils.sameVtx(vtx, vtxs[k])) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    vtxs.push(vtx);
                }
            }
        }
        return vtxs;
    };  

    static uniqEdges(faces) {
        let edges = [];
        for (let i = 0; i < faces.length; i++) {
            let face = faces[i];
            let faceEdges = GeometryUtils.getFaceEdges(face);
            for (let j = 0; j < faceEdges.length; j++) {
                let edge = faceEdges[j];
                let found = false;
                for (let k = 0; k < edges.length; k++) {
                    if (GeometryUtils.sameEdge(edge, edges[k])) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    edges.push(edge);
                }
            }
        }
        return edges;
    }

    static parityCheck(edges, vertices) {
        let z2Laplacian = [];
        // get upper triangular matrix
        for (let i = 0; i < vertices.length; i++) {
            z2Laplacian.push(new Array(vertices.length).fill(0));
            for (let j = i+1; j < vertices.length; j++) {
                let thisEdge = new Edge(vertices[i], vertices[j]);
                for (let k = 0; k < edges.length; k++) {
                    if (GeometryUtils.sameEdge(thisEdge, edges[k])) {
                        z2Laplacian[i][j] = 1;
                        break;
                    }
                }
            }
        }
        // fill in lower triangular matrix
        for (let i = 0; i < vertices.length; i++) {
            for (let j = 0; j < i; j++) {
                z2Laplacian[i][j] = z2Laplacian[j][i];
            }
        }
        // fill in diagonal
        for (let i = 0; i < vertices.length; i++) {
            let degree = 0;
            for (let j = 0; j < vertices.length; j++) {
                degree += z2Laplacian[i][j];
            }
            z2Laplacian[i][i] = degree % 2;
        }
        return z2Laplacian;
    }
}