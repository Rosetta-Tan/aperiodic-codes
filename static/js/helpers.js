export class Vtx {
    constructor(x, y) {
        this.x = x;
        this.y = y;
    }
}

export class Trig {
    constructor(v1, v2, v3) {
        this.v1 = v1;
        this.v2 = v2;
        this.v3 = v3;
    }
}

export class GeometryUtils {
    static sameVtx(v1, v2) {
        return abs(v1.x - v2.x) < 0.00001 && abs(v1.y - v2.y) < 0.00001;
    }

    static sameEdge(e1, e2) {
        return GeometryUtils.sameVtx(e1.v1, e2.v1) && GeometryUtils.sameVtx(e1.v2, e2.v2) || GeometryUtils.sameVtx(e1.v1, e2.v2) && GeometryUtils.sameVtx(e1.v2, e2.v1);
    }

    static getTrigCenter(trig) {
        return createVector((trig.v1.x + trig.v2.x + trig.v3.x) / 3, (trig.v1.y + trig.v2.y + trig.v3.y) / 3);
    }

    static getTrigEdges(trig) {
        return [new Edge(trig.v1, trig.v2), new Edge(trig.v2, trig.v3), new Edge(trig.v3, trig.v1)];
    }

    static subdivide(trigs) {
        let newTrigs = [];
        for (let i = 0; i < trigs.length; i++) {
            let trig = trigs[i];
            let center = GeometryUtils.getTrigCenter(trig);
            let edges = GeometryUtils.getTrigEdges(trig);
            for (let j = 0; j < edges.length; j++) {
                let edge = edges[j];
                let newVtx = createVector((edge.v1.x + edge.v2.x) / 2, (edge.v1.y + edge.v2.y) / 2);
                newTrigs.push(new Trig(edge.v1, edge.v2, newVtx));
                newTrigs.push(new Trig(edge.v2, edge.v1, newVtx));
            }
            newTrigs.push(new Trig(trig.v1, center, trig.v3));
            newTrigs.push(new Trig(trig.v2, center, trig.v1));
            newTrigs.push(new Trig(trig.v3, center, trig.v2));
        }
        return newTrigs;
    }
}

class Tiling {
    static uniqVtxs(trigs) {
        let vtxs = [];
        for (let i = 0; i < trigs.length; i++) {
            let trig = trigs[i];
            vtxs.push(trig.v1);
            vtxs.push(trig.v2);
            vtxs.push(trig.v3);
        }
    }
}