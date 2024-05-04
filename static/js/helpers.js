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
export class Trig {
    constructor(v1, v2, v3) {
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

    static getTrigCenter(trig) {
        return new Vtx((trig.v1.x + trig.v2.x + trig.v3.x) / 3, (trig.v1.y + trig.v2.y + trig.v3.y) / 3);
    }

    static getTrigEdges(trig) {
        return [new Edge(trig.v1, trig.v2), new Edge(trig.v2, trig.v3), new Edge(trig.v3, trig.v1)];
    }

    static subdivide(trigs) {
        let newTrigs = [];
        for (let i = 0; i < trigs.length; i++) {
            let trig = trigs[i];
            let p1 = trig.v1.add(trig.v2.add(trig.v1.scale(-1)).scale(0.4));
            let p2 = trig.v1.add(trig.v2.add(trig.v1.scale(-1)).scale(0.8));
            let p3 = (trig.v1.add(trig.v3)).scale(0.5);
            let p4 = (p2.add(trig.v3)).scale(0.5);

            let f1 = new Trig(trig.v1, p3, p1);
            let f2 = new Trig(p2, p3, p1);
            let f3 = new Trig(p3, p2, p4);
            let f4 = new Trig(p3, trig.v3, p4);
            let f5 = new Trig(trig.v3, trig.v2, p2);
            newTrigs.push(f1);
            newTrigs.push(f2);
            newTrigs.push(f3);
            newTrigs.push(f4);
            newTrigs.push(f5);
        }
        return newTrigs;
    }
}

export class Tiling {
    static uniqVtxs(trigs) {
        let vtxs = [];
        for (let i = 0; i < trigs.length; i++) {
            let trig = trigs[i];
            let trigVtxs = [trig.v1, trig.v2, trig.v3];
            for (let j = 0; j < trigVtxs.length; j++) {
                let vtx = trigVtxs[j];
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

    static uniqEdges(trigs) {
        let edges = [];
        for (let i = 0; i < trigs.length; i++) {
            let trig = trigs[i];
            let trigEdges = GeometryUtils.getTrigEdges(trig);
            for (let j = 0; j < trigEdges.length; j++) {
                let edge = trigEdges[j];
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

    static draw(trigs, vertices, ctx) {
        ctx.stroke(0);
        ctx.strokeWeight(1);
        for (let i = 0; i < trigs.length; i++) {
            let trig = trigs[i];
            ctx.fill(255);
            ctx.triangle(trig.v1.x, trig.v1.y, trig.v2.x, trig.v2.y, trig.v3.x, trig.v3.y);
        }
        ctx.stroke(0);
        ctx.strokeWeight(2);
        for (let i = 0; i < vertices.length; i++) {
            let vtx = vertices[i];
            ctx.point(vtx.x, vtx.y);
        }
    }

}