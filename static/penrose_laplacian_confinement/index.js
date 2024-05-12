import * as helpers from './helpers.js';
const PADDING = 20,
    form = document.getElementById('genForm'),
    input = document.getElementById('genInput'),
    reset = document.getElementById('resetButton'),
    chooseChecks = document.getElementById('chooseChecks'),
    chooseBits = document.getElementById('chooseBits'),
    bitCanvas = setCanvasStyle(document.getElementById('bitCanvas')),
    checkCanvas = setCanvasStyle(document.getElementById('checkCanvas')),
    bitCanvasNxt = setCanvasStyle(document.getElementById('bitCanvasNxt')),
    checkCanvasNxt = setCanvasStyle(document.getElementById('checkCanvasNxt'));

const initFaces = [];
for (let i = 0; i < 10; i++) {
    const angleB = (2 * i - 1) * Math.PI / 10;
    const angleC = (2 * i + 1) * Math.PI / 10;
    let B = new helpers.Vtx(Math.cos(angleB), Math.sin(angleB));
    let C = new helpers.Vtx(Math.cos(angleC), Math.sin(angleC));
    if (i % 2 === 0) {
        let tmp = B;
        B = C;
        C = tmp;
    }
    initFaces.push(new helpers.Face(1, new helpers.Vtx(0, 0), B, C));
}
const initVtxs = helpers.Tiling.uniqVtxs(initFaces);
const initEdges = helpers.Tiling.uniqEdges(initFaces);

function setCanvasStyle(canvas) {
    let style = window.getComputedStyle(canvas);
    [canvas.width, canvas.height] = [parseInt(style.width, 10), parseInt(style.height, 10)];
    return canvas;
}

function getCanvasScale(canvas, vertices) {
    let minX = Number.MAX_VALUE;
    let maxX = Number.MIN_VALUE;
    let minY = Number.MAX_VALUE;
    let maxY = Number.MIN_VALUE;
    for (let i = 0; i < vertices.length; i++) {
        let vertex = vertices[i];
        minX = Math.min(minX, vertex.x);
        maxX = Math.max(maxX, vertex.x);
        minY = Math.min(minY, vertex.y);
        maxY = Math.max(maxY, vertex.y);
    }
    let scaleX = (canvas.width - 2*PADDING) / (maxX - minX);
    let scaleY = (canvas.height - 2*PADDING) / (maxY - minY);
    return [minX, minY, scaleX, scaleY];
}

function removeEventListeners() {
    checkCanvas.onclick = null;
    bitCanvas.onclick = null;
    chooseChecks.onclick = null;
    chooseBits.onclick = null;
}

const config = {
    "current": {
        "bit": {
            "canvas": bitCanvas,
            "faces": initFaces,
            "edges": initEdges,
            "vertices": initVtxs,
            "hlgtVtxs": [],
            "stateVec": Array(initVtxs.length).fill(0)
        },
        "check": {
            "canvas": checkCanvas,
            "faces": initFaces,
            "edges": initEdges,
            "vertices": initVtxs,
            "hlgtVtxs": [],
            "stateVec": Array(initVtxs.length).fill(0)
        }
    },
    "next": {
        "bit": {
            "canvas": bitCanvasNxt,
            "faces": helpers.GeometryUtils.subdivide(initFaces),
            "edges": helpers.Tiling.uniqEdges(helpers.GeometryUtils.subdivide(initFaces)),
            "vertices": helpers.Tiling.uniqVtxs(helpers.GeometryUtils.subdivide(initFaces)),
            "hlgtVtxs": [],
            "stateVec": Array(initVtxs.length).fill(0)
        },
        "check": {
            "canvas": checkCanvasNxt,
            "faces": helpers.GeometryUtils.subdivide(initFaces),
            "edges": helpers.Tiling.uniqEdges(helpers.GeometryUtils.subdivide(initFaces)),
            "vertices": helpers.Tiling.uniqVtxs(helpers.GeometryUtils.subdivide(initFaces)),
            "hlgtVtxs": [],
            "stateVec": Array(initVtxs.length).fill(0)
        }
    }
}

function drawDefault() {
    let stateVec = Array(initVtxs.length).fill(0);
    draw(config.current.bit);
    draw(config.current.check);
    
    let facesNxt = helpers.GeometryUtils.subdivide(initFaces);
    config.next.bit.faces = facesNxt;
    config.next.check.faces = facesNxt;
    let verticesNxt = helpers.Tiling.uniqVtxs(facesNxt);
    config.next.bit.vertices = verticesNxt;
    config.next.check.vertices = verticesNxt;
    let edgesNxt = helpers.Tiling.uniqEdges(facesNxt);
    config.next.bit.edges = edgesNxt;
    config.next.check.edges = edgesNxt;
    stateVec = Array(verticesNxt.length).fill(0);
    config.next.bit.stateVec = stateVec;
    config.next.check.stateVec = stateVec;
    draw(config.next.bit);
    draw(config.next.check);
}

function draw(obj) {
    let canvas = obj.canvas;
    let ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    let vertices = obj.vertices;
    let edges = obj.edges;
    let stateVec = obj.stateVec;
    let hlgtVtxs = obj.hlgtVtxs;

    // Calculate bounds
    let [minX, minY, scaleX, scaleY] = getCanvasScale(canvas, vertices);
    
    // Draw vertices
    if (canvas.id.includes('bit')) {
        ctx.fillStyle = 'blue';
        for (let i = 0; i < vertices.length; i++) {
            if (stateVec[i] === 0) {
                ctx.globalAlpha = 0.2;
            }
            else {
                ctx.globalAlpha = 1;
            }
            ctx.beginPath();
            let x = (vertices[i].x - minX) * scaleX + PADDING;
            let y = (vertices[i].y - minY) * scaleY + PADDING;
            ctx.arc(x, y, 5, 0, 2 * Math.PI);
            ctx.fill();
        }
    } else {
        ctx.fillStyle = 'green';
        for (let i = 0; i < vertices.length; i++) {
            if (stateVec[i] === 0) {
                ctx.globalAlpha = 0.2;
            }
            else {
                ctx.globalAlpha = 1;
            }
            ctx.beginPath();
            var x = (vertices[i].x - minX) * scaleX + PADDING;
            var y = (vertices[i].y - minY) * scaleY + PADDING;
            // draw a square centered at (x,y)
            ctx.rect(x-7.5, y-7.5, 15, 15);
            ctx.fill();
        }
    }

    // Draw hightlighted vertices
    if (obj.hlgtVtxs.length > 0) {
        ctx.globalAlpha = 1;
        ctx.lineWidth = 2;
        if (obj.canvas.id.includes('bit')) {
            for (let i = 0; i < hlgtVtxs.length; i++) {
                ctx.beginPath();
                let x = (hlgtVtxs[i].x - minX) * scaleX + PADDING;
                let y = (hlgtVtxs[i].y - minY) * scaleY + PADDING;
                ctx.arc(x, y, 5+1, 0, 2 * Math.PI);
                ctx.stroke();
            }
        }
        else {
            for (let i = 0; i < hlgtVtxs.length; i++) {
                ctx.beginPath();
                let x = (hlgtVtxs[i].x - minX) * scaleX + PADDING;
                let y = (hlgtVtxs[i].y - minY) * scaleY + PADDING;
                ctx.rect(x-7.5-1, y-7.5-1, 15+2, 15+2);
                ctx.stroke();
            }
        }
    }
    
    // Draw edges
    ctx.strokeStyle = 'black';
    ctx.globalAlpha = 0.2;
    for (let i = 0; i < edges.length; i++) {
        let edge = edges[i];
        ctx.beginPath();
        var startX = (edge.v1.x - minX) * scaleX + PADDING;
        var startY = (edge.v1.y - minY) * scaleY + PADDING;
        var endX = (edge.v2.x - minX) * scaleX + PADDING;
        var endY = (edge.v2.y - minY) * scaleY + PADDING;
        ctx.moveTo(startX, startY);
        ctx.lineTo(endX, endY);
        ctx.stroke();
    }
}

function nearestVtxCoords(vertices, x, y) {
    let minDist = Number.MAX_VALUE;
    let minIdx = -1;
    for (let i = 0; i < vertices.length; i++) {
        let vertex = vertices[i];
        let dist = Math.sqrt((vertex.x - x) * (vertex.x - x) + (vertex.y - y) * (vertex.y - y));
        if (dist < minDist) {
            minDist = dist;
            minIdx = i;
        }
    }
    return [vertices[minIdx].x, vertices[minIdx].y];
}

function updatehlgtCheckIndsNxt(faces, hlghtVtxs) {
    let hlgtVtxsNxt = [];
    let hlgtVtxInds2faceIndsMap = {};
    for (let faceInd = 0; faceInd < faces.length; faceInd++) {
        let face = faces[faceInd];
        let faceVtxs = [face.v1, face.v2, face.v3];
        for (let j = 0; j < hlghtVtxs.length; j++) { // for each highlighted vertex
            if (hlgtVtxInds2faceIndsMap[j] === undefined) {
                hlgtVtxInds2faceIndsMap[j] = [];
                for (let k = 0; k < faceVtxs.length; k++) {
                    if (helpers.GeometryUtils.sameVtx(hlghtVtxs[j], faceVtxs[k])) {
                        hlgtVtxInds2faceIndsMap[j].push(faceInd);
                        break;
                    }
                }
            }
            // if the highlighted vertex is in the face, add the vertex following a rule to hlgtVtxsNxt
            if (hlgtVtxInds2faceIndsMap[j] == faceInd) {
                let ctg = faces[faceInd].ctg;
                // TODO
            }
        }

    }
}

function handleFormSubmit() {
    let gen = parseInt(input.value, 10);
    if (isNaN(gen)) {
        alert('Please enter a valid integer number');
        return;
    } else if (gen > 4) {
        alert('Please enter a number less than or equal to 4');
        return;
    }
    let faces = initFaces;
    for (let i = 0; i < gen; i++) {
        faces = helpers.GeometryUtils.subdivide(faces);
    }
    for (let type in config.current) {
        obj = config.current[type];
        obj.faces = faces;
        obj.edges = helpers.Tiling.uniqEdges(faces);
        obj.vertices = helpers.Tiling.uniqVtxs(faces);
        obj.hlgtVtxs = [];
        obj.stateVec = Array(obj.vertices.length).fill(0);
    }

    faces = helpers.GeometryUtils.subdivide(faces);
    for (let type in config.next) {
        obj = config.next[type];
        obj.faces = faces;
        obj.edges = helpers.Tiling.uniqEdges(faces);
        obj.vertices = helpers.Tiling.uniqVtxs(faces);
        obj.hlgtVtxs = [];
        obj.stateVec = Array(obj.vertices.length).fill(0);
    }
}

function handleChooseSubmit() {
    // removeEventListeners();
    console.log('type chosen:', document.activeElement.id);
    let id = document.activeElement.id;
    let type = id.includes("Check") ? "check" : "bit";
    let othertype = type === "check" ? "bit" : "check";
    let obj = config.current[type];
    let othobj = config.current[othertype];
    // set the z order of the current canvas to be higher than the other canvas
    obj.canvas.style.zIndex = 1;
    console.log(obj.canvas);
    othobj.canvas.style.zIndex = 0;
    console.log(othobj.canvas);
}

function setHlgtChecksFromCurrent(x, y) {
    obj = config.current.check;
    objNxt = config.next.check;
    objNxt.hlgtVtxs = [new helpers.Vtx(x, y)];
    // look for the vertex in faces
    let faces = obj.faces;
    let foundFaceInd = -1;
    let foundFaceCtg = -1;
    let foundFaceVtxInd = -1;
    for (let i = 0; i < faces.length; i++) {
        let face = faces[i];
        let faceVtxs = [face.v1, face.v2, face.v3];
        for (let j = 0; j < faceVtxs.length; j++) {
            let vtx = faceVtxs[j];
            if (vtx.x === x && vtx.y === y) {
                console.log('foundFaceInd:', i);
                foundFaceInd = i;
                foundFaceCtg = face.ctg;
                foundFaceVtxInd = j+1;
                break;
            }
        }
    }
    if (foundFaceCtg == 1) {
        if (foundFaceVtxInd != 2) {
            alert('face ctg is 1, but the vertex is not the second vertex');
            return;
        }
        let v1 = faces[foundFaceInd].v1;
        let v2 = faces[foundFaceInd].v2;
        let P = v1.add((v2.add(v1.scale(-1))).scale(1/helpers.goldenRatio));
        objNxt.hlgtVtxs.push(P);
    } else {
        if (foundFaceVtxInd == 2) {
            alert('face ctg is 2, but the vertex is not the first or third vertex');
            return;
        } else {
            let v2 = faces[foundFaceInd].v2;
            let v3 = faces[foundFaceInd].v3;
            if (foundFaceVtxInd == 1) {
                let Q = v2.add((v3.add(v2.scale(-1))).scale(1/helpers.goldenRatio));
                objNxt.hlgtVtxs.push(Q);
            } else {
                let R = v2.add((v1.add(v2.scale(-1))).scale(1/helpers.goldenRatio));
                objNxt.hlgtVtxs.push(R);
            }
        }
    }
}

function handleCanvasClick(event, obj) {
    console.log('getCoords');
    let vertices = obj.vertices;
    let [x, y] = [event.offsetX, event.offsetY];
    let [minX, minY, scaleX, scaleY] = getCanvasScale(bitCanvas, vertices);
    x = (x - PADDING) / scaleX + minX;
    y = (y - PADDING) / scaleY + minY;
    [x, y] = nearestVtxCoords(vertices, x, y);
    // look for the vertex in hlghtVtxs
    let hlghtVtxs = obj.hlgtVtxs;
    let foundInd = -1;
    for (let i = 0; i < hlghtVtxs.length; i++) {
        if (hlghtVtxs[i].x === x && hlghtVtxs[i].y === y) {
            foundInd = i;
            break;
        }
    }
    if (foundInd == -1) {  // if the vertex is not already highlighted
        hlghtVtxs.push(new helpers.Vtx(x, y));
    } else {
        hlghtVtxs.splice(foundInd, 1);
    }
    console.log('hlghtVtxs:', hlghtVtxs);
    obj.hlghtVtxs = hlghtVtxs;
    draw(obj);
    
    if (obj.canvas.id.includes('check')) {
        setHlgtChecksFromCurrent(x, y);
        draw(config.next.check);
    }
}

function resetCanvas(obj) {
    obj.hlgtVtxs = [];
    obj.stateVec = Array(obj.vertices.length).fill(0);
    draw(obj);
}

window.onload = () => {
    // removeEventListeners();
    drawDefault();
}

form.onsubmit = (event) => {
    event.preventDefault();
    handleFormSubmit();
    for (let key in config) {
        for (let type in config[key]) {
            draw(config[key][type]);
        }
    }
}

reset.onclick = () => {
    for (let key in config) {
        for (let type in config[key]) {
            resetCanvas(config[key][type]);
        }
    }
}

chooseChecks.onclick = () => {
    handleChooseSubmit();
}

chooseBits.onclick = () => {
    handleChooseSubmit();
}

checkCanvas.onclick = (event) => {
    console.log('checkCanvas clicked');
    handleCanvasClick(event, config.current.check);
}

bitCanvas.onclick = (event) => {
    console.log('bitCanvas clicked');
    handleCanvasClick(event, config.current.bit);
}