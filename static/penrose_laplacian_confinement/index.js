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
}

function drawDefault() {
    let stateVec = Array(initVtxs.length).fill(0);
    draw(initEdges, initVtxs, stateVec, [], bitCanvas);
    draw(initEdges, initVtxs, stateVec, [], checkCanvas, 'check');
    let facesNxt = helpers.GeometryUtils.subdivide(initFaces);
    let [edgesNxt, verticesNxt, parityCheckNxt, bitVectorNxt, checkVectorNxt, hlgtBitIndsNxt, hlgtCheckIndsNxt] = initData(facesNxt);
    stateVec = Array(verticesNxt.length).fill(0);
    draw(edgesNxt, verticesNxt, stateVec, [], bitCanvasNxt);
    draw(edgesNxt, verticesNxt, stateVec, [], checkCanvasNxt, 'check');
}

function draw(edges, vertices, stateVec, hlgtInds, canvas, type = 'bit') {
    let ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Calculate bounds
    let [minX, minY, scaleX, scaleY] = getCanvasScale(canvas, vertices);
    
    // Draw vertices
    if (type === 'bit') {
        ctx.fillStyle = 'blue';
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
            ctx.arc(x, y, 5, 0, 2 * Math.PI);
            ctx.fill();
            if (hlgtInds.includes(i)) {
                ctx.lineWidth = 2;
                ctx.globalAlpha = 1;
                ctx.beginPath();
                ctx.arc(x, y, 5+1, 0, 2 * Math.PI);
                ctx.stroke();
            }
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
            if (hlgtInds.includes(i)) {
                ctx.lineWidth = 2;
                ctx.globalAlpha = 1;
                ctx.beginPath();
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

function nearestVtxIdx(vertices, x, y) {
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
    return minIdx;
}

function initData(faces) {
    let edges = helpers.Tiling.uniqEdges(faces);
    let vertices = helpers.Tiling.uniqVtxs(faces);
    let parityCheck = helpers.Tiling.parityCheck(edges, vertices);
    let bitVector = Array(vertices.length).fill(0);
    let checkVector = Array(vertices.length).fill(0);
    let hlgtBitInds = [];
    let hlgtCheckInds = [];
    return [edges, vertices, parityCheck, bitVector, checkVector, hlgtBitInds, hlgtCheckInds];
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

function updatehlgtBitIndsNxt(verticesNxt, hlghtVtx, hlgtBitIndsNxt) {
    let idx = verticesNxt.indexOf(hlghtVtx);
    if (!hlgtBitIndsNxt.includes(idx)) {
        hlgtBitIndsNxt.push(idx);
    } else {
        hlgtBitIndsNxt = hlgtBitIndsNxt.filter((v) => v !== idx);
    }
    return hlgtBitIndsNxt;
}

form.addEventListener('submit', function(event) {
    event.preventDefault();

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
    let facesNxt = helpers.GeometryUtils.subdivide(faces);

    // initialize data for display and calculation for this generation
    let [edges, vertices, parityCheck, bitVector, checkVector, hlgtBitInds, hlgtCheckInds] = initData(faces);
    draw(edges, vertices, bitVector, [], bitCanvas);
    draw(edges, vertices, checkVector, [] ,checkCanvas, 'check');
    let [edgesNxt, verticesNxt, parityCheckNxt, bitVectorNxt, checkVectorNxt, hlgtBitIndsNxt, hlgtCheckIndsNxt] = initData(facesNxt);
    draw(edgesNxt, verticesNxt, bitVectorNxt, [], bitCanvasNxt);
    draw(edgesNxt, verticesNxt, checkVectorNxt, [], checkCanvasNxt, 'check');

    // add event listeners
    let currentCanvas = null;

    chooseChecks.onclick = (currentCanvas) => {
        removeEventListeners();
        currentCanvas = 'checkCanvas';
        checkCanvas.style.zIndex = 1;
        bitCanvas.style.zIndex = 0;

        checkCanvas.onclick = (event) => {
            let [x, y] = [event.offsetX, event.offsetY]
            let [minX, minY, scaleX, scaleY] = getCanvasScale(checkCanvas, vertices);
            x = (x - PADDING) / scaleX + minX;
            y = (y - PADDING) / scaleY + minY;
            let idx = nearestVtxIdx(vertices, x, y);            
            if (!hlgtCheckInds.includes(idx)) {
                hlgtCheckInds.push(idx);
            } else {
                hlgtCheckInds = hlgtCheckInds.filter((v) => v !== idx);
            }
            draw(edges, vertices, checkVector, hlgtCheckInds, checkCanvas, 'check');
            let hlgtChecks = [];
            for (let i = 0; i < hlgtCheckInds.length; i++) {
                hlgtChecks.push(vertices[hlgtCheckInds[i]]);
            }
            hlgtCheckIndsNxt = updatehlgtCheckIndsNxt(verticesNxt, hlgtChecks);
            draw(edgesNxt, verticesNxt, checkVectorNxt, hlgtCheckIndsNxt, checkCanvasNxt, 'check');
        }
    }

    chooseBits.onclick = (currentCanvas) => {
        removeEventListeners();
        currentCanvas = 'bitCanvas';
        checkCanvas.style.zIndex = 0;
        bitCanvas.style.zIndex = 1;

        bitCanvas.onclick = (event) => {
            let [x, y] = [event.offsetX, event.offsetY]
            let [minX, minY, scaleX, scaleY] = getCanvasScale(bitCanvas, vertices);
            x = (x - PADDING) / scaleX + minX;
            y = (y - PADDING) / scaleY + minY;
            let idx = nearestVtxIdx(vertices, x, y);
            if (!hlgtBitInds.includes(idx)) {
                hlgtBitInds.push(idx);
            } else {
                hlgtBitInds = hlgtBitInds.filter((v) => v !== idx);
            }
            draw(edges, vertices, bitVector, hlgtBitInds, bitCanvas);
            // hlgtBitIndsNxt = updatehlgtBitIndsNxt(verticesNxt, vertices[idx], hlgtBitIndsNxt);
            // draw(edgesNxt, verticesNxt, bitVectorNxt, hlgtBitIndsNxt, bitCanvasNxt);
        }
    }

    // reset the bit and check vectors
    reset.onclick = function() {
        bitVector = Array(vertices.length).fill(0);
        checkVector = Array(vertices.length).fill(0);
        draw(edges, vertices, bitVector, [], bitCanvas);
        draw(edges, vertices, checkVector, [], checkCanvas, 'check');
        draw(edgesNxt, verticesNxt, bitVectorNxt, [], bitCanvasNxt);
        draw(edgesNxt, verticesNxt, checkVectorNxt, [], checkCanvasNxt, 'check');
    }
}
);

window.onload = () => {
    removeEventListeners();
    drawDefault();
}