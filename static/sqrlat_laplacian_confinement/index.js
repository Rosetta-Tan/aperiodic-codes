import * as helpers from './helpers.js';
const PADDING = 20,
    form = document.getElementById('genForm'),
    input = document.getElementById('genInput'),
    reset = document.getElementById('resetButton'),
    chooseChecks = document.getElementById('chooseChecks'),
    chooseBits = document.getElementById('chooseBits'),
    initVtxs = [new helpers.Vtx(0, 0), new helpers.Vtx(1, 0), new helpers.Vtx(1, 1), new helpers.Vtx(0, 1)],
    bitCanvas = setCanvasStyle(document.getElementById('bitCanvas')),
    checkCanvas = setCanvasStyle(document.getElementById('checkCanvas')),
    bitCanvasNxt = setCanvasStyle(document.getElementById('bitCanvasNxt')),
    checkCanvasNxt = setCanvasStyle(document.getElementById('checkCanvasNxt'));


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
    let initFace = new helpers.Face(initVtxs[0], initVtxs[1], initVtxs[2], initVtxs[3]);
    let initEdges = helpers.GeometryUtils.getFaceEdges(initFace);
    let stateVec = Array(initVtxs.length).fill(0);
    draw(initEdges, initVtxs, stateVec, [], bitCanvas);
    draw(initEdges, initVtxs, stateVec, [], checkCanvas, 'check');
    draw(initEdges, initVtxs, stateVec, [], bitCanvasNxt);
    draw(initEdges, initVtxs, stateVec, [], checkCanvasNxt, 'check');
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

function initData(faces, thisBitCanvas, thisCheckCanvas) {
    let edges = helpers.Tiling.uniqEdges(faces);
    let vertices = helpers.Tiling.uniqVtxs(faces);
    let parityCheck = helpers.Tiling.parityCheck(edges, vertices);
    let bitVector = Array(vertices.length).fill(0);
    let checkVector = Array(vertices.length).fill(0);
    let hlgtBits = [];
    let hlgtChecks = [];
    return [edges, vertices, parityCheck, bitVector, checkVector, hlgtBits, hlgtChecks];
}

function updateHlgtChecksNxt(verticesNxt, hlghtVtx, hlgtChecksNxt) {
    let idx = verticesNxt.indexOf(hlghtVtx);
    let [x, y] = [verticesNxt[idx].x, verticesNxt[idx].y];
    // find the current vertex
    if (!hlgtChecksNxt.includes(idx)) {
        hlgtChecksNxt.push(idx);
    } else {
        hlgtChecksNxt = hlgtChecksNxt.filter((v) => v !== idx);
    }
    // find the upper neighbor
    let upperNbrY = Number.MAX_VALUE;
    let upperNbrIdx = -1;
    for (let i = 0; i < verticesNxt.length; i++) {
        if (verticesNxt[i].x == x && verticesNxt[i].y > y) {
            if (verticesNxt[i].y < upperNbrY) {
                upperNbrY = verticesNxt[i].y;
                upperNbrIdx = i;
            }
        }
    }
    if (upperNbrIdx !== -1) {
        if (!hlgtChecksNxt.includes(upperNbrIdx)) {
            hlgtChecksNxt.push(upperNbrIdx);
        } else {
            hlgtChecksNxt = hlgtChecksNxt.filter((v) => v !== upperNbrIdx);
        }
    }
    // find the right neighbor
    let rightNbrX = Number.MAX_VALUE;
    let rightNbrIdx = -1;
    for (let i = 0; i < verticesNxt.length; i++) {
        if (verticesNxt[i].y == y && verticesNxt[i].x > x) {
            if (verticesNxt[i].x < rightNbrX) {
                rightNbrX = verticesNxt[i].x;
                rightNbrIdx = i;
            }
        }
    }
    if (rightNbrIdx !== -1) {
        if (!hlgtChecksNxt.includes(rightNbrIdx)) {
            hlgtChecksNxt.push(rightNbrIdx);
        } else {
            hlgtChecksNxt = hlgtChecksNxt.filter((v) => v !== rightNbrIdx);
        }
    }
    // find the upper-right neighbor
    if (upperNbrIdx != -1 && rightNbrIdx != -1) {
        for (let i = 0; i < verticesNxt.length; i++) {
            if (verticesNxt[i].x == rightNbrX && verticesNxt[i].y == upperNbrY) {
                if (!hlgtChecksNxt.includes(i)) {
                    hlgtChecksNxt.push(i);
                } else {
                    hlgtChecksNxt = hlgtChecksNxt.filter((v) => v !== i);
                }
            }
        }
    }
    return hlgtChecksNxt;
}

function updateHlgtBitsNxt(verticesNxt, hlghtVtx, hlgtBitsNxt) {
    let idx = verticesNxt.indexOf(hlghtVtx);
    if (!hlgtBitsNxt.includes(idx)) {
        hlgtBitsNxt.push(idx);
    } else {
        hlgtBitsNxt = hlgtBitsNxt.filter((v) => v !== idx);
    }
    return hlgtBitsNxt;
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
    let faces = [new helpers.Face(initVtxs[0], initVtxs[1], initVtxs[2], initVtxs[3])];
    for (let i = 0; i < gen; i++) {
        faces = helpers.GeometryUtils.subdivide(faces);
    }
    let facesNxt = helpers.GeometryUtils.subdivide(faces);

    // initialize data for display and calculation for this generation
    let [edges, vertices, parityCheck, bitVector, checkVector, hlgtBits, hlgtChecks] = initData(faces);
    draw(edges, vertices, bitVector, [], bitCanvas);
    draw(edges, vertices, checkVector, [] ,checkCanvas, 'check');
    let [edgesNxt, verticesNxt, parityCheckNxt, bitVectorNxt, checkVectorNxt, hlgtBitsNxt, hlgtChecksNxt] = initData(facesNxt, bitCanvasNxt, checkCanvasNxt);
    draw(edgesNxt, verticesNxt, bitVectorNxt, [], bitCanvasNxt);
    draw(edgesNxt, verticesNxt, checkVectorNxt, [], checkCanvasNxt, 'check');

    // add event listeners
    let currentCanvas = null;

    chooseChecks.onclick = () => {
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
            if (!hlgtChecks.includes(idx)) {
                hlgtChecks.push(idx);
            } else {
                hlgtChecks = hlgtChecks.filter((v) => v !== idx);
            }
            hlgtChecksNxt = updateHlgtChecksNxt(verticesNxt, vertices[idx], hlgtChecksNxt);
            draw(edges, vertices, checkVector, hlgtChecks, checkCanvas, 'check');
            draw(edgesNxt, verticesNxt, checkVectorNxt, hlgtChecksNxt, checkCanvasNxt, 'check');
        }
    }

    chooseBits.onclick = () => {
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
            if (!hlgtBits.includes(idx)) {
                hlgtBits.push(idx);
            } else {
                hlgtBits = hlgtBits.filter((v) => v !== idx);
            }
            hlgtBitsNxt = updateHlgtBitsNxt(verticesNxt, vertices[idx], hlgtBitsNxt);
            draw(edges, vertices, bitVector, hlgtBits, bitCanvas);
            draw(edgesNxt, verticesNxt, bitVectorNxt, hlgtBitsNxt, bitCanvasNxt);
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
    drawDefault();
    removeEventListeners();
}