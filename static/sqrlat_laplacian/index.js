import * as helpers from './helpers.js';
const PADDING = 20,
    form = document.getElementById('genForm'),
    input = document.getElementById('genInput'),
    reset = document.getElementById('resetButton'),
    initVtxs = [new helpers.Vtx(0, 0), new helpers.Vtx(1, 0), new helpers.Vtx(1, 1), new helpers.Vtx(0, 1)],
    bitCanvas = setCanvasStyle(document.getElementById('bitCanvas')),
    checkCanvas = setCanvasStyle(document.getElementById('checkCanvas'));


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

function drawDefault() {
    let initFace = new helpers.Face(initVtxs[0], initVtxs[1], initVtxs[2], initVtxs[3]);
    let initEdges = helpers.GeometryUtils.getFaceEdges(initFace);
    let stateVec = Array(initVtxs.length).fill(0);
    draw(initEdges, initVtxs, stateVec, bitCanvas);
    draw(initEdges, initVtxs, stateVec, checkCanvas, 'check');
}

function draw(edges, vertices, stateVec, canvas, type = 'bit') {
    let ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // Calculate edges
    console.log('edges')
    console.log(edges);
    console.log('vertices')
    console.log(vertices);
    // Calculate bounds
    let [minX, minY, scaleX, scaleY] = getCanvasScale(canvas, vertices);
    
    // Draw vertices
    if (type === 'bit') {
        ctx.fillStyle = 'blue';
        for (let i = 0; i < vertices.length; i++) {
            if (stateVec[i] === 0) {
                // set alpha of this filling to 0.5
                ctx.globalAlpha = 0.2;
            }
            else {
                // set alpha of this filling to 1
                ctx.globalAlpha = 1;
            }
            ctx.beginPath();
            var x = (vertices[i].x - minX) * scaleX + PADDING;
            var y = (vertices[i].y - minY) * scaleY + PADDING;
            ctx.arc(x, y, 5, 0, 2 * Math.PI);
            ctx.fill();
        }
    } else {
        ctx.fillStyle = 'green';
        for (let i = 0; i < vertices.length; i++) {
            if (stateVec[i] === 0) {
                // set alpha of this filling to 0.5
                ctx.globalAlpha = 0.2;
            }
            else {
                // set alpha of this filling to 1
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

form.addEventListener('submit', function(event) {
    event.preventDefault();

    let gen = parseInt(input.value, 10);
    if (isNaN(gen)) {
        alert('Please enter a valid integer number');
        return;
    } else if (gen > 5) {
        alert('Please enter a number less than or equal to 5');
        return;
    }
    let faces = [new helpers.Face(initVtxs[0], initVtxs[1], initVtxs[2], initVtxs[3])];
    for (let i = 0; i < gen; i++) {
        faces = helpers.GeometryUtils.subdivide(faces);
    }

    // initialize data for display and calculation for this generation
    let edges = helpers.Tiling.uniqEdges(faces);
    let vertices = helpers.Tiling.uniqVtxs(faces);
    let parityCheck = helpers.Tiling.parityCheck(edges, vertices);
    let bitVector = Array(vertices.length).fill(0);
    let checkVector = Array(vertices.length).fill(0);
    draw(edges, vertices, bitVector, bitCanvas);
    draw(edges, vertices, checkVector, checkCanvas, 'check');

    // set the clicked vertex to 1 in the bit and check vectors
    bitCanvas.onclick = function(event) {
        let x = event.offsetX;
        let y = event.offsetY;
        let [minX, minY, scaleX, scaleY] = getCanvasScale(bitCanvas, vertices);
        x = (x - PADDING) / scaleX + minX;
        y = (y - PADDING) / scaleY + minY;
        console.log('click at (' + x + ', ' + y + ')');
        
        let nearestIdx = nearestVtxIdx(vertices, x, y);  // query the nearest vertex of the clicked point
        bitVector[nearestIdx] = 1 - bitVector[nearestIdx];  // flip the bit at the nearest vertex
        // update the check vector
        checkVector = Array(vertices.length).fill(0);
        for (let checkIdx = 0; checkIdx < vertices.length; checkIdx++) {
            for (let bitIdx = 0; bitIdx < vertices.length; bitIdx++) {
                checkVector[checkIdx] ^= (parityCheck[checkIdx][bitIdx] * bitVector[bitIdx]);
            }
        }

        draw(edges, vertices, bitVector, bitCanvas);
        draw(edges, vertices, checkVector, checkCanvas, 'check');
    }

    // reset the bit and check vectors
    reset.onclick = function() {
        bitVector = Array(vertices.length).fill(0);
        checkVector = Array(vertices.length).fill(0);
        draw(edges, vertices, bitVector, bitCanvas);
        draw(edges, vertices, checkVector, checkCanvas, 'check');
    }
}
);

window.onload = drawDefault