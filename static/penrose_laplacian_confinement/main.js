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

const config = {
    "current": {
        "bit": {
            "canvas": bitCanvas,
            "faces": initFaces,
            "edges": initEdges,
            "vertices": initVtxs,
            "hlgtVtxs": [],
            "ambientBits": new Set(),
            "stateVec": Array(initVtxs.length).fill(0)
        },
        "check": {
            "canvas": checkCanvas,
            "faces": initFaces,
            "edges": initEdges,
            "vertices": initVtxs,
            "hlgtVtxs": [],
            "ambientBits": new Set(),
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
            "ambientBits": new Set(),
            "stateVec": Array(initVtxs.length).fill(0)
        },
        "check": {
            "canvas": checkCanvasNxt,
            "faces": helpers.GeometryUtils.subdivide(initFaces),
            "edges": helpers.Tiling.uniqEdges(helpers.GeometryUtils.subdivide(initFaces)),
            "vertices": helpers.Tiling.uniqVtxs(helpers.GeometryUtils.subdivide(initFaces)),
            "hlgtVtxs": [],
            "ambientBits": new Set(),
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

    // Draw hightlighted checks
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

    // Draw ambient bits
    if (obj.ambientBits.size > 0) {
        console.log('current canvas:', obj.canvas.id);
        console.log('ambientBits:', obj.ambientBits);
        ctx.globalAlpha = 1;
        ctx.lineWidth = 2;
        for (let vertex of obj.ambientBits) {
            ctx.beginPath();
            let x = (vertex.x - minX) * scaleX + PADDING;
            let y = (vertex.y - minY) * scaleY + PADDING;
            ctx.arc(x, y, 5+1, 0, 2 * Math.PI);
            ctx.stroke();
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

/*
    * Handle the form submission
    * @return {void}
    * @effect: initialize the faces, edges, vertices, hlgtVtxs, and stateVec of the current and next canvases
*/
function handleFormSubmit() {
    let gen = parseInt(input.value, 10);
    if (isNaN(gen)) {
        alert('Please enter a valid integer number');
        return;
    } else if (gen > 5) {
        alert('Please enter a number less than or equal to 5');
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

/*
    * Set the highlighted vertices of the next check canvas based on the highlighted vertices of the current check canvas
    * @param {number} x - x coordinate of the clicked vertex
    * @param {number} y - y coordinate of the clicked vertex
    * @return {void}
    * @sideEffect: modifies the highlighted vertices of the next check canvas
    * 
    * @example
    * setHlgtChecksFromCurrent(0, 0);
*/
function setHlgtChecksFromCurrent() {
    obj = config.current.check;
    objNxt = config.next.check;

    // look for the vertices in faces
    let hlgtVtxs = obj.hlgtVtxs;
    let faces = obj.faces;
    let hlghVtxsNxt = [];
    for (let hlgtInd = 0; hlgtInd < hlgtVtxs.length; hlgtInd++) {
        let [x, y] = [hlgtVtxs[hlgtInd].x, hlgtVtxs[hlgtInd].y];
        hlghVtxsNxt.push(hlgtVtxs[hlgtInd]);

        // for every face, check if the vertex is in the face
        for (let i = 0; i < faces.length; i++) {
            let foundCtg = -1;
            let foundVtxInd = -1;

            let face = faces[i];
            let faceVtxs = [face.v1, face.v2, face.v3];
            for (let j = 0; j < faceVtxs.length; j++) {
                if (faceVtxs[j].x === x && faceVtxs[j].y === y) {
                    console.log('foundFaceInd:', i);
                    foundCtg = face.ctg;
                    foundVtxInd = j + 1;
                    break;
                }
            }
            // for every face, check if the vertex is in the face
            if (foundCtg == 1) {
                if (foundVtxInd == 2) {
                    let v1 = face.v1;
                    let v2 = face.v2;
                    let P = v1.add((v2.add(v1.scale(-1))).scale(1/helpers.goldenRatio));
                    hlghVtxsNxt.push(P);
                }
            } else if (foundCtg == 2) {
                if (foundVtxInd == 1) {
                    let v1 = face.v1;
                    let v2 = face.v2;
                    let Q = v2.add((v1.add(v2.scale(-1))).scale(1/helpers.goldenRatio));
                    hlghVtxsNxt.push(Q);
                } else {
                    let v2 = face.v2;
                    let v3 = face.v3;
                    let R = v2.add((v3.add(v2.scale(-1))).scale(1/helpers.goldenRatio));
                    hlghVtxsNxt.push(R);
                }
            }
        } 
    }

    objNxt.hlgtVtxs = hlghVtxsNxt;
}

function setAmbientBitsFromChecks() {
    obj = config.current.check;
    objNxt = config.next.check;

    let ambientBits = new Set();
    let ambientBitsNxt = new Set();

    // look for the vertices in faces
    let hlgtVtxs = obj.hlgtVtxs;
    let faces = obj.faces;
    for (let hlgtInd = 0; hlgtInd < hlgtVtxs.length; hlgtInd++) {
        let [x, y] = [hlgtVtxs[hlgtInd].x, hlgtVtxs[hlgtInd].y];
        ambientBits.add(hlgtVtxs[hlgtInd]);

        // for every face, check if the vertex is in the face
        for (let i = 0; i < faces.length; i++) {
            let foundCtg = -1;
            let foundVtxInd = -1;

            let face = faces[i];
            let faceVtxs = [face.v1, face.v2, face.v3];
            for (let j = 0; j < faceVtxs.length; j++) {
                if (faceVtxs[j].x === x && faceVtxs[j].y === y) {
                    console.log('foundFaceInd:', i);
                    foundCtg = face.ctg;
                    foundVtxInd = j + 1;
                    break;
                }
            }
            if (foundVtxInd != -1) {
                for (let j = 0; j < faceVtxs.length; j++) {
                    if (
                        (j+1) != foundVtxInd
                    ) {
                        ambientBits.add(faceVtxs[j]);
                    }
                }
            }
        }
    }

    // look for the vertices in faces in the next canvas
    let facesNxt = objNxt.faces;
    for (let hlgtInd = 0; hlgtInd < hlgtVtxs.length; hlgtInd++) {
        let [x, y] = [hlgtVtxs[hlgtInd].x, hlgtVtxs[hlgtInd].y];
        ambientBitsNxt.add(hlgtVtxs[hlgtInd]);

        // for every face, check if the vertex is in the face
        for (let i = 0; i < facesNxt.length; i++) {
            let foundCtg = -1;
            let foundVtxInd = -1;

            let face = facesNxt[i];
            let faceVtxs = [face.v1, face.v2, face.v3];
            for (let j = 0; j < faceVtxs.length; j++) {
                if (faceVtxs[j].x === x && faceVtxs[j].y === y) {
                    console.log('foundFaceInd:', i);
                    foundCtg = face.ctg;
                    foundVtxInd = j + 1;
                    break;
                }
            }
            if (foundVtxInd != -1) {
                for (let j = 0; j < faceVtxs.length; j++) {
                    if (
                        (j+1) != foundVtxInd
                    ) {
                        ambientBitsNxt.add(faceVtxs[j]);
                    }
                }
            }
        }
    }
    
    config.current.bit.ambientBits = ambientBits;
    config.next.bit.ambientBits = ambientBitsNxt;
}

function handleCanvasClick(event, obj) {
    console.log('getCoords');
    let vertices = obj.vertices;
    let [x, y] = [event.offsetX, event.offsetY];
    let [minX, minY, scaleX, scaleY] = getCanvasScale(bitCanvas, vertices);
    x = (x - PADDING) / scaleX + minX;
    y = (y - PADDING) / scaleY + minY;
    [x, y] = nearestVtxCoords(vertices, x, y);
    // look for the vertex in hlgtVtxs
    let hlgtVtxs = obj.hlgtVtxs;
    console.log('type of hlgtVtxs:', Object.prototype.toString.call(hlgtVtxs));
    let foundInd = -1;
    for (let i = 0; i < hlgtVtxs.length; i++) {
        if (hlgtVtxs[i].x === x && hlgtVtxs[i].y === y) {
            foundInd = i;
            break;
        }
    }
    if (foundInd == -1) {  // if the vertex is not already highlighted
        hlgtVtxs.push(new helpers.Vtx(x, y));
    } else {
        hlgtVtxs.splice(foundInd, 1);
    }
    console.log('hlgtVtxs:', hlgtVtxs);
    obj.hlgtVtxs = hlgtVtxs;
}

function resetCanvas(obj) {
    obj.hlgtVtxs = [];
    obj.stateVec = Array(obj.vertices.length).fill(0);
    draw(obj);
}

window.onload = () => {
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
    setHlgtChecksFromCurrent();
    setAmbientBitsFromChecks();
    draw(config.current.check);
    draw(config.current.bit);
    draw(config.next.check);
    draw(config.next.bit);
}

bitCanvas.onclick = (event) => {
    console.log('bitCanvas clicked');
    handleCanvasClick(event, config.current.bit);
    // draw(config.current.bit);
    // draw(config.next.bit);
}