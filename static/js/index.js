import * as helpers from './helpers.js';
let canvas = document.getElementById('myCanvas');
const ctx = canvas.getContext('2d');
let style = canvas.currentStyle || window.getComputedStyle(canvas);
let width = style.width;
let height = style.height;
let marginTop = style.marginTop;
let marginBottom = style.marginBottom;
let CANVAS_WIDTH = parseInt(width, 10); // Parse numeric values from the properties
let CANVAS_HEIGHT = parseInt(height, 10);
let numericMarginTop = parseInt(marginTop, 10);
let numericMarginBottom = parseInt(marginBottom, 10);
[canvas.width, canvas.height] = [CANVAS_WIDTH, CANVAS_HEIGHT];

// Add a form to let user input an integer number
const form = document.getElementById('genForm');
const input = document.getElementById('genInput');

function draw_default() {
    let v1 = new helpers.Vtx(0, 0);
    let v2 = new helpers.Vtx(2, 1);
    let v3 = new helpers.Vtx(2, 0);
    let trigs = [new helpers.Trig(v1, v2, v3)];
    let edges = helpers.Tiling.uniqEdges(trigs);
    draw(edges, [v1, v2, v3]);
}

function draw(edges, vertices) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // Calculate edges
    console.log(edges);
    console.log(vertices);
    // Calculate bounds
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
    console.log(minX, maxX, minY, maxY);
    // Calculate scale and offset
    var scaleX = CANVAS_WIDTH / (maxX - minX);
    var scaleY = CANVAS_HEIGHT / (maxY - minY);
    
    // Draw vertices
    ctx.fillStyle = 'green';
    for (let i = 0; i < vertices.length; i++) {
        ctx.beginPath();
        var x = (vertices[i].x - minX) * scaleX;
        var y = (vertices[i].y - minY) * scaleY;
        ctx.arc(x, y, 5, 0, 2 * Math.PI);
        ctx.fill();
    }

    // Draw edges
    ctx.strokeStyle = 'blue';
    for (let i = 0; i < edges.length; i++) {
        let edge = edges[i];
        ctx.beginPath();
        var startX = (edge.v1.x - minX) * scaleX;
        var startY = (edge.v1.y - minY) * scaleY;
        var endX = (edge.v2.x - minX) * scaleX;
        var endY = (edge.v2.y - minY) * scaleY;
        ctx.moveTo(startX, startY);
        ctx.lineTo(endX, endY);
        ctx.stroke();
    }
}

form.addEventListener('submit', function(event) {
    event.preventDefault();
    let num = parseInt(input.value, 10);
    if (isNaN(num)) {
        alert('Please enter a valid integer number');
        return;
    }
    let trigs = [new helpers.Trig(new helpers.Vtx(0, 0), new helpers.Vtx(2, 1), new helpers.Vtx(2, 0))];
    for (let i = 0; i < num; i++) {
        trigs = helpers.GeometryUtils.subdivide(trigs);
    }
    console.log('trigs');
    console.log(trigs);
    let edges = helpers.Tiling.uniqEdges(trigs);
    let vertices = helpers.Tiling.uniqVtxs(trigs);
    draw(edges, vertices);
}
);

window.onload = draw_default;