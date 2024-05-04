import * as helpers from './helpers.js';
let canvas = document.getElementById('myCanvas');
let ctx = canvas.getContext('2d');
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

function draw() {
    ctx.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
    ctx.beginPath();
    ctx.strokeStyle = 'black';
    ctx.moveTo(0, CANVAS_HEIGHT);
    ctx.lineTo(CANVAS_WIDTH, CANVAS_HEIGHT);
    ctx.lineTo(CANVAS_WIDTH, 0);
    ctx.lineTo(0, CANVAS_HEIGHT);
    ctx.closePath();
    ctx.stroke();
}

form.addEventListener('submit', function(event) {
    event.preventDefault();
    let num = parseInt(input.value, 10);
    if (isNaN(num)) {
        alert('Please enter a valid integer number');
        return;
    }
    let trigs = [new helpers.Trig(new helpers.Vtx(0, 0), new helpers.Vtx(2, 1), new helpers.Vtx(2, 0))];
    
    ctx.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
    ctx.beginPath();
    ctx.strokeStyle = 'black';
    ctx.moveTo(0, CANVAS_HEIGHT/2);
    ctx.lineTo(CANVAS_WIDTH/2, CANVAS_HEIGHT/2);
    ctx.lineTo(CANVAS_WIDTH/2, 0);
    ctx.lineTo(0, CANVAS_HEIGHT/2);
    ctx.closePath();
    ctx.stroke();
}
);

window.onload = draw;