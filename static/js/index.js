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
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('genForm')
    form.action = '/data/';  // Set to AJAX submission URL
    form.method = 'POST';
    form.addEventListener('submit', handleSubmit);
});

function handleSubmit(event) {
    event.preventDefault(); // Prevent default form submission
    const number = document.getElementById('genInput').value;

    fetch('/data/', {  // Sending data to Flask
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ gen: number })  // Send number as JSON
    })
    .then(response => {
        if (response.ok) {
            return response.json();
        } else {
            throw new Error('Server response was not ok.');
        }
    })
    .then(data => {
        console.log('Success:', data);
        drawData(data);  // Redraw the canvas with new data
    })
    .catch((error) => {
        console.error('Error:', error);
        alert("Failed to fetch data: " + error.message);
    });
}

async function fetchData(gen = 0) {  // Default generation parameter set to 0
    let options = {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ gen: gen })
    };

    const response = await fetch('/data/', options);
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response.json();
}

function drawData(data) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    var vertices = data.vertices;
    var edges = data.edges;
    // Calculate bounds
    var minX = Math.min(...vertices.map(v => v.x));
    var maxX = Math.max(...vertices.map(v => v.x));
    var minY = Math.min(...vertices.map(v => v.y));
    var maxY = Math.max(...vertices.map(v => v.y));
    // Calculate scale and offset
    var scaleX = CANVAS_WIDTH / (maxX - minX);
    var scaleY = CANVAS_HEIGHT / (maxY - minY);
    var offsetX = (CANVAS_WIDTH - (maxX + minX) * scaleX) / 2;
    var offsetY = (CANVAS_HEIGHT - (maxY + minY) * scaleY) / 2;
    
    // Draw vertices
    ctx.fillStyle = 'green';
    vertices.forEach(vertex => {
        ctx.beginPath();
        var x = (vertex.x - minX) * scaleX + offsetX;
        var y = (vertex.y - minY) * scaleY + offsetY;
        ctx.arc(x, y, 5, 0, 2 * Math.PI);
        ctx.fill();
    });
    // Draw edges, use gradient color to denot different edges
    ctx.strokeStyle = 'blue';
    edges.forEach(edge => {
        ctx.beginPath();
        var startX = (edge.start.x - minX) * scaleX * 1 + offsetX;
        var startY = (edge.start.y - minY) * scaleY * 1 + offsetY;
        var endX = (edge.end.x - minX) * scaleX * 1 + offsetX;
        var endY = (edge.end.y - minY) * scaleY * 1 + offsetY;
        ctx.moveTo(startX, startY);
        ctx.lineTo(endX, endY);
        ctx.stroke();
    });
}

function main() {
    fetchData(0).
    then(data => {
        drawData(data);
    })
    .catch(error => {
        console.error('Error during initial fetch:', error);
    });
}

window.onload = main;