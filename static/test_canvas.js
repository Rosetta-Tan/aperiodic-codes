// Get the canvas element
var canvas = document.getElementById('canvas');

// Check if the canvas is supported by the browser
if (canvas.getContext) {
    // Get the 2D drawing context
    var ctx = canvas.getContext('2d');

    // Set font properties
    ctx.font = '20px Arial';
    ctx.fillStyle = 'black'; // Text color

    // Display text on the canvas
    ctx.fillText('Current stock price: $3.15 + 0.15', 20, 30);
} else {
    // Canvas is not supported
    console.log("Canvas is not supported in this browser.");
}