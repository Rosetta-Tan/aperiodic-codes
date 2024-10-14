import * as THREE from 'three';
import { WaterRefractionShader } from 'three/examples/jsm/Addons.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 0.1, 1000 );
const renderer = new THREE.WebGLRenderer();
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
const controls = new OrbitControls(camera, renderer.domElement);
camera.position.z = 12;
renderer.setSize( window.innerWidth, window.innerHeight );
renderer.setAnimationLoop( animate );
controls.enableDamping = true;
controls.dampingFactor = 0.25;
controls.enableZoom = true;
controls.enablePan = true;
document.body.appendChild( renderer.domElement );

function animate() {
	controls.update();
	renderer.render( scene, camera );
}

function parsePoints(text) {
    const lines = text.split('\n');
    const points = lines.map(line => {
        const coords = line.split(',').map(Number);
        return coords;
    });
    return points;
}

function createSpheres(pointsArray) {
    for (let i = 0; i < pointsArray.length; i++) {
        let position = pointsArray[i];
        if (position.length === 3) {
            const geometry = new THREE.SphereGeometry(0.08, 32, 32);
            let material = new THREE.MeshBasicMaterial({ color: 0xffffff }); // Initial white color
            let sphere = new THREE.Mesh(geometry, material);

            sphere.position.set(position[0], position[1], position[2]);            
            sphere.userData.isClicked = false; // Start with unclicked state

            if (logicalOp[i] == 1 || logicalOp[i+logicalOp.length/2] == 1) {
                sphere.material.color.set(0x00ff00);
                sphere.userData.isClicked = true;
            }

            scene.add(sphere);
        }
    }
}

function ptArrToIndArr() {
	for (let i = 0; i < scene.children.length; i++) {
		let point = scene.children[i];
		p2i.set(point, i);
	}
	return p2i;
}

async function loadPoints(filepath) {
    const response = await fetch(filepath);
    const text = await response.text();
    const pointsArray = parsePoints(text);
    createSpheres(pointsArray);
	p2i = ptArrToIndArr(pointsArray);
}

async function loadAdjacencyMatrix(filepath) {
	const response = await fetch(filepath);
	const text = await response.text();
	const matrix = parseMatrix(text);
	return matrix;
}

function parseMatrix(text) {
	const lines = text.split('\n');
	const matrix = lines.map(line => {
		const row = line.split(',').map(Number);
		return row;
	});
	return matrix;
}

function parseLogicalOp(text) {
    const line = text.split('\n');
    // console.log("parseLogicalOp");
    // console.log(line);
    const op = line[0].split(',').map(Number);
    return op;
}

async function loadLogicalOp(filepath) {
    const response = await fetch(filepath);
    const text = await response.text();
    const op = parseLogicalOp(text);
    return op;
}

function onMouseClick(event) {
	mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObjects(scene.children);
	let point = intersects[0].object;
	let index = p2i.get(point);
	let neighbor_inds = matrix[index];
    for (let i = 0; i < neighbor_inds.length; i++) {
        let neighbor_ind = neighbor_inds[i];
        if (neighbor_ind >= logicalOp.length/2) {
            neighbor_ind -= logicalOp.length/2;
        }
        let neighbor = scene.children[neighbor_ind];
        console.log(neighbor);
        console.log(neighbor.userData);
        neighbor.userData.isClicked = !neighbor.userData.isClicked;
        if (neighbor.userData.isClicked) {
            neighbor.material.color.set(0x00ff00);
        } else {
            neighbor.material.color.set(0xffffff);
        }
    }
}

window.addEventListener('click', onMouseClick);
const matrix = await loadAdjacencyMatrix('hz.txt');
const logicalOp = await loadLogicalOp('L_symm.txt');
let p2i = new Map();
loadPoints('proj_pts.txt');
