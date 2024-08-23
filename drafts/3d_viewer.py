import base64
import numpy as np
from pygltflib import GLTF2, Asset, Scene, Node, Mesh, Primitive, Attributes, Buffer, BufferView, Accessor

# input:
# output: a gtlf file

def create_gltf(filepath):
    assert filepath.endswith('.gltf')
    # Generate some random 3D points
    num_points = 1000
    points = np.random.rand(num_points, 3).astype(np.float32)

    # Flatten the points array for buffer creation
    vertices = points.flatten()

    # Create buffer data
    buffer_data = vertices.tobytes()

    # Encode buffer data to base64
    buffer_base64 = base64.b64encode(buffer_data).decode('utf-8')

    # Create buffer
    buffer = Buffer(uri="data:application/octet-stream;base64," + buffer_base64)

    # Create buffer view
    buffer_view = BufferView(buffer=0, byteOffset=0, byteLength=len(buffer_data))

    # Create accessor
    accessor = Accessor(bufferView=0, byteOffset=0, componentType=5126, count=num_points, type="VEC3")

    # Create mesh
    mesh = Mesh(primitives=[Primitive(attributes=Attributes(POSITION=0), mode=0)])

    # Create node
    node = Node(mesh=0)

    # Create scene
    scene = Scene(nodes=[0])

    # Create GLTF asset
    gltf = GLTF2(
        asset=Asset(),
        scenes=[scene],
        nodes=[node],
        meshes=[mesh],
        buffers=[buffer],
        bufferViews=[buffer_view],
        accessors=[accessor]
    )

    # Save GLTF to file
    gltf.save(filepath)

    return gltf

def gltf_to_html(gltf):
    # Generate HTML to display the GLTF file
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>3D Viewer</title>
        <script src="https://cdn.babylonjs.com/babylon.js"></script>
        <style>
            canvas {{
                width: 100%;
                height: 100%;
            }}
        </style>
    </head>
    <body>
        <canvas id="renderCanvas"></canvas>
        <script>
            var canvas = document.getElementById("renderCanvas");
            var engine = new BABYLON.Engine(canvas, true);
            var createScene = function() {{
                var scene = new BABYLON.Scene(engine);
                var camera = new BABYLON.ArcRotateCamera("Camera", Math.PI / 2, Math.PI / 2, 2, BABYLON.Vector3.Zero(), scene);
                camera.attachControl(canvas, true);
                var light = new BABYLON.HemisphericLight("light", new BABYLON.Vector3(0, 1, 0), scene);
                var gltf = BABYLON.SceneLoader.Append("", "{gltf}", scene, function() {{
                    scene.createDefaultCameraOrLight(true);
                    scene.createDefaultEnvironment();
                    scene.activeCamera.alpha += Math.PI;
                }});
                return scene;
            }};
            var scene = createScene();
            engine.runRenderLoop(function() {{
                scene.render();
            }});
            window.addEventListener("resize", function() {{
                engine.resize();
            }});
        </script>
    </body>
    </html>
    """

    return html

gltf = create_gltf("output.gltf")
html = gltf_to_html("output.gltf")
with open("point_cloud.html", "w") as f:
    f.write(html)