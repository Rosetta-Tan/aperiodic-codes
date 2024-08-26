import base64
import numpy as np
from pygltflib import *
import logging
logging.basicConfig(level=logging.DEBUG)
# input: a npz file
# output: a gtlf file

def load_npz(inpath):
    data = np.load(inpath)
    proj_pts = data['proj_pts']
    hx_vv = data['hx_vv']
    hx_cc = data['hx_cc']
    hz_vv = data['hz_vv']
    hz_cc = data['hz_cc']
    return proj_pts, hx_vv, hx_cc, hz_vv, hz_cc

def check_comm_after_proj(hx_vv, hx_cc, hz_vv, hz_cc):
    '''
    Check commutation of all pairs of stabilizers.
    '''
    assert hx_vv.shape == hx_cc.shape == hz_vv.shape == hz_cc.shape
    hx = np.hstack((hx_vv, hx_cc))
    hz = np.hstack((hz_vv, hz_cc))
    return (hx @ hz.T) % 2

def npz_to_gltf(proj_pts,
                hx_vv,
                hx_cc,
                hz_vv,
                hz_cc,
                comm_mat,
                outpath='output.gltf'):
    
    # convert to float32 is necessary for buffer creation
    vertices = proj_pts.astype(np.float32).flatten()

    line_vertices = []
    line_vertices_red = []
    for i in range(hx_cc.shape[0]):
        for j in range(hx_cc.shape[1]):
            if hx_cc[i, j] != 0:
                line_vertices.extend(proj_pts[i])
                line_vertices.extend(proj_pts[j])

    for i in range(hz_cc.shape[0]):
        for j in range(hz_cc.shape[1]):
            if hz_cc[i, j] != 0:
                line_vertices.extend(proj_pts[i])
                line_vertices.extend(proj_pts[j])

    for i in range(comm_mat.shape[0]):
        for j in range(comm_mat.shape[1]):
            if comm_mat[i, j] != 0:
                line_vertices_red.extend(proj_pts[i])
                line_vertices_red.extend(proj_pts[j])

    line_vertices = np.array(line_vertices, dtype=np.float32).flatten()
    line_vertices_red = np.array(line_vertices_red, dtype=np.float32).flatten()

    buffer_data = vertices.tobytes()
    buffer_data_lines = line_vertices.tobytes()
    buffer_data_lines_red = line_vertices_red.tobytes() 

    buffer_base64 = base64.b64encode(buffer_data).decode('utf-8')
    buffer_base64_lines = base64.b64encode(buffer_data_lines).decode('utf-8')
    buffer_base64_lines_red = base64.b64encode(buffer_data_lines_red).decode('utf-8')

    buffer = Buffer(uri="data:application/octet-stream;base64," + buffer_base64, byteLength=len(buffer_data))
    buffer_lines = Buffer(uri="data:application/octet-stream;base64," + buffer_base64_lines)
    buffer_lines_red = Buffer(uri="data:application/octet-stream;base64," + buffer_base64_lines_red)

    buffer_view = BufferView(buffer=0, byteOffset=0, byteLength=len(buffer_data))
    buffer_view_lines = BufferView(buffer=1, byteOffset=0, byteLength=len(buffer_data_lines))
    buffer_view_lines_red = BufferView(buffer=2, byteOffset=0, byteLength=len(buffer_data_lines_red))

    accessor = Accessor(bufferView=0, byteOffset=0, componentType=5126, count=proj_pts.shape[0], type="VEC3")
    accessor_lines = Accessor(bufferView=1, byteOffset=0, componentType=5126, count=len(line_vertices) // 3, type="VEC3")
    accessor_lines_red = Accessor(bufferView=2, byteOffset=0, componentType=5126, count=len(line_vertices_red) // 3, type="VEC3")

    material_red = Material(pbrMetallicRoughness=PbrMetallicRoughness(baseColorFactor=[1, 0, 0, 1]))  # Red

    mesh = Mesh(primitives=[Primitive(attributes=Attributes(POSITION=0), mode=0)])
    mesh_lines = Mesh(primitives=[Primitive(attributes=Attributes(POSITION=1), mode=1)])
    mesh_lines_red = Mesh(primitives=[Primitive(attributes=Attributes(POSITION=2), mode=1, material=0)])

    # Create node
    node = Node(mesh=0)
    node_lines = Node(mesh=1)
    node_lines_red = Node(mesh=2)

    # Create scene
    scene = Scene(nodes=[0, 1, 2])

    # Create GLTF asset
    gltf = GLTF2(
        asset=Asset(),
        scenes=[scene],
        nodes=[node, node_lines, node_lines_red],
        meshes=[mesh, mesh_lines, mesh_lines_red],
        materials=[material_red],
        buffers=[buffer, buffer_lines, buffer_lines_red],
        bufferViews=[buffer_view, buffer_view_lines, buffer_view_lines_red],
        accessors=[accessor, accessor_lines, accessor_lines_red]
    )

    # Save GLTF to file
    gltf.save(outpath)

    return gltf

def create_example_gltf(filepath):
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
        <script src="https://cdn.babylonjs.com/loaders/babylonjs.loaders.min.js"></script>
        <style>
            canvas {{
                width: 65%;
                height: 65%;
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

if __name__ == "__main__":
    inpath = "../data/apc/6d_to_3d/294549_opt.npz"
    proj_pts, hx_vv, hx_cc, hz_vv, hz_cc = load_npz(inpath)
    comm_after_proj = check_comm_after_proj(hx_vv, hx_cc, hz_vv, hz_cc)

    gltf = npz_to_gltf(proj_pts, hx_vv, hx_cc, hz_vv, hz_cc, comm_after_proj)
    html = gltf_to_html("output.gltf")

    # gltf = create_example_gltf("point_cloud.gltf")
    # html = gltf_to_html("point_cloud.gltf")

    with open("point_cloud.html", "w") as f:
        f.write(html)