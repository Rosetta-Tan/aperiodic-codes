import numpy as np
import json
import os
import tempfile
import time
import uuid
import webbrowser
from typing import Dict, Optional, Union, cast


from jinja2 import Environment, PrefixLoader, FileSystemLoader, nodes
from jinja2.ext import Extension
from jinja2.utils import markupsafe
from jinja2.parser import Parser

dirname = os.path.dirname(__file__)
html_loader = FileSystemLoader(searchpath=os.path.join(dirname, "static"))
js_loader = FileSystemLoader(searchpath=os.path.join(dirname, "js"))

loader = PrefixLoader(
    {
        "html": html_loader,
        "js": js_loader,
    }
)

readdir = '/Users/yitan/Library/CloudStorage/GoogleDrive-yitan@g.harvard.edu/My Drive/from_cannon/qmemory_simulation/data/qc_code/psi_tiling/'
data = np.load(os.path.join(readdir, 'psi_tiling_gen_15.npz'))
h = data['h']  # equivalent to face_to_vertex
vertices_pos = data['vertices_pos']
faces_pos = data['faces_pos']
edges = data['edges']

num_faces, num_vertices = h.shape
nodes_vis = [{'id': i, 'label': f'Node {i}'} for i in range(num_vertices)]
edges_vis = [{'from': int(edge[0]), 'to': int(edge[1])} for edge in edges]
# edges_vis = [{'from': 1, 'to': 2}, {'from': 2, 'to': 3}]

jinja_env = Environment(loader=loader)

uid = uuid.uuid4()
html_template = jinja_env.get_template("html/psi_quad_tiling.html")
html = html_template.render(
    nodes=nodes_vis,
    edges=edges_vis,
    uid=uid,
)
# view the html file in a browser
fp = tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False, dir=os.getcwd())
try:
    fp.write(html)
    fp.close()

    webbrowser.open("file://" + os.path.realpath(fp.name), new=2)

    # give browser enough time to open before deleting file
    time.sleep(5)
finally:
    os.remove(fp.name)