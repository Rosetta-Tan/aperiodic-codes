# Python (Flask)
from flask import Flask, render_template, send_from_directory, jsonify
from flask_cors import CORS
from src.tus.helpers import GeometryUtils, Tiling, Trig, Edge
from typing import List, Tuple
import os
import logging

app = Flask(__name__)
CORS(app, resources={r"/data/*": {"origins": "*"}})
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')

# @app.route('/assets/<path:path>')
# def serve_static(path):
#     return send_from_directory(os.path.join(app.static_folder, 'assets'), path)

@app.route('/data/', methods=['GET'])
def get_data():
    print("Fetching data...")
    logging.info("Received request for /data/")
    faces = [(0, 0, 2+1j, 2)]
    for _ in range(3):
        faces = GeometryUtils.subdivide(faces)
    vertices = Tiling.uniq_vertices(faces)
    edges = Tiling.uniq_edges(faces)
    vertices = [{"x": v.real, "y": v.imag} for v in vertices]
    edges = [{"start": {"x": e[0].real, "y": e[0].imag}, "end": {"x": e[1].real, "y": e[1].imag}} for e in edges]
    response = jsonify({'vertices': vertices, 'edges': edges})
    return response

if __name__ == "__main__":
    print('Running server...')
    app.run(debug=True, port=5000)