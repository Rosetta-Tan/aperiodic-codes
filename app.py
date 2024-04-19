from flask import Flask, render_template, request, jsonify, render_template_string
from flask_cors import CORS
from src.helpers import GeometryUtils, Tiling, Face, Edge
from typing import List, Tuple
import os
import logging

app = Flask(__name__)
CORS(app, resources={r"/data/*": {"origins": "*"}})
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/data/', methods=['POST', 'GET'])
def get_data():
    print("Fetching data...")
    logging.info("Received request for /data/")
    try:
        data = request.get_json()
        gen = int(data['gen'])  # Get the generation number from JSON

        faces = [(0, 0, 2+1j, 2)]
        for _ in range(gen):
            faces = GeometryUtils.subdivide(faces)
        vertices = Tiling.uniq_vertices(faces)
        edges = Tiling.uniq_edges(faces)
        vertices = [{"x": v.real, "y": v.imag} for v in vertices]
        edges = [{"start": {"x": e[0].real, "y": e[0].imag}, "end": {"x": e[1].real, "y": e[1].imag}} for e in edges]
        
        response = jsonify({'vertices': vertices, 'edges': edges})
        return response
    
    except Exception as e:
        logging.error(f"Error: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    print('Running server...')
    app.run(debug=True, port=5000)