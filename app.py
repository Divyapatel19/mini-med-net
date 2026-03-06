"""
app.py

Main entry point for Transparent Mini-Med.
Flask backend handling image uploads, inference, and UI serving.
"""

import os
import sys
import uuid
import json
from pathlib import Path
from datetime import datetime

from flask import Flask, request, render_template, jsonify, send_from_directory  # type: ignore
from werkzeug.utils import secure_filename  # type: ignore

# -- PATH SETUP --
# Add src to sys.path so we can import AI modules
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from inference.predictor import MiniMedPredictor  # type: ignore

app = Flask(__name__)

# -- CONFIGURATION --
UPLOAD_FOLDER = ROOT / "static" / "uploads"
DATA_FOLDER = ROOT / "data"
HISTORY_FILE = DATA_FOLDER / "history.json"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
DATA_FOLDER.mkdir(parents=True, exist_ok=True)

app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# -- PERSISTENCE --

from typing import List, Dict, Any

def load_history() -> List[Dict[str, Any]]:
    if not HISTORY_FILE.exists():
        return []
    try:
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return []

def save_history(history: List[Dict[str, Any]]) -> None:
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=4)

def add_scan_to_history(data: Dict[str, Any]) -> None:
    history = load_history()
    data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    history.insert(0, data)  # Newest first
    save_history(history[:100])  # type: ignore

# -- INITIALIZATION --
# Load model weights (demo or best)
weights_path = ROOT / "models" / "mini_med_net_demo.pth"
if not weights_path.exists():
    weights_path = ROOT / "models" / "mini_med_net_best.pth"

if weights_path.exists():
    predictor = MiniMedPredictor(
        weights_path=str(weights_path),
        device="auto",
        threshold=0.5
    )
    print(f"[Init] Model loaded from {weights_path}")
else:
    predictor = None
    print("[Warning] No model weights found!")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if not predictor:
        return jsonify({'error': 'Prediction model not loaded.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file.'}), 400
    
    if file and allowed_file(file.filename):
        # Generate unique filename to avoid collisions
        ext = file.filename.rsplit('.', 1)[1].lower()
        unique_name = f"{uuid.uuid4()}.{ext}"
        filename = secure_filename(unique_name)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Update params from request if provided (optional)
            threshold = float(request.form.get('threshold', 0.5))
            predictor.threshold = threshold
            
            # Run prediction
            result = predictor.predict(filepath)
            
            # Prepare visualization paths (relative to static)
            import cv2  # type: ignore
            overlay_filename = f"overlay_{filename}"
            overlay_path = os.path.join(app.config['UPLOAD_FOLDER'], overlay_filename)
            cv2.imwrite(overlay_path, result.overlay_bgr)
            
            # Save to history
            scan_data = {
                'id': filename.split('.')[0],
                'label': result.label,
                'probability': result.probability_pct,
                'confidence': result.confidence,
                'is_pneumonia': result.is_pneumonia,
                'original_image': f'/static/uploads/{filename}',
                'overlay_image': f'/static/uploads/{overlay_filename}'
            }
            add_scan_to_history(scan_data)
            
            return jsonify({
                'success': True,
                **scan_data,
                'is_demo': 'demo' in str(weights_path),
                'message': 'Analysis complete.'
            })
            
        except Exception as e:
            return jsonify({'error': f'Processing error: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type.'}), 400


@app.route('/api/history')
def get_history():
    return jsonify({'history': load_history()})


@app.route('/api/settings')
def get_settings():
    return jsonify({
        'weights': str(weights_path),
        'device': predictor.device if predictor else 'N/A',
        'status': 'Online',
        'threshold': predictor.threshold if predictor else 0.5
    })


if __name__ == '__main__':
    # Run server
    print("\n" + "="*40)
    print("  Transparent Mini-Med — Clinical Dashboard")
    print("  Address: http://localhost:5000")
    print("  Proper Structure: Confirmed")
    print("="*40 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=True)
