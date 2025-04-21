import os
import sys
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import cv2
import numpy as np
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('app.file')

# ——————————————————————————————
# Inisialisasi Flask & CORS
# ——————————————————————————————
app = Flask(__name__)
CORS(app)

# ——————————————————————————————
# Load YOLOv5 model
# ——————————————————————————————
ROOT_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(ROOT_DIR, "bisindo_best.pt")
if not os.path.isfile(MODEL_PATH):
    logger.error(f"❌ Model not found at {MODEL_PATH}")
    sys.exit(1)

DEVICE = os.getenv("YOLO_DEVICE", "cpu")
logger.info(f"Loading YOLOv5 model from {MODEL_PATH} on {DEVICE}...")

try:
    # Load YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=False)
    model.to(DEVICE)

    # Precision handling
    if DEVICE == 'cuda':
        model.half()  # Gunakan half precision jika GPU
    else:
        model.float()  # Paksa ke float32 jika CPU

    # Threshold dan setting lain
    model.conf = float(os.getenv("YOLO_CONF", "0.1"))  # Lower threshold
    model.iou = float(os.getenv("YOLO_IOU", "0.45"))   # NMS IOU threshold
    model.max_det = int(os.getenv("YOLO_MAX_DET", "100"))  # Maximum detections

    names = model.names
    logger.info(f"✅ Loaded model with {len(model.names)} classes: {model.names}")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    sys.exit(1)

# ——————————————————————————————
# Memory-optimized detection
# ——————————————————————————————
def detect_objects(image_url):
    """Memory-optimized object detection from URL"""
    try:
        with requests.get(image_url, stream=True, timeout=10) as r:
            r.raise_for_status()
            image_array = np.asarray(bytearray(r.content), dtype=np.uint8)
            frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if frame is None:
            logger.error("Failed to decode image")
            return {"error": "Failed to decode image"}, 400

        h, w = frame.shape[:2]
        max_dim = 640
        if h > max_dim or w > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h))
            logger.info(f"Resized image from {w}x{h} to {new_w}x{new_h}")

        try:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            lab = cv2.merge((cl, a, b))
            frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        except Exception as e:
            logger.warning(f"Contrast enhancement failed: {str(e)}")

        # Run inference dengan handling tipe precision
        with torch.no_grad():
            if DEVICE == 'cuda':
                frame_tensor = torch.from_numpy(frame).to(DEVICE).half()
            else:
                frame_tensor = torch.from_numpy(frame).to(DEVICE).float()

            results = model(frame)

        dets = results.xyxy[0].cpu().numpy()
        logger.info(f"Found {len(dets)} raw detections")

        predictions = []
        for x1, y1, x2, y2, conf, cls in dets:
            if conf < model.conf:
                continue
            label = names[int(cls)]
            predictions.append({
                "label": label,
                "confidence": float(conf),
                "bbox": [int(x1), int(y1), int(x2), int(y2)]
            })

        return {
            "success": True,
            "predictions": predictions,
            "count": len(predictions)
        }

    except Exception as e:
        logger.error(f"Detection error: {str(e)}")
        return {"error": str(e)}, 500

# ——————————————————————————————
# Routes
# ——————————————————————————————
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "running",
        "model": "YOLOv5",
        "classes": list(model.names.values()),
        "device": DEVICE,
        "confidence_threshold": model.conf
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        start_time = time.time()
        logger.info(f"Received request: {request.method} {request.path}")

        if request.is_json:
            data = request.get_json()
            if not data or 'url' not in data:
                return jsonify({"error": "URL parameter required in JSON body"}), 400
            image_url = data['url']
        elif request.form:
            if 'url' not in request.form:
                return jsonify({"error": "URL parameter required in form data"}), 400
            image_url = request.form['url']
        else:
            return jsonify({"error": "Please send URL parameter as JSON or form data"}), 400

        logger.info(f"Processing image from URL: {image_url}")
        result = detect_objects(image_url)

        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
            return jsonify(result[0]), result[1]

        result["processing_time"] = f"{time.time() - start_time:.2f} seconds"
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/test', methods=['GET'])
def test():
    return jsonify({"status": "ok", "message": "YOLOv5 API is running"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    os.environ["WORKER_TIMEOUT"] = os.environ.get("WORKER_TIMEOUT", "300")
    debug_mode = os.environ.get("DEBUG", "False").lower() == "true"
    logger.info(f"Starting Flask app on port {port}, debug={debug_mode}")
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
