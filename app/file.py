
import os
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import cv2
import numpy as np
import requests
import cloudinary
import cloudinary.uploader
import tempfile
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ——————————————————————————————
# Inisialisasi Flask & CORS
# ——————————————————————————————
app = Flask(__name__)
CORS(app)

# ——————————————————————————————
# Setup Cloudinary
# ——————————————————————————————
cloudinary.config(
    cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME", ""),
    api_key    = os.getenv("CLOUDINARY_API_KEY", ""),
    api_secret = os.getenv("CLOUDINARY_API_SECRET", "")
)

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
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=False)
    model.to(DEVICE)
    model.conf = float(os.getenv("YOLO_CONF", "0.25"))  # Set confidence threshold
    names = model.names
    logger.info(f"✅ Loaded model with {len(model.names)} classes: {model.names}")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    sys.exit(1)

# ——————————————————————————————
# Deteksi + Anotasi
# ——————————————————————————————
def detect_and_annotate(frame: np.ndarray, target_w=320):
    """Process an image frame and return detections"""
    try:
        h, w = frame.shape[:2]
        new_h = int(h * target_w / w)
        small = cv2.resize(frame, (target_w, new_h))
        
        # Run detection
        results = model(small)
        dets = results.xyxy[0].cpu().numpy()
        
        # Process results
        output = []
        scale_x = w / target_w
        scale_y = h / new_h
        
        for x1, y1, x2, y2, conf, cls in dets:
            if conf < model.conf:
                continue
                
            x1o, y1o = int(x1 * scale_x), int(y1 * scale_y)
            x2o, y2o = int(x2 * scale_x), int(y2 * scale_y)
            label = names[int(cls)]
            
            output.append({
                "label": label,
                "confidence": float(conf),
                "bbox": [x1o, y1o, x2o, y2o]
            })
            
            # Draw on frame (only used if we're saving debug images)
            cv2.rectangle(frame, (x1o, y1o), (x2o, y2o), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1o, y1o - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return output
        
    except Exception as e:
        logger.error(f"Error in detection: {str(e)}")
        raise

# ——————————————————————————————
# Routes
# ——————————————————————————————
@app.route('/', methods=['GET'])
def home():
    """Basic health check endpoint"""
    return jsonify({
        "status": "running",
        "model": "YOLOv5",
        "classes": list(model.names.values()),
        "device": DEVICE,
        "confidence_threshold": model.conf
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to perform object detection on an image URL"""
    try:
        # Log request info
        logger.info(f"Received request: {request.method} {request.path}")
        
        # Check if there's a URL in the request
        if request.is_json:
            # If JSON data
            data = request.get_json()
            if not data or 'url' not in data:
                logger.warning("No URL found in JSON data")
                return jsonify({"error": "URL parameter required in JSON body"}), 400
            image_url = data['url']
        elif request.form:
            # If form data
            if 'url' not in request.form:
                logger.warning("No URL found in form data")
                return jsonify({"error": "URL parameter required in form data"}), 400
            image_url = request.form['url']
        else:
            logger.warning("No appropriate data found in request")
            return jsonify({"error": "Please send URL parameter as JSON or form data"}), 400
            
        logger.info(f"Processing image from URL: {image_url}")
        
        # Download image from URL
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()  # Raise exception for HTTP errors
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch image: {str(e)}")
            return jsonify({"error": f"Failed to fetch image: {str(e)}"}), 400
            
        # Convert to OpenCV format
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            logger.error("Failed to decode image")
            return jsonify({"error": "Failed to decode image"}), 400
        
        # Detect objects - use larger target width for better detection
        detections = detect_and_annotate(frame, target_w=640)
        logger.info(f"Detection complete. Found {len(detections)} objects.")
        
        # Return detection results
        return jsonify({
            "success": True,
            "predictions": detections,
            "count": len(detections)
        })
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500

# ——————————————————————————————
# Run the app
# ——————————————————————————————
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("DEBUG", "False").lower() == "true"
    logger.info(f"Starting Flask app on port {port}, debug={debug_mode}")
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
