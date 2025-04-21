import os
import sys
import time
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
logger = logging.getLogger('app.file')

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
    model.conf = float(os.getenv("YOLO_CONF", "0.1"))  # Lowered threshold for better detection
    names = model.names
    logger.info(f"✅ Loaded model with {len(model.names)} classes: {model.names}")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    sys.exit(1)

# ——————————————————————————————
# Deteksi + Anotasi
# ——————————————————————————————
def detect_and_annotate(frame: np.ndarray, target_w=640):  # Increased default target width
    """Process an image frame and return detections"""
    try:
        original_frame = frame.copy()  # Save original frame for debug
        
        # Pre-processing to improve detection
        # 1. Apply contrast enhancement
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        lab = cv2.merge((cl, a, b))
        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # 2. Resize image for detection
        h, w = frame.shape[:2]
        new_h = int(h * target_w / w)
        small = cv2.resize(frame, (target_w, new_h))
        
        # 3. Run detection with augmentation
        results = model(small, augment=True)  # Use test-time augmentation for better results
        
        # 4. Process results
        dets = results.xyxy[0].cpu().numpy()
        logger.info(f"Raw detections (before filtering): {len(dets)}")
        
        # 5. Process and filter detections
        output = []
        all_detections = []
        scale_x = w / target_w
        scale_y = h / new_h
        
        for x1, y1, x2, y2, conf, cls in dets:
            label = names[int(cls)]
            
            # Log all detections including those below threshold
            detection_info = {
                "label": label,
                "confidence": float(conf),
                "bbox": [int(x1 * scale_x), int(y1 * scale_y), 
                         int(x2 * scale_x), int(y2 * scale_y)]
            }
            all_detections.append(detection_info)
            
            # Only include detections above threshold in output
            if conf < model.conf:
                logger.info(f"Detected {label} with confidence {conf:.4f} (below threshold {model.conf})")
                continue
                
            x1o, y1o = int(x1 * scale_x), int(y1 * scale_y)
            x2o, y2o = int(x2 * scale_x), int(y2 * scale_y)
            
            output.append({
                "label": label,
                "confidence": float(conf),
                "bbox": [x1o, y1o, x2o, y2o]
            })
            
            # Draw bounding box on original frame
            cv2.rectangle(original_frame, (x1o, y1o), (x2o, y2o), (0, 255, 0), 2)
            cv2.putText(original_frame, f"{label} {conf:.2f}", (x1o, y1o - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Save debug information
        if os.getenv("DEBUG_IMAGES", "True").lower() == "true":
            try:
                debug_dir = os.path.join(os.getcwd(), "debug_images")
                os.makedirs(debug_dir, exist_ok=True)
                
                timestamp = int(time.time())
                
                # Save original input image
                input_path = os.path.join(debug_dir, f"input_{timestamp}.jpg")
                cv2.imwrite(input_path, frame)
                
                # Save processed image with detections
                output_path = os.path.join(debug_dir, f"output_{timestamp}.jpg")
                cv2.imwrite(output_path, original_frame)
                
                logger.info(f"Debug images saved to {debug_dir}")
            except Exception as e:
                logger.error(f"Error saving debug images: {str(e)}")
        
        # Log all detections for debugging
        logger.info(f"All detections (including below threshold): {all_detections}")
        logger.info(f"Filtered detections (above threshold): {output}")
        
        return output
        
    except Exception as e:
        logger.error(f"Error in detection: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

# ——————————————————————————————
# Image processing helpers
# ——————————————————————————————
def preprocess_image(image):
    """Apply additional preprocessing to improve detection quality"""
    # Try multiple preprocessing techniques
    preprocessed_images = []
    
    # 1. Original image
    preprocessed_images.append(image.copy())
    
    # 2. Contrast enhancement
    try:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        lab = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        preprocessed_images.append(enhanced)
    except Exception as e:
        logger.error(f"Error in contrast enhancement: {str(e)}")
    
    # 3. Grayscale conversion (if model accepts gray images)
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_3c = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        preprocessed_images.append(gray_3c)
    except Exception as e:
        logger.error(f"Error in grayscale conversion: {str(e)}")
    
    return preprocessed_images

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

@app.route('/health', methods=['GET'])
def health():
    """Detailed health check"""
    import psutil
    return jsonify({
        "status": "ok",
        "memory_usage": {
            "percent": psutil.virtual_memory().percent,
            "available_mb": psutil.virtual_memory().available / 1024 / 1024
        },
        "cpu_usage": psutil.cpu_percent(),
        "model": {
            "name": "YOLOv5",
            "classes": len(model.names),
            "confidence": model.conf,
            "device": DEVICE
        }
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
        
        # Save original image for debugging
        if os.getenv("DEBUG_IMAGES", "True").lower() == "true":
            try:
                debug_dir = os.path.join(os.getcwd(), "debug_images")
                os.makedirs(debug_dir, exist_ok=True)
                timestamp = int(time.time())
                original_path = os.path.join(debug_dir, f"original_{timestamp}.jpg")
                cv2.imwrite(original_path, frame)
                logger.info(f"Original image saved to {original_path}")
            except Exception as e:
                logger.error(f"Error saving original image: {str(e)}")
        
        # Try multiple preprocessing techniques for better detection
        best_detections = []
        preprocessed_frames = preprocess_image(frame)
        
        for i, processed_frame in enumerate(preprocessed_frames):
            logger.info(f"Running detection on preprocessed frame {i}")
            
            # Try different target widths for resize
            for target_w in [640, 1280]:  # Try standard and higher resolution
                logger.info(f"Detecting with target width {target_w}")
                detections = detect_and_annotate(processed_frame.copy(), target_w=target_w)
                
                # Keep the best result (most detections or highest confidence)
                if len(detections) > len(best_detections):
                    best_detections = detections
                elif len(detections) == len(best_detections) and len(detections) > 0:
                    # If same number of detections, keep the one with highest average confidence
                    curr_avg_conf = sum(d["confidence"] for d in detections) / len(detections)
                    best_avg_conf = sum(d["confidence"] for d in best_detections) / len(best_detections)
                    if curr_avg_conf > best_avg_conf:
                        best_detections = detections
        
        logger.info(f"Detection complete. Found {len(best_detections)} objects.")
        
        # Return detection results
        return jsonify({
            "success": True,
            "predictions": best_detections,
            "count": len(best_detections)
        })
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/test', methods=['GET'])
def test():
    """Simple test endpoint"""
    return jsonify({
        "message": "YOLOv5 API is running correctly",
        "timestamp": time.time()
    })

# ——————————————————————————————
# Run the app
# ——————————————————————————————
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("DEBUG", "False").lower() == "true"
    logger.info(f"Starting Flask app on port {port}, debug={debug_mode}")
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
