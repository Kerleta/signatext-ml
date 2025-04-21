import os
import sys
import time
import pathlib
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import requests
import torch

# Fix untuk Windows loading model
if sys.platform == "win32":
    pathlib.PosixPath = pathlib.WindowsPath

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('railway.app')

# Inisialisasi Flask
app = Flask(__name__)
CORS(app)

# Path model
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")
MODEL_PATH = os.path.join(ROOT_DIR, "model", "bisindo_best.pt")  # Pastikan struktur folder benar

# Validasi model
if not os.path.isfile(MODEL_PATH):
    logger.critical(f"Model tidak ditemukan di {MODEL_PATH}")
    sys.exit(1)

# Load model
try:
    from models.common import DetectMultiBackend
    from utils.general import non_max_suppression, scale_coords, letterbox
    from utils.torch_utils import select_device
    
    # Device configuration
    DEVICE = select_device('cpu')  # Railway biasanya tidak support GPU
    logger.info(f"Memuat model dari {MODEL_PATH} pada device {DEVICE}...")
    
    model = DetectMultiBackend(MODEL_PATH, device=DEVICE, dnn=False)
    model.eval()
    
    stride = model.stride
    names = model.names
    conf_thres = 0.5  # Threshold confidence lebih tinggi
    iou_thres = 0.45
    max_det = 100
    
    logger.info(f"✅ Model berhasil dimuat. Kelas: {names}")
except Exception as e:
    logger.error(f"Gagal memuat model: {str(e)}")
    sys.exit(1)

def detect_objects(image_url):
    try:
        # Download gambar
        logger.info(f"Mengunduh gambar dari {image_url}")
        response = requests.get(image_url, timeout=15)
        response.raise_for_status()
        
        # Decode gambar
        img_array = np.frombuffer(response.content, dtype=np.uint8)
        img0 = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img0 is None:
            logger.error("Gagal mendecode gambar")
            return {"error": "Invalid image"}, 400

        # Preprocessing
        img = letterbox(img0, 640, stride=stride, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(DEVICE).float() / 255.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        # Inference
        with torch.no_grad():
            pred = model(img)
            pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=max_det)

        # Postprocessing
        detections = []
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    label = names[int(cls)]
                    detections.append({
                        "label": label,
                        "confidence": float(conf),
                        "bbox": [int(x) for x in xyxy]
                    })

        # Ambil deteksi dengan confidence tertinggi
        if detections:
            best_detection = max(detections, key=lambda x: x['confidence'])
            return {
                "success": True,
                "predictions": [best_detection],
                "count": 1
            }
        return {"success": True, "predictions": [], "count": 0}

    except Exception as e:
        logger.error(f"Error deteksi: {str(e)}", exc_info=True)
        return {"error": str(e)}, 500

# Routes
@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    try:
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({"error": "Parameter 'url' diperlukan"}), 400
        
        result = detect_objects(data['url'])
        result['processing_time'] = f"{time.time() - start_time:.2f}s"
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error request: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    return jsonify({
        "status": "running",
        "model": "YOLOv5",
        "classes": list(names.values()),
        "conf_threshold": conf_thres
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
