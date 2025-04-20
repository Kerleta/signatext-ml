import os
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import cv2
import numpy as np
from io import BytesIO
import requests
import cloudinary
import cloudinary.uploader

# ——————————————————————————————
# Inisialisasi Flask & CORS
# ——————————————————————————————
app = Flask(__name__)
CORS(app)

# ——————————————————————————————
# Setup Cloudinary (via ENV)
# ——————————————————————————————
cloudinary.config(
    cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME", ""),
    api_key    = os.getenv("CLOUDINARY_API_KEY", ""),
    api_secret = os.getenv("CLOUDINARY_API_SECRET", "")
)

# ——————————————————————————————
# Load YOLOv5 custom model
# ——————————————————————————————
ROOT_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(ROOT_DIR, "bisindo_best.pt")

if not os.path.isfile(MODEL_PATH):
    print(f"❌ Model not found at {MODEL_PATH}")
    sys.exit(1)

DEVICE = os.getenv("YOLO_DEVICE", "cpu")

# NOTE: pastikan torch.hub bisa akses internet atau
# kamu sudah bundle ultralytics/yolov5 di Docker.
model = torch.hub.load(
    'ultralytics/yolov5', 'custom',
    path=MODEL_PATH,
    force_reload=False  # False agar tidak reload tiap start
)
model.to(DEVICE)
names = model.names

# ——————————————————————————————
# Utility: deteksi satu frame OpenCV
# ——————————————————————————————
def detect_frame(frame: np.ndarray):
    results = model(frame)  # YOLOv5 AutoShape
    output = []
    for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
        x1, y1, x2, y2 = map(int, xyxy)
        output.append({
            "label": names[int(cls)],
            "confidence": float(conf),
            "bbox": [x1, y1, x2, y2]
        })
    return output

# ——————————————————————————————
# Routes
# ——————————————————————————————
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Signatext ML API is running ✅"})

@app.route("/predict", methods=["POST"])
def predict():
    # 1) File upload
    if "file" in request.files:
        data  = request.files["file"].read()
        arr   = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    # 2) URL upload
    elif url := request.form.get("url"):
        resp  = requests.get(url, timeout=5)
        ctype = resp.headers.get("Content-Type", "")
        if "image" not in ctype:
            return jsonify({"error": "Unsupported content type"}), 400
        arr   = np.frombuffer(resp.content, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    else:
        return jsonify({"error": "No file or url provided"}), 400

    # jalankan deteksi
    detections = detect_frame(frame)

    # upload ke Cloudinary
    upload_source = BytesIO(data) if "file" in request.files else url
    res = cloudinary.uploader.upload(upload_source)

    return jsonify({
        "detections": detections,
        "media_url": res.get("secure_url")
    })

# ——————————————————————————————
# Run
# ——————————————————————————————
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
