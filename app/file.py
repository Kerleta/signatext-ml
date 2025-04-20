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
import tempfile

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
    print(f"❌ Model not found at {MODEL_PATH}")
    sys.exit(1)

DEVICE = os.getenv("YOLO_DEVICE", "cpu")
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=False)
print(f"✅ Loaded model with {len(model.names)} classes: {model.names}")
model.to(DEVICE)
model.conf = 0.1  # Set default confidence threshold
names = model.names

# ——————————————————————————————
# Deteksi + Anotasi
# ——————————————————————————————
def detect_and_annotate(frame: np.ndarray, target_w=320):
    h, w = frame.shape[:2]
    new_h = int(h * target_w / w)
    small = cv2.resize(frame, (target_w, new_h))

    results = model(small)

    dets = results.xyxy[0].cpu().numpy()
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

        cv2.rectangle(frame, (x1o, y1o), (x2o, y2o), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1o, y1o - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

