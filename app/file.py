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
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=False)
model.to(DEVICE)
names = model.names

# ——————————————————————————————
# Utility: deteksi dan gambar kotak
# ——————————————————————————————
def detect_and_annotate(frame: np.ndarray):
    # deteksi
    results = model(frame)
    dets = results.xyxy[0].cpu().numpy()  # [x1,y1,x2,y2,conf,cls] per det
    output = []
    for x1,y1,x2,y2,conf,cls in dets:
        x1,y1,x2,y2 = map(int, (x1,y1,x2,y2))
        label = names[int(cls)]
        output.append({"label": label, "confidence": float(conf), "bbox": [x1,y1,x2,y2]})
        # gambar kotak & label
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}",
                    (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    return output, frame

# ——————————————————————————————
# Routes
# ——————————————————————————————
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Signatext ML API is running ✅"})

@app.route("/predict", methods=["POST"])
def predict():
    # — file upload
    if "file" in request.files:
        data = request.files["file"].read()
        arr  = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Invalid image data"}), 400

    # — URL upload
    elif (url := request.form.get("url")):
        resp = requests.get(url, timeout=5)
        if "image" not in resp.headers.get("Content-Type",""):
            return jsonify({"error": "Unsupported content type"}), 400
        arr   = np.frombuffer(resp.content, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    else:
        return jsonify({"error": "No file or url provided"}), 400

    # deteksi + annotate
    detections, annotated = detect_and_annotate(frame)

    # encode annotated image ke JPEG
    success, jpg = cv2.imencode(".jpg", annotated)
    if not success:
        return jsonify({"error": "Failed to encode image"}), 500
    annotated_bytes = jpg.tobytes()

    # upload annotated image ke Cloudinary
    res = cloudinary.uploader.upload(
        BytesIO(annotated_bytes),
        folder="signatext",               # opsional folder
        resource_type="image"
    )

    return jsonify({
        "detections": detections,
        "media_url": res["secure_url"]
    })

@app.route("/predict-video", methods=["POST"])
def predict_video():
    file = request.files.get("file")
    if not file or not file.mimetype.startswith("video/"):
        return jsonify({"error": "No video file provided"}), 400

    # simpan sementara
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1])
    file.save(tmp.name)

    cap = cv2.VideoCapture(tmp.name)
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        dets, _ = detect_and_annotate(frame)
        frames.append({"frame": idx, "detections": dets})
        idx += 1
    cap.release()

    return jsonify({"frames": frames})

# ——————————————————————————————
# Run
# ——————————————————————————————
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
