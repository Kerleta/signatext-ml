import os
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import requests
import cloudinary
import cloudinary.uploader

# Inisialisasi Flask
app = Flask(__name__)
CORS(app)

# Setup Cloudinary (ambil dari env)
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME", ""),
    api_key=os.getenv("CLOUDINARY_API_KEY", ""),
    api_secret=os.getenv("CLOUDINARY_API_SECRET", "")
)

# Load YOLOv5 lewat package yang sudah di-install (-e yolov5)
from yolov5.models.common import DetectMultiBackend

# Tentukan path model (asumsi bisindo_best.pt diletakkan di root repo)
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir    = os.path.abspath(os.path.join(current_dir, ".."))
model_path  = os.path.join(root_dir, "bisindo_best.pt")

if not os.path.isfile(model_path):
    print(f"❌ Model not found at {model_path}")
    sys.exit(1)

device = os.getenv("YOLO_DEVICE", "cpu")
model  = DetectMultiBackend(model_path, device=device)
names  = model.names

def detect_frame(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float() / 255.0
    pred = model(tensor)[0]
    output = []
    for *xyxy, conf, cls in pred:
        x1,y1,x2,y2 = map(int, xyxy)
        output.append({
            "label": names[int(cls)],
            "confidence": float(conf),
            "bbox": [x1, y1, x2, y2]
        })
    return output

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Signatext ML API is running ✅"})

@app.route("/predict", methods=["POST"])
def predict():
    # file upload
    if "file" in request.files:
        file = request.files["file"]
        data = file.read()
        arr  = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        detections = detect_frame(frame)

        # upload ke Cloudinary
        upload = BytesIO(data)
        res = cloudinary.uploader.upload(upload)
        return jsonify({
            "detections": detections,
            "media_url": res.get("secure_url")
        })

    # url form
    url = request.form.get("url")
    if url:
        resp = requests.get(url)
        ctype = resp.headers.get("Content-Type", "")
        if "image" in ctype:
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            detections = detect_frame(frame)
            res = cloudinary.uploader.upload(url)
            return jsonify({"detections": detections, "media_url": res.get("secure_url")})
        else:
            return jsonify({"error": "Unsupported content type"}), 400

    return jsonify({"error": "No file or url provided"}), 400

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
