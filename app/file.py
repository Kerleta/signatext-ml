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

# Make sure YOLOv5 utils are available
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# Inisialisasi Flask
app = Flask(__name__)
CORS(app)

# Setup Cloudinary (ambil dari env)
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME", ""),
    api_key=os.getenv("CLOUDINARY_API_KEY", ""),
    api_secret=os.getenv("CLOUDINARY_API_SECRET", "")
)

# Load YOLOv5 model directly to avoid module errors
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir    = os.path.abspath(os.path.join(current_dir, ".."))
model_path  = os.path.join(root_dir, "bisindo_best.pt")

if not os.path.isfile(model_path):
    print(f"❌ Model not found at {model_path}")
    sys.exit(1)

device = os.getenv("YOLO_DEVICE", "cpu")

# Load model with torch directly instead of using YOLOv5 package
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
model.to(device)
names = model.names if hasattr(model, 'names') else None

def detect_frame(frame):
    # Process frame directly with the model
    results = model(frame)
    
    # Extract predictions from results
    output = []
    for detection in results.xyxy[0]:  # First image in batch
        x1, y1, x2, y2, conf, cls = detection.cpu().numpy()
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        output.append({
            "label": names[int(cls)] if names else f"Class {int(cls)}",
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
