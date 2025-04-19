import os
import sys

# Add yolov5 to Python path if not already there
current_dir = os.path.dirname(os.path.abspath(__file__))
yolo_dir = os.path.join(current_dir, "yolov5")
if yolo_dir not in sys.path:
    sys.path.insert(0, yolo_dir)

import torch
os.environ["OPENCV_OPENGL_RUNTIME"] = "none"
import cv2
import cloudinary
import cloudinary.uploader
from flask import Flask, request, jsonify
from flask_cors import CORS
try:
    from yolov5.models.common import DetectMultiBackend
except ImportError:
    print("Error importing yolov5. Current sys.path:", sys.path)
    raise
from io import BytesIO
from PIL import Image
import numpy as np
import argparse
import json
import requests

# Inisialisasi Flask app
app = Flask(__name__)
CORS(app)

# Konfigurasi Cloudinary
cloudinary.config(
    cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME", ""),
    api_key=os.environ.get("CLOUDINARY_API_KEY", ""),
    api_secret=os.environ.get("CLOUDINARY_API_SECRET", "")
)

# Logging untuk pengecekan startup
print("🚀 Starting Signatext ML Backend...")
print(f"Current working directory: {os.getcwd()}")
print(f"Files in current directory: {os.listdir('.')}")
if os.path.exists('yolov5'):
    print(f"Files in yolov5 directory: {os.listdir('yolov5')}")
else:
    print("yolov5 directory not found!")

# Load model YOLOv5
device = os.getenv("YOLO_DEVICE", "cpu")
model_path = "bisindo_best.pt"

if not os.path.exists(model_path):
    print(f"❌ Model file not found: {model_path}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Files in current directory: {os.listdir('.')}")
    sys.exit(1)

try:
    model = DetectMultiBackend(model_path, device=device)
    names = model.names
    print("✅ YOLOv5 model loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    sys.exit(1)

def detect_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    results = model(tensor)[0]
    output = []
    for *xyxy, conf, cls in results:
        x1, y1, x2, y2 = [int(x.item()) for x in xyxy]
        label = names[int(cls.item())]
        output.append({
            'label': label,
            'confidence': float(conf),
            'bbox': [x1, y1, x2, y2]
        })
    return output

def predict_image(path):
    frame = cv2.imread(path)
    return {"file": path, "detections": detect_frame(frame)}

def predict_video(path, frame_skip=10):
    cap = cv2.VideoCapture(path)
    results = []
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        if frame_id % frame_skip == 0:
            results.append({'frame': frame_id, 'detections': detect_frame(frame)})
    cap.release()
    return {"file": path, "video_detections": results}

def download_from_url(url):
    resp = requests.get(url)
    ctype = resp.headers.get('Content-Type', '')
    if 'image' in ctype:
        img = Image.open(BytesIO(resp.content)).convert('RGB')
        frame = np.array(img)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        temp = "temp_image.jpg"
        cv2.imwrite(temp, frame_bgr)
        return predict_image(temp)
    elif 'video' in ctype:
        temp = "temp_video.mp4"
        with open(temp, 'wb') as f:
            f.write(resp.content)
        return predict_video(temp)
    else:
        return {"error": "Unsupported file type"}

# Health check
@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Signatext ML API is running ✅'})

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' in request.files:
        file = request.files['file']
        data = file.read()
        arr = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        detections = detect_frame(frame)

        upload = BytesIO(data)
        res = cloudinary.uploader.upload(upload)
        media_url = res.get('secure_url')

        return jsonify({'detections': detections, 'media_url': media_url})

    url = request.form.get('url')
    if url:
        result = download_from_url(url)
        detections = result.get('detections') or result.get('video_detections', [])

        res = cloudinary.uploader.upload(url, resource_type='video')
        media_url = res.get('secure_url')

        return jsonify({'detections': detections, 'media_url': media_url})

    return jsonify({'error': 'No file or url provided'}), 400

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLOv5 BISINDO Service")
    parser.add_argument('--image', type=str)
    parser.add_argument('--video', type=str)
    parser.add_argument('--url', type=str)
    parser.add_argument('--cli', action='store_true')
    args = parser.parse_args()
    if args.cli:
        if args.image:
            print(json.dumps(predict_image(args.image), indent=2))
        elif args.video:
            print(json.dumps(predict_video(args.video), indent=2))
        elif args.url:
            print(json.dumps(download_from_url(args.url), indent=2))
        else:
            print(json.dumps({"error": "Provide --image, --video, or --url"}, indent=2))
    else:
        port = int(os.getenv("PORT", 5000))
        print(f"🌐 Running server on port {port}")
        app.run(host='0.0.0.0', port=port)
