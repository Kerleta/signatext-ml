FROM python:3.10-slim
WORKDIR /app

# 1) System deps
RUN apt-get update && apt-get install -y \
     git ffmpeg libsm6 libxext6 libxrender-dev libgl1-mesa-glx libglib2.0-0 \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

# 2) Copy everything (your code + vendored yolov5)
COPY . .

# 3) Install YOLOv5 in editable mode
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r yolov5/requirements.txt \
 && pip install --no-cache-dir -e yolov5

# 4) Install your app’s Python deps
RUN pip install --no-cache-dir -r requirements.txt

# 5) Launch
CMD ["python", "file.py"]
