FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone YOLOv5 dan install dengan numpy versi lama
RUN git clone https://github.com/ultralytics/yolov5.git && \
    pip install --upgrade pip && \
    pip install "numpy<2" && \  # Pastikan numpy 1.x terinstall dulu
    pip install -r yolov5/requirements.txt && \
    pip install opencv-python-headless==4.9.0.80  # Force install versi kompatibel

# Copy dan install requirements aplikasi
COPY . .
RUN pip install -r requirements.txt

CMD ["python", "file.py"]
