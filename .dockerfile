FROM python:3.10-slim

# Install dependencies (incl. libGL for OpenCV)
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean

# Set workdir
WORKDIR /app

# Copy all files
COPY . .

# Clone YOLOv5 (atau bisa copy lokal kalau udah dimasukin repo)
RUN git clone https://github.com/ultralytics/yolov5.git && \
    pip install --upgrade pip && \
    pip install -r yolov5/requirements.txt && \
    pip install -r requirements.txt

# Run your app
CMD ["python", "file.py"]
