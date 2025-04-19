FROM python:3.10-slim

# 1) System deps
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

# 2) Clone & install YOLOv5 as an editable package
RUN git clone https://github.com/ultralytics/yolov5.git /opt/yolov5
RUN pip install --upgrade pip \
  && pip install --no-cache-dir -r /opt/yolov5/requirements.txt \
  && pip install --no-cache-dir -e /opt/yolov5

# 3) Install your own app requirements
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4) Copy your code and run
COPY . .
CMD ["python", "file.py"]
