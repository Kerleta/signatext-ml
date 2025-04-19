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

# Clone dan install YOLOv5 sebagai editable package
RUN git clone https://github.com/ultralytics/yolov5.git && \
    cd yolov5 && \
    pip install --upgrade pip && \
    pip install "numpy<2" && \
    pip install -r requirements.txt && \
    pip install -e .  # Ini yang paling kritis!

# Install app requirements
COPY . .
RUN pip install -r requirements.txt

CMD ["python", "file.py"]
