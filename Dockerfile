FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone YOLOv5 repository
RUN git clone https://github.com/ultralytics/yolov5 /yolov5

# Setup virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /app

# Install PyTorch CPU version first
RUN pip install --upgrade pip && \
    pip install torch==2.0.0 torchvision==0.15.1 --extra-index-url https://download.pytorch.org/whl/cpu

# Copy requirements
COPY requirements.txt .

# Install remaining dependencies
RUN pip install -r requirements.txt && \
    pip install -r /yolov5/requirements.txt

COPY . .

# Setup environment
RUN mkdir -p /app/model && \
    if [ -f "bisindo_best.pt" ]; then \
        mv bisindo_best.pt /app/model/; \
    fi

ENV PYTHONPATH "${PYTHONPATH}:/yolov5"
ENV YOLOV5_DIR="/yolov5"
ENV MODEL_PATH="/app/model/bisindo_best.pt"

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app", "--timeout", "300", "--workers", "1"]
