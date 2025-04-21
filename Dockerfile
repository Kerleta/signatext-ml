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

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install -r /yolov5/requirements.txt

# Copy application code
COPY . .

# Verify model path
RUN mkdir -p /app/model && \
    if [ -f "bisindo_best.pt" ]; then \
        mv bisindo_best.pt /app/model/; \
    fi

# Set environment variables
ENV PYTHONPATH "${PYTHONPATH}:/yolov5"
ENV YOLOV5_DIR="/yolov5"
ENV MODEL_PATH="/app/model/bisindo_best.pt"

# Gunicorn command
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app", "--timeout", "300", "--workers", "1"]
