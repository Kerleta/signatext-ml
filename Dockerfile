FROM python:3.10-slim
WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y git ffmpeg libgl1-mesa-glx libglib2.0-0

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Clone YOLOv5 repository
RUN git clone --depth 1 -b master https://github.com/Kerleta/yolov5.git yolov5 && \
    cd yolov5 && pip install -e .

# Copy the rest of the application
COPY . .

# Set environment variables
ENV PYTHONPATH="${PYTHONPATH}:./yolov5"

# Command to run
CMD gunicorn --bind "0.0.0.0:${PORT:-5000}" file:app
