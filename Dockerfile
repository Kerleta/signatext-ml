FROM python:3.10-slim
WORKDIR /app

# install system deps…
RUN apt-get update && apt-get install -y git ffmpeg libsm6 libxext6 libxrender-dev libgl1-mesa-glx libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# first install your Python libs (so Docker caches this)
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install git+https://github.com/ultralytics/yolov5.git@master  # <-- pulls yolov5 + its utils, models, etc.

# then copy your own code
COPY . .

CMD ["python", "file.py"]
