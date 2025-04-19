FROM python:3.10-slim
WORKDIR /app

# system deps…
RUN apt-get update && apt-get install -y git ffmpeg libsm6 libxext6 libxrender-dev libgl1-mesa-glx libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# install everything in one go (including yolov5 from GitHub)
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# copy your code and run
COPY . .
CMD ["python", "file.py"]
