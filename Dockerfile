FROM python:3.10-slim

# Install dependencies sistem untuk OpenCV & thread lib
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Buat virtual env
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy requirements dan install
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy semua file ke image
COPY . .

# Ensure the model file exists or download it
RUN if [ ! -f "bisindo_best.pt" ]; then \
    echo "Warning: Model file not found in the repository. You need to make sure it's available during deployment"; \
    fi

# Jalankan aplikasi menggunakan Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app.file:app", "--timeout", "300"]
