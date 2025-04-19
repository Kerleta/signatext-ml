FROM python:3.10-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Use shell form to ensure environment variable substitution works
CMD gunicorn --bind "0.0.0.0:${PORT:-5000}" file:app
