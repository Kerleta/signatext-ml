FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Use shell form for variable expansion
CMD gunicorn --bind 0.0.0.0:${PORT:-5000} file:app
