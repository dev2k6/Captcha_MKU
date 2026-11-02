FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

# Install minimal system libs needed by Pillow / ONNX Runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY captcha_ocr.py evaluate_ocr.py ./
COPY app ./app
COPY models/captcha ./models/captcha

EXPOSE 28186

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "28186"]
