# Hướng dẫn Vận hành, Giám sát & Triển khai (Operations & Deployment Guide)

Tài liệu hướng dẫn đội ngũ Vận hành (DevOps / SysAdmin) triển khai, cấu hình và giám sát dịch vụ **MKU CAPTCHA OCR**.

---

## 1. Yêu cầu Hệ thống (System Requirements)

- **Hệ điều hành:** Linux (Ubuntu 20.04+, Debian 11+, RHEL/Rocky 8+), macOS (Apple Silicon hoặc Intel), Windows (WSL2).
- **Phần cứng tối thiểu:**
  - CPU: 1 vCPU (Kiến trúc x86_64 hoặc ARM64).
  - RAM: 512 MB (Service runtime tiêu thụ ~90–140 MB).
  - Ổ đĩa: 1 GB trống.
- **Phần mềm:** Docker 24+ và Docker Compose v2 (nếu chạy container), hoặc Python 3.10+ (nếu chạy trực tiếp).

---

## 2. Triển khai Bằng Docker & Docker Compose (Khuyến nghị)

### 2.1. Khởi chạy nhanh bằng Docker Compose
Trong thư mục gốc dự án:

```bash
# Build image và khởi chạy ngầm
docker compose up -d --build

# Xem logs dịch vụ thời gian thực
docker compose logs -f

# Dừng dịch vụ
docker compose down
```

### 2.2. Build và Run Docker Image thủ công
```bash
# Build image độc lập
docker build -t mku-captcha-ocr:latest .

# Chạy container bind cổng 28186
docker run -d \
  --name mku-captcha-ocr \
  --restart unless-stopped \
  -p 28186:28186 \
  mku-captcha-ocr:latest
```

---

## 3. Triển khai Local / Trực tiếp với Systemd

### 3.1. Cài đặt môi trường ảo
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-build-isolation --no-deps
```

### 3.2. Cấu hình dịch vụ Systemd (`/etc/systemd/system/captcha-mku.service`)
```ini
[Unit]
Description=MKU CAPTCHA OCR FastAPI Microservice
After=network.target

[Service]
Type=simple
User=appuser
WorkingDirectory=/opt/captcha-mku
ExecStart=/opt/captcha-mku/.venv/bin/captcha-mku serve --host 0.0.0.0 --port 28186
Restart=always
RestartSec=3
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

Kích hoạt dịch vụ:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now captcha-mku
sudo systemctl status captcha-mku
```

---

## 4. Giám sát & Xử lý Sự cố (Monitoring & Troubleshooting)

### 4.1. Giám sát Sức khỏe (Healthchecks)
Cấu hình healthcheck định kỳ tới `http://localhost:28186/health`:
- Trạng thái HTTP mong đợi: `200 OK`.
- Kiểm tra trường `"engine": "onnx"` và `"model_exists": true`. Nếu chuyển sang `"engine": "ddddocr"`, nghĩa là file model ONNX bị thiếu hoặc hỏng.

### 4.2. Giám sát Hiệu năng (Latency & Errors)
- **Thời gian phản hồi bình thường:** `< 15 ms` với endpoint `/api/v1/solve/base64`, `< 500 ms` với endpoint `/api/v1/solve/mku` (phụ thuộc tốc độ mạng tới máy chủ MKU).
- **Lỗi 400 Bad Request:** Thường do client truyền ảnh Base64 bị cắt cụt hoặc không hợp lệ.
- **Lỗi 502/504 Bad Gateway:** Thường xảy ra khi máy chủ trường `apiportal.mku.edu.vn` bị quá tải hoặc chặn IP.
