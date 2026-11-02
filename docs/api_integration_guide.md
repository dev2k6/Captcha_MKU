# Hướng dẫn Tích hợp & Đặc tả Giao diện API (Integration & API Guide)

Tài liệu hướng dẫn các lập trình viên và nhóm phát triển tích hợp dịch vụ OCR vào hệ thống tự động hóa hoặc ứng dụng nghiệp vụ.

---

## 1. Các Phương thức Tích hợp Hỗ trợ

Hệ thống cung cấp 3 phương thức tích hợp tùy theo kiến trúc của ứng dụng gọi:

1. **HTTP REST API Microservice (Khuyến nghị cho Web / Backend đa ngôn ngữ / Microservices):** Gọi qua HTTP JSON endpoints.
2. **Python SDK (Khuyến nghị cho các script RPA, bot, crawling bằng Python):** Import trực tiếp hàm module.
3. **Unified CLI (Khuyến nghị cho Bash scripting / Cron job / CI/CD):** Chạy lệnh terminal `captcha-mku`.

---

## 2. Đặc tả Chi tiết HTTP REST API

Dịch vụ chạy mặc định tại cổng `28186`. Swagger UI tài liệu trực quan có tại: `http://localhost:28186/docs`.

### 2.1. Kiểm tra trạng thái hệ thống: `GET /health`
- **Mục đích:** Health-check cho Load Balancer, Kubernetes liveness/readiness probe.
- **Phản hồi mẫu:**
```json
{
  "status": "ok",
  "version": "1.0.0",
  "engine": "onnx",
  "model_path": "/app/models/captcha/model.onnx",
  "model_exists": true,
  "expected_length": 6,
  "uptime_seconds": 1542.2
}
```

---

### 2.2. Tự động lấy & giải mã CAPTCHA MKU: `POST /api/v1/solve/mku`
- **Mục đích:** Microservice tự động kết nối cổng MKU, tải ảnh và token mới nhất về giải mã trong ~5ms.
- **Request Body (JSON, tùy chọn):**
```json
{
  "builtin": false,
  "include_image": false,
  "debug": false
}
```
- **Response (`200 OK`):**
```json
{
  "status": "success",
  "captcha": "aQDsNd",
  "text": "aQDsNd",
  "valid_length": true,
  "latency_ms": 5.12,
  "engine": "onnx",
  "token": "7T5/x/2GgjvZRcC2R3r76EXe7+j7/o9S9H55V3TfCEs=",
  "image": null,
  "confidence": null,
  "mean_confidence": null,
  "needs_review": null,
  "candidates": null
}
```

---

### 2.3. Giải mã ảnh Base64: `POST /api/v1/solve/base64`
- **Mục đích:** Dành cho trường hợp client đã tự fetch ảnh về máy hoặc nhận ảnh từ webhook/browser canvas.
- **Request Body (JSON):**
```json
{
  "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAAA8CAIAAA...",
  "builtin": false,
  "debug": true
}
```
- **Response (`200 OK`):**
```json
{
  "status": "success",
  "captcha": "KdQRZl",
  "text": "KdQRZl",
  "valid_length": true,
  "latency_ms": 4.88,
  "engine": "onnx",
  "token": null,
  "image": null,
  "confidence": [0.999998, 1.0, 1.0, 1.0, 0.999999, 1.0],
  "mean_confidence": 0.9999995,
  "needs_review": false,
  "candidates": []
}
```

---

### 2.4. Giải mã file ảnh tải lên: `POST /api/v1/solve/file`
- **Content-Type:** `multipart/form-data`
- **Tham số Form:**
  - `file` (bắt buộc): File ảnh PNG hoặc JPEG.
  - `debug` (tùy chọn, boolean): Bật chuẩn đoán điểm tin cậy.

---

## 3. Hướng dẫn Tích hợp Python SDK

```python
import captcha_ocr
from PIL import Image

# 1. Giải mã trực tiếp từ file ảnh cục bộ
image = Image.open("captcha.png")
captcha_text = captcha_ocr.solve_captcha(image)
print("Kết quả:", captcha_text)  # Ví dụ: 'KdQRZl'

# 2. Giải mã chi tiết kèm điểm tin cậy từng ký tự
info = captcha_ocr.solve_captcha_detailed(image)
print("Text:", info["text"])
print("Độ tin cậy 6 ký tự:", info["confidence"])
print("Engine:", info["engine"])

# 3. Lấy ảnh trực tiếp từ API MKU
data = captcha_ocr.fetch_captcha()
# data['token'] là session token, data['img'] là chuỗi data URL
img = captcha_ocr.decode_data_image(data["img"])
result = captcha_ocr.solve_captcha(img)
print("CAPTCHA MKU mới:", result, "với token:", data["token"])
```

---

## 4. Hướng dẫn Tích hợp Dòng lệnh CLI (`captcha-mku`)

Dùng trong các kịch bản shell hoặc tự động hóa:

```bash
# Lấy và giải mã tự động CAPTCHA live
captcha-mku solve

# Giải mã ảnh có sẵn và in dạng JSON
captcha-mku solve --image /path/to/captcha.png

# Bật chế độ debug để theo dõi độ tin cậy
captcha-mku solve --image /path/to/captcha.png --debug
```
