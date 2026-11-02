# Technical Specifications (`SPECS.md`)

Tài liệu đặc tả kỹ thuật toàn diện cho hệ thống **MKU CAPTCHA OCR Microservice & CLI** (`captcha-mku`).

---

## 1. Tổng quan Hệ thống & Mục tiêu Thiết kế

`captcha-mku` là hệ thống nhận diện ký tự quang học (OCR) chuyên biệt cho ảnh CAPTCHA của cổng thông tin trường Đại học Cửu Long (`https://apiportal.mku.edu.vn/api/authenticate/GetCaptcha`).

### Ràng buộc nghiệp vụ (Domain Invariants)
- **Định dạng ảnh đầu vào:** Ảnh PNG RGB kích thước gốc thường `~170×60 px`, nền xanh lá nhạt kèm nhiễu đường kẻ và chấm mảnh.
- **Độ dài chuỗi chuẩn:** Luôn gồm đúng **6 ký tự ASCII** (`EXPECTED_CAPTCHA_LENGTH = 6`).
- **Tập ký tự (`CASE_SENSITIVE_CHARSET`):** 62 ký tự phân biệt chữ hoa/chữ thường (`0-9`, `a-z`, `A-Z`) cộng với 1 token `blank` ở vị trí index `0` cho giải thuật CTC (tổng cộng 63 lớp đầu ra).
- **Đặc trưng bộ font CAPTCHA MKU:**
  - Bộ sinh CAPTCHA của MKU **không bao giờ sinh chữ số `0`** (số không); mọi ký tự tròn khép kín đều là chữ `O` hoa hoặc `o` thường.
  - Phân biệt `9` và `q`: Chữ số `9` nằm hoàn toàn trên đường cơ sở (baseline) với đuôi cong tròn bên dưới; chữ `q` thường có nét sổ thẳng kéo dài xuống dưới đường cơ sở (`y > 48` trên ảnh chuẩn hóa cao 64px).
  - Không bao giờ tự động điền thêm (pad) hoặc đoán mò ký tự nếu kết quả giải mã CTC khác 6 ký tự; mọi kết quả khác 6 ký tự đều bị từ chối và trả về chuỗi rỗng `""` (`status: false`).

---

## 2. Kiến trúc Mô hình Học sâu (CRNN + ResidualCTCHead)

### 2.1. Kiến trúc Mạng (`training/model.py`)
Mô hình sử dụng kiến trúc **CRNN (Convolutional Recurrent Neural Network)** kết hợp **Residual CTC Classification Head** ([`CaptchaModel`](file:///Users/dev2k6/Workspace/tooling/ai/captcha-mku/training/model.py#L46-L65)):
1. **Backbone & Sequence Encoder (`ddddocr` Frozen/Fine-tuned Feature Extractor):**
   - Trích xuất đặc trưng không gian 2D qua chuỗi khối tích chập (CNN) và gom đặc trưng theo chiều dọc thành chuỗi thời gian `T` bước (width sequence), mỗi bước là vector đặc trưng `256` chiều (`feature_dim = 256`).
2. **`ResidualCTCHead` ([`training/model.py`](file:///Users/dev2k6/Workspace/tooling/ai/captcha-mku/training/model.py#L22-L43)):**
   - Nhận tensor đầu vào kích thước `(T, B, 256)` hoặc `(B, T, 256)`.
   - Nhánh chính (Linear Projection): Ánh xạ trực tiếp `Linear(256 -> 63)`.
   - Nhánh dư (Residual Context Adapter): `LayerNorm(256) -> Linear(256 -> 256) -> GELU -> Dropout(p) -> Linear(256 -> 63)` được nhân với hệ số học được `residual_scale` (khởi tạo bằng `0.0` để bảo toàn điểm xuất phát từ head chữ thường sang chữ hoa/thường).
   - Đầu ra: Tensor logits kích thước `(T, B, 63)` phục vụ hàm mất mát `nn.CTCLoss(blank=0, zero_infinity=True)`.

### 2.2. Tiền xử lý Ảnh (`_preprocess_custom_onnx`)
Quy trình chuẩn hóa ảnh đầu vào trong [`captcha_ocr.py`](file:///Users/dev2k6/Workspace/tooling/ai/captcha-mku/captcha_ocr.py#L140-L151) và [`training/train.py`](file:///Users/dev2k6/Workspace/tooling/ai/captcha-mku/training/train.py):
1. Chuyển ảnh đầu vào sang hệ màu `RGB`.
2. Nội suy kích thước theo tỷ lệ khung hình về chiều cao cố định `64 px`:
   $$W_{\text{new}} = \max\left(1, \left\lfloor W_{\text{orig}} \times \frac{64}{H_{\text{orig}}} \right\rfloor\right)$$
   sử dụng bộ lọc `PIL.Image.Resampling.LANCZOS`.
3. Chuyển sang ảnh đa mức xám 1 kênh (`L` — grayscale).
4. Chuẩn hóa giá trị điểm ảnh về đoạn `[0.0, 1.0]` kiểu `float32` (`pixel / 255.0`) và đóng gói thành tensor 4 chiều `(1, 1, 64, W_new)`.

### 2.3. Giải mã CTC & Độ tin cậy theo Ký tự (`decode_ctc_logits`)
Thuật toán giải mã CTC trong [`captcha_ocr.py`](file:///Users/dev2k6/Workspace/tooling/ai/captcha-mku/captcha_ocr.py#L154-L209):
- Áp dụng hàm `softmax` ổn định số học theo trục lớp ký tự tại mỗi bước thời gian $t \in [1, T]$.
- Thực hiện **Greedy CTC Collapse**: Gộp các bước thời gian liên tiếp có cùng nhãn dự đoán $\arg\max_c P_t(c)$ và loại bỏ các bước rơi vào token `blank` (`index = 0`).
- **Điểm tin cậy ký tự ($i \in \{1..6\}$):** Với mỗi đoạn bước thời gian liên tiếp $S_i$ tạo nên ký tự thứ $i$, độ tin cậy của ký tự được tính bằng trung bình xác suất đỉnh trên đoạn đó:
  $$\text{conf}_i = \frac{1}{|S_i|} \sum_{t \in S_i} P_t(c_i)$$
- **Cờ cảnh báo `needs_review`:** Bật `True` nếu chuỗi giải mã không đủ 6 ký tự hoặc tồn tại bất kỳ ký tự nào có $\text{conf}_i < 0.85$ (`MIN_CONFIDENCE_FOR_AUTO_ACCEPT`).

---

## 3. Giao diện Dòng lệnh Hợp nhất (Unified CLI `captcha-mku`)

Entry point chính được khai báo trong [`pyproject.toml`](file:///Users/dev2k6/Workspace/tooling/ai/captcha-mku/pyproject.toml) (`captcha-mku = "captcha_ocr:main"`) và thực thi tại [`captcha_ocr.py`](file:///Users/dev2k6/Workspace/tooling/ai/captcha-mku/captcha_ocr.py#L406-L458).

### Các Subcommands hỗ trợ
1. **`captcha-mku [solve]` (Mặc định, tương thích ngược 100%):**
   - `captcha-mku` hoặc `captcha-mku solve`: Gọi trực tiếp API MKU lấy ảnh mới và giải mã.
   - `--image <path>`: Giải mã file ảnh cục bộ.
   - `--save-image <path>`: Lưu ảnh CAPTCHA gốc ra đĩa.
   - `--model <path>`: Chỉ định đường dẫn file `.onnx` tùy chỉnh.
   - `--builtin`: Ép sử dụng engine `ddddocr` gốc thay vì model ONNX đã tinh chỉnh.
   - `--debug`: Bổ sung trường `ocr` gồm `confidence` (6 điểm float), `needs_review`, `candidates`, và `engine`.
2. **`captcha-mku serve`:**
   - Khởi chạy FastAPI Microservice qua Uvicorn (`--host 0.0.0.0 --port 28186 --reload`).
3. **`captcha-mku benchmark`:**
   - Chạy benchmark đối đầu giữa Custom ONNX và `ddddocr` trên 500 ảnh thực tế (`--report-only` để chỉ kết xuất lại báo cáo HTML Light Mode).
4. **`captcha-mku evaluate`:**
   - Đánh giá độ chính xác offline trên file manifest bất kỳ (`--manifest <path> --report <path>`).
5. **`captcha-mku train`:**
   - Huấn luyện / tinh chỉnh model CRNN và xuất file ONNX (`--manifest <path> --output <dir> --epochs 80 --train-all`).

---

## 4. Đặc tả REST API Microservice (`app/`)

Ứng dụng FastAPI được định nghĩa tại [`app/main.py`](file:///Users/dev2k6/Workspace/tooling/ai/captcha-mku/app/main.py), sử dụng Singleton [`CaptchaOCRService`](file:///Users/dev2k6/Workspace/tooling/ai/captcha-mku/app/service.py#L22-L120) nạp sẵn ONNX Runtime session vào RAM và thực hiện warmup ngay trong sự kiện `lifespan`.

### 4.1. `GET /health`
- **Mô tả:** Kiểm tra trạng thái sống của container và thông tin model ONNX đang nạp.
- **Response (`200 OK` - [`HealthResponse`](file:///Users/dev2k6/Workspace/tooling/ai/captcha-mku/app/schemas.py#L10-L20)):**
```json
{
  "status": "ok",
  "version": "1.0.0",
  "engine": "onnx",
  "model_path": "/app/models/captcha/model.onnx",
  "model_exists": true,
  "expected_length": 6,
  "uptime_seconds": 124.5
}
```

### 4.2. `POST /api/v1/solve/mku`
- **Mô tả:** Tự động gọi API MKU lấy CAPTCHA mới và trả về kết quả giải mã kèm `token` phiên.
- **Request Body ([`SolveMkuRequest`](file:///Users/dev2k6/Workspace/tooling/ai/captcha-mku/app/schemas.py#L40-L55)):** `{"builtin": false, "include_image": false, "debug": true}`
- **Response (`200 OK` - [`SolveResponse`](file:///Users/dev2k6/Workspace/tooling/ai/captcha-mku/app/schemas.py#L57-L83)).**

### 4.3. `POST /api/v1/solve/base64`
- **Request Body ([`SolveBase64Request`](file:///Users/dev2k6/Workspace/tooling/ai/captcha-mku/app/schemas.py#L22-L38)):**
```json
{
  "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg...",
  "builtin": false,
  "debug": true
}
```
- **Response (`200 OK` - [`SolveResponse`](file:///Users/dev2k6/Workspace/tooling/ai/captcha-mku/app/schemas.py#L57-L83)):**
```json
{
  "status": "success",
  "captcha": "fMGuVA",
  "text": "fMGuVA",
  "valid_length": true,
  "latency_ms": 4.82,
  "engine": "onnx",
  "token": null,
  "image": null,
  "confidence": [0.9999, 0.9998, 0.9965, 0.9991, 0.9982, 0.9995],
  "mean_confidence": 0.9988,
  "needs_review": false,
  "candidates": []
}
```

### 4.4. `POST /api/v1/solve/file`
- **Request (`multipart/form-data`):** Trường `file` (ảnh PNG/JPEG) cùng form fields tùy chọn `builtin=false`, `debug=false`.
- **Response (`200 OK` - [`SolveResponse`](file:///Users/dev2k6/Workspace/tooling/ai/captcha-mku/app/schemas.py#L57-L83)).**

---

## 5. Cấu trúc Dữ liệu & Báo cáo Chuẩn (Canonical Datasets & Reports)

| Thành phần | Đường dẫn | Quy mô | Mục đích & Kết quả |
|---|---|---:|---|
| **Production Model** | [`models/captcha/model.onnx`](file:///Users/dev2k6/Workspace/tooling/ai/captcha-mku/models/captcha/model.onnx) | `6.43 MB` | Model ONNX sản xuất (kèm [`charsets.json`](file:///Users/dev2k6/Workspace/tooling/ai/captcha-mku/models/captcha/charsets.json)) |
| **Tập dữ liệu huấn luyện chính** | `data/training/` | `5,000` ảnh | Tập rà soát chuẩn (`4,000` train / `500` val / `500` test) — **100.0% (5,000/5,000)** |
| **Tập kiểm thử 1,000 ảnh** | `data/test1000/` | `1,000` ảnh | Tập kiểm thử độc lập 1,000 ảnh — **100.0% (1,000/1,000)** |
| **Tập Benchmark 500 ảnh** | `data/benchmark/` | `500` ảnh | Tập đối đầu trực tiếp Custom ONNX (**100.0%**) vs `ddddocr` (**17.6%**) |
| **Báo cáo HTML Light Mode** | [`reports/benchmark_500_overview.html`](file:///Users/dev2k6/Workspace/tooling/ai/captcha-mku/reports/benchmark_500_overview.html) | `500` mẫu | Dashboard trực quan thân thiện non-tech so sánh Custom AI và `ddddocr` |

---

## 6. Chỉ số Cam kết Chất lượng (SLA & Benchmarks)

Trên phần cứng CPU tiêu chuẩn (Apple Silicon / x86_64 2 vCPU):
- **Độ chính xác khớp tuyệt đối cả 6 ký tự hoa/thường (Exact Match):** **100.0%** trên `data/training/` (5,000 ảnh), `data/test1000/` (1,000 ảnh) và `data/benchmark/` (500 ảnh).
- **Độ trễ suy luận (Inference Latency):** Trung bình **4.8 – 7.4 ms / ảnh** (P95 `< 8.5 ms`), thông lượng **135 – 200+ ảnh/giây** trên 1 nhân CPU.
- **Dung lượng bộ nhớ (Memory Footprint):** Kích thước model ONNX `6.43 MB`, RAM tiến trình khi chạy dịch vụ `< 150 MB`.
