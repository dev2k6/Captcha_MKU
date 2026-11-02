# AI Agent Handbook & Operational Rules (`AGENTS.md`)

Tài liệu hướng dẫn vận hành, quy tắc bất biến và quy trình làm việc dành cho các AI Coding Agents khi đọc hiểu, bảo trì hoặc mở rộng dự án **MKU CAPTCHA OCR** (`captcha-mku`).

---

## 1. Bản đồ Kho mã nguồn (Repository Map)

- [`captcha_ocr.py`](file:///Users/dev2k6/Workspace/tooling/ai/captcha-mku/captcha_ocr.py): Trái tim của SDK giải mã OCR và Unified CLI (`captcha-mku`). Chứa toàn bộ logic nạp ONNX, tiền xử lý ảnh, giải mã CTC và tương tác với API MKU.
- [`app/`](file:///Users/dev2k6/Workspace/tooling/ai/captcha-mku/app/main.py): Tầng dịch vụ HTTP Microservice sử dụng FastAPI:
  - [`app/main.py`](file:///Users/dev2k6/Workspace/tooling/ai/captcha-mku/app/main.py): Định nghĩa các route `/health`, `/api/v1/solve/mku`, `/api/v1/solve/base64`, `/api/v1/solve/file`.
  - [`app/service.py`](file:///Users/dev2k6/Workspace/tooling/ai/captcha-mku/app/service.py): Lớp [`CaptchaOCRService`](file:///Users/dev2k6/Workspace/tooling/ai/captcha-mku/app/service.py#L22-L120) quản lý vòng đời model và warmup.
  - [`app/schemas.py`](file:///Users/dev2k6/Workspace/tooling/ai/captcha-mku/app/schemas.py): Các Pydantic models kiểm tra kiểu dữ liệu đầu vào/đầu ra.
- [`training/`](file:///Users/dev2k6/Workspace/tooling/ai/captcha-mku/training/train.py): Pipeline học sâu và đánh giá:
  - [`training/model.py`](file:///Users/dev2k6/Workspace/tooling/ai/captcha-mku/training/model.py): Định nghĩa [`CaptchaModel`](file:///Users/dev2k6/Workspace/tooling/ai/captcha-mku/training/model.py#L46-L65) và [`ResidualCTCHead`](file:///Users/dev2k6/Workspace/tooling/ai/captcha-mku/training/model.py#L22-L43).
  - [`training/train.py`](file:///Users/dev2k6/Workspace/tooling/ai/captcha-mku/training/train.py): Kịch bản huấn luyện, chọn checkpoint và xuất ONNX.
  - [`training/prepare_data.py`](file:///Users/dev2k6/Workspace/tooling/ai/captcha-mku/training/prepare_data.py) & [`training/curate.py`](file:///Users/dev2k6/Workspace/tooling/ai/captcha-mku/training/curate.py): Thu thập dữ liệu, tạo contact sheet và gộp nhãn rà soát.
  - [`training/run_benchmark.py`](file:///Users/dev2k6/Workspace/tooling/ai/captcha-mku/training/run_benchmark.py) & [`training/generate_report.py`](file:///Users/dev2k6/Workspace/tooling/ai/captcha-mku/training/generate_report.py): Chạy benchmark 500 ảnh và sinh báo cáo HTML Light Mode.
- [`evaluate_ocr.py`](file:///Users/dev2k6/Workspace/tooling/ai/captcha-mku/evaluate_ocr.py): Công cụ đánh giá độ chính xác offline theo manifest.
- [`tests/`](file:///Users/dev2k6/Workspace/tooling/ai/captcha-mku/tests/test_captcha_ocr.py): Bộ kiểm thử tự động gồm [`tests/test_captcha_ocr.py`](file:///Users/dev2k6/Workspace/tooling/ai/captcha-mku/tests/test_captcha_ocr.py) và [`tests/test_api.py`](file:///Users/dev2k6/Workspace/tooling/ai/captcha-mku/tests/test_api.py).

---

## 2. Các Quy tắc Bất biến (Core Invariants — KHÔNG ĐƯỢC VI PHẠM)

1. **Không bao giờ tự chèn hoặc đoán ký tự thiếu:**
   - CAPTCHA MKU luôn có đúng 6 ký tự (`EXPECTED_CAPTCHA_LENGTH = 6`). Nếu chuỗi giải mã CTC có độ dài $\neq 6$ hoặc chứa ký tự ngoài `0-9a-zA-Z`, hàm [`solve_captcha`](file:///Users/dev2k6/Workspace/tooling/ai/captcha-mku/captcha_ocr.py#L351-L358) **bắt buộc** trả về chuỗi rỗng `""` và `status: False`. Tuyệt đối không pad hoặc tạo biến thể giả tạo.
2. **Bảo toàn tuyệt đối thứ tự tiền xử lý ảnh ONNX:**
   - Mọi luồng suy luận và huấn luyện cho model custom ONNX phải giữ đúng thứ tự trong [`_preprocess_custom_onnx`](file:///Users/dev2k6/Workspace/tooling/ai/captcha-mku/captcha_ocr.py#L140-L151):
     `image.convert("RGB")` $\rightarrow$ `resize((new_w, 64), LANCZOS)` $\rightarrow$ `convert("L")` $\rightarrow$ `float32 / 255.0`.
   - Việc thay đổi bộ lọc nội suy (ví dụ sang `BILINEAR`) hoặc chuyển sang `L` trước khi resize sẽ làm lệch phân bố mức xám biên chữ và gây giảm độ chính xác.
3. **Bảo vệ Ground-Truth và Artifact Sản xuất:**
   - Không bao giờ tự động ghi đè nhãn đã xác nhận (`label` / `expected`) bằng kết quả dự đoán của model nếu không kiểm tra trực quan từng ảnh.
   - Không xóa hoặc ghi đè [`models/captcha/model.onnx`](file:///Users/dev2k6/Workspace/tooling/ai/captcha-mku/models/captcha/model.onnx) khi chưa xác nhận model mới đạt **100.0% Exact Match** trên cả `data/training/manifest.json`, `data/test1000/manifest.json` và `data/benchmark/images/`.
4. **Giữ tương thích ngược cho CLI & SDK:**
   - Lệnh `python captcha_ocr.py --image captcha.png` (không truyền tên subcommand) phải luôn hoạt động giống hệt `captcha-mku solve --image captcha.png`.
   - Lưu ý khóa từ điển của [`solve_captcha_detailed`](file:///Users/dev2k6/Workspace/tooling/ai/captcha-mku/captcha_ocr.py#L307-L349) là `"confidence"` (danh sách 6 số thực float) và `"engine"` (`"custom_onnx"` hoặc `"builtin_ddddocr"`).

---

## 3. Quy tắc Nhận diện Hình học Font chữ MKU (Glyph Disambiguation Rules)

Khi rà soát nhãn dữ liệu (human/agent visual curation) trên ảnh CAPTCHA MKU, luôn áp dụng các quy tắc hình học sau:

- **Quy tắc số `0` vs chữ `O`/`o`:**
  - Hệ thống CAPTCHA MKU **không bao giờ sinh chữ số `0`**. Mọi ký tự hình bầu dục/tròn khép kín đều là chữ `O` hoa (nếu cao tương đương chữ hoa/số bên cạnh) hoặc `o` thường (nếu thấp bằng nửa thân chữ).
- **Quy tắc số `9` vs chữ `q` thường vs chữ `g` thường:**
  - **Số `9`:** Toàn bộ thân chữ nằm **trên đường cơ sở (baseline)**, phần đuôi dưới uốn cong sang trái (`y_max <= 46` trên ảnh chuẩn hóa cao 64px).
  - **Chữ `q` thường:** Bầu tròn nằm ở vùng x-height và có **nét sổ thẳng đứng** bên phải kéo dài sâu xuống dưới đường cơ sở (`y_max >= 50`), không uốn cong sang trái.
  - **Chữ `g` thường:** Có nét móc cong tròn dưới đường cơ sở uốn ngược về bên trái.
- **Quy tắc phân biệt Hoa / Thường cho các cặp đồng dạng (`c/C`, `s/S`, `v/V`, `w/W`, `x/X`, `z/Z`, `p/P`, `k/K`):**
  - So sánh chiều cao đỉnh (top y-coordinate) và chiều cao thân (bounding-box height) với các chữ số hoặc chữ cái có hình dáng phân biệt rõ (`A, B, D, E, F, G, H, L, M, N, Q, R, T` vs `a, b, d, e, f, h, i, j, m, n, r, t, u`) trong cùng một ảnh.
  - Riêng chữ `p` thường có nét sổ trái kéo xuống dưới đường cơ sở, trong khi chữ `P` hoa đứng hoàn toàn trên đường cơ sở.

---

## 4. Lệnh Phát triển, Kiểm thử & Vận hành Chuẩn

Luôn sử dụng môi trường ảo `.venv` của dự án (`/Users/dev2k6/Workspace/tooling/ai/captcha-mku/.venv`):

```bash
# 1. Cài đặt gói ở chế độ editable để cập nhật CLI `captcha-mku`
.venv/bin/pip install -e . --no-build-isolation --no-deps

# 2. Chạy toàn bộ bộ kiểm thử tự động (Unit + FastAPI Endpoints + CLI)
.venv/bin/python -m unittest discover -s tests -v

# 3. Kiểm tra nhanh CLI giải mã ảnh mẫu
.venv/bin/captcha-mku --image captcha.png --debug
.venv/bin/captcha-mku solve --image captcha.png

# 4. Đánh giá độ chính xác trên các bộ dữ liệu chuẩn
.venv/bin/captcha-mku evaluate --manifest data/training/manifest.json --report reports/evaluation_5000.json

# 5. Kết xuất lại báo cáo HTML Light Mode từ kết quả benchmark 500 ảnh
.venv/bin/captcha-mku benchmark --report-only

# 6. Khởi chạy FastAPI Microservice cục bộ
.venv/bin/captcha-mku serve --host 0.0.0.0 --port 28186
```

---

## 5. Checklist Trước khi Hoàn tất Tác vụ (Pre-Completion Checklist)

Trước khi kết thúc bất kỳ phiên chỉnh sửa mã nguồn nào, Agent bắt buộc kiểm tra:
- [ ] Tất cả các bài test trong `.venv/bin/python -m unittest discover -s tests -v` đều **PASS 100%**.
- [ ] Không tạo ra file rác, thư mục dữ liệu tạm thời hoặc script dùng một lần chưa dọn dẹp ở thư mục gốc, `training/` hay `reports/`.
- [ ] Nếu thay đổi logic API hoặc CLI, đã cập nhật đồng bộ tài liệu tại [`SPECS.md`](file:///Users/dev2k6/Workspace/tooling/ai/captcha-mku/SPECS.md) và [`README.md`](file:///Users/dev2k6/Workspace/tooling/ai/captcha-mku/README.md).
