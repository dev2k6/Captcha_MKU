# Mục lục Tài liệu Dự án (Documentation Index)

Chào mừng bạn đến với trung tâm tài liệu kỹ thuật và nghiệp vụ của dự án **MKU CAPTCHA OCR**.

---

## 📚 Danh mục Tài liệu

1. **[Tài liệu Nghiệp vụ & Khái niệm Hệ thống (`business_requirements.md`)](./business_requirements.md)**
   - Bối cảnh bài toán xác thực CAPTCHA tại cổng trường Đại học Cửu Long.
   - Các quy tắc hình học ký tự: không có số `0`, phân biệt `9`/`q`/`g`, quy tắc chiều cao hoa/thường.
   - Cam kết mức dịch vụ (SLA) về độ chính xác 100% và độ trễ ~5 ms.

2. **[Hướng dẫn Tích hợp & Đặc tả API (`api_integration_guide.md`)](./api_integration_guide.md)**
   - Hướng dẫn gọi HTTP REST API Microservice (`/health`, `/api/v1/solve/*`).
   - Hướng dẫn nhúng thư viện Python SDK trong các script/bot.
   - Hướng dẫn sử dụng Unified CLI `captcha-mku`.

3. **[Hướng dẫn Triển khai & Vận hành (`deployment_and_ops.md`)](./deployment_and_ops.md)**
   - Triển khai sản xuất bằng Docker và Docker Compose.
   - Cấu hình dịch vụ Systemd trên Linux server.
   - Giám sát sức khỏe, SLA độ trễ và xử lý sự cố thường gặp.

4. **[Đặc tả Kỹ thuật Toàn diện (`../SPECS.md`)](../SPECS.md)**
   - Kiến trúc mạng CRNN + ResidualCTCHead, công thức tiền xử lý ảnh LANCZOS, giải thuật CTC decoding.

5. **[Sổ tay Vận hành Dành cho AI Coding Agents (`../AGENTS.md`)](../AGENTS.md)**
   - Các quy tắc bất biến (Core Invariants), quy chuẩn chỉnh sửa code và checklist nghiệm thu dành cho AI.
