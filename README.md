# Bypass Captcha MKU

Script Python `Bypass Captcha MKU` gọi API để lấy captcha, OCR thành text và trả về kết quả ở định dạng JSON.

![Captcha mẫu](./captcha.png)

Phát triển: Thái Nguyên - Telegram: [@tn2k6](https://t.me/tn2k6)

## Mục tiêu

Dự án này tự động hóa 3 bước chính:

1. Gọi API để lấy captcha mới nhất.
2. Giải mã ảnh captcha từ chuỗi `data:image/...;base64,...`.
3. Chạy OCR trên nhiều phiên bản tiền xử lý của cùng một ảnh, sau đó chọn ra kết quả hợp lý nhất.

## Yêu cầu

- Python 3.10+
- Các thư viện trong `requirements.txt`

## Cài đặt

```bash
pip install -r requirements.txt
```

## Cách dùng

In ra JSON kết quả captcha:

```bash
python captcha_ocr.py
```

Lấy captcha và lưu ảnh ra file:

```bash
python captcha_ocr.py --save-image captcha.png
```

## Flow xử lý

Luồng xử lý tổng thể của script:

1. `fetch_captcha()`
   Gọi API `GetCaptcha` để nhận về JSON chứa `img` và `token`.
2. `decode_data_image()`
   Tách phần base64 trong trường `img`, giải mã thành bytes và nạp thành ảnh RGB.
3. `solve_captcha()`
   Chạy nhiều pipeline tiền xử lý trên cùng một ảnh gốc.
4. `run_ddddocr()`
   OCR từng ảnh đã tiền xử lý bằng `ddddocr`.
5. `vote_text()`
   Tổng hợp tất cả kết quả OCR, mở rộng các ký tự dễ nhầm lẫn, tính điểm và chọn chuỗi tốt nhất.
6. `main()`
   In ra stdout đối tượng JSON gồm `status`, `captcha`, `text`, `token` để dễ sử dụng tiếp trong script khác.

## Thuật toán OCR và bỏ phiếu

Script không chỉ OCR ảnh gốc một lần, mà áp dụng chiến lược nhiều pipeline để tăng độ ổn định:

1. Chuyển xám, tăng tương phản, lọc nhiễu và phóng to ảnh.
2. Tạo ảnh nhị phân để làm rõ foreground/background.
3. Đảo màu ảnh nhị phân để thử trường hợp OCR phân tích tốt hơn với polarity ngược lại.
4. Tăng cường biên để giữ nét chữ khi nền có nhiều nhiễu.

Mỗi pipeline trả về một kết quả OCR. Hệ thống sau đó:

1. Chuẩn hóa kết quả, chỉ giữ lại chữ và số.
2. Chấm điểm heuristic dựa trên:
   - độ dài chuỗi,
   - mức độ phù hợp với độ dài captcha kỳ vọng,
   - việc chuỗi có cả chữ và số hay không,
   - mức độ lặp lại ký tự,
   - phạt nhẹ các ký tự dễ OCR nhầm như `0`, `1`.
3. Sinh thêm các biến thể cho nhóm ký tự dễ nhầm lẫn như `0/O/o`, `1/I/l`, `5/S/s`, `2/Z/z`.
4. Cộng điểm đồng thuận cho các chuỗi được nhiều pipeline hoặc nhiều biến thể cùng ủng hộ.
5. Chọn chuỗi có tổng điểm cao nhất làm kết quả cuối.

## Tại sao cần nhiều pipeline

Captcha thường thay đổi về:

- độ đậm/nhạt của nét chữ,
- độ tương phản giữa chữ và nền,
- nhiễu hạt hoặc viền mờ,
- kiểu ký tự dễ nhầm lẫn.

Một pipeline có thể tốt với ảnh này nhưng lại kém với ảnh khác. Cách tiếp cận hiện tại giảm phụ thuộc vào một kiến trúc tiền xử lý duy nhất bằng cách để nhiều pipeline cùng "bỏ phiếu".

## Mô tả nhanh các hàm chính

- `fetch_captcha()`: Gọi API và kiểm tra payload hợp lệ.
- `decode_data_image()`: Chuyển data URL base64 thành đối tượng ảnh Pillow.
- `preprocess_gray()`: Pipeline cơ sở để làm sạch ảnh OCR.
- `preprocess_binary()`: Tạo ảnh đen-trắng rõ foreground/background.
- `preprocess_invert_binary()`: Đảo màu ảnh binary.
- `preprocess_edges()`: Tăng cường đường biên ký tự.
- `run_ddddocr()`: OCR 1 ảnh đã xử lý.
- `score_text()`: Chấm điểm chất lượng chuỗi OCR.
- `generate_text_variants()`: Tạo biến thể cho ký tự dễ nhầm.
- `vote_text()`: Hợp nhất kết quả từ nhiều pipeline.
- `solve_captcha()`: Điều phối toàn bộ quá trình OCR.

## Đầu ra

Script in ra JSON với 4 trường:

- `status`: `true` nếu OCR ra được text không rỗng, ngược lại là `false`.
- `captcha`: Chuỗi ảnh captcha gốc dạng data URL từ API.
- `text`: Kết quả OCR sau khi bỏ phiếu.
- `token`: Token captcha trả về từ API.

Ví dụ:

```json
{
  "status": true,
  "captcha": "data:image/png;base64,iVBORw0KGgo...",
  "text": "Ab3dEf",
  "token": "example-token"
}
```

## Tệp trong dự án

- `captcha_ocr.py`: Script lấy captcha, tiền xử lý ảnh, OCR và bỏ phiếu kết quả.
- `requirements.txt`: Danh sách thư viện cần cài.
- `captcha.png`: Ảnh captcha mẫu được nhúng trực tiếp trong README.
- `.gitignore`: Bỏ qua file tạm, cache và file hệ thống/editor.

## Giới hạn

- Script hiện tại sử dụng `ddddocr` làm OCR engine chính.
- OCR captcha không đảm bảo đúng 100% cho mọi ảnh.
- Heuristic scoring và mapping ký tự dễ nhầm đang được tối ưu cho captcha hiện tại; nếu cấu trúc captcha thay đổi mạnh, cần điều chỉnh lại ngưỡng và pipeline.
