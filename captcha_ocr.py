from __future__ import annotations

import argparse
import base64
import importlib
import io
import json
import re
from collections import Counter
from typing import TYPE_CHECKING, Any, Callable, Iterable

import requests

if TYPE_CHECKING:
    from PIL import Image
else:
    Image = Any


API_URL = "https://apiportal.mku.edu.vn/api/authenticate/GetCaptcha"
EXPECTED_CAPTCHA_LENGTH = 6
HEADERS = {
    "accept": "application/json, text/plain, */*",
    "accept-language": "vi;q=0.9",
    "apikey": "mkupscRBF0zT2Mqo6vMw69YMOH43IrB2RtXBS0EHit2kzvL2auxaFJBvw==",
    "clientid": "mku",
    "origin": "https://portal.mku.edu.vn",
    "priority": "u=1, i",
    "referer": "https://portal.mku.edu.vn/",
    "sec-ch-ua": '"Brave";v="147", "Not.A/Brand";v="8", "Chromium";v="147"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-site",
    "sec-gpc": "1",
    "user-agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36"
    ),
}
AMBIGUOUS_CHAR_MAP = {
    "0": "oO",
    "O": "0o",
    "o": "0O",
    "1": "ilI",
    "I": "1il",
    "i": "1lI",
    "l": "1iI",
    "5": "Ss",
    "S": "5s",
    "s": "5S",
    "2": "Zz",
    "Z": "2z",
    "z": "2Z",
}
_DDDDOCR = None


def _import_pillow_modules() -> tuple[Any, Any, Any, Any]:
    """Lazy import Pillow để script vẫn báo lỗi phụ thuộc rõ ràng khi thiếu thư viện."""
    try:
        pil_image = importlib.import_module("PIL.Image")
        pil_image_enhance = importlib.import_module("PIL.ImageEnhance")
        pil_image_filter = importlib.import_module("PIL.ImageFilter")
        pil_image_ops = importlib.import_module("PIL.ImageOps")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'Pillow'. Install it with 'py -m pip install -r requirements.txt'."
        ) from exc
    return pil_image, pil_image_enhance, pil_image_filter, pil_image_ops


def _import_ddddocr_module() -> Any:
    """Lazy import ddddocr để tránh lỗi import ngay khi chỉ đọc mã nguồn."""
    try:
        return importlib.import_module("ddddocr")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'ddddocr'. Install it with 'py -m pip install -r requirements.txt'."
        ) from exc


def get_ddddocr_instance() -> Any:
    """Khởi tạo singleton OCR engine để tái sử dụng model giữa các lần suy luận."""
    global _DDDDOCR
    if _DDDDOCR is None:
        ddddocr_module = _import_ddddocr_module()
        _DDDDOCR = ddddocr_module.DdddOcr(show_ad=False)
    return _DDDDOCR


def fetch_captcha() -> dict:
    """Gọi API captcha và trả về payload chứa ảnh dạng data URL cùng token phiên."""
    response = requests.get(API_URL, headers=HEADERS, timeout=30)
    response.raise_for_status()
    data = response.json()
    if "img" not in data or "token" not in data:
        raise ValueError(f"Phan hoi API khong hop le: {data}")
    return data


def decode_data_image(data_url: str) -> Image.Image:
    """Tách phần base64 từ data URL và chuyển thành ảnh RGB để xử lý tiếp."""
    pil_image, _, _, _ = _import_pillow_modules()
    match = re.match(r"^data:image/[^;]+;base64,(.+)$", data_url)
    if not match:
        raise ValueError("Chuoi img khong dung dinh dang data URL base64")
    image_bytes = base64.b64decode(match.group(1))
    return pil_image.open(io.BytesIO(image_bytes)).convert("RGB")


def normalize_text(text: str) -> str:
    """Loại bỏ ký tự OCR nhiễu, chỉ giữ lại chữ và số."""
    text = re.sub(r"[^A-Za-z0-9]", "", text)
    return text.strip()


def score_text(text: str, expected_min: int = EXPECTED_CAPTCHA_LENGTH, expected_max: int = EXPECTED_CAPTCHA_LENGTH) -> float:
    """Chấm điểm heuristic cho chuỗi OCR dựa trên độ dài và mức hợp lý của captcha."""
    if not text:
        return 0.0
    score = len(text) * 12.0
    if expected_min <= len(text) <= expected_max:
        score += 30.0
    if text.isalnum():
        score += 20.0
    if re.search(r"[A-Za-z]", text):
        score += 10.0
    if re.search(r"\d", text):
        score += 10.0
    unique_ratio = len(set(text)) / max(len(text), 1)
    score += unique_ratio * 10.0
    score -= text.count("0") * 2.0
    score -= text.count("1") * 2.0
    return score


def generate_text_variants(text: str, limit: int = 64) -> list[str]:
    """Sinh các biến thể ký tự dễ nhầm để tăng cơ hội khớp khi OCR sai nhẹ."""
    variants = [text]
    for index, char in enumerate(text):
        replacements = AMBIGUOUS_CHAR_MAP.get(char, "")
        if not replacements:
            continue

        # Mở rộng theo từng vị trí ký tự mơ hồ, nhưng giới hạn số biến thể
        # để tránh bùng nổ tổ hợp không cần thiết.
        current = list(variants)
        for variant in current:
            for replacement in replacements:
                candidate = variant[:index] + replacement + variant[index + 1 :]
                if candidate not in variants:
                    variants.append(candidate)
                if len(variants) >= limit:
                    return variants
    return variants


def upscale(image: Image.Image, scale: int = 3) -> Image.Image:
    """Phóng to ảnh trước OCR để nét chữ dày và dễ nhận dạng hơn."""
    pil_image, _, _, _ = _import_pillow_modules()
    return image.resize((image.width * scale, image.height * scale), pil_image.Resampling.LANCZOS)


def preprocess_gray(image: Image.Image) -> Image.Image:
    """Pipeline cơ sở: grayscale, tăng tương phản, lọc median, rồi upscale."""
    _, pil_image_enhance, pil_image_filter, pil_image_ops = _import_pillow_modules()
    gray = pil_image_ops.grayscale(image)
    gray = pil_image_enhance.Contrast(gray).enhance(2.0)
    gray = gray.filter(pil_image_filter.MedianFilter(size=3))
    return upscale(gray)


def preprocess_binary(image: Image.Image) -> Image.Image:
    """Nhị phân hóa ảnh để tách foreground/background khi captcha có nền đơn giản."""
    gray = preprocess_gray(image)
    bw = gray.point(lambda x: 255 if x > 170 else 0, mode="1")
    return bw.convert("L")


def preprocess_invert_binary(image: Image.Image) -> Image.Image:
    """Đảo màu ảnh nhị phân để bắt các trường hợp OCR nhạy với polarity của ảnh."""
    _, _, _, pil_image_ops = _import_pillow_modules()
    bw = preprocess_binary(image)
    return pil_image_ops.invert(bw)


def preprocess_edges(image: Image.Image) -> Image.Image:
    """Tăng cường biên nhằm giữ nét ký tự khi nền ảnh còn nhiều nhiễu."""
    _, _, pil_image_filter, pil_image_ops = _import_pillow_modules()
    gray = preprocess_gray(image)
    edge = gray.filter(pil_image_filter.EDGE_ENHANCE_MORE)
    edge = pil_image_ops.autocontrast(edge)
    return edge


def run_ddddocr(image: Image.Image) -> tuple[str, float]:
    """Chạy OCR trên một ảnh đã tiền xử lý và trả về text cùng điểm heuristic."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    text = get_ddddocr_instance().classification(buffer.getvalue())
    text = normalize_text(text)
    return text, score_text(text)


def vote_text(candidates: list[tuple[str, float, str]]) -> str:
    """Hợp nhất kết quả OCR từ nhiều pipeline bằng cơ chế mở rộng và bỏ phiếu có trọng số."""
    valid_candidates = [item for item in candidates if item[0]]
    if not valid_candidates:
        return ""

    expanded_candidates = []
    for text, conf, pipeline in valid_candidates:
        # Mỗi kết quả OCR được mở rộng thành các biến thể ký tự dễ nhầm,
        # sau đó cộng thêm điểm heuristic để tăng xác suất chọn đúng captcha.
        for variant in generate_text_variants(text):
            variant_score = conf + score_text(variant)
            expanded_candidates.append((variant, variant_score, pipeline))

    counts = Counter(text for text, _, _ in expanded_candidates)
    scored = []
    for text, conf, pipeline in expanded_candidates:
        # Chuỗi nào được nhiều pipeline/biến thể cùng ủng hộ sẽ nhận bonus đồng thuận.
        consensus_bonus = counts[text] * 25.0
        scored.append((text, conf + consensus_bonus, pipeline))

    best_text, _, _ = max(scored, key=lambda item: item[1])
    return best_text


def solve_captcha(image: Image.Image) -> str:
    """Điều phối toàn bộ quá trình giải captcha bằng nhiều pipeline tiền xử lý."""
    pipelines: Iterable[tuple[str, Callable[[Image.Image], Image.Image]]] = (
        ("ddddocr_gray", preprocess_gray),
        ("ddddocr_binary", preprocess_binary),
        ("ddddocr_invert_binary", preprocess_invert_binary),
        ("ddddocr_edges", preprocess_edges),
    )

    candidates = []
    for name, processor in pipelines:
        # Cùng một ảnh gốc được biến đổi theo nhiều cách để bù cho việc
        # không có một pipeline đơn lẻ nào luôn tối ưu cho mọi captcha.
        processed = processor(image)
        text, conf = run_ddddocr(processed)
        candidates.append((text, conf, name))

    return vote_text(candidates)


def main() -> None:
    """Entry point CLI: lấy captcha, OCR và in kết quả JSON ra stdout."""
    parser = argparse.ArgumentParser(description="Bypass Captcha MKU - Lay captcha tu API MKU va OCR thanh text")
    parser.add_argument("--save-image", help="Luu anh captcha goc ra file")
    args = parser.parse_args()

    payload = fetch_captcha()
    image = decode_data_image(payload["img"])

    if args.save_image:
        image.save(args.save_image)

    result_text = solve_captcha(image)
    result = {
        "status": bool(result_text),
        "captcha": payload["img"],
        "text": result_text,
        "token": payload["token"],
    }

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
