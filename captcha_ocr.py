from __future__ import annotations

import argparse
import base64
import importlib
import io
import json
import re
from typing import TYPE_CHECKING, Any

import requests

if TYPE_CHECKING:
    from PIL import Image
else:
    Image = Any


API_URL = "https://apiportal.mku.edu.vn/api/authenticate/GetCaptcha"
EXPECTED_CAPTCHA_LENGTH = 6
TRAINED_CHARSET = ["", *"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"]
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
_DDDDOCR = None
_ONNX_MODELS: dict[str, dict[str, Any]] = {}


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


def _load_custom_onnx_model(model_path: str | Any) -> dict[str, Any]:
    """Load a trained captcha ONNX model and its adjacent charset metadata."""
    from pathlib import Path

    path = Path(model_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"OCR model file does not exist: {path}")
    cache_key = str(path)
    metadata_path = path.with_name("charsets.json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"OCR model metadata file does not exist: {metadata_path}")
    signature = (
        path.stat().st_mtime_ns,
        path.stat().st_size,
        metadata_path.stat().st_mtime_ns,
        metadata_path.stat().st_size,
    )
    cached = _ONNX_MODELS.get(cache_key)
    if cached is not None and cached["signature"] == signature:
        return cached

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    charset = metadata.get("charset") if isinstance(metadata, dict) else None
    if (
        not isinstance(charset, list)
        or charset != TRAINED_CHARSET
    ):
        raise ValueError("Trained OCR model must use the 63-class case-sensitive ASCII charset")

    try:
        ort = importlib.import_module("onnxruntime")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'onnxruntime'. Install it with 'pip install onnxruntime'."
        ) from exc
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if not inputs or not outputs:
        raise ValueError("Trained OCR model must have at least one input and output")
    engine = {
        "kind": "trained_onnx",
        "path": str(path),
        "session": session,
        "input_name": inputs[0].name,
        "output_name": outputs[0].name,
        "charset": charset,
        "signature": signature,
    }
    _ONNX_MODELS[cache_key] = engine
    return engine


def _select_engine(model_path: str | Any | None = None, *, builtin: bool = False) -> dict[str, Any]:
    """Select the trained model when available, otherwise the built-in OCR."""
    from pathlib import Path

    if model_path is not None and builtin:
        raise ValueError("Choose either --model or --builtin, not both")
    if builtin:
        return {"kind": "builtin_ddddocr", "instance": get_ddddocr_instance()}
    if model_path is not None:
        return _load_custom_onnx_model(model_path)

    automatic_model = Path(__file__).resolve().parent / "models" / "captcha" / "model.onnx"
    if automatic_model.is_file():
        # A present default artifact is authoritative; load errors should be visible.
        return _load_custom_onnx_model(automatic_model)
    return {"kind": "builtin_ddddocr", "instance": get_ddddocr_instance()}


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
    """Chuẩn hóa chuỗi captcha: chỉ giữ chữ/số ASCII và giữ nguyên hoa/thường."""
    return re.sub(r"[^A-Za-z0-9]", "", text)


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


def _image_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _probability_rows(result: dict[str, Any]) -> tuple[Any, list[str]]:
    """Đưa đầu ra ddddocr về ma trận (thời gian, ký tự) và bộ ký tự."""
    np = importlib.import_module("numpy")
    probabilities = np.asarray(result["probabilities"], dtype=float)
    if probabilities.ndim == 3:
        if probabilities.shape[0] == 1:  # batch, time, classes
            probabilities = probabilities[0]
        elif probabilities.shape[1] == 1:  # time, batch, classes
            probabilities = probabilities[:, 0, :]
        else:
            raise ValueError("Unexpected OCR probability batch dimensions")
    if probabilities.ndim != 2:
        raise ValueError("OCR probabilities must have two dimensions after batch removal")
    if probabilities.shape[0] == 0 or probabilities.shape[1] == 0:
        raise ValueError("OCR probabilities must contain time steps and classes")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("OCR probabilities must be finite and non-negative")
    if (probabilities.sum(axis=1) <= 0).any():
        raise ValueError("Each OCR probability row must have positive mass")
    return probabilities, list(result["charset"])


def _decode_with_confidence(result: dict[str, Any]) -> tuple[str, list[float]]:
    """Greedy CTC decode and collect the selected case-sensitive class probability."""
    np = importlib.import_module("numpy")
    probabilities, charset = _probability_rows(result)
    if not len(charset) or probabilities.shape[1] != len(charset):
        raise ValueError("OCR probabilities do not match the model charset")

    indices = np.argmax(probabilities, axis=1)
    decoded: list[str] = []
    confidences: list[float] = []
    cursor = 0
    while cursor < len(indices):
        index = int(indices[cursor])
        run_end = cursor + 1
        while run_end < len(indices) and int(indices[run_end]) == index:
            run_end += 1

        if index != 0:
            char = charset[index]
            if len(char) == 1 and char.isascii() and char.isalnum():
                decoded.append(char)
                confidence_by_timestep = probabilities[cursor:run_end, index]
                confidences.append(float(confidence_by_timestep.max()))
            else:
                # Preserve the position as invalid so unsupported classes cannot
                # disappear and make a shortened sequence look like a valid answer.
                decoded.append("?")
                confidences.append(
                    float(probabilities[cursor:run_end, index].max())
                )
        cursor = run_end
    return "".join(decoded), confidences


def _trained_image_input(image: Image.Image) -> Any:
    """Preprocess as the training encoder did: RGB resize, grayscale, [0, 1]."""
    np = importlib.import_module("numpy")
    pil_image, _, _, _ = _import_pillow_modules()
    rgb = image.convert("RGB")
    width = max(1, int(rgb.width * 64 / rgb.height))
    resized = rgb.resize((width, 64), pil_image.Resampling.LANCZOS).convert("L")
    pixels = np.asarray(resized, dtype=np.float32) / 255.0
    return pixels[None, None, :, :].copy()


def _trained_probabilities(image: Image.Image, engine: dict[str, Any]) -> dict[str, Any]:
    np = importlib.import_module("numpy")
    logits = engine["session"].run(
        [engine["output_name"]],
        {engine["input_name"]: _trained_image_input(image)},
    )[0]
    if logits.ndim != 3 or logits.shape[1] != 1 or logits.shape[2] != len(engine["charset"]):
        raise ValueError(f"Unexpected trained OCR output shape: {logits.shape}")
    if not np.isfinite(logits).all():
        raise ValueError("Trained OCR model returned non-finite logits")
    shifted = logits - logits.max(axis=2, keepdims=True)
    exponentials = np.exp(shifted)
    probabilities = exponentials / exponentials.sum(axis=2, keepdims=True)
    return {"probabilities": probabilities, "charset": engine["charset"]}


def _recognize_view(image: Image.Image, name: str, engine: dict[str, Any] | None = None) -> dict[str, Any]:
    engine = engine or {"kind": "builtin_ddddocr", "instance": get_ddddocr_instance()}
    if engine["kind"] == "trained_onnx":
        result = _trained_probabilities(image, engine)
    else:
        result = engine["instance"].classification(_image_bytes(image), probability=True)
    text, confidences = _decode_with_confidence(result)
    return {
        "view": name,
        "text": text,
        "character_confidences": confidences,
        "valid_length": len(text) == EXPECTED_CAPTCHA_LENGTH
        and all(char.isascii() and char.isalnum() for char in text),
    }


def solve_captcha_detailed(
    image: Image.Image,
    model_path: str | Any | None = None,
    *,
    builtin: bool = False,
) -> dict[str, Any]:
    """Recognize a captcha and return diagnostics from the selected OCR model."""
    engine = _select_engine(model_path, builtin=builtin)
    candidates = [_recognize_view(image, "raw", engine)]
    if engine["kind"] == "builtin_ddddocr":
        candidates.append(_recognize_view(preprocess_gray(image), "gray", engine))
    elif not candidates[0]["valid_length"]:
        pil_image, _, _, _ = _import_pillow_modules()
        for scale, tag in ((1.05, "scale_up"), (0.95, "scale_down")):
            scaled_w = max(1, int(image.width * scale))
            scaled_img = image.resize((scaled_w, image.height), pil_image.Resampling.LANCZOS)
            view_res = _recognize_view(scaled_img, tag, engine)
            candidates.append(view_res)
            if view_res["valid_length"]:
                break
    eligible = [candidate for candidate in candidates if candidate["valid_length"]]

    if not eligible:
        text = ""
        confidence = []
    else:
        text = "".join(
            max(
                (candidate for candidate in eligible),
                key=lambda candidate: candidate["character_confidences"][position],
            )["text"][position]
            for position in range(EXPECTED_CAPTCHA_LENGTH)
        )
        confidence = [
            max(
                candidate["character_confidences"][position]
                for candidate in eligible
            )
            for position in range(EXPECTED_CAPTCHA_LENGTH)
        ]

    disagreement = len({candidate["text"] for candidate in candidates}) > 1
    return {
        "text": normalize_text(text),
        "confidence": confidence,
        "needs_review": (
            len(text) != EXPECTED_CAPTCHA_LENGTH
            or disagreement
            or not confidence
            or min(confidence) < 0.8
        ),
        "candidates": candidates,
        "engine": engine["kind"],
    }


def solve_captcha(
    image: Image.Image,
    model_path: str | Any | None = None,
    *,
    builtin: bool = False,
) -> str:
    """Return six case-sensitive characters, or empty if no valid result exists."""
    return solve_captcha_detailed(image, model_path, builtin=builtin)["text"]


def _add_solve_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--save-image", help="Luu anh captcha goc ra file")
    parser.add_argument("--image", help="Doc anh captcha tu file thay vi goi API")
    parser.add_argument("--debug", action="store_true", help="Them diem tin cay OCR de chan doan")
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--model", help="Duong dan den model ONNX OCR da train")
    model_group.add_argument("--builtin", action="store_true", help="Ep dung model OCR ddddocr mac dinh")


def _run_solve_cli(args: argparse.Namespace) -> None:
    if args.image:
        _, _, _, _ = _import_pillow_modules()
        pil_image = importlib.import_module("PIL.Image")
        image = pil_image.open(args.image).convert("RGB")
        buffer = _image_bytes(image)
        image_data = "data:image/png;base64," + base64.b64encode(buffer).decode("ascii")
        token = None
    else:
        payload = fetch_captcha()
        image_data = payload["img"]
        image = decode_data_image(image_data)
        token = payload["token"]

    if args.save_image:
        image.save(args.save_image)

    diagnostics = solve_captcha_detailed(image, args.model, builtin=args.builtin)
    result_text = diagnostics["text"]
    result = {
        "status": len(result_text) == EXPECTED_CAPTCHA_LENGTH,
        "captcha": image_data,
        "text": result_text,
        "token": token,
    }
    if args.debug:
        result["ocr"] = {
            "confidence": diagnostics["confidence"],
            "needs_review": diagnostics["needs_review"],
            "candidates": diagnostics["candidates"],
            "engine": diagnostics["engine"],
        }

    print(json.dumps(result, ensure_ascii=False))


def main(argv: list[str] | None = None) -> None:
    """Unified CLI entry point: solve (default), serve, benchmark, evaluate, train."""
    import sys

    args_list = list(sys.argv[1:] if argv is None else argv)
    subcommands = {"solve", "serve", "benchmark", "evaluate", "train"}

    if args_list and args_list[0] in subcommands:
        cmd = args_list[0]
        rest = args_list[1:]
        if cmd == "solve":
            parser = argparse.ArgumentParser(prog="captcha-mku solve", description="Giai ma anh CAPTCHA MKU")
            _add_solve_arguments(parser)
            _run_solve_cli(parser.parse_args(rest))
            return
        if cmd == "serve":
            parser = argparse.ArgumentParser(prog="captcha-mku serve", description="Khoi chay FastAPI Microservice")
            parser.add_argument("--host", default="0.0.0.0", help="Host bind address (default: 0.0.0.0)")
            parser.add_argument("--port", type=int, default=28186, help="Port number (default: 28186)")
            parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
            serve_args = parser.parse_args(rest)
            uvicorn = importlib.import_module("uvicorn")
            uvicorn.run("app.main:app", host=serve_args.host, port=serve_args.port, reload=serve_args.reload)
            return
        if cmd == "benchmark":
            parser = argparse.ArgumentParser(prog="captcha-mku benchmark", description="Chay benchmark 500 anh va xuat HTML")
            parser.add_argument("--report-only", action="store_true", help="Chi xuat lai file HTML tu JSON co san")
            bench_args = parser.parse_args(rest)
            if bench_args.report_only:
                from training.generate_report import build_light_html
                build_light_html()
            else:
                from training.run_benchmark import main as run_bench_main
                run_bench_main()
            return
        if cmd == "evaluate":
            from evaluate_ocr import main as eval_main
            eval_main(rest)
            return
        if cmd == "train":
            from training.train import main as train_main
            train_main(rest)
            return

    parser = argparse.ArgumentParser(
        description=(
            "MKU CAPTCHA OCR CLI & Microservice. "
            "Subcommands: solve (default), serve, benchmark, evaluate, train."
        )
    )
    _add_solve_arguments(parser)
    _run_solve_cli(parser.parse_args(args_list))


if __name__ == "__main__":
    main()
