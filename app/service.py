"""Core OCR inference service layer for the MKU CAPTCHA Microservice."""

from __future__ import annotations

import base64
import io
import time
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

import captcha_ocr
from app import __version__
from app.schemas import HealthResponse, SolveResponse


class CaptchaOcrService:
    """Thread-safe OCR inference service with automatic model warmup."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        self.started_at = time.time()
        default_path = Path(captcha_ocr.__file__).resolve().parent / "models" / "captcha" / "model.onnx"
        self.model_path = (
            Path(model_path).expanduser().resolve()
            if model_path is not None
            else default_path
        )

    def warmup(self) -> None:
        """Pre-load the ONNX session into memory so the first request has zero cold-start latency."""
        dummy = Image.new("RGB", (200, 60), color=(255, 255, 255))
        if self.model_path.exists():
            captcha_ocr.solve_captcha_detailed(dummy, self.model_path, builtin=False)
        else:
            captcha_ocr.solve_captcha_detailed(dummy, None, builtin=True)

    def health(self) -> HealthResponse:
        """Return service health and active model information."""
        exists = self.model_path.exists()
        return HealthResponse(
            status="ok",
            version=__version__,
            engine="onnx" if exists else "ddddocr",
            model_path=str(self.model_path) if exists else None,
            model_exists=exists,
            expected_length=captcha_ocr.EXPECTED_CAPTCHA_LENGTH,
            uptime_seconds=round(time.time() - self.started_at, 2),
        )

    def solve_image(
        self,
        image: Image.Image,
        *,
        builtin: bool = False,
        debug: bool = False,
        token: str | None = None,
        raw_image: str | None = None,
    ) -> SolveResponse:
        """Run OCR on a decoded PIL Image and format the standardized response."""
        target_model = None if builtin else (self.model_path if self.model_path.exists() else None)
        t0 = time.perf_counter()
        result = captcha_ocr.solve_captcha_detailed(image, target_model, builtin=builtin)
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        text = result["text"]
        valid = len(text) == captcha_ocr.EXPECTED_CAPTCHA_LENGTH
        conf_list = result.get("confidence") or []
        mean_conf = round(sum(conf_list) / len(conf_list), 4) if conf_list else None

        return SolveResponse(
            status="success" if valid else "invalid_length",
            captcha=text,
            text=text,
            valid_length=valid,
            latency_ms=latency_ms,
            engine=result["engine"],
            token=token,
            image=raw_image,
            confidence=[round(c, 4) for c in conf_list] if debug else None,
            mean_confidence=mean_conf if debug else None,
            needs_review=result["needs_review"] if debug else None,
            candidates=result["candidates"] if debug else None,
        )

    def decode_bytes(self, raw_bytes: bytes) -> Image.Image:
        """Validate and decode image bytes into a PIL Image."""
        if not raw_bytes:
            raise ValueError("Uploaded image payload is empty")
        try:
            image = Image.open(io.BytesIO(raw_bytes))
            image.load()
            return image
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError(f"Invalid or unsupported image data: {exc}") from exc

    def decode_base64(self, b64_or_data_url: str) -> Image.Image:
        """Decode either a raw base64 string or a data:image/...;base64,... URL."""
        cleaned = b64_or_data_url.strip()
        if "," in cleaned and cleaned.lower().startswith("data:image"):
            cleaned = cleaned.split(",", 1)[1].strip()
        try:
            raw_bytes = base64.b64decode(cleaned, validate=True)
        except Exception as exc:
            raise ValueError(f"Invalid base64 image string: {exc}") from exc
        return self.decode_bytes(raw_bytes)

    def solve_live_mku(
        self,
        *,
        builtin: bool = False,
        debug: bool = False,
        include_image: bool = False,
    ) -> SolveResponse:
        """Fetch a fresh CAPTCHA from MKU portal and solve it immediately."""
        payload = captcha_ocr.fetch_captcha()
        img_b64 = payload["img"]
        token = payload["token"]
        image = captcha_ocr.decode_data_image(img_b64)
        return self.solve_image(
            image,
            builtin=builtin,
            debug=debug,
            token=token,
            raw_image=img_b64 if include_image else None,
        )
