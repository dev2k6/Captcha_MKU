"""FastAPI entrypoint for the MKU CAPTCHA OCR Microservice."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.schemas import (
    HealthResponse,
    SolveBase64Request,
    SolveMkuRequest,
    SolveResponse,
)
from app.service import CaptchaOcrService

ocr_service = CaptchaOcrService()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Warm up the ONNX inference engine on server startup."""
    ocr_service.warmup()
    yield


app = FastAPI(
    title="MKU CAPTCHA OCR API",
    description=(
        "High-speed, 100% case-sensitive accuracy CAPTCHA recognition microservice "
        "powered by a custom fine-tuned CRNN + Temporal Residual CTC ONNX model."
    ),
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["System"])
def root() -> dict[str, str]:
    """Service overview and documentation links."""
    return {
        "service": "MKU CAPTCHA OCR Microservice",
        "version": __version__,
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check() -> HealthResponse:
    """Return server health and loaded model status."""
    return ocr_service.health()


@app.post("/api/v1/solve/base64", response_model=SolveResponse, tags=["OCR"])
def solve_base64(payload: SolveBase64Request) -> SolveResponse:
    """Recognize a CAPTCHA from a Base64 string or Data URL."""
    try:
        image = ocr_service.decode_base64(payload.image)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ocr_service.solve_image(image, builtin=payload.builtin, debug=payload.debug)


@app.post("/api/v1/solve/file", response_model=SolveResponse, tags=["OCR"])
async def solve_file(
    file: UploadFile = File(..., description="CAPTCHA image file (PNG/JPEG/WebP)"),
    builtin: bool = Form(default=False, description="Force builtin ddddocr engine"),
    debug: bool = Form(default=False, description="Include confidence details"),
) -> SolveResponse:
    """Recognize a CAPTCHA from an uploaded image file."""
    raw_bytes = await file.read()
    try:
        image = ocr_service.decode_bytes(raw_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ocr_service.solve_image(image, builtin=builtin, debug=debug)


@app.post("/api/v1/solve/mku", response_model=SolveResponse, tags=["OCR"])
def solve_mku_live(payload: SolveMkuRequest = SolveMkuRequest()) -> SolveResponse:
    """Fetch a live CAPTCHA directly from the MKU portal and solve it in one call."""
    try:
        return ocr_service.solve_live_mku(
            builtin=payload.builtin,
            debug=payload.debug,
            include_image=payload.include_image,
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"MKU upstream API error: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
