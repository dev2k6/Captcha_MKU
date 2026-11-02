"""Pydantic request and response schemas for the MKU CAPTCHA OCR API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health and model readiness status."""

    status: str = Field(default="ok", description="Service health status")
    version: str = Field(description="API product version")
    engine: str = Field(description="Active OCR engine ('onnx' or 'ddddocr')")
    model_path: str | None = Field(default=None, description="Resolved ONNX model path")
    model_exists: bool = Field(description="Whether custom ONNX model artifact is present")
    expected_length: int = Field(default=6, description="Expected CAPTCHA character length")
    uptime_seconds: float = Field(description="Server uptime in seconds")


class SolveBase64Request(BaseModel):
    """Request payload for solving a base64-encoded CAPTCHA image."""

    image: str = Field(
        ...,
        min_length=16,
        description="Base64-encoded image string or data URL (e.g. 'data:image/png;base64,...')",
    )
    builtin: bool = Field(
        default=False,
        description="Force builtin ddddocr engine instead of the fine-tuned ONNX model",
    )
    debug: bool = Field(
        default=False,
        description="Include per-character confidence scores and candidate views",
    )


class SolveMkuRequest(BaseModel):
    """Request payload for fetching and solving a live CAPTCHA from MKU portal."""

    builtin: bool = Field(
        default=False,
        description="Force builtin ddddocr engine instead of the fine-tuned ONNX model",
    )
    include_image: bool = Field(
        default=False,
        description="Include the raw base64 data URL from MKU API in the response",
    )
    debug: bool = Field(
        default=False,
        description="Include per-character confidence scores and candidate views",
    )


class SolveResponse(BaseModel):
    """Standardized OCR prediction response."""

    status: str = Field(description="'success' when 6 valid characters are recognized, else 'invalid_length'")
    captcha: str = Field(description="Recognized 6-character case-sensitive CAPTCHA text (alias of text)")
    text: str = Field(description="Recognized 6-character case-sensitive CAPTCHA text")
    valid_length: bool = Field(description="True if recognized text is exactly 6 characters")
    latency_ms: float = Field(description="OCR inference latency in milliseconds")
    engine: str = Field(description="Engine used for inference ('onnx' or 'ddddocr')")
    token: str | None = Field(default=None, description="MKU session token when fetched from live API")
    image: str | None = Field(default=None, description="Original base64 image when include_image=True")
    confidence: list[float] | None = Field(
        default=None,
        description="Per-character confidence scores [0..1] (included when debug=True)",
    )
    mean_confidence: float | None = Field(
        default=None,
        description="Mean character confidence score [0..1] (included when debug=True)",
    )
    needs_review: bool | None = Field(
        default=None,
        description="True if low confidence or candidate disagreement detected (included when debug=True)",
    )
    candidates: list[dict[str, Any]] | None = Field(
        default=None,
        description="Candidate view predictions (included when debug=True)",
    )
