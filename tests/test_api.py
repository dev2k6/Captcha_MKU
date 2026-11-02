"""Unit tests for the FastAPI CAPTCHA OCR Microservice (app.main)."""

from __future__ import annotations

import base64
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from app.main import app

SAMPLE_IMAGE = Path(__file__).resolve().parent.parent / "captcha.png"


class ApiMicroserviceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)
        cls.sample_bytes = SAMPLE_IMAGE.read_bytes()
        cls.sample_b64 = base64.b64encode(cls.sample_bytes).decode("utf-8")
        cls.sample_data_url = f"data:image/png;base64,{cls.sample_b64}"

    def test_root_returns_service_links(self) -> None:
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["service"], "MKU CAPTCHA OCR Microservice")
        self.assertEqual(body["docs"], "/docs")

    def test_health_endpoint_reports_ready_model(self) -> None:
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["status"], "ok")
        self.assertIn(body["engine"], ("onnx", "ddddocr"))
        self.assertEqual(body["expected_length"], 6)

    def test_solve_base64_accepts_raw_and_data_url_with_debug(self) -> None:
        resp = self.client.post(
            "/api/v1/solve/base64",
            json={"image": self.sample_data_url, "debug": True},
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["status"], "success")
        self.assertEqual(len(body["text"]), 6)
        self.assertEqual(body["captcha"], body["text"])
        self.assertTrue(body["valid_length"])
        self.assertIsInstance(body["confidence"], list)
        self.assertEqual(len(body["confidence"]), 6)
        self.assertIsNotNone(body["mean_confidence"])

    def test_solve_base64_rejects_corrupt_payload(self) -> None:
        resp = self.client.post(
            "/api/v1/solve/base64",
            json={"image": "not-valid-base64-image-bytes!!!"},
        )
        self.assertEqual(resp.status_code, 400)

    def test_solve_file_upload_returns_six_characters(self) -> None:
        resp = self.client.post(
            "/api/v1/solve/file",
            files={"file": ("captcha.png", self.sample_bytes, "image/png")},
            data={"debug": "false"},
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["status"], "success")
        self.assertEqual(len(body["text"]), 6)
        self.assertIsNone(body["confidence"])

    def test_solve_mku_endpoint_uses_upstream_fetcher(self) -> None:
        with mock.patch(
            "captcha_ocr.fetch_captcha",
            return_value={"img": self.sample_data_url, "token": "mock-token-123"},
        ):
            resp = self.client.post(
                "/api/v1/solve/mku",
                json={"include_image": True, "debug": True},
            )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["status"], "success")
        self.assertEqual(body["token"], "mock-token-123")
        self.assertEqual(body["image"], self.sample_data_url)
        self.assertEqual(len(body["text"]), 6)

    def test_unified_cli_solve_subcommand_and_default_mode(self) -> None:
        import io
        import json
        from contextlib import redirect_stdout
        import captcha_ocr

        buf_default = io.StringIO()
        with redirect_stdout(buf_default):
            captcha_ocr.main(["--image", str(SAMPLE_IMAGE), "--debug"])
        out_default = json.loads(buf_default.getvalue())

        buf_sub = io.StringIO()
        with redirect_stdout(buf_sub):
            captcha_ocr.main(["solve", "--image", str(SAMPLE_IMAGE), "--debug"])
        out_sub = json.loads(buf_sub.getvalue())

        self.assertTrue(out_default["status"])
        self.assertEqual(len(out_default["text"]), 6)
        self.assertEqual(out_default["text"], out_sub["text"])
        self.assertIn("ocr", out_sub)

    def test_unified_cli_serve_dispatches_to_uvicorn(self) -> None:
        import captcha_ocr

        with mock.patch("uvicorn.run") as mock_run:
            captcha_ocr.main(["serve", "--host", "127.0.0.1", "--port", "9001"])
            mock_run.assert_called_once_with(
                "app.main:app", host="127.0.0.1", port=9001, reload=False
            )

    def test_solve_file_rejects_empty_or_corrupt_upload(self) -> None:
        resp = self.client.post(
            "/api/v1/solve/file",
            files={"file": ("corrupt.png", b"not-an-image", "image/png")},
        )
        self.assertEqual(resp.status_code, 400)

    def test_solve_mku_handles_upstream_request_failure(self) -> None:
        import requests
        with mock.patch("captcha_ocr.fetch_captcha", side_effect=requests.RequestException("MKU down")):
            resp = self.client.post("/api/v1/solve/mku")
        self.assertEqual(resp.status_code, 502)

    def test_solve_mku_handles_decode_error(self) -> None:
        with mock.patch("captcha_ocr.fetch_captcha", return_value={"img": "invalid-data", "token": "t"}):
            resp = self.client.post("/api/v1/solve/mku")
        self.assertEqual(resp.status_code, 502)

    def test_app_lifespan_warms_up_service(self) -> None:
        import asyncio
        from app.main import lifespan
        async def run_lifespan():
            async with lifespan(app):
                pass
        asyncio.run(run_lifespan())


if __name__ == "__main__":
    unittest.main()
