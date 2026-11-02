"""Unit tests for training/generate_report.py (HTML dashboard generator)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from training import generate_report


class GenerateReportTests(unittest.TestCase):
    def test_build_light_html_generates_valid_html_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            mock_json = tmp_path / "benchmark_500_overview.json"
            mock_html = tmp_path / "benchmark_500_overview.html"
            mock_images = tmp_path / "images"
            mock_images.mkdir(parents=True)

            source_json = Path(__file__).resolve().parent.parent / "reports" / "benchmark_500_overview.json"
            data = json.loads(source_json.read_text(encoding="utf-8"))
            mock_json.write_text(json.dumps(data), encoding="utf-8")

            # Create dummy sample images
            from PIL import Image
            img = Image.new("RGB", (100, 35), color=(200, 240, 200))
            for i in range(1, 25):
                img.save(mock_images / f"{i:06d}.png")

            with mock.patch("training.generate_report.PROJECT_ROOT", tmp_path), \
                 mock.patch("captcha_ocr.solve_captcha_detailed", return_value={"text": "KdQRZl", "confidence": [0.99]*6}), \
                 mock.patch("ddddocr.DdddOcr.classification", return_value="kdqrzl"):
                generate_report.build_light_html(
                    summary_path=mock_json,
                    images_dir=mock_images,
                    output_html=mock_html,
                )

            self.assertTrue(mock_html.exists())
            content = mock_html.read_text(encoding="utf-8")
            self.assertIn("So Sánh Hiệu Quả Nhận Diện CAPTCHA", content)


if __name__ == "__main__":
    unittest.main()
