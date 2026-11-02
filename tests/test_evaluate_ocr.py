import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import evaluate_ocr


class EvaluateOcrTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        (self.root / "images").mkdir()
        for name in ("1.png", "2.png", "3.png"):
            Image.new("RGB", (8, 8), "white").save(self.root / "images" / name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_manifest(self, samples):
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps({"samples": samples}), encoding="utf-8")
        return manifest

    def test_uppercases_answers_and_counts_matches_against_baseline(self):
        manifest = self.write_manifest(
            [
                {"image": "images/1.png", "expected": "AaB123", "baseline": "aab123"},
                {"image": "images/2.png", "expected": "Q9x5zA", "baseline": "Q9x5zA"},
            ]
        )
        outputs = iter(("aAb123", "Q9X5zA"))

        report = evaluate_ocr.evaluate_manifest(manifest, lambda _image: next(outputs), case_insensitive=True)

        self.assertEqual(report["labelled_samples"], 2)
        self.assertEqual(report["baseline_matches"], 2)
        self.assertEqual(report["new_matches"], 2)
        self.assertEqual(report["baseline_accuracy"], 1.0)
        self.assertEqual(report["accuracy"], 1.0)
        self.assertEqual(report["samples"][0]["prediction"], "AAB123")
        self.assertEqual(report["samples"][0]["expected"], "AAB123")

    def test_unlabelled_samples_are_reported_but_excluded_from_accuracy(self):
        manifest = self.write_manifest(
            [
                {"image": "images/1.png", "expected": "ABC123", "baseline": "ABC123"},
                {"image": "images/2.png", "expected": None, "baseline": "uncertain"},
            ]
        )
        outputs = iter(("ABC123", "maybe"))

        report = evaluate_ocr.evaluate_manifest(manifest, lambda _image: next(outputs))

        self.assertEqual(report["labelled_samples"], 1)
        self.assertEqual(report["unlabelled_samples"], 1)
        self.assertEqual(report["new_matches"], 1)
        self.assertEqual(report["accuracy"], 1.0)
        self.assertIsNone(report["samples"][1]["match"])
        self.assertIsNone(report["samples"][1]["baseline_match"])

    def test_case_and_repeated_characters_survive_normalization(self):
        self.assertEqual(evaluate_ocr.normalize_answer("aAbB"), "aAbB")
        self.assertEqual(evaluate_ocr.normalize_answer("aAbB", True), "AABB")

    def test_default_comparison_is_case_sensitive(self):
        manifest = self.write_manifest(
            [{"image": "images/1.png", "expected": "Abc123", "baseline": "ABC123"}]
        )
        report = evaluate_ocr.evaluate_manifest(manifest, lambda _image: "abc123")
        self.assertEqual(report["new_matches"], 0)
        self.assertEqual(report["baseline_matches"], 0)

    def test_cli_writes_report_without_network_access(self):
        manifest = self.write_manifest(
            [{"image": "images/1.png", "expected": "ABC123", "baseline": "abc123"}]
        )
        report_path = self.root / "reports" / "evaluation.json"
        with patch("captcha_ocr.solve_captcha", return_value="ABC123"):
            with patch("sys.argv", ["evaluate_ocr.py", "--manifest", str(manifest), "--report", str(report_path)]):
                with patch("builtins.print"):
                    evaluate_ocr.main()

        written = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(written["accuracy"], 1.0)
        self.assertEqual(written["samples"][0]["prediction"], "ABC123")


if __name__ == "__main__":
    unittest.main()
