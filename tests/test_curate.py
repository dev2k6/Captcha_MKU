import json
import tempfile
import unittest
from pathlib import Path

from training.curate import curate


class CurationTests(unittest.TestCase):
    def test_preserves_case_split_and_uncertainty(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"samples": [{
                "id": "000001", "image": "a.png", "sha256": "abc", "split": "test"
            }]}))
            review = root / "review.json"
            review.write_text(json.dumps({"000001": {
                "label": "ABC123", "label_case": "aBc123", "uncertain": True
            }}))
            curate(manifest, [review])
            sample = json.loads(manifest.read_text())["samples"][0]
            self.assertEqual(sample["label"], "aBc123")
            self.assertEqual(sample["split"], "test")
            self.assertTrue(sample["label_uncertain"])

    def test_missing_labels_fail_without_rewriting_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"samples": [{"id": "1"}]}))
            original = manifest.read_bytes()
            review = root / "review.json"
            review.write_text('{"1": {"label": null}}')
            with self.assertRaises(ValueError):
                curate(manifest, [review])
            self.assertEqual(manifest.read_bytes(), original)

    def test_unknown_case_override_id_fails_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"samples": [{"id": "000001"}]}))
            review = root / "review.json"
            review.write_text(json.dumps({"000001": {"label": "ABC123"}}))
            override = root / "case.json"
            override.write_text(json.dumps({"labels": {"000002": "aBc123"}}))

            with self.assertRaisesRegex(ValueError, "no corresponding reviewed label"):
                curate(manifest, [review], case_overrides_paths=[override])

    def test_case_override_shape_is_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"samples": [{"id": "000001"}]}))
            review = root / "review.json"
            review.write_text(json.dumps({"000001": {"label": "ABC123"}}))
            override = root / "case.json"
            override.write_text(json.dumps({"labels": {"000001": ["aBc123"]}}))

            with self.assertRaisesRegex(ValueError, "must be a string or object"):
                curate(manifest, [review], case_overrides_paths=[override])

    def test_review_file_shape_is_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"samples": [{"id": "000001"}]}))
            review = root / "review.json"
            review.write_text(json.dumps([{"id": "000001", "label": "ABC123"}]))

            with self.assertRaisesRegex(ValueError, "object of sample IDs"):
                curate(manifest, [review])


if __name__ == "__main__":
    unittest.main()
