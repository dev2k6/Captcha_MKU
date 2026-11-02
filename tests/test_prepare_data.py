import base64
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from training.prepare_data import collect, make_contact_sheets, propose_labels


def png_data(color):
    image = Image.new("RGB", (200, 60), color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def api_payload(image_bytes):
    return {
        "img": "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii"),
        "token": "must-not-be-saved",
    }


class PrepareDataTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name) / "dataset"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_collect_deduplicates_and_resume_does_not_fetch_when_complete(self):
        red = png_data("red")
        blue = png_data("blue")
        green = png_data("green")
        responses = iter((api_payload(red), api_payload(red), api_payload(blue), api_payload(green)))
        result = collect(
            count=3,
            directory=self.root,
            delay=0,
            fetch_payload=lambda: next(responses),
            progress_fn=lambda _message: None,
        )

        self.assertEqual(result["total_unique"], 3)
        self.assertEqual(result["added_this_run"], 3)
        self.assertEqual(result["api_attempts"], 4)
        manifest_path = self.root / "manifest.json"
        raw_manifest = manifest_path.read_text(encoding="utf-8")
        manifest = json.loads(raw_manifest)
        self.assertNotIn("must-not-be-saved", raw_manifest)
        self.assertEqual(len({sample["sha256"] for sample in manifest["samples"]}), 3)
        self.assertTrue(all(sample["label"] is None for sample in manifest["samples"]))
        self.assertEqual(result["split_counts"], {"train": 2, "validation": 0, "test": 1})

        resumed = collect(
            count=3,
            directory=self.root,
            delay=0,
            fetch_payload=lambda: self.fail("complete collection must not issue a request"),
            progress_fn=lambda _message: None,
        )
        self.assertEqual(resumed["added_this_run"], 0)
        self.assertEqual(resumed["api_attempts"], 0)

    def test_proposals_are_separate_and_confirmed_labels_are_not_overwritten(self):
        image_bytes = png_data("white")
        collect(
            count=1,
            directory=self.root,
            delay=0,
            fetch_payload=lambda: api_payload(image_bytes),
            progress_fn=lambda _message: None,
        )
        manifest_path = self.root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["samples"][0]["label"] = "HUMAN1"
        manifest["samples"].append(
            {
                "id": "000002",
                "image": "images/000001.png",
                "sha256": manifest["samples"][0]["sha256"],
                "label": None,
            }
        )
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        class FakeOcr:
            def __init__(self, **_kwargs):
                pass

            def classification(self, _image_bytes, probability=False):
                self_probability = probability
                return {"text": "raw123", "confidence": 0.91 if self_probability else 0.2}

        fake_module = types.SimpleNamespace(DdddOcr=FakeOcr)
        with patch.dict(sys.modules, {"ddddocr": fake_module}):
            result = propose_labels(self.root)

        after = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(result["samples_updated"], 1)
        self.assertTrue(result["confirmed_labels_preserved"])
        self.assertEqual(after["samples"][0]["label"], "HUMAN1")
        self.assertNotIn("proposed", after["samples"][0])
        self.assertIsNone(after["samples"][1]["label"])
        self.assertEqual(after["samples"][1]["proposed"]["raw"]["text"], "raw123")
        self.assertEqual(after["samples"][1]["proposed"]["raw"]["confidence"], 0.91)

    def test_contact_sheet_is_two_columns_and_scales_images_to_review_size(self):
        collect(
            count=1,
            directory=self.root,
            delay=0,
            fetch_payload=lambda: api_payload(png_data("black")),
            progress_fn=lambda _message: None,
        )

        paths = make_contact_sheets(self.root)

        self.assertEqual(len(paths), 1)
        with Image.open(paths[0]) as sheet:
            self.assertEqual(sheet.size, (840, 1500))


if __name__ == "__main__":
    unittest.main()
