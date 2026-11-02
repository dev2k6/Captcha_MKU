import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock
from unittest.mock import patch

import numpy as np
from PIL import Image

import captcha_ocr


def probability_result(charset, timestep_chars):
    """Build deterministic (time, batch, class) probabilities for CTC tests."""
    matrix = np.full((len(timestep_chars), 1, len(charset)), 0.001)
    for timestep, char in enumerate(timestep_chars):
        matrix[timestep, 0, charset.index(char)] = 0.9
    matrix /= matrix.sum(axis=2, keepdims=True)
    return {"probabilities": matrix, "charset": charset}


class CaptchaOcrTests(unittest.TestCase):
    def test_ctc_removes_repeats_but_blank_separates_equal_glyphs(self):
        charset = ["", "A", "a", "B"]
        result = probability_result(charset, ["A", "A", "", "A"])

        text, confidence = captcha_ocr._decode_with_confidence(result)

        self.assertEqual(text, "AA")
        self.assertEqual(len(confidence), 2)

    def test_preserves_case_without_deduplicating_repeated_letters(self):
        charset = ["", "A", "a", "B"]
        result = probability_result(charset, ["a", "", "A", "B"])

        text, confidence = captcha_ocr._decode_with_confidence(result)

        self.assertEqual(text, "aAB")
        self.assertEqual(len(confidence), 3)
        self.assertGreater(confidence[0], 0.8)

    def test_probability_tensor_layouts_supported(self):
        charset = ["", "A", "B"]
        base = probability_result(charset, ["A", "", "B"])
        time_batch_class = base["probabilities"]
        batch_time_class = np.transpose(time_batch_class, (1, 0, 2))
        time_class = time_batch_class[:, 0, :]

        for matrix in (time_batch_class, batch_time_class, time_class):
            with self.subTest(shape=matrix.shape):
                text, _ = captcha_ocr._decode_with_confidence(
                    {"probabilities": matrix, "charset": charset}
                )
                self.assertEqual(text, "AB")

    def test_run_confidence_uses_peak_same_case_mass(self):
        charset = ["", "A", "a"]
        matrix = np.full((3, 1, len(charset)), 0.01)
        matrix[0, 0] = [0.01, 0.70, 0.29]
        matrix[1, 0] = [0.01, 0.55, 0.44]
        matrix[2, 0] = [0.21, 0.40, 0.39]

        text, confidence = captcha_ocr._decode_with_confidence(
            {"probabilities": matrix, "charset": charset}
        )

        self.assertEqual(text, "A")
        self.assertAlmostEqual(confidence[0], 0.70)

    def test_unsupported_glyph_is_preserved_as_invalid_position(self):
        charset = ["", "A", "中", "B", "C", "D", "E"]
        result = probability_result(charset, ["A", "中", "B", "C", "D", "E"])
        engine = type("Engine", (), {"classification": lambda _self, *_args, **_kwargs: result})()
        with patch.object(captcha_ocr, "get_ddddocr_instance", return_value=engine):
            candidate = captcha_ocr._recognize_view(Image.new("RGB", (5, 5)), "raw")

        self.assertEqual(candidate["text"], "A?BCDE")
        self.assertEqual(len(candidate["text"]), 6)
        self.assertFalse(candidate["valid_length"])

    def test_invalid_probability_rows_are_rejected(self):
        result = {
            "probabilities": np.array([[[np.nan, 0.5]]]),
            "charset": ["", "A"],
        }
        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            captcha_ocr._decode_with_confidence(result)

    def test_wrong_length_is_rejected_and_not_padded(self):
        candidate = {
            "view": "raw",
            "text": "ABCDE",
            "character_confidences": [0.99] * 5,
            "valid_length": False,
        }
        with patch.object(captcha_ocr, "_recognize_view", return_value=candidate):
            result = captcha_ocr.solve_captcha_detailed(Image.new("RGB", (5, 5)))

        self.assertEqual(result["text"], "")
        self.assertTrue(result["needs_review"])

    def test_confident_six_character_view_selected_by_character_probability(self):
        raw = {
            "view": "raw",
            "text": "AZC123",
            "character_confidences": [0.9, 0.4, 0.9, 0.9, 0.9, 0.9],
            "valid_length": True,
        }
        gray = {
            "view": "gray",
            "text": "ABC123",
            "character_confidences": [0.8, 0.95, 0.8, 0.8, 0.8, 0.8],
            "valid_length": True,
        }
        with patch.object(captcha_ocr, "_recognize_view", side_effect=[raw, gray]):
            result = captcha_ocr.solve_captcha_detailed(
                Image.new("RGB", (5, 5)), builtin=True
            )

        self.assertEqual(result["text"], "ABC123")
        self.assertTrue(result["needs_review"])

    def test_does_not_invent_ambiguous_character_replacements(self):
        raw = {
            "view": "raw",
            "text": "AZJPWR",
            "character_confidences": [0.9] * 6,
            "valid_length": True,
        }
        gray = {
            "view": "gray",
            "text": "AZJPWR",
            "character_confidences": [0.8] * 6,
            "valid_length": True,
        }
        with patch.object(captcha_ocr, "_recognize_view", side_effect=[raw, gray]):
            self.assertEqual(
                captcha_ocr.solve_captcha(Image.new("RGB", (5, 5)), builtin=True),
                "AZJPWR",
            )

    def test_offline_cli_uses_local_file_and_reports_six_character_status(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "captcha.png"
            Image.new("RGB", (10, 10), "white").save(image_path)
            output = io.StringIO()
            diagnostics = {
                "text": "ABC123",
                "confidence": [0.95] * 6,
                "needs_review": False,
                "candidates": [],
                "engine": "builtin_ddddocr",
            }
            with patch("sys.argv", ["captcha_ocr.py", "--image", str(image_path), "--debug"]):
                with patch("captcha_ocr.solve_captcha_detailed", return_value=diagnostics):
                    with patch("builtins.print", side_effect=lambda value: output.write(value)):
                        captcha_ocr.main()

            payload = json.loads(output.getvalue())
            self.assertTrue(payload["status"])
            self.assertEqual(payload["text"], "ABC123")
            self.assertIsNone(payload["token"])
            self.assertTrue(payload["ocr"]["needs_review"] is False)
            self.assertTrue(payload["captcha"].startswith("data:image/png;base64,"))

    def test_trained_image_preprocessing_matches_training_layout(self):
        image = Image.new("RGB", (20, 10), (255, 0, 0))

        value = captcha_ocr._trained_image_input(image)

        self.assertEqual(value.shape, (1, 1, 64, 128))
        self.assertEqual(value.dtype, np.float32)
        self.assertTrue(np.isfinite(value).all())
        self.assertTrue(((value >= 0) & (value <= 1)).all())

    def test_custom_onnx_loader_validates_metadata_and_caches_session(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model.onnx"
            model_path.write_bytes(b"fake onnx")
            charset = ["", *"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"]
            (Path(temp_dir) / "charsets.json").write_text(
                json.dumps({"charset": charset}), encoding="utf-8"
            )
            fake_session = types.SimpleNamespace(
                get_inputs=lambda: [types.SimpleNamespace(name="input1")],
                get_outputs=lambda: [types.SimpleNamespace(name="logits")],
            )
            ort = types.SimpleNamespace(
                InferenceSession=lambda *_args, **_kwargs: fake_session
            )
            captcha_ocr._ONNX_MODELS.clear()
            with patch.object(captcha_ocr.importlib, "import_module", return_value=ort):
                first = captcha_ocr._load_custom_onnx_model(model_path)
                second = captcha_ocr._load_custom_onnx_model(model_path)

        self.assertIs(first, second)
        self.assertEqual(first["kind"], "trained_onnx")
        self.assertEqual(first["input_name"], "input1")
        self.assertEqual(first["output_name"], "logits")

    def test_custom_onnx_loader_reloads_when_artifact_changes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model.onnx"
            metadata_path = Path(temp_dir) / "charsets.json"
            model_path.write_bytes(b"fake onnx one")
            charset = ["", *"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"]
            metadata_path.write_text(json.dumps({"charset": charset}), encoding="utf-8")
            sessions = []

            def make_session(*_args, **_kwargs):
                session = types.SimpleNamespace(
                    get_inputs=lambda: [types.SimpleNamespace(name="input1")],
                    get_outputs=lambda: [types.SimpleNamespace(name="logits")],
                )
                sessions.append(session)
                return session

            ort = types.SimpleNamespace(InferenceSession=make_session)
            captcha_ocr._ONNX_MODELS.clear()
            with patch.object(captcha_ocr.importlib, "import_module", return_value=ort):
                first = captcha_ocr._load_custom_onnx_model(model_path)
                self.assertIs(first, captcha_ocr._load_custom_onnx_model(model_path))
                model_path.write_bytes(b"fake onnx replacement with new size")
                second = captcha_ocr._load_custom_onnx_model(model_path)

        self.assertIsNot(first, second)
        self.assertEqual(len(sessions), 2)

    def test_custom_onnx_inference_softmax_and_ctc_decode(self):
        charset = ["", *"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"]
        text = "a1B2c3"
        logits = np.full((6, 1, len(charset)), -5.0, dtype=np.float32)
        for timestep, char in enumerate(text):
            logits[timestep, 0, charset.index(char)] = 5.0
        captured = {}

        class Session:
            def run(self, outputs, inputs):
                captured["outputs"] = outputs
                captured["inputs"] = inputs
                return [logits]

        engine = {
            "kind": "trained_onnx",
            "session": Session(),
            "input_name": "input1",
            "output_name": "logits",
            "charset": charset,
        }

        result = captcha_ocr._trained_probabilities(Image.new("RGB", (200, 60)), engine)
        decoded, confidence = captcha_ocr._decode_with_confidence(result)

        self.assertEqual(decoded, text)
        self.assertEqual(len(confidence), 6)
        self.assertAlmostEqual(float(result["probabilities"].sum(axis=2).mean()), 1.0, places=6)
        self.assertEqual(captured["outputs"], ["logits"])
        self.assertEqual(captured["inputs"]["input1"].shape, (1, 1, 64, 213))

    def test_explicit_missing_model_fails_instead_of_falling_back(self):
        with self.assertRaises(FileNotFoundError):
            captcha_ocr._select_engine("/does/not/exist/model.onnx")

    def test_builtin_flag_forces_ddddocr_even_if_default_model_exists(self):
        instance = object()
        with patch.object(captcha_ocr, "get_ddddocr_instance", return_value=instance):
            engine = captcha_ocr._select_engine(builtin=True)

        self.assertEqual(engine["kind"], "builtin_ddddocr")
        self.assertIs(engine["instance"], instance)

    def test_fetch_captcha_handles_successful_request(self):
        mock_response = mock.MagicMock()
        mock_response.json.return_value = {
            "img": "data:image/png;base64,mock",
            "token": "tok123"
        }
        with patch("captcha_ocr.requests.get", return_value=mock_response):
            res = captcha_ocr.fetch_captcha()
            self.assertEqual(res["token"], "tok123")
            self.assertEqual(res["img"], "data:image/png;base64,mock")

    def test_fetch_captcha_raises_on_invalid_structure(self):
        mock_response = mock.MagicMock()
        mock_response.json.return_value = {"error": "bad request"}
        with patch("captcha_ocr.requests.get", return_value=mock_response):
            with self.assertRaises(ValueError):
                captcha_ocr.fetch_captcha()

    def test_cli_subcommands_dispatching(self):
        with patch("training.generate_report.build_light_html") as mock_rep:
            captcha_ocr.main(["benchmark", "--report-only"])
            mock_rep.assert_called_once()

        with patch("training.run_benchmark.main") as mock_bench:
            captcha_ocr.main(["benchmark"])
            mock_bench.assert_called_once()

        with patch("evaluate_ocr.main") as mock_eval:
            captcha_ocr.main(["evaluate", "--manifest", "fake.json"])
            mock_eval.assert_called_once_with(["--manifest", "fake.json"])

        with patch("training.train.main") as mock_train:
            captcha_ocr.main(["train", "--manifest", "fake.json"])
            mock_train.assert_called_once_with(["--manifest", "fake.json"])


if __name__ == "__main__":
    unittest.main()
