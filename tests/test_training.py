import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

from training.model import CHARSET, ctc_greedy_decode
from training.train import FeatureDataset, FeatureExample, collate_features, evaluate, load_manifest


class _FixedHead(torch.nn.Module):
    def __init__(self, predicted_text: str) -> None:
        super().__init__()
        self.predicted_text = predicted_text

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = torch.full(
            (features.shape[0], features.shape[1], len(CHARSET)),
            -10.0,
            dtype=features.dtype,
            device=features.device,
        )
        for time, char in enumerate(self.predicted_text):
            logits[time, :, CHARSET.index(char)] = 10.0
        return logits


class TrainingCaseSensitivityTests(unittest.TestCase):
    def test_charset_has_distinct_ascii_case_classes(self):
        self.assertEqual(len(CHARSET), 63)
        self.assertNotEqual(CHARSET.index("A"), CHARSET.index("a"))
        self.assertEqual(CHARSET[0], "")

    def test_ctc_decoder_preserves_original_case(self):
        indices = [CHARSET.index(char) for char in "AaB3"]
        logits = torch.full((len(indices), 1, len(CHARSET)), -10.0)
        for time, index in enumerate(indices):
            logits[time, 0, index] = 10.0
        self.assertEqual(ctc_greedy_decode(logits), ["AaB3"])

    def test_manifest_keeps_mixed_case_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "images").mkdir()
            Image.new("RGB", (10, 10), "white").save(root / "images" / "one.png")
            manifest = {
                "samples": [
                    {
                        "id": "one",
                        "image": "images/one.png",
                        "label": "aBc123",
                        "split": "train",
                    },
                    {
                        "id": "two",
                        "image": "images/one.png",
                        "label": "AbC123",
                        "split": "validation",
                        "sha256": "unique-second-hash",
                    },
                    {
                        "id": "three",
                        "image": "images/one.png",
                        "label": "ABC123",
                        "split": "test",
                        "sha256": "unique-third-hash",
                    },
                ]
            }
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")

            samples = load_manifest(path)

        self.assertEqual(samples[0].label, "aBc123")
        self.assertEqual(samples[1].label, "AbC123")

    def test_evaluation_reports_case_sensitive_and_case_insensitive_metrics(self):
        example = FeatureExample("sample", "aBc123", torch.zeros((6, 1024)).numpy())
        loader = DataLoader(
            FeatureDataset([example]), batch_size=1, collate_fn=collate_features
        )

        metrics = evaluate(_FixedHead("ABC123"), loader, torch.device("cpu"))

        exact = metrics["all_labels"]
        self.assertEqual(exact["exact_correct"], 0)
        self.assertEqual(exact["exact_accuracy"], 0.0)
        self.assertEqual(exact["character_errors"], 2)
        self.assertEqual(exact["case_insensitive_exact_correct"], 1)
        self.assertEqual(exact["case_insensitive_character_errors"], 0)
        self.assertEqual(metrics["clear_labels"]["samples"], 1)


if __name__ == "__main__":
    unittest.main()
