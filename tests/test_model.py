"""Unit tests for training/model.py (ResidualCTCHead and helper functions)."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from training import model


class ModelArchitectureTests(unittest.TestCase):
    def test_residual_ctc_head_forward(self) -> None:
        head = model.ResidualCTCHead(hidden_dim=128, dropout=0.1)
        # Expected input shape: [time, batch, 1024]
        x = torch.randn(15, 2, 1024)
        out = head(x)
        self.assertEqual(out.shape, (15, 2, len(model.CHARSET)))
        self.assertEqual(len(model.CHARSET), 63)

    def test_residual_property_accessor(self) -> None:
        head = model.ResidualCTCHead(hidden_dim=64, dropout=0.0)
        res_modules = head.residual
        self.assertEqual(len(res_modules), 4)

    def test_feature_encoder_requires_valid_dimensions(self) -> None:
        encoder = object.__new__(model.CaptchaFeatureEncoder)
        # Pass invalid dimensions
        bad_shape = np.zeros((1, 3, 64, 100), dtype=np.float32)
        with self.assertRaises(ValueError):
            encoder.extract(bad_shape)


if __name__ == "__main__":
    unittest.main()
