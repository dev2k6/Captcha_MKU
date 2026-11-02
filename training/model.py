"""Frozen ddddocr ONNX encoder and trainable case-sensitive CTC head."""

from __future__ import annotations

import string
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


CHARSET = ("", *string.digits, *string.ascii_uppercase, *string.ascii_lowercase)
IMAGE_HEIGHT = 64
FEATURE_NODE = "366"
PRETRAINED_MODEL = "common_old.onnx"


class CaptchaFeatureEncoder:
    """Expose the penultimate 1024-wide features from ddddocr's ONNX graph."""

    def __init__(self, model_path: str | Path | None = None, intra_op_threads: int = 2) -> None:
        import onnx
        import onnxruntime as ort

        if model_path is None:
            import ddddocr

            model_path = Path(ddddocr.__file__).parent / PRETRAINED_MODEL
        self.model_path = Path(model_path).expanduser().resolve()
        graph = onnx.load(str(self.model_path))
        if not any(FEATURE_NODE in node.output for node in graph.graph.node):
            raise ValueError(f"Encoder graph does not contain feature tensor {FEATURE_NODE!r}")
        feature_output = onnx.helper.make_tensor_value_info(
            FEATURE_NODE,
            onnx.TensorProto.FLOAT,
            ["time", "batch", 1024],
        )
        del graph.graph.output[:]
        graph.graph.output.extend([feature_output])
        # Prune the original classifier tail so feature extraction doesn't run
        # the 8,210-way output layer for every augmentation.
        feature_node_index = next(
            i for i, node in enumerate(graph.graph.node) if FEATURE_NODE in node.output
        )
        del graph.graph.node[feature_node_index + 1 :]
        used_names = {name for node in graph.graph.node for name in node.input}
        del graph.graph.initializer[:]
        for initializer in onnx.load(str(self.model_path)).graph.initializer:
            if initializer.name in used_names:
                graph.graph.initializer.append(initializer)
        self._temporary_graph = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
        self._temporary_graph.close()
        self._temporary_path = Path(self._temporary_graph.name)
        onnx.save(graph, str(self._temporary_path))
        options = ort.SessionOptions()
        options.intra_op_num_threads = intra_op_threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(self._temporary_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Run one [1,1,64,width] float32 image and return [time,1024]."""
        if image.ndim != 4 or image.shape[:3] != (1, 1, IMAGE_HEIGHT):
            raise ValueError(f"Expected image batch [1,1,{IMAGE_HEIGHT},width], got {image.shape}")
        features = self.session.run([FEATURE_NODE], {self.input_name: image})[0]
        if features.ndim != 3 or features.shape[1:] != (1, 1024):
            raise ValueError(f"Unexpected ddddocr feature shape {features.shape}")
        if not np.isfinite(features).all():
            raise ValueError("ddddocr encoder returned non-finite features")
        return np.asarray(features[:, 0, :], dtype=np.float32)

    def close(self) -> None:
        self.session = None
        self._temporary_path.unlink(missing_ok=True)


class ResidualCTCHead(nn.Module):
    """Pretrained linear character head adapted by a temporal + pointwise residual block."""

    def __init__(self, hidden_dim: int = 256, dropout: float = 0.2) -> None:
        super().__init__()
        self.pretrained = nn.Linear(1024, len(CHARSET))
        self.in_proj = nn.Linear(1024, hidden_dim)
        self.temporal_conv = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
            groups=4,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim, len(CHARSET))
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    @property
    def residual(self) -> nn.ModuleList:
        """Parameter group accessor for the trainable residual adapter."""
        return nn.ModuleList([self.in_proj, self.temporal_conv, self.norm, self.out_proj])

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        base_logits = self.pretrained(features)
        hidden = F.gelu(self.in_proj(features))
        # Apply 1D temporal convolution across the time dimension [time, batch, hidden] -> [batch, hidden, time]
        temporal = self.temporal_conv(hidden.permute(1, 2, 0)).permute(2, 0, 1)
        hidden = self.norm(hidden + F.gelu(temporal))
        hidden = self.dropout(hidden)
        return base_logits + self.out_proj(hidden)


def initialize_head_from_ddddocr(head: ResidualCTCHead, model_path: str | Path | None = None) -> dict[str, Any]:
    """Map ddddocr's finite quantized output layer to case-sensitive ASCII.

    The original PyTorch checkpoint available from upstream was downloaded
    from its official repository and checked with ``weights_only=True``; all
    floating-point weights were NaN. It is intentionally not used. This path
    instead dequantizes finite ``common_old.onnx`` classifier weights.
    """
    import ddddocr
    import onnx
    from onnx import numpy_helper

    if model_path is None:
        model_path = Path(ddddocr.__file__).parent / PRETRAINED_MODEL
    graph = onnx.load(str(model_path))
    initializers = {item.name: item for item in graph.graph.initializer}
    expected = {"135_quantized", "135_scale", "135_zero_point", "136"}
    if not expected.issubset(initializers):
        raise ValueError(f"Quantized ddddocr classifier is missing tensors: {sorted(expected - set(initializers))}")
    quantized = numpy_helper.to_array(initializers["135_quantized"]).astype(np.float32)
    weight_scale = float(numpy_helper.to_array(initializers["135_scale"]))
    weight_zero = float(numpy_helper.to_array(initializers["135_zero_point"]))
    bias = numpy_helper.to_array(initializers["136"]).astype(np.float32)
    weight = (quantized - weight_zero) * weight_scale
    if weight.shape != (1024, len(bias)) or not np.isfinite(weight).all() or not np.isfinite(bias).all():
        raise ValueError(f"Invalid ONNX classifier tensors: weight={weight.shape} bias={bias.shape}")

    source_charset = ddddocr.DdddOcr(show_ad=False).ocr_engine.charset_manager.get_charset()
    if len(source_charset) != len(bias) or source_charset[0] != "":
        raise ValueError(
            f"Installed ddddocr charset does not match the ONNX classifier: "
            f"charset={len(source_charset)} head={len(bias)}"
        )
    class_indices: dict[str, list[int]] = {char: [] for char in CHARSET[1:]}
    blank_indices: list[int] = []
    for index, char in enumerate(source_charset):
        if char == "":
            blank_indices.append(index)
        elif len(char) == 1 and char in string.ascii_letters + string.digits:
            class_indices[char].append(index)
    invalid = {char: indices for char, indices in class_indices.items() if len(indices) != 1}
    if invalid or len(blank_indices) != 1:
        raise ValueError(f"Could not map ddddocr classifier charset; invalid={invalid}, blank={blank_indices}")

    target_weight = np.empty((len(CHARSET), 1024), dtype=np.float32)
    target_bias = np.empty((len(CHARSET),), dtype=np.float32)
    target_weight[0] = weight[:, blank_indices[0]]
    target_bias[0] = bias[blank_indices[0]]
    for target_index, char in enumerate(CHARSET[1:], start=1):
        source_index = class_indices[char][0]
        target_weight[target_index] = weight[:, source_index]
        target_bias[target_index] = bias[source_index]
    with torch.no_grad():
        head.pretrained.weight.copy_(torch.from_numpy(target_weight))
        head.pretrained.bias.copy_(torch.from_numpy(target_bias))
    return {
        "encoder": str(Path(model_path).resolve()),
        "source_classes": len(source_charset),
        "target_classes_including_blank": len(CHARSET),
        "case_preserving": True,
        "head_initialization": "dequantized finite common_old.onnx MatMulInteger weights and bias",
    }


def ctc_greedy_decode(logits: torch.Tensor, lengths: torch.Tensor | None = None) -> list[str]:
    """Decode a batch of [time,batch,classes] CTC logits."""
    if logits.ndim != 3:
        raise ValueError("Expected logits with shape [time, batch, classes]")
    rows = logits.argmax(dim=2).transpose(0, 1).detach().cpu().tolist()
    valid_lengths = lengths.tolist() if lengths is not None else [len(row) for row in rows]
    decoded: list[str] = []
    for row, length in zip(rows, valid_lengths):
        result: list[str] = []
        previous = -1
        for index in row[: int(length)]:
            if index != previous and index != 0:
                result.append(CHARSET[index])
            previous = index
        decoded.append("".join(result))
    return decoded
