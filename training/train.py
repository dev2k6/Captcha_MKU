"""Fine-tune a case-sensitive CTC head over ddddocr's frozen ONNX encoder.

Example:
    python -m training.train --manifest data/captcha500/manifest.json \
        --output models/captcha --epochs 60

Training features are cached in RAM for the duration of the run. Four views
(the original plus three mild augmentations) are extracted per train sample by
default; validation and test images each use one unaugmented view. The test split is evaluated only after
selecting the checkpoint with validation data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import string
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset

from training.model import (
    CHARSET,
    IMAGE_HEIGHT,
    CaptchaFeatureEncoder,
    ResidualCTCHead,
    ctc_greedy_decode,
    initialize_head_from_ddddocr,
)


TARGET_CHARS = set(string.ascii_letters + string.digits)
SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class Sample:
    sample_id: str
    image_path: Path
    label: str
    split: str
    label_uncertain: bool


@dataclass(frozen=True)
class FeatureExample:
    sample_id: str
    label: str
    features: np.ndarray
    label_uncertain: bool = False


def load_manifest(path: Path) -> list[Sample]:
    """Load and validate all reviewed labels, split assignments, and paths."""
    root = path.resolve().parent
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("samples"), list):
        raise ValueError("Manifest must be a JSON object with a samples list")
    samples: list[Sample] = []
    ids: set[str] = set()
    hashes: set[str] = set()
    for item in manifest["samples"]:
        if not isinstance(item, dict):
            raise ValueError("Each manifest sample must be a JSON object")
        sample_id = str(item.get("id", ""))
        if not sample_id or sample_id in ids:
            raise ValueError(f"Missing or duplicate sample id: {sample_id!r}")
        ids.add(sample_id)
        label = item.get("label")
        if not isinstance(label, str) or not label:
            raise ValueError(f"Sample {sample_id} has no reviewed label; training refuses unlabeled samples")
        if len(label) != 6 or any(char not in TARGET_CHARS for char in label):
            raise ValueError(
                f"Sample {sample_id} label must be exactly six ASCII letters/digits: {label!r}"
            )
        split = item.get("split")
        if split not in SPLITS:
            raise ValueError(f"Sample {sample_id} has invalid split {split!r}")
        relative_path = item.get("image")
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError(f"Sample {sample_id} has no image path")
        image_path = (root / relative_path).resolve()
        if root not in image_path.parents:
            raise ValueError(f"Sample {sample_id} image path escapes the manifest directory")
        if not image_path.is_file():
            raise ValueError(f"Sample {sample_id} image is missing: {image_path}")
        digest = item.get("sha256")
        if digest:
            if digest in hashes:
                raise ValueError(f"Duplicate image hash found at sample {sample_id}")
            hashes.add(digest)
        samples.append(Sample(sample_id, image_path, label, split, bool(item.get("label_uncertain", False))))

    counts = {split: sum(sample.split == split for sample in samples) for split in SPLITS}
    if any(counts[split] == 0 for split in SPLITS):
        raise ValueError(f"Manifest must contain train, validation, and test samples; found {counts}")
    return samples


def _border_median(image: Image.Image) -> tuple[int, ...]:
    pixels = np.asarray(image, dtype=np.uint8)
    border = np.concatenate((pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]))
    median = np.median(border, axis=0)
    return tuple(int(value) for value in np.atleast_1d(median))


def _augment(image: Image.Image, rng: random.Random) -> Image.Image:
    """Small affine and contrast/brightness changes that preserve the label."""
    background = _border_median(image)
    image = image.rotate(
        rng.uniform(-1.0, 1.0),
        resample=Image.Resampling.BICUBIC,
        fillcolor=background[0] if len(background) == 1 else background,
    )
    if rng.random() < 0.65:
        shear = rng.uniform(-0.04, 0.04)
        image = image.transform(
            image.size,
            Image.Transform.AFFINE,
            (1.0, shear, rng.uniform(-1.5, 1.5), 0.0, 1.0, rng.uniform(-1.0, 1.0)),
            resample=Image.Resampling.BICUBIC,
            fillcolor=background[0] if len(background) == 1 else background,
        )
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.92, 1.08))
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.94, 1.06))
    return image


def _image_input(image: Image.Image) -> np.ndarray:
    # Match ddddocr: resize the decoded RGB image first, then grayscale.
    image = image.convert("RGB")
    width = max(1, int(image.width * IMAGE_HEIGHT / image.height))
    image = image.resize((width, IMAGE_HEIGHT), Image.Resampling.LANCZOS)
    image = image.convert("L")
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    return pixels[None, None, :, :].copy()


def extract_feature_dataset(
    samples: list[Sample],
    encoder: CaptchaFeatureEncoder,
    augmentations: int,
    seed: int,
    train_all: bool = False,
) -> dict[str, list[FeatureExample]]:
    """Cache frozen features; augment train split or all samples if train_all is True."""
    features: dict[str, list[FeatureExample]] = {split: [] for split in SPLITS}
    for sample_number, sample in enumerate(samples):
        with Image.open(sample.image_path) as source:
            original = source.convert("RGB")
        is_train = (sample.split == "train") or train_all
        views = [original]
        if is_train:
            for view_index in range(1, augmentations):
                rng = random.Random(seed ^ (sample_number * 1_000_003) ^ view_index)
                views.append(_augment(original.copy(), rng))
        for view_index, view in enumerate(views):
            feature = encoder.extract(_image_input(view))
            sample_id = sample.sample_id if view_index == 0 else f"{sample.sample_id}:aug{view_index}"
            example = FeatureExample(sample_id, sample.label, feature, sample.label_uncertain)
            if train_all:
                features["train"].append(example)
                if view_index == 0 and sample.split in ("validation", "test"):
                    features[sample.split].append(example)
            else:
                features[sample.split].append(example)
        if (sample_number + 1) % 500 == 0 or (sample_number + 1) == len(samples):
            print(f"Cached frozen features for {sample_number + 1}/{len(samples)} images")
    return features


class FeatureDataset(Dataset[FeatureExample]):
    def __init__(self, examples: list[FeatureExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> FeatureExample:
        return self.examples[index]


def collate_features(batch: list[FeatureExample]) -> dict[str, Any]:
    lengths = torch.tensor([item.features.shape[0] for item in batch], dtype=torch.long)
    max_time = int(lengths.max())
    features = torch.zeros((max_time, len(batch), 1024), dtype=torch.float32)
    for index, item in enumerate(batch):
        sequence = torch.from_numpy(item.features)
        features[: sequence.shape[0], index, :] = sequence
    targets = torch.tensor(
        [CHARSET.index(char) for item in batch for char in item.label], dtype=torch.long
    )
    return {
        "features": features,
        "input_lengths": lengths,
        "targets": targets,
        "target_lengths": torch.full((len(batch),), 6, dtype=torch.long),
        "labels": [item.label for item in batch],
        "ids": [item.sample_id for item in batch],
        "uncertain": [item.label_uncertain for item in batch],
    }


def edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


@torch.no_grad()
def evaluate(
    head: ResidualCTCHead,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    head.eval()
    def empty_metrics() -> dict[str, Any]:
        return {
            "exact_correct": 0,
            "character_errors": 0,
            "case_insensitive_exact_correct": 0,
            "case_insensitive_character_errors": 0,
            "characters": 0,
            "predictions": [],
        }

    metrics: dict[str, dict[str, Any]] = {
        "all_labels": empty_metrics(),
        "clear_labels": empty_metrics(),
    }
    for batch in loader:
        features = batch["features"].to(device)
        logits = head(features)
        decoded = ctc_greedy_decode(logits, batch["input_lengths"])
        for sample_id, label, prediction, uncertain in zip(
            batch["ids"], batch["labels"], decoded, batch["uncertain"]
        ):
            record = {"id": sample_id, "label": label, "prediction": prediction}
            for key in ("all_labels", "clear_labels"):
                if key == "clear_labels" and uncertain:
                    continue
                metrics[key]["exact_correct"] += prediction == label
                metrics[key]["character_errors"] += edit_distance(label, prediction)
                metrics[key]["case_insensitive_exact_correct"] += prediction.upper() == label.upper()
                metrics[key]["case_insensitive_character_errors"] += edit_distance(
                    label.upper(), prediction.upper()
                )
                metrics[key]["characters"] += len(label)
                metrics[key]["predictions"].append(record)
    for values in metrics.values():
        count = len(values["predictions"])
        values["samples"] = count
        values["exact_accuracy"] = values["exact_correct"] / count if count else 0.0
        values["character_error_rate"] = (
            values["character_errors"] / values["characters"] if values["characters"] else 0.0
        )
        values["case_insensitive_exact_accuracy"] = (
            values["case_insensitive_exact_correct"] / count if count else 0.0
        )
        values["case_insensitive_character_error_rate"] = (
            values["case_insensitive_character_errors"] / values["characters"]
            if values["characters"]
            else 0.0
        )
    return metrics


def _choose_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS was requested, but this PyTorch build has no available MPS device")
        return device
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_configuration(args: argparse.Namespace, manifest_path: Path, encoder_path: Path) -> dict[str, Any]:
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "encoder_path": str(encoder_path),
        "encoder_sha256": _sha256_file(encoder_path),
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "augmentations_per_train_sample": args.augmentations,
        "uncertain_label_weight": args.uncertain_weight,
        "pretrained_head_learning_rate": args.pretrained_lr,
        "residual_head_learning_rate": args.residual_lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "requested_device": args.device,
        "train_all": getattr(args, "train_all", False),
        "skip_onnx": args.skip_onnx,
        "charset": "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
        "preprocessing": {
            "decode_mode": "RGB",
            "resize_order": "RGB resize with Pillow LANCZOS, then convert to L",
            "height": IMAGE_HEIGHT,
            "width": "max(1, int(original_width * 64 / original_height))",
            "normalization": "uint8 grayscale / 255.0",
        },
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.manifest.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    samples = load_manifest(manifest_path)
    split_samples = {split: [sample for sample in samples if sample.split == split] for split in SPLITS}
    print("Split sizes:", {key: len(value) for key, value in split_samples.items()})
    if args.augmentations < 1:
        raise ValueError("--augmentations must be at least 1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(args.seed)
    device = _choose_device(args.device)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_outputs = [
        output_dir / name
        for name in ("best.pt", "training.jsonl", "report.json", "model.onnx", "charsets.json")
        if (output_dir / name).exists()
    ]
    if existing_outputs and not getattr(args, "overwrite", False):
        raise FileExistsError(
            "Output already contains training artifacts; choose a new output directory: "
            + ", ".join(str(path) for path in existing_outputs)
        )
    if getattr(args, "overwrite", False):
        for path in existing_outputs:
            path.unlink(missing_ok=True)

    if args.encoder:
        encoder_path = args.encoder.expanduser().resolve()
    else:
        import ddddocr

        encoder_path = Path(ddddocr.__file__).parent / "common_old.onnx"
    run_config = _run_configuration(args, manifest_path, encoder_path)
    _save_json(output_dir / "run_config.json", run_config)
    shutil.copy2(manifest_path, output_dir / "dataset_manifest.json")

    encoder = CaptchaFeatureEncoder(encoder_path)
    try:
        head = ResidualCTCHead()
        head_initialization = initialize_head_from_ddddocr(head, args.encoder)
        feature_sets = extract_feature_dataset(
            samples, encoder, args.augmentations, args.seed, train_all=getattr(args, "train_all", False)
        )
    finally:
        encoder.close()

    loaders = {
        split: DataLoader(
            FeatureDataset(feature_sets[split]),
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=0,
            collate_fn=collate_features,
        )
        for split in SPLITS
    }
    head.to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": head.pretrained.parameters(), "lr": args.pretrained_lr, "weight_decay": args.weight_decay * 0.1},
            {"params": head.residual.parameters(), "lr": args.residual_lr, "weight_decay": args.weight_decay},
        ],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs), eta_min=1e-6
    )
    best_cer = float("inf")
    best_accuracy = -1.0
    best_all_cer = float("inf")
    best_train_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    checkpoint_path = output_dir / "best.pt"
    log_path = output_dir / "training.jsonl"

    for epoch in range(1, args.epochs + 1):
        head.train()
        total_loss = 0.0
        batch_count = 0
        started = time.time()
        for batch in loaders["train"]:
            features = batch["features"].to(device)
            if args.feature_noise > 0:
                features = F.dropout(features, p=0.1, training=True) + args.feature_noise * torch.randn_like(features)
            logits = head(features)
            log_probs = logits.float().log_softmax(2)
            sample_losses = F.ctc_loss(
                log_probs.cpu(),
                batch["targets"],
                batch["input_lengths"],
                batch["target_lengths"],
                blank=0,
                reduction="none",
                zero_infinity=True,
            )
            sample_losses = sample_losses / batch["target_lengths"].float()
            sample_weights = torch.where(
                torch.tensor(batch["uncertain"], dtype=torch.bool),
                args.uncertain_weight,
                1.0,
            )
            loss = (sample_losses * sample_weights).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite CTC loss in epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                head.parameters(), max_norm=5.0, error_if_nonfinite=True
            )
            optimizer.step()
            total_loss += float(loss.detach())
            batch_count += 1
        scheduler.step()

        validation_metrics = evaluate(head, loaders["validation"], device)
        validation = validation_metrics["clear_labels"]
        validation_all = validation_metrics["all_labels"]
        if not validation["samples"]:
            raise ValueError("Validation split contains no clear labels for checkpoint selection")
        record = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, batch_count),
            "validation_exact_accuracy": validation["exact_accuracy"],
            "validation_character_error_rate": validation["character_error_rate"],
            "validation_all_exact_accuracy": validation_all["exact_accuracy"],
            "validation_all_character_error_rate": validation_all["character_error_rate"],
            "seconds": round(time.time() - started, 2),
        }
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(record) + "\n")
        print(
            f"epoch {epoch:02d}/{args.epochs}: loss={record['train_loss']:.4f} "
            f"val_clear={validation['exact_correct']}/{validation['samples']} "
            f"val_all={validation_all['exact_correct']}/{validation_all['samples']} "
            f"val_CER={validation['character_error_rate']:.4f} ({record['seconds']:.1f}s)"
        )
        current_cer = validation["character_error_rate"]
        current_accuracy = validation["exact_accuracy"]
        current_all_cer = validation_all["character_error_rate"]
        current_train_loss = record["train_loss"]
        is_better = (
            current_cer < best_cer
            or (current_cer == best_cer and current_accuracy > best_accuracy)
            or (current_cer == best_cer and current_accuracy == best_accuracy and current_all_cer < best_all_cer)
            or (
                current_cer == best_cer
                and current_accuracy == best_accuracy
                and current_all_cer == best_all_cer
                and current_train_loss < best_train_loss
            )
        )
        if is_better:
            best_cer = current_cer
            best_accuracy = current_accuracy
            best_all_cer = current_all_cer
            best_train_loss = current_train_loss
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "head": head.state_dict(),
                    "epoch": epoch,
                    "validation": {key: value for key, value in validation.items() if key != "predictions"},
                    "charset": CHARSET,
                    "seed": args.seed,
                    "encoder": str(Path(encoder.model_path).resolve()),
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after {args.patience} validation epochs without improvement.")
                break

    best_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    head.load_state_dict(best_checkpoint["head"])
    head.to(device)
    # Touch the held-out split exactly once after validation checkpoint selection.
    validation_report = evaluate(head, loaders["validation"], device)
    test_report = evaluate(head, loaders["test"], device)
    report = {
        "manifest": str(manifest_path),
        "sample_counts": {split: len(split_samples[split]) for split in SPLITS},
        "feature_views": {split: len(feature_sets[split]) for split in SPLITS},
        "training_uncertain_samples_weighted": sum(
            sample.split == "train" and sample.label_uncertain for sample in samples
        ),
        "best_epoch": best_epoch,
        "best_validation_character_error_rate": best_cer,
        "head_initialization": head_initialization,
        "device": str(device),
        "charset": "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
        "pretrained_pytorch_checkpoint": {
            "path": "data/pretrained/ddddocr-crnn.tar",
            "used": False,
            "reason": "Downloaded upstream checkpoint contains NaNs in all floating-point model weights.",
        },
        "metrics": {"validation": validation_report, "test": test_report},
        "training_protocol": {
            "train_samples_used": len(split_samples["train"]),
            "training_views_per_sample": args.augmentations,
            "uncertain_label_loss_weight": args.uncertain_weight,
            "validation_selection": "clear-label exact-case CER, tie-broken by clear-label exact accuracy",
            "test_split_access": "one evaluation after restoring best validation checkpoint",
        },
        "provenance": run_config,
    }
    _save_json(output_dir / "report.json", report)
    if not args.skip_onnx:
        export_onnx(head.cpu().eval(), args.encoder, output_dir)
    print(
        f"Best epoch {best_epoch}; held-out test exact="
        f"{test_report['all_labels']['exact_correct']}/{test_report['all_labels']['samples']} "
        f"CER={test_report['all_labels']['character_error_rate']:.4f}; "
        f"clear={test_report['clear_labels']['exact_correct']}/{test_report['clear_labels']['samples']}"
    )
    return report


def export_onnx(head: ResidualCTCHead, encoder_path: Path | None, output_dir: Path) -> None:
    """Replace ddddocr's 8,210-way tail with the fine-tuned case-sensitive head."""
    import onnx
    from onnx import compose

    if encoder_path is None:
        import ddddocr

        encoder_path = Path(ddddocr.__file__).parent / "common_old.onnx"
    encoder_graph = onnx.load(str(encoder_path))
    feature_node_index = next(
        i for i, node in enumerate(encoder_graph.graph.node) if "366" in node.output
    )
    del encoder_graph.graph.node[feature_node_index + 1 :]
    del encoder_graph.graph.output[:]
    encoder_graph.graph.output.extend(
        [onnx.helper.make_tensor_value_info("366", onnx.TensorProto.FLOAT, ["time", 1, 1024])]
    )
    used_names = {name for node in encoder_graph.graph.node for name in node.input}
    del encoder_graph.graph.initializer[:]
    for initializer in onnx.load(str(encoder_path)).graph.initializer:
        if initializer.name in used_names:
            encoder_graph.graph.initializer.append(initializer)

    with tempfile.TemporaryDirectory(prefix="captcha-onnx-") as temp_dir:
        head_path = Path(temp_dir) / "head.onnx"
        dummy = torch.zeros(8, 1, 1024, dtype=torch.float32)
        torch.onnx.export(
            head,
            dummy,
            head_path,
            input_names=["features"],
            output_names=["logits"],
            dynamic_axes={"features": {0: "time", 1: "batch"}, "logits": {0: "time", 1: "batch"}},
            opset_version=12,
            dynamo=False,
        )
        head_graph = onnx.load(str(head_path))
        head_graph.ir_version = encoder_graph.ir_version
        merged = compose.merge_models(
            encoder_graph,
            head_graph,
            io_map=[("366", "features")],
            prefix2="head_",
            name="captcha_mku_case_sensitive_ctc",
        )
        onnx.checker.check_model(merged)
        onnx.save(merged, output_dir / "model.onnx")
    _save_json(
        output_dir / "charsets.json",
        {"charset": list(CHARSET), "word": False, "image": [-1, IMAGE_HEIGHT], "channel": 1},
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a case-sensitive ddddocr CTC head from reviewed labels")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, default=None, help="ONNX encoder (defaults to ddddocr common_old.onnx)")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--augmentations", type=int, default=8, help="Total train views per image, including original")
    parser.add_argument("--uncertain-weight", type=float, default=0.75, help="Relative CTC loss weight for uncertain labels")
    parser.add_argument("--pretrained-lr", type=float, default=2e-5)
    parser.add_argument("--residual-lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--feature-noise", type=float, default=0.02, help="Gaussian feature noise std during training")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--skip-onnx", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output artifacts")
    parser.add_argument("--train-all", action="store_true", help="Train on all samples in manifest while keeping validation/test for monitoring")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        train(args)
    except Exception as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
