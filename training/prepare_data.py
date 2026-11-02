"""Collect, review, and split local captcha images for OCR experiments.

Collection and review commands operate on local files. ``collect`` makes
sequential requests to the MKU captcha endpoint; collection should only be
started deliberately with that subcommand.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import random
import re
import time
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageFont


DEFAULT_COUNT = 500
DEFAULT_DIRECTORY = Path("data/captcha500")
DEFAULT_DELAY = 0.2
DEFAULT_SEED = 42
MAX_REQUESTS_PER_MISSING_SAMPLE = 20
HTTP_TIMEOUT = (5, 15)
DATA_URL_PATTERN = re.compile(r"^data:image/([^;]+);base64,(.+)$", re.DOTALL)
IMAGE_EXTENSIONS = {
    "png": ".png",
    "jpeg": ".jpg",
    "jpg": ".jpg",
    "webp": ".webp",
    "gif": ".gif",
    "bmp": ".bmp",
}


def decode_image_data_url(data_url: str) -> tuple[bytes, str]:
    """Decode and validate a base64 data URL, returning its bytes and suffix."""
    match = DATA_URL_PATTERN.fullmatch(data_url)
    if not match:
        raise ValueError("Captcha image is not a base64 data URL")
    image_bytes = base64.b64decode(match.group(2), validate=True)
    with Image.open(io.BytesIO(image_bytes)) as image:
        image_format = (image.format or match.group(1)).lower()
        image.verify()
    return image_bytes, IMAGE_EXTENSIONS.get(image_format, ".img")


def _read_manifest(manifest_path: Path, seed: int) -> dict[str, Any]:
    if not manifest_path.exists():
        return {"version": 1, "seed": seed, "target_count": 0, "samples": []}
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("samples"), list):
        raise ValueError(f"Invalid dataset manifest: {manifest_path}")
    data.setdefault("version", 1)
    data.setdefault("seed", seed)
    data.setdefault("target_count", len(data["samples"]))
    return data


def _write_manifest(manifest_path: Path, manifest: dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = manifest_path.with_name(manifest_path.name + ".tmp")
    temporary_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary_path.replace(manifest_path)


def assign_splits(samples: list[dict[str, Any]], seed: int = DEFAULT_SEED) -> None:
    """Assign a reproducible 80/10/10 split without using labels or proposals."""
    shuffled_ids = [str(sample["id"]) for sample in samples]
    random.Random(seed).shuffle(shuffled_ids)
    train_end = int(len(samples) * 0.8)
    validation_end = train_end + int(len(samples) * 0.1)
    split_by_id = {
        sample_id: (
            "train" if index < train_end else "validation" if index < validation_end else "test"
        )
        for index, sample_id in enumerate(shuffled_ids)
    }
    for sample in samples:
        sample["split"] = split_by_id[str(sample["id"])]


def _fetch_payload(session: Any) -> dict[str, Any]:
    """Fetch one response without persisting any session token."""
    import captcha_ocr as c

    response = session.get(c.API_URL, headers=c.HEADERS, timeout=HTTP_TIMEOUT)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or not isinstance(payload.get("img"), str):
        raise ValueError("Captcha response is missing its image")
    # The API token is intentionally neither copied into the dataset nor logged.
    return payload


def collect(
    count: int = DEFAULT_COUNT,
    directory: str | Path = DEFAULT_DIRECTORY,
    delay: float = DEFAULT_DELAY,
    seed: int = DEFAULT_SEED,
    *,
    fetch_payload: Callable[[], dict[str, Any]] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    progress_fn: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Resume collecting until the directory contains ``count`` unique images.

    Requests run sequentially. The total number of API attempts is bounded to
    20 times the number of currently missing unique images.
    """
    if count < 1:
        raise ValueError("count must be at least 1")
    if delay < 0:
        raise ValueError("delay cannot be negative")

    root = Path(directory).expanduser().resolve()
    images_dir = root / "images"
    manifest_path = root / "manifest.json"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest = _read_manifest(manifest_path, seed)
    samples = manifest["samples"]
    seen_hashes = {sample.get("sha256") for sample in samples if sample.get("sha256")}
    next_id = max((int(sample["id"]) for sample in samples), default=0) + 1
    missing_at_start = max(0, count - len(samples))
    max_requests = missing_at_start * MAX_REQUESTS_PER_MISSING_SAMPLE
    if missing_at_start and fetch_payload is None:
        import requests

        session = requests.Session()
        payload_fetcher = lambda: _fetch_payload(session)
    else:
        payload_fetcher = fetch_payload

    api_attempts = 0
    added = 0
    while len(samples) < count and api_attempts < max_requests:
        api_attempts += 1
        try:
            payload = payload_fetcher()  # type: ignore[misc]
            image_bytes, extension = decode_image_data_url(payload["img"])
        except Exception as exc:
            if delay and api_attempts % max(1, int(1 / delay)) == 0:
                progress_fn(f"Request attempt {api_attempts} failed ({type(exc).__name__})")
            if delay:
                sleep_fn(delay)
            continue

        image_hash = hashlib.sha256(image_bytes).hexdigest()
        if image_hash not in seen_hashes:
            sample_id = f"{next_id:06d}"
            relative_path = Path("images") / f"{sample_id}{extension}"
            (root / relative_path).write_bytes(image_bytes)
            sample = {
                "id": sample_id,
                "image": relative_path.as_posix(),
                "sha256": image_hash,
                "label": None,
            }
            samples.append(sample)
            seen_hashes.add(image_hash)
            next_id += 1
            added += 1
            assign_splits(samples, seed)
            manifest["seed"] = seed
            manifest["target_count"] = count
            manifest["samples"] = samples
            _write_manifest(manifest_path, manifest)
            if len(samples) % 25 == 0:
                progress_fn(f"Collected {len(samples)}/{count} unique images")

        if delay:
            sleep_fn(delay)

    manifest["seed"] = seed
    manifest["target_count"] = count
    assign_splits(samples, seed)
    _write_manifest(manifest_path, manifest)
    if len(samples) % 25:
        progress_fn(f"Collected {len(samples)}/{count} unique images")
    if len(samples) < count:
        raise RuntimeError(
            f"Stopped after {api_attempts} bounded attempts; collected {len(samples)} of {count} unique images"
        )
    return {
        "directory": str(root),
        "manifest": str(manifest_path),
        "target_count": count,
        "total_unique": len(samples),
        "added_this_run": added,
        "api_attempts": api_attempts,
        "split_counts": _split_counts(samples),
    }


def _split_counts(samples: list[dict[str, Any]]) -> dict[str, int]:
    return {
        split: sum(sample.get("split") == split for sample in samples)
        for split in ("train", "validation", "test")
    }


def _raw_ocr_proposal(ocr: Any, image_bytes: bytes) -> dict[str, Any]:
    result = ocr.classification(image_bytes, probability=True)
    if isinstance(result, dict):
        return {
            "text": str(result.get("text", "")),
            "confidence": float(result["confidence"]) if result.get("confidence") is not None else None,
        }
    return {"text": str(result), "confidence": None}


def propose_labels(
    directory: str | Path = DEFAULT_DIRECTORY,
    *,
    include_baseline: bool = False,
    seed: int = DEFAULT_SEED,
) -> dict[str, int]:
    """Store OCR suggestions separately from the human-confirmed ``label``."""
    root = Path(directory).expanduser().resolve()
    manifest_path = root / "manifest.json"
    manifest = _read_manifest(manifest_path, seed)
    import ddddocr

    ocr = ddddocr.DdddOcr(show_ad=False)
    baseline_solver = None
    if include_baseline:
        import captcha_ocr as c

        baseline_solver = c.solve_captcha

    updated = 0
    for sample in manifest["samples"]:
        if sample.get("label") not in (None, ""):
            continue
        image_path = root / sample["image"]
        image_bytes = image_path.read_bytes()
        proposed = {"raw": _raw_ocr_proposal(ocr, image_bytes)}
        if baseline_solver is not None:
            with Image.open(io.BytesIO(image_bytes)) as source:
                proposed["baseline"] = str(baseline_solver(source.convert("RGB")))
        sample["proposed"] = proposed
        updated += 1

    _write_manifest(manifest_path, manifest)
    return {"samples_updated": updated, "confirmed_labels_preserved": True}


def make_contact_sheets(
    directory: str | Path = DEFAULT_DIRECTORY,
    per_page: int = 20,
) -> list[str]:
    """Create review sheets with sample IDs only, never OCR suggestions/labels."""
    if per_page < 1:
        raise ValueError("per_page must be at least 1")
    root = Path(directory).expanduser().resolve()
    manifest = _read_manifest(root / "manifest.json", DEFAULT_SEED)
    samples = sorted(manifest["samples"], key=lambda sample: str(sample["id"]))
    output_dir = root / "contact_sheets"
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = 2
    rows = math.ceil(per_page / columns)
    image_width = 400
    image_height = 120
    cell_width = 420
    cell_height = 150
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 18)
    except OSError:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 18)
        except OSError:
            font = ImageFont.load_default()
    paths: list[str] = []

    for page_start in range(0, len(samples), per_page):
        page_samples = samples[page_start : page_start + per_page]
        page_number = page_start // per_page + 1
        sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
        draw = ImageDraw.Draw(sheet)
        for index, sample in enumerate(page_samples):
            x = (index % columns) * cell_width + 10
            y = (index // columns) * cell_height
            draw.text((x + 5, y + 3), str(sample["id"]), fill="black", font=font)
            with Image.open(root / sample["image"]) as source:
                thumbnail = source.convert("RGB").resize(
                    (image_width, image_height), Image.Resampling.LANCZOS
                )
                sheet.paste(thumbnail, (x + 5, y + 25))
        output_path = output_dir / f"page-{page_number:03d}.png"
        sheet.save(output_path)
        paths.append(str(output_path))
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare local captcha OCR training data")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser("collect", help="Collect unique captcha images")
    collect_parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    collect_parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    collect_parser.add_argument("--delay", type=float, default=DEFAULT_DELAY)
    collect_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    propose_parser = subparsers.add_parser("propose", help="Add non-authoritative OCR suggestions")
    propose_parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    propose_parser.add_argument("--include-baseline", action="store_true")
    propose_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    sheets_parser = subparsers.add_parser("contact-sheets", help="Build ID-only blind-review sheets")
    sheets_parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    sheets_parser.add_argument("--per-page", type=int, default=20)

    args = parser.parse_args(argv)
    if args.command == "collect":
        result = collect(args.count, args.directory, args.delay, args.seed)
    elif args.command == "propose":
        result = propose_labels(args.directory, include_baseline=args.include_baseline, seed=args.seed)
    else:
        result = {"contact_sheets": make_contact_sheets(args.directory, args.per_page)}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
