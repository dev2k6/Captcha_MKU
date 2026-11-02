"""Evaluate the OCR solver on a local, labelled image manifest.

This utility never fetches captchas or makes network requests. The manifest is
JSON with a ``samples`` array; image paths are resolved relative to the
manifest file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from PIL import Image


def normalize_answer(value: str | None, case_insensitive: bool = False) -> str | None:
    """Keep case and repeated glyphs unless comparison explicitly ignores case."""
    if value is None:
        return None
    value = value.strip()
    return value.upper() if case_insensitive else value


def _load_solver() -> Callable[[Image.Image], str]:
    # Lazy import keeps manifest/report handling independently testable.
    from captcha_ocr import solve_captcha

    return solve_captcha


def evaluate_manifest(
    manifest_path: str | Path,
    solve_fn: Callable[[Image.Image], str] | None = None,
    case_insensitive: bool = False,
) -> dict[str, Any]:
    """Run the local OCR solver and compare it with available labels."""
    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("samples"), list):
        raise ValueError("Manifest must be a JSON object with a 'samples' array")

    solver = solve_fn or _load_solver()
    sample_reports: list[dict[str, Any]] = []
    labelled = 0
    baseline_matches = 0
    new_matches = 0

    for index, sample in enumerate(manifest["samples"], start=1):
        if not isinstance(sample, dict):
            raise ValueError(f"Sample {index} must be a dictionary")

        raw_image_rel = sample.get("image") or (f"images/{sample['filename']}" if sample.get("filename") else None)
        if not raw_image_rel:
            raise ValueError(f"Sample {index} must contain an image path or filename")

        image_path = (manifest_file.parent / raw_image_rel).resolve()
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        prediction = normalize_answer(solver(image), case_insensitive) or ""
        expected = normalize_answer(sample.get("expected") or sample.get("label"), case_insensitive)
        baseline = normalize_answer(sample.get("baseline"), case_insensitive)

        match = None
        baseline_match = None
        if expected is not None:
            labelled += 1
            match = prediction == expected
            new_matches += int(match)
            if baseline is not None:
                baseline_match = baseline == expected
                baseline_matches += int(baseline_match)

        sample_reports.append(
            {
                "image": raw_image_rel,
                "expected": expected,
                "baseline": baseline,
                "prediction": prediction,
                "match": match,
                "baseline_match": baseline_match,
            }
        )

    baseline_labelled = sum(row["baseline_match"] is not None for row in sample_reports)
    return {
        "manifest": str(manifest_file),
        "case_insensitive": case_insensitive,
        "labelled_samples": labelled,
        "unlabelled_samples": len(sample_reports) - labelled,
        "baseline_labelled_samples": baseline_labelled,
        "baseline_matches": baseline_matches,
        "new_matches": new_matches,
        "baseline_accuracy": (
            baseline_matches / baseline_labelled if baseline_labelled else None
        ),
        "accuracy": new_matches / labelled if labelled else None,
        "samples": sample_reports,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate captcha OCR against a local labelled image manifest"
    )
    parser.add_argument("--manifest", required=True, help="Path to the JSON manifest")
    parser.add_argument("--report", help="Optional path for a JSON report")
    parser.add_argument("--case-insensitive", action="store_true", help="Ignore case for this comparison")
    args = parser.parse_args(argv)

    report = evaluate_manifest(args.manifest, case_insensitive=args.case_insensitive)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        report_path = Path(args.report).expanduser()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
