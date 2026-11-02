"""Merge recorded visual reviews without converting OCR proposals to labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from training.prepare_data import _write_manifest


def curate(manifest_path: Path, review_paths: list[Path], replacements_path: Path | None = None,
           case_overrides_paths: list[Path] | None = None) -> dict:
    manifest = json.loads(manifest_path.read_text())
    reviews = {}
    for path in review_paths:
        entries = json.loads(path.read_text())
        if isinstance(entries, dict):
            entries = entries.get("samples", entries.get("labels", entries))
        if not isinstance(entries, dict):
            raise ValueError(f"Review file must contain an object of sample IDs: {path}")
        overlap = reviews.keys() & entries.keys()
        if overlap:
            raise ValueError(f"Duplicate review IDs: {sorted(overlap)}")
        malformed = [sample_id for sample_id, entry in entries.items() if not isinstance(entry, dict)]
        if malformed:
            raise ValueError(f"Review entries must be objects; malformed IDs: {sorted(malformed)}")
        reviews.update(entries)
    for path in case_overrides_paths or []:
        entries = json.loads(path.read_text())
        if isinstance(entries, dict):
            entries = entries.get("labels", entries)
        if not isinstance(entries, dict):
            raise ValueError(f"Case override file must contain an object of sample IDs: {path}")
        for sample_id, entry in entries.items():
            if sample_id not in reviews:
                raise ValueError(f"Case override ID {sample_id!r} has no corresponding reviewed label")
            if not isinstance(entry, (dict, str)):
                raise ValueError(f"Case override for {sample_id!r} must be a string or object")
            label_case = entry.get("label_case") if isinstance(entry, dict) else entry
            reviews[sample_id]["label_case"] = label_case
            if isinstance(entry, dict):
                reviews[sample_id]["uncertain"] = entry.get("uncertain", reviews[sample_id].get("uncertain", False))
                if entry.get("note"):
                    reviews[sample_id]["note"] = entry["note"]
            original = reviews[sample_id].get("label")
            if label_case and original and original != label_case.upper():
                reviews[sample_id]["uncertain"] = True
                reviews[sample_id]["note"] = "Identity changed during visual case review; needs independent confirmation."
    replacements = json.loads(replacements_path.read_text()) if replacements_path else {}
    if not isinstance(replacements, dict):
        raise ValueError("Replacement file must contain an object of sample IDs")
    unresolved = []
    for sample in manifest["samples"]:
        review = reviews.get(sample["id"], {})
        replacement = replacements.get(sample["id"])
        if replacement:
            source = Path(replacement["image"]).resolve()
            raw_bytes = source.read_bytes()
            relative = Path("images") / f"replacement_{sample['id']}{source.suffix}"
            (manifest_path.parent / relative).write_bytes(raw_bytes)
            sample.setdefault("rejected_original", {
                "image": sample["image"], "sha256": sample["sha256"],
                "reason": review.get("note", "Unclear glyph count"),
            })
            sample["image"] = str(relative)
            sample["sha256"] = hashlib.sha256(raw_bytes).hexdigest()
            sample.pop("proposed", None)
            review = replacement
        label = review.get("label_case", review.get("label"))
        if not isinstance(label, str) or not re.fullmatch(r"[A-Za-z0-9]{6}", label):
            unresolved.append(sample["id"])
            continue
        sample["label"] = label
        sample["label_source"] = "assistant_visual_review"
        sample["label_uncertain"] = bool(review.get("uncertain", False))
        sample["label_note"] = review.get("note", "")
    if unresolved:
        raise ValueError(f"Unresolved six-character labels: {unresolved}")
    hashes = [s["sha256"] for s in manifest["samples"]]
    if len(hashes) != len(set(hashes)):
        raise ValueError("Duplicate images after curation")
    manifest["label_policy"] = (
        "Case-sensitive assistant visual transcriptions, reviewed with OCR suggestions; not server ground truth. "
        "Ambiguous labels are flagged and must be reported separately in evaluation."
    )
    _write_manifest(manifest_path, manifest)
    return {
        "samples": len(manifest["samples"]),
        "uncertain": sum(s["label_uncertain"] for s in manifest["samples"]),
        "replacements": len(replacements),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--reviews", nargs="+", required=True, type=Path)
    parser.add_argument("--replacements", type=Path)
    parser.add_argument("--case-overrides", type=Path, nargs="+")
    args = parser.parse_args()
    print(json.dumps(curate(args.manifest, args.reviews, args.replacements, args.case_overrides), indent=2))


if __name__ == "__main__":
    main()
