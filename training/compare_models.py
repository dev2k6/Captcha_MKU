"""Compare built-in and trained OCR on the same fixed, reviewed test split."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from PIL import Image

import captcha_ocr
from training.train import edit_distance, load_manifest


def summarize(rows: list[dict], key: str) -> dict:
    count = len(rows)
    errors = sum(edit_distance(row["label"], row[key]) for row in rows)
    return {
        "samples": count,
        "exact_correct": sum(row["label"] == row[key] for row in rows),
        "exact_accuracy": sum(row["label"] == row[key] for row in rows) / count if count else None,
        "case_insensitive_correct": sum(row["label"].upper() == row[key].upper() for row in rows),
        "character_error_rate": errors / (count * 6) if count else None,
    }


def compare(manifest: Path, model: Path, output: Path) -> dict:
    samples = [sample for sample in load_manifest(manifest) if sample.split == "test"]
    rows = []
    for sample in samples:
        with Image.open(sample.image_path) as source:
            image = source.convert("RGB")
        result = captcha_ocr.solve_captcha_detailed(image, model)
        rows.append({
            "id": sample.sample_id, "image": str(sample.image_path),
            "label": sample.label, "label_uncertain": sample.label_uncertain,
            "builtin": captcha_ocr.solve_captcha(image, builtin=True),
            "trained": result["text"], "needs_review": result["needs_review"],
            "character_confidences": result["confidence"],
        })
    clear = [row for row in rows if not row["label_uncertain"]]
    report = {
        "manifest": str(manifest.resolve()), "model": str(model.resolve()),
        "label_caveat": "Assistant visual labels; not verified against server answers.",
        "all_labels": {key: summarize(rows, key) for key in ("builtin", "trained")},
        "clear_labels": {key: summarize(clear, key) for key in ("builtin", "trained")},
        "samples": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    table_rows = []
    for row in rows:
        matches = row["label"] == row["trained"]
        table_rows.append(
            '<tr><td>{}</td><td><img src="{}" alt="captcha {}"></td><td>{}{}</td>'
            '<td>{}</td><td class="{}">{}</td></tr>'.format(
                row["id"], Path(row["image"]).as_uri(), row["id"], html.escape(row["label"]),
                " ⚑" if row["label_uncertain"] else "", html.escape(row["builtin"]),
                "correct" if matches else "incorrect", html.escape(row["trained"]),
            )
        )
    output.with_suffix(".html").write_text(
        '<!doctype html><meta charset="utf-8"><title>So sánh OCR trên 50 ảnh kiểm tra</title>'
        '<style>body{font:16px system-ui;max-width:1100px;margin:30px auto;background:#fafafa}'
        'table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #ddd}'
        'td:nth-child(n+3){font-family:monospace;font-size:20px}img{width:300px;height:90px}'
        '.correct{color:#087b35}.incorrect{color:#c32929}</style>'
        '<h1>OCR — tập kiểm tra cố định</h1>'
        '<p>So sánh có phân biệt hoa/thường. Nhãn do trợ lý đọc trực quan; ⚑ là nhãn còn mơ hồ. '
        'Đây không phải đáp án đã được máy chủ xác nhận.</p>'
        '<table><thead><tr><th>ID</th><th>Ảnh</th><th>Nhãn</th><th>OCR trước train</th>'
        '<th>OCR sau train</th></tr></thead><tbody>' + "".join(table_rows) + '</tbody></table>'
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=Path("reports/comparison.json"))
    args = parser.parse_args()
    report = compare(args.manifest, args.model, args.report)
    print(json.dumps({key: value for key, value in report.items() if key != "samples"}, indent=2))


if __name__ == "__main__":
    main()
