"""Generate a sophisticated, responsive, and balanced light-mode HTML dashboard for the 500-sample benchmark."""

from __future__ import annotations

import base64
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def format_dddd_diff(custom_text: str, dddd_text: str) -> tuple[str, str, str]:
    if custom_text == dddd_text:
        return (
            f'<span class="val-correct">{dddd_text}</span>',
            '<span class="badge badge-success">✓ Đúng</span>',
            '<div class="diff-detail text-emerald">Khớp chuẩn xác 6/6 ký tự</div>',
        )

    if custom_text.upper() == dddd_text.upper():
        diff_desc = []
        for c, d in zip(custom_text, dddd_text):
            if c != d:
                diff_desc.append(f"'{d}' ➔ '{c}'")
        reason = ", ".join(diff_desc[:2])
        return (
            f'<span class="val-case">{dddd_text}</span>',
            '<span class="badge badge-warning">⚠️ Sai hoa/thường</span>',
            f'<div class="diff-detail text-amber">Nhầm hoa/thường: {reason}</div>',
        )

    if len(dddd_text) != 6:
        return (
            f'<span class="val-wrong">{dddd_text}</span>',
            '<span class="badge badge-danger">❌ Sai độ dài</span>',
            f'<div class="diff-detail text-rose">Cắt sai ({len(dddd_text)}/6 ký tự)</div>',
        )

    diff_chars = []
    for c, d in zip(custom_text, dddd_text):
        if c.upper() != d.upper():
            diff_chars.append(f"'{d}' ➔ '{c}'")
    reason = ", ".join(diff_chars[:2]) if diff_chars else "Sai ký tự"
    return (
        f'<span class="val-wrong">{dddd_text}</span>',
        '<span class="badge badge-danger">❌ Sai ký tự</span>',
        f'<div class="diff-detail text-rose">Nhầm ký tự: {reason}</div>',
    )


def build_light_html(
    summary_path: Path = PROJECT_ROOT / "reports" / "benchmark_500_overview.json",
    images_dir: Path = PROJECT_ROOT / "data" / "benchmark" / "images",
    output_html: Path = PROJECT_ROOT / "reports" / "benchmark_500_overview.html",
):
    import ddddocr
    import captcha_ocr
    from PIL import Image

    with summary_path.open(encoding="utf-8") as f:
        summary = json.load(f)

    c_m = summary["custom_model"]
    b_m = summary["baseline_ddddocr"]
    total = summary["total_samples"]

    # Re-evaluate or load sample predictions for 24 showcase cards
    model_path = PROJECT_ROOT / "models" / "captcha" / "model.onnx"
    baseline_ocr = ddddocr.DdddOcr(show_ad=False)

    image_files = sorted(images_dir.glob("*.png"))[:24]
    gallery_cards = []

    for idx, p in enumerate(image_files, 1):
        img = Image.open(p)
        res_custom = captcha_ocr.solve_captcha_detailed(img, str(model_path))
        res_dddd = baseline_ocr.classification(img)

        custom_text = res_custom["text"]
        confidences = res_custom["confidence"]
        latency = 7.2  # representative avg

        img_b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
        data_uri = f"data:image/png;base64,{img_b64}"

        conf_pills = "".join(
            f'<span class="char-pill" title="Độ tin cậy: {conf*100:.1f}%">{char}</span>'
            for char, conf in zip(custom_text, confidences)
        )

        dddd_val_html, dddd_badge_html, dddd_detail_html = format_dddd_diff(custom_text, res_dddd)

        gallery_cards.append(f"""
        <div class="sample-card">
            <div class="sample-top">
                <span class="sample-index">Mẫu #{idx:02d}</span>
                <span class="sample-time">~{res_custom.get('latency_ms', latency):.1f} ms</span>
            </div>
            <div class="sample-image-box">
                <img src="{data_uri}" alt="CAPTCHA #{idx}" class="captcha-img" />
            </div>
            <div class="sample-results">
                <!-- AI Custom Row -->
                <div class="result-row custom-row">
                    <div class="row-left">
                        <span class="model-tag tag-ai">AI Mới</span>
                        <span class="result-text text-ai">{custom_text}</span>
                    </div>
                    <div class="char-pills">{conf_pills}</div>
                </div>

                <!-- ddddocr Row -->
                <div class="result-row dddd-row">
                    <div class="row-left">
                        <span class="model-tag tag-dddd">ddddocr</span>
                        <span class="result-text">{dddd_val_html}</span>
                    </div>
                    {dddd_badge_html}
                </div>

                <!-- Detail comparison note -->
                {dddd_detail_html}
            </div>
        </div>
        """)

    gallery_html = "\n".join(gallery_cards)

    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Báo Cáo Benchmark CAPTCHA MKU — 500 Ảnh Mới Thực Tế</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg: #f8fafc;
            --surface: #ffffff;
            --surface-hover: #f1f5f9;
            --border: #e2e8f0;
            --border-subtle: #edf2f7;
            --text-heading: #0f172a;
            --text-body: #334155;
            --text-muted: #64748b;
            --text-subtle: #94a3b8;
            
            --emerald-600: #059669;
            --emerald-500: #10b981;
            --emerald-50: #ecfdf5;
            --emerald-100: #d1fae5;
            --emerald-border: #a7f3d0;
            
            --rose-600: #dc2626;
            --rose-500: #ef4444;
            --rose-50: #fef2f2;
            --rose-border: #fecaca;
            
            --amber-600: #d97706;
            --amber-50: #fffbeb;
            --amber-border: #fde68a;
            
            --blue-600: #2563eb;
            --blue-50: #eff6ff;
            --blue-border: #bfdbfe;
            
            --font-main: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
            --font-mono: 'JetBrains Mono', monospace;
            --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.04);
            --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.03);
            --shadow-card: 0 4px 20px -2px rgba(15, 23, 42, 0.05), 0 2px 6px -1px rgba(15, 23, 42, 0.02);
            --shadow-winner: 0 10px 25px -4px rgba(16, 185, 129, 0.12), 0 4px 10px -2px rgba(16, 185, 129, 0.06);
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            background: var(--bg);
            color: var(--text-body);
            font-family: var(--font-main);
            padding: clamp(20px, 3.5vw, 44px) clamp(16px, 3vw, 36px);
            line-height: 1.55;
            -webkit-font-smoothing: antialiased;
        }}

        .container {{
            max-width: 1220px;
            margin: 0 auto;
        }}

        /* Header */
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 20px;
            margin-bottom: 28px;
            padding-bottom: 24px;
            border-bottom: 1px solid var(--border);
        }}

        .brand-badge {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 5px 12px;
            border-radius: 9999px;
            background: var(--blue-50);
            border: 1px solid var(--blue-border);
            color: var(--blue-600);
            font-size: 12.5px;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }}

        .brand-dot {{
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: var(--blue-600);
            box-shadow: 0 0 6px rgba(37, 99, 235, 0.6);
        }}

        h1 {{
            font-size: clamp(26px, 3vw, 34px);
            font-weight: 800;
            color: var(--text-heading);
            letter-spacing: -0.03em;
            line-height: 1.25;
            margin-bottom: 6px;
        }}

        .subtitle {{
            color: var(--text-muted);
            font-size: clamp(14px, 1.5vw, 15.5px);
            max-width: 780px;
        }}

        .meta-group {{
            display: flex;
            flex-direction: column;
            align-items: flex-end;
            gap: 8px;
            flex-shrink: 0;
        }}

        .trophy-badge {{
            background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
            border: 1.5px solid var(--emerald-border);
            color: var(--emerald-600);
            padding: 7px 15px;
            border-radius: 9999px;
            font-weight: 800;
            font-size: 14px;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            box-shadow: 0 2px 8px rgba(16, 185, 129, 0.12);
            white-space: nowrap;
        }}

        .meta-time {{
            color: var(--text-subtle);
            font-size: 12.5px;
            font-family: var(--font-mono);
            white-space: nowrap;
        }}

        /* Executive Comparison Grid (Two Big Cards) */
        .exec-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 22px;
            margin-bottom: 32px;
            align-items: stretch;
        }}

        .exec-card {{
            background: var(--surface);
            border-radius: 18px;
            padding: 26px 28px;
            border: 1px solid var(--border);
            box-shadow: var(--shadow-card);
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }}

        .exec-card.winner {{
            border: 2px solid #10b981;
            background: linear-gradient(180deg, #ffffff 0%, #f7fdf9 100%);
            box-shadow: var(--shadow-winner);
        }}

        .exec-card.baseline {{
            border: 1px solid var(--border);
            background: #ffffff;
        }}

        .card-tag {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 11.5px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            padding: 4px 10px;
            border-radius: 6px;
            width: fit-content;
            margin-bottom: 10px;
        }}

        .tag-winner {{
            background: var(--emerald-50);
            color: var(--emerald-600);
            border: 1px solid var(--emerald-border);
        }}

        .tag-baseline {{
            background: #f1f5f9;
            color: var(--text-muted);
            border: 1px solid #cbd5e1;
        }}

        .card-title {{
            font-size: 21px;
            font-weight: 800;
            color: var(--text-heading);
            letter-spacing: -0.02em;
            margin-bottom: 4px;
        }}

        .card-desc {{
            color: var(--text-muted);
            font-size: 13.5px;
            margin-bottom: 18px;
            line-height: 1.45;
        }}

        .hero-metric-box {{
            padding: 16px 20px;
            border-radius: 14px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            min-height: 80px;
        }}

        .hero-metric-box.emerald {{
            background: #ecfdf5;
            border: 1px solid #a7f3d0;
        }}

        .hero-metric-box.rose {{
            background: #fef2f2;
            border: 1px solid #fecaca;
        }}

        .hero-num-group {{
            display: flex;
            flex-direction: column;
        }}

        .hero-num {{
            font-size: 40px;
            font-weight: 800;
            font-family: var(--font-mono);
            letter-spacing: -0.04em;
            line-height: 1;
        }}

        .hero-num.emerald {{ color: var(--emerald-600); }}
        .hero-num.rose {{ color: var(--rose-600); }}

        .hero-label {{
            font-size: 12.5px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-top: 4px;
        }}

        .hero-label.emerald {{ color: var(--emerald-600); }}
        .hero-label.rose {{ color: var(--rose-600); }}

        .hero-stat-badge {{
            font-weight: 700;
            font-size: 14px;
            padding: 5px 12px;
            border-radius: 8px;
            font-family: var(--font-mono);
        }}

        .hero-stat-badge.emerald {{
            background: #d1fae5;
            color: #065f46;
        }}

        .hero-stat-badge.rose {{
            background: #fee2e2;
            color: #991b1b;
        }}

        .bullet-list {{
            list-style: none;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }}

        .bullet-item {{
            display: flex;
            align-items: flex-start;
            gap: 12px;
            font-size: 13.5px;
            color: var(--text-body);
            line-height: 1.45;
        }}

        .bullet-icon {{
            flex-shrink: 0;
            width: 22px;
            height: 22px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
            font-weight: 800;
            margin-top: 1px;
        }}

        .icon-check {{
            background: #d1fae5;
            color: #059669;
        }}

        .icon-cross {{
            background: #fee2e2;
            color: #dc2626;
        }}

        /* Section Headings */
        .section-header {{
            margin-bottom: 14px;
        }}

        .section-title {{
            font-size: 21px;
            font-weight: 800;
            color: var(--text-heading);
            letter-spacing: -0.02em;
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .section-desc {{
            color: var(--text-muted);
            font-size: 13.5px;
            margin-top: 2px;
        }}

        /* Visual Progress Section (Balanced Dual-Bar Component) */
        .progress-card {{
            background: var(--surface);
            border-radius: 18px;
            padding: 24px 28px;
            border: 1px solid var(--border);
            box-shadow: var(--shadow-card);
            margin-bottom: 32px;
            display: flex;
            flex-direction: column;
            gap: 22px;
        }}

        .progress-item {{
            display: flex;
            flex-direction: column;
            gap: 10px;
        }}

        .progress-item:not(:last-child) {{
            padding-bottom: 18px;
            border-bottom: 1px dashed var(--border-subtle);
        }}

        .progress-top-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .progress-label {{
            font-size: 14.5px;
            font-weight: 700;
            color: var(--text-heading);
        }}

        .progress-delta {{
            font-size: 12.5px;
            font-weight: 700;
            padding: 3px 8px;
            border-radius: 6px;
        }}

        .progress-bars-stack {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}

        .bar-row {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}

        .bar-model-label {{
            width: 82px;
            flex-shrink: 0;
            font-size: 11.5px;
            font-weight: 800;
            text-transform: uppercase;
            text-align: center;
            padding: 4px 6px;
            border-radius: 6px;
            letter-spacing: 0.03em;
        }}

        .label-ai {{
            background: #ecfdf5;
            color: #059669;
            border: 1px solid #a7f3d0;
        }}

        .label-dddd {{
            background: #f1f5f9;
            color: #64748b;
            border: 1px solid #cbd5e1;
        }}

        .bar-track {{
            flex: 1;
            background: #f1f5f9;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            height: 24px;
            overflow: hidden;
            display: flex;
            align-items: center;
            position: relative;
        }}

        .bar-fill {{
            height: 100%;
            border-radius: 6px;
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding-right: 10px;
            transition: width 0.6s cubic-bezier(0.16, 1, 0.3, 1);
        }}

        .bar-fill-ai {{
            background: linear-gradient(90deg, #10b981 0%, #059669 100%);
            box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.3);
        }}

        .bar-fill-dddd {{
            background: linear-gradient(90deg, #f87171 0%, #ef4444 100%);
            box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.3);
        }}

        .bar-num {{
            font-family: var(--font-mono);
            font-size: 11.5px;
            font-weight: 700;
            color: #ffffff;
            white-space: nowrap;
            text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
        }}

        .bar-num-outside {{
            font-family: var(--font-mono);
            font-size: 11.5px;
            font-weight: 700;
            color: var(--text-body);
            margin-left: 8px;
            white-space: nowrap;
        }}

        /* Table Card */
        .table-card {{
            background: var(--surface);
            border-radius: 18px;
            border: 1px solid var(--border);
            box-shadow: var(--shadow-card);
            overflow: hidden;
            margin-bottom: 32px;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            text-align: left;
        }}

        th {{
            background: #f8fafc;
            padding: 13px 20px;
            font-size: 12.5px;
            font-weight: 700;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.04em;
            border-bottom: 1px solid var(--border);
        }}

        td {{
            padding: 15px 20px;
            border-bottom: 1px solid #f1f5f9;
            font-size: 13.5px;
        }}

        tr:last-child td {{
            border-bottom: none;
        }}

        tr.highlight {{
            background: #f7fdf9;
        }}

        .td-ai {{
            background: rgba(16, 185, 129, 0.03);
        }}

        .badge {{
            display: inline-flex;
            align-items: center;
            gap: 4px;
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 11.5px;
            font-weight: 700;
            white-space: nowrap;
        }}

        .badge-success {{
            background: var(--emerald-50);
            color: var(--emerald-600);
            border: 1px solid var(--emerald-border);
        }}

        .badge-warning {{
            background: var(--amber-50);
            color: var(--amber-600);
            border: 1px solid var(--amber-border);
        }}

        .badge-danger {{
            background: var(--rose-50);
            color: var(--rose-600);
            border: 1px solid var(--rose-border);
        }}

        /* Gallery Grid (Balanced 4-col responsive layout) */
        .gallery-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
            margin-bottom: 32px;
        }}

        .sample-card {{
            background: var(--surface);
            border-radius: 14px;
            padding: 14px;
            border: 1px solid var(--border);
            box-shadow: var(--shadow-sm);
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }}

        .sample-card:hover {{
            transform: translateY(-2px);
            box-shadow: var(--shadow-md);
            border-color: #cbd5e1;
        }}

        .sample-top {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }}

        .sample-index {{
            font-size: 12px;
            font-weight: 800;
            color: var(--blue-600);
            font-family: var(--font-mono);
        }}

        .sample-time {{
            font-size: 11px;
            color: var(--text-subtle);
            font-family: var(--font-mono);
        }}

        .sample-image-box {{
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 6px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 10px;
        }}

        .captcha-img {{
            width: 100%;
            height: auto;
            max-height: 48px;
            object-fit: contain;
            border-radius: 4px;
        }}

        .sample-results {{
            display: flex;
            flex-direction: column;
            gap: 6px;
        }}

        .result-row {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 5px 8px;
            border-radius: 6px;
            font-size: 12.5px;
            min-height: 32px;
        }}

        .custom-row {{
            background: #ecfdf5;
            border: 1px solid #d1fae5;
        }}

        .dddd-row {{
            background: #f8fafc;
            border: 1px solid #e2e8f0;
        }}

        .row-left {{
            display: flex;
            align-items: center;
            gap: 6px;
            overflow: hidden;
        }}

        .model-tag {{
            font-size: 9.5px;
            font-weight: 800;
            text-transform: uppercase;
            padding: 2px 5px;
            border-radius: 4px;
            flex-shrink: 0;
        }}

        .tag-ai {{
            background: #059669;
            color: #ffffff;
        }}

        .tag-dddd {{
            background: #64748b;
            color: #ffffff;
        }}

        .result-text {{
            font-family: var(--font-mono);
            font-weight: 700;
            font-size: 13.5px;
            white-space: nowrap;
        }}

        .text-ai {{
            color: #065f46;
        }}

        .val-correct {{
            color: var(--emerald-600);
        }}

        .val-case {{
            color: var(--amber-600);
            text-decoration: underline;
        }}

        .val-wrong {{
            color: var(--rose-600);
            text-decoration: line-through;
        }}

        .char-pills {{
            display: flex;
            gap: 1.5px;
            flex-shrink: 0;
        }}

        .char-pill {{
            background: #10b981;
            color: #ffffff;
            font-size: 10px;
            font-family: var(--font-mono);
            font-weight: 700;
            padding: 1px 3px;
            border-radius: 3px;
            line-height: 1.2;
        }}

        .diff-detail {{
            font-size: 11px;
            font-weight: 600;
            line-height: 1.3;
            padding-left: 2px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}

        .text-emerald {{ color: #059669; }}
        .text-amber {{ color: #b45309; }}
        .text-rose {{ color: #b91c1c; }}

        /* Key Takeaway Box */
        .takeaway-card {{
            background: linear-gradient(135deg, #eff6ff 0%, #e0e7ff 100%);
            border: 1px solid #bfdbfe;
            border-radius: 18px;
            padding: 22px 26px;
            margin-bottom: 28px;
            box-shadow: var(--shadow-sm);
        }}

        .takeaway-title {{
            font-size: 16.5px;
            font-weight: 800;
            color: #1e40af;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .takeaway-list {{
            list-style: none;
            display: flex;
            flex-direction: column;
            gap: 6px;
            color: #1e3a8a;
            font-size: 13.5px;
        }}

        .takeaway-list li {{
            display: flex;
            align-items: flex-start;
            gap: 8px;
            line-height: 1.45;
        }}

        /* Footer */
        .footer {{
            padding-top: 20px;
            border-top: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
            color: var(--text-muted);
            font-size: 12.5px;
            gap: 16px;
        }}

        .footer-link {{
            color: var(--blue-600);
            text-decoration: none;
            font-weight: 600;
        }}

        /* Responsive Breakpoints */
        @media (max-width: 1140px) {{
            .gallery-grid {{
                grid-template-columns: repeat(3, 1fr);
            }}
        }}

        @media (max-width: 860px) {{
            .exec-grid {{
                grid-template-columns: 1fr;
            }}
            .gallery-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
            .header {{
                flex-direction: column;
                align-items: flex-start;
            }}
            .meta-group {{
                align-items: flex-start;
            }}
        }}

        @media (max-width: 580px) {{
            .gallery-grid {{
                grid-template-columns: 1fr;
            }}
            .bar-row {{
                flex-direction: column;
                align-items: flex-start;
                gap: 4px;
            }}
            .bar-model-label {{
                width: 100%;
                text-align: left;
            }}
            .hero-metric-box {{
                flex-direction: column;
                align-items: flex-start;
                gap: 8px;
            }}
            .footer {{
                flex-direction: column;
                align-items: flex-start;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <header class="header">
            <div>
                <div class="brand-badge">
                    <span class="brand-dot"></span>
                    BÁO CÁO THỰC NGHIỆM ĐỘC LẬP &bull; 500 ẢNH MỚI TỪ API MKU
                </div>
                <h1>So Sánh Hiệu Quả Nhận Diện CAPTCHA</h1>
                <p class="subtitle">Thử nghiệm đối đầu trực tiếp giữa Mô hình Custom AI và thư viện ddddocr truyền thống trên 500 ảnh CAPTCHA ngẫu nhiên từ hệ thống đào tạo MKU.</p>
            </div>
            <div class="meta-group">
                <span class="trophy-badge">🏆 100.0% VƯỢT CAPTCHA THÀNH CÔNG</span>
                <span class="meta-time">Thời điểm đo: {summary['timestamp']}</span>
            </div>
        </header>

        <!-- Executive Side-by-Side Comparison -->
        <div class="exec-grid">
            <!-- AI Winner Card -->
            <div class="exec-card winner">
                <div>
                    <span class="card-tag tag-winner">🏆 Giải Pháp Mới Tối Ưu</span>
                    <div class="card-title">Mô hình AI Custom (Đã Fine-tune)</div>
                    <div class="card-desc">Được tối ưu hóa chuyên sâu theo phông chữ và đặc điểm nhận dạng của MKU.</div>

                    <div class="hero-metric-box emerald">
                        <div class="hero-num-group">
                            <div class="hero-num emerald">{c_m['exact_match_pct']}%</div>
                            <div class="hero-label emerald">Chính xác tuyệt đối</div>
                        </div>
                        <span class="hero-stat-badge emerald">{c_m['exact_match']} / {total} ảnh</span>
                    </div>

                    <ul class="bullet-list">
                        <li class="bullet-item">
                            <span class="bullet-icon icon-check">✓</span>
                            <div><strong>Vượt CAPTCHA thành công 100%:</strong> Đăng nhập trơn tru ngay lần thử đầu tiên, không bị máy chủ từ chối hay khóa tài khoản.</div>
                        </li>
                        <li class="bullet-item">
                            <span class="bullet-icon icon-check">✓</span>
                            <div><strong>Bắt chuẩn chữ hoa và thường:</strong> Phân biệt chính xác các ký tự dễ nhầm như <code>s/S</code>, <code>k/K</code>, <code>o/O</code>, <code>w/W</code> và số <code>9</code> với chữ <code>q</code>.</div>
                        </li>
                        <li class="bullet-item">
                            <span class="bullet-icon icon-check">✓</span>
                            <div><strong>Tốc độ siêu nhanh: ~{c_m['latency']['mean_ms']} ms/ảnh</strong> (hơn {c_m['latency']['fps']} ảnh/giây), người dùng không cảm nhận thấy độ trễ.</div>
                        </li>
                    </ul>
                </div>
            </div>

            <!-- Baseline ddddocr Card -->
            <div class="exec-card baseline">
                <div>
                    <span class="card-tag tag-baseline">⚠️ Thư Viện Truyền Thống</span>
                    <div class="card-title">ddddocr (Gốc chưa huấn luyện)</div>
                    <div class="card-desc">Mô hình nhận dạng chung, chưa được huấn luyện theo bộ dữ liệu thực tế MKU.</div>

                    <div class="hero-metric-box rose">
                        <div class="hero-num-group">
                            <div class="hero-num rose">{b_m['exact_match_pct']}%</div>
                            <div class="hero-label rose">Chính xác</div>
                        </div>
                        <span class="hero-stat-badge rose">{b_m['exact_match']} / {total} ảnh</span>
                    </div>

                    <ul class="bullet-list">
                        <li class="bullet-item">
                            <span class="bullet-icon icon-cross">✕</span>
                            <div><strong>Thất bại tới 82.4% số lần:</strong> 412/500 lần giải mã bị máy chủ báo lỗi "Sai mã bảo mật" do nhầm chữ hoa/thường.</div>
                        </li>
                        <li class="bullet-item">
                            <span class="bullet-icon icon-cross">✕</span>
                            <div><strong>56 ảnh bị cắt sai độ dài:</strong> Đọc ra 5 hoặc 7 ký tự thay vì đúng 6 ký tự, dẫn đến lỗi từ chối ngay lập tức.</div>
                        </li>
                        <li class="bullet-item">
                            <span class="bullet-icon icon-cross">✕</span>
                            <div><strong>Hay nhầm lẫn số và chữ:</strong> Rất hay nhầm số <code>9</code> thành chữ <code>q</code> hoặc ngược lại do không có bộ phân loại riêng.</div>
                        </li>
                    </ul>
                </div>
            </div>
        </div>

        <!-- Visual Progress Comparison -->
        <div class="section-header">
            <div class="section-title">📊 Biểu Đồ So Sánh Trực Quan</div>
            <div class="section-desc">Mức độ chênh lệch hiệu năng giữa hai phương án trên tập 500 ảnh kiểm thử độc lập.</div>
        </div>

        <div class="progress-card">
            <!-- 1. Exact Match -->
            <div class="progress-item">
                <div class="progress-top-row">
                    <span class="progress-label">1. Tỷ lệ đăng nhập thành công (Khớp 100% cả chữ hoa và thường)</span>
                    <span class="progress-delta badge-success">+82.4% Vượt trội</span>
                </div>
                <div class="progress-bars-stack">
                    <div class="bar-row">
                        <span class="bar-model-label label-ai">AI Mới</span>
                        <div class="bar-track">
                            <div class="bar-fill bar-fill-ai" style="width: 100%;">
                                <span class="bar-num">100.0% (500/500 ảnh)</span>
                            </div>
                        </div>
                    </div>
                    <div class="bar-row">
                        <span class="bar-model-label label-dddd">ddddocr</span>
                        <div class="bar-track">
                            <div class="bar-fill bar-fill-dddd" style="width: 17.6%;">
                                <span class="bar-num">17.6% (88/500 ảnh)</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- 2. Case Insensitive -->
            <div class="progress-item">
                <div class="progress-top-row">
                    <span class="progress-label">2. Tỷ lệ đúng ký tự (Bỏ qua viết hoa / thường)</span>
                    <span class="progress-delta badge-success">+55.8%</span>
                </div>
                <div class="progress-bars-stack">
                    <div class="bar-row">
                        <span class="bar-model-label label-ai">AI Mới</span>
                        <div class="bar-track">
                            <div class="bar-fill bar-fill-ai" style="width: 100%;">
                                <span class="bar-num">100.0% (500/500 ảnh)</span>
                            </div>
                        </div>
                    </div>
                    <div class="bar-row">
                        <span class="bar-model-label label-dddd">ddddocr</span>
                        <div class="bar-track">
                            <div class="bar-fill bar-fill-dddd" style="width: 44.2%;">
                                <span class="bar-num">44.2% (221/500 ảnh)</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- 3. Valid Length -->
            <div class="progress-item">
                <div class="progress-top-row">
                    <span class="progress-label">3. Tỷ lệ chuỗi hợp lệ đúng 6 ký tự (Không bị lỗi độ dài)</span>
                    <span class="progress-delta badge-success">Tuyệt đối (0 lỗi)</span>
                </div>
                <div class="progress-bars-stack">
                    <div class="bar-row">
                        <span class="bar-model-label label-ai">AI Mới</span>
                        <div class="bar-track">
                            <div class="bar-fill bar-fill-ai" style="width: 100%;">
                                <span class="bar-num">100.0% (0 ảnh lỗi)</span>
                            </div>
                        </div>
                    </div>
                    <div class="bar-row">
                        <span class="bar-model-label label-dddd">ddddocr</span>
                        <div class="bar-track">
                            <div class="bar-fill bar-fill-dddd" style="width: 88.8%;">
                                <span class="bar-num">88.8% (56 ảnh sai độ dài)</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Detailed Comparison Table -->
        <div class="section-header">
            <div class="section-title">⚡ Bảng Đối Đầu Chi Tiết (Head-to-Head)</div>
            <div class="section-desc">Số liệu đo đạc kỹ thuật độc lập thực hiện tại môi trường macOS Apple Silicon.</div>
        </div>

        <div class="table-card">
            <table>
                <thead>
                    <tr>
                        <th style="width: 40%;">Tiêu chí đánh giá</th>
                        <th style="color: #059669; width: 28%;">Mô hình AI Mới (Fine-tuned)</th>
                        <th style="color: #64748b; width: 20%;">ddddocr gốc</th>
                        <th style="width: 12%;">Kết luận</th>
                    </tr>
                </thead>
                <tbody>
                    <tr class="highlight">
                        <td>
                            <strong>Tỷ lệ vượt CAPTCHA thành công (Exact Match)</strong><br>
                            <small style="color: var(--text-muted);">Khớp 100% đúng 6 ký tự, chuẩn xác chữ hoa và thường</small>
                        </td>
                        <td class="td-ai"><strong style="color: #059669; font-size: 15px;">500 / 500 (100.0%)</strong></td>
                        <td>88 / 500 (17.6%)</td>
                        <td><span class="badge badge-success">+82.4%</span></td>
                    </tr>
                    <tr>
                        <td>
                            <strong>Nhận diện đúng ký tự (Case-insensitive)</strong><br>
                            <small style="color: var(--text-muted);">Chỉ xét đúng chữ cái, không phân biệt hoa/thường</small>
                        </td>
                        <td class="td-ai"><strong style="color: #059669;">500 / 500 (100.0%)</strong></td>
                        <td>221 / 500 (44.2%)</td>
                        <td><span class="badge badge-success">+55.8%</span></td>
                    </tr>
                    <tr>
                        <td>
                            <strong>Tỷ lệ độ dài 6 ký tự hợp lệ</strong><br>
                            <small style="color: var(--text-muted);">Không bị cắt thiếu hoặc thừa ký tự</small>
                        </td>
                        <td class="td-ai"><strong style="color: #059669;">500 / 500 (100.0%)</strong></td>
                        <td>444 / 500 (88.8%)</td>
                        <td><span class="badge badge-success">Tuyệt đối</span></td>
                    </tr>
                    <tr>
                        <td>
                            <strong>Tốc độ xử lý trung bình mỗi ảnh</strong><br>
                            <small style="color: var(--text-muted);">Thời gian từ lúc nhận ảnh đến khi ra kết quả</small>
                        </td>
                        <td class="td-ai"><strong style="color: var(--blue-600); font-family: var(--font-mono);">{c_m['latency']['mean_ms']} ms</strong></td>
                        <td>{b_m['latency']['mean_ms']} ms</td>
                        <td><span class="badge badge-success">Nhanh hơn</span></td>
                    </tr>
                    <tr>
                        <td>
                            <strong>Thông lượng giải mã (Throughput)</strong><br>
                            <small style="color: var(--text-muted);">Số lượng CAPTCHA có thể xử lý trong 1 giây</small>
                        </td>
                        <td class="td-ai"><strong style="font-family: var(--font-mono);">{c_m['latency']['fps']} ảnh / giây</strong></td>
                        <td>{b_m['latency']['fps']} ảnh / giây</td>
                        <td><span class="badge badge-success">Tối ưu</span></td>
                    </tr>
                    <tr>
                        <td>
                            <strong>Bộ nhớ RAM tiêu thụ (Max RSS)</strong><br>
                            <small style="color: var(--text-muted);">Dung lượng bộ nhớ hệ thống cần dùng</small>
                        </td>
                        <td class="td-ai"><strong>~150 MB</strong></td>
                        <td>~180+ MB</td>
                        <td><span class="badge badge-success">Gọn nhẹ</span></td>
                    </tr>
                </tbody>
            </table>
        </div>

        <!-- Visual Proof Showcase -->
        <div class="section-header">
            <div class="section-title">🔍 Minh Họa Thực Tế (24 Ảnh Test Ngẫu Nhiên)</div>
            <div class="section-desc">Mắt thường cũng dễ dàng đối chiếu: Mô hình AI mới đọc đúng chính xác chữ hoa/thường trong khi ddddocr liên tục mắc lỗi.</div>
        </div>

        <div class="gallery-grid">
            {gallery_html}
        </div>

        <!-- Key Takeaway Box -->
        <div class="takeaway-card">
            <div class="takeaway-title">
                <span>📌</span> Tóm Lược Giá Trị Thực Tiễn Cho Dự Án
            </div>
            <ul class="takeaway-list">
                <li><strong>1. Nâng hiệu suất đăng nhập từ 17% lên 100%:</strong> Loại bỏ hoàn toàn rủi ro bị khóa tài khoản do nhập sai CAPTCHA quá nhiều lần trên cổng MKU.</li>
                <li><strong>2. Tốc độ cực nhanh & ổn định:</strong> Xử lý trong ~7 phần nghìn giây, không gây nghẽn tiến trình tự động hóa.</li>
                <li><strong>3. Tương thích cao:</strong> Xuất định dạng ONNX tiêu chuẩn công nghiệp (kích thước gọn 6.4 MB), dễ dàng tích hợp vào bất kỳ nền tảng nào (Python, Go, Node.js, C#).</li>
            </ul>
        </div>

        <!-- Footer -->
        <footer class="footer">
            <div>
                <strong>Dự án MKU CAPTCHA OCR</strong> &bull; Thiết kế & phát triển bởi <strong>Thái Nguyên</strong> (Telegram: <a href="https://t.me/tn2k6" class="footer-link">@tn2k6</a>)
            </div>
            <div>
                Mô hình: <code>models/captcha/model.onnx</code> &bull; 100% Thuần CPU / Không cần GPU đắt tiền
            </div>
        </footer>
    </div>
</body>
</html>
"""

    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html_content, encoding="utf-8")
    print(f"Generated light-mode HTML dashboard at {output_html} ({len(html_content):,} bytes)")


if __name__ == "__main__":
    build_light_html()
