"""EasyOCR adapter for the 65-photo OCR benchmark."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from benchmark_ocr_65 import OUTPUT_DIR, ROOT, append_jsonl, crop_view, load_manifest, normalize_text, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    import easyocr

    config = "easyocr_ru_en_detail"
    output = OUTPUT_DIR / "raw" / f"{config}.jsonl"
    if args.force and output.exists():
        output.unlink()
    completed = {row["name"] for row in read_jsonl(output)}
    manifest = load_manifest()[: args.limit or None]
    reader = easyocr.Reader(["ru", "en"], gpu=True, verbose=False)
    for index, item in enumerate(manifest, start=1):
        if item["name"] in completed:
            continue
        started = time.perf_counter()
        error = None
        try:
            with Image.open(ROOT / item["path"]) as source:
                image = crop_view(ImageOps.exif_transpose(source).convert("RGB"), "detail")
                image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
            result = reader.readtext(
                np.asarray(image), detail=1, paragraph=False, canvas_size=1600,
                text_threshold=0.55, low_text=0.30, link_threshold=0.35,
            )
            lines = [
                {
                    "text": str(text),
                    "normalized": normalize_text(str(text)),
                    "score": float(score),
                    "view": "detail",
                    "box": np.asarray(box).round(1).tolist(),
                }
                for box, text, score in result
                if normalize_text(str(text))
            ]
        except Exception as exception:  # pragma: no cover - long-running adapter
            lines = []
            error = f"{type(exception).__name__}: {exception}"
        elapsed = (time.perf_counter() - started) * 1000
        append_jsonl(
            output,
            {
                "engine": "easyocr",
                "config": config,
                "config_details": {"languages": ["ru", "en"], "view": "detail", "max_size": 1600},
                "name": item["name"],
                "lines": lines,
                "timing_ms": {"total": elapsed},
                "error": error,
            },
        )
        print(f"[{config}] {index}/{len(manifest)} lines={len(lines)} ms={elapsed:.0f} error={error}", flush=True)


if __name__ == "__main__":
    main()
