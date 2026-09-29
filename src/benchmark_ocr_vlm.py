"""Run modern vision-language OCR models on the 65-photo benchmark.

Each backend is loaded in its own process so its GPU memory is released before
the next model is tested.  Output follows ``benchmark_ocr_65.py``'s raw JSONL
schema and is scored by that script.
"""

from __future__ import annotations

import argparse
import gc
import re
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps

from benchmark_ocr_65 import (
    OUTPUT_DIR,
    ROOT,
    append_jsonl,
    crop_view,
    load_manifest,
    normalize_text,
    read_jsonl,
)


MODELS = {
    "glm": "zai-org/GLM-OCR",
    "paddle_vl15": "PaddlePaddle/PaddleOCR-VL-1.5",
}


def clean_lines(text: str) -> list[dict[str, Any]]:
    # VLM OCR sometimes wraps plain text in Markdown.  Keep the content, but
    # discard formatting-only lines so coverage cannot be inflated by syntax.
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in text.replace("<|endoftext|>", "").splitlines():
        value = re.sub(r"^\s*(?:```\w*|```|[-*#>]+)\s*", "", raw).strip()
        normalized = normalize_text(value)
        if len(normalized) < 2 or normalized in seen:
            continue
        seen.add(normalized)
        # Generative OCR models do not expose line-level confidence.  A value
        # of 1.0 lets the existing OCR similarity function use their output;
        # model quality is measured independently by catalog retrieval below.
        result.append({"text": value, "normalized": normalized, "score": 1.0, "view": "detail", "box": None})
    return result


def load_glm() -> tuple[Any, Any]:
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(MODELS["glm"])
    model = AutoModelForMultimodalLM.from_pretrained(
        MODELS["glm"], dtype=torch.bfloat16
    ).to("cuda").eval()
    return model, processor


def infer_glm(model: Any, processor: Any, image: Image.Image) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Text Recognition:"},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    inputs.pop("token_type_ids", None)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    generated = output[0][inputs["input_ids"].shape[-1] :]
    return processor.decode(generated, skip_special_tokens=True)


def load_paddle() -> tuple[Any, Any]:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(MODELS["paddle_vl15"])
    model = AutoModelForImageTextToText.from_pretrained(
        MODELS["paddle_vl15"], dtype=torch.bfloat16
    ).to("cuda").eval()
    return model, processor


def infer_paddle(model: Any, processor: Any, image: Image.Image) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "OCR:"},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        images_kwargs={
            "size": {
                "shortest_edge": processor.image_processor.size.shortest_edge,
                "longest_edge": 1280 * 28 * 28,
            }
        },
    ).to(model.device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    generated = output[0][inputs["input_ids"].shape[-1] :]
    return processor.decode(generated, skip_special_tokens=True)


def run(model_name: str, limit: int | None, force: bool) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required for this benchmark")
    config = f"{model_name}_detail"
    output_path = OUTPUT_DIR / "raw" / f"{config}.jsonl"
    if force and output_path.exists():
        output_path.unlink()
    completed = {row["name"] for row in read_jsonl(output_path)}
    manifest = load_manifest()
    if limit:
        manifest = manifest[:limit]

    model, processor = load_glm() if model_name == "glm" else load_paddle()
    for index, item in enumerate(manifest, start=1):
        if item["name"] in completed:
            continue
        started = time.perf_counter()
        error = None
        raw_text = ""
        try:
            with Image.open(ROOT / item["path"]) as source:
                image = crop_view(ImageOps.exif_transpose(source).convert("RGB"), "detail")
                image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
            raw_text = (
                infer_glm(model, processor, image)
                if model_name == "glm"
                else infer_paddle(model, processor, image)
            )
            lines = clean_lines(raw_text)
        except Exception as exception:  # pragma: no cover - long-running model path
            lines = []
            error = f"{type(exception).__name__}: {exception}"
        elapsed = (time.perf_counter() - started) * 1000
        append_jsonl(
            output_path,
            {
                "engine": "vlm",
                "config": config,
                "config_details": {"model": MODELS[model_name], "view": "detail", "prompt": "OCR"},
                "name": item["name"],
                "lines": lines,
                "raw_text": raw_text,
                "timing_ms": {"total": elapsed},
                "error": error,
            },
        )
        print(f"[{config}] {index}/{len(manifest)} lines={len(lines)} ms={elapsed:.0f} error={error}", flush=True)

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=sorted(MODELS))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(args.model, args.limit, args.force)


if __name__ == "__main__":
    main()
