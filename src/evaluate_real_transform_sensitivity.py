from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from stage3.embeddings import Dinov2Embedder
from stage3.index import VisualIndex


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SELECTION_MANIFEST = PROJECT_ROOT / "data/test/recheck_fuzzy_v2/selection_manifest.jsonl"
DATABASE = PROJECT_ROOT / "data/app/stage6/events.sqlite3"
INDEX_PATH = PROJECT_ROOT / "data/artifacts/stage3/dinov2-small/index.npz"
OUTPUT_PATH = PROJECT_ROOT / "data/artifacts/stage7/synthetic_real_gap/transform_sensitivity.json"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def latest_feedback() -> dict[str, dict[str, Any]]:
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        """
        SELECT r.original_name, r.created_at, f.verdict, f.correct_slug
        FROM recognitions r
        JOIN feedback f ON f.recognition_id = r.id
        ORDER BY r.created_at DESC
        """
    ).fetchall()
    connection.close()
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = str(row["original_name"])
        if name not in result:
            result[name] = dict(row)
    return result


def known_samples() -> list[dict[str, str]]:
    feedback = latest_feedback()
    samples: list[dict[str, str]] = []
    for record in read_jsonl(SELECTION_MANIFEST):
        if record["reason"] == "ambiguous_not_in_store":
            continue
        if record["reason"] == "stable_confirmed_top1":
            expected_slug = record["new_top1"]
        else:
            item = feedback.get(record["file"])
            if not item or item["verdict"] not in {"correct", "incorrect"}:
                continue
            expected_slug = item["correct_slug"] if item["verdict"] == "incorrect" else record["new_top1"]
        if not expected_slug:
            continue
        samples.append(
            {
                "name": record["file"],
                "path": str(PROJECT_ROOT / record["source"]),
                "expected_slug": str(expected_slug),
            }
        )
    if len(samples) != 52:
        raise ValueError(f"Expected 52 real samples with exact slugs, got {len(samples)}")
    return samples


def resize_long_side(image: np.ndarray, target: int = 512) -> np.ndarray:
    height, width = image.shape[:2]
    scale = target / max(height, width)
    return cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def motion_blur(image: np.ndarray, length: int = 9, angle: float = 0.0) -> np.ndarray:
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0 / length
    rotation = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rotation, (length, length))
    kernel /= max(float(kernel.sum()), 1e-6)
    return cv2.filter2D(image, -1, kernel)


def low_resolution(image: np.ndarray, scale: float) -> np.ndarray:
    height, width = image.shape[:2]
    reduced = cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return cv2.resize(reduced, (width, height), interpolation=cv2.INTER_CUBIC)


def jpeg_roundtrip(image: np.ndarray, quality: int) -> np.ndarray:
    success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise ValueError("JPEG encoding failed")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("JPEG decoding failed")
    return decoded


def transform(path: Path, variant: str) -> Image.Image:
    with Image.open(path) as source:
        rgb = np.asarray(source.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgr = resize_long_side(bgr)
    if variant == "gaussian_sigma_1_7":
        bgr = cv2.GaussianBlur(bgr, (0, 0), 1.7)
    elif variant == "motion_9px":
        bgr = motion_blur(bgr, 9, 0.0)
    elif variant == "compression_median":
        bgr = jpeg_roundtrip(low_resolution(bgr, 0.526), 44)
    elif variant == "blur_plus_compression":
        bgr = motion_blur(bgr, 9, 0.0)
        bgr = jpeg_roundtrip(low_resolution(bgr, 0.618), 55)
    elif variant != "original":
        raise ValueError(variant)
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def query_view(image: Image.Image, index: int) -> Image.Image:
    width, height = image.size
    if index == 0:
        return image
    if index == 1:
        return image.crop(
            (round(width * 0.10), round(height * 0.10), round(width * 0.90), round(height * 0.99))
        )
    return image.crop(
        (round(width * 0.22), round(height * 0.30), round(width * 0.78), round(height * 0.91))
    )


def ranked(scores: np.ndarray, top_k: int = 5) -> np.ndarray:
    candidates = np.argpartition(-scores, top_k - 1, axis=1)[:, :top_k]
    values = np.take_along_axis(scores, candidates, axis=1)
    order = np.argsort(-values, axis=1)
    return np.take_along_axis(candidates, order, axis=1)


def evaluate_variant(
    variant: str,
    samples: list[dict[str, str]],
    index: VisualIndex,
    embedder: Dinov2Embedder,
    batch_size: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    images = [transform(Path(sample["path"]), variant) for sample in samples]
    views = []
    embedding_ms = 0.0
    for view_index in range(3):
        vectors, elapsed_ms = embedder.encode(
            [query_view(image, view_index) for image in images], batch_size
        )
        views.append(vectors)
        embedding_ms += elapsed_ms
    _, _, scores = index.cls_scores(views[0], np.stack((views[1], views[2]), axis=1))
    indices = ranked(scores, 5)
    predictions = index.slugs[indices]
    top1 = 0
    top5 = 0
    reciprocal_ranks = []
    rows = []
    for sample, predicted in zip(samples, predictions, strict=True):
        slugs = [str(value) for value in predicted]
        expected = sample["expected_slug"]
        rank = slugs.index(expected) + 1 if expected in slugs else None
        top1 += rank == 1
        top5 += rank is not None
        reciprocal_ranks.append(1 / rank if rank else 0.0)
        rows.append({"name": sample["name"], "expected_slug": expected, "rank": rank, "top5": slugs})
    return {
        "sample_count": len(samples),
        "top1_correct": top1,
        "top1_accuracy": top1 / len(samples),
        "top5_correct": top5,
        "top5_recall": top5 / len(samples),
        "mean_reciprocal_rank": float(np.mean(reciprocal_ranks)),
        "embedding_ms": embedding_ms,
        "wall_ms": (time.perf_counter() - started) * 1000,
        "predictions": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure DINO sensitivity on 52 labeled real photos.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--local-files-only", action="store_true", default=True)
    args = parser.parse_args()

    samples = known_samples()
    index = VisualIndex.load(INDEX_PATH)
    embedder = Dinov2Embedder(index.model_name, args.device, args.local_files_only)
    embedder.warmup()
    variants = (
        "original",
        "gaussian_sigma_1_7",
        "motion_9px",
        "compression_median",
        "blur_plus_compression",
    )
    results = {
        variant: evaluate_variant(variant, samples, index, embedder, args.batch_size)
        for variant in variants
    }
    payload = {
        "schema_version": 1,
        "description": "Controlled DINOv2-small CLS multiscale sensitivity on real labeled photos",
        "sample_count": len(samples),
        "normalization": "input long side resized to 512 before controlled transform",
        "variants": {
            "original": "resize only",
            "gaussian_sigma_1_7": "Gaussian sigma=1.7 (median stage2 blur)",
            "motion_9px": "horizontal motion kernel=9 (median stage2 blur)",
            "compression_median": "downscale=0.526, upscale, JPEG q=44 (median compression)",
            "blur_plus_compression": "motion=9, downscale=0.618, JPEG q=55 (controlled combined subset)",
        },
        "results": results,
        "runtime": embedder.runtime_info(),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    concise = {
        name: {
            "top1": result["top1_accuracy"],
            "top5": result["top5_recall"],
            "wall_ms": result["wall_ms"],
        }
        for name, result in results.items()
    }
    print(json.dumps(concise, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
