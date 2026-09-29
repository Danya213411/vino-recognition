"""Benchmark OCR coverage and catalog-field fidelity on the 65 field photos.

The benchmark is deliberately independent from DINO/SIFT.  Every engine writes
raw JSONL output first; scoring can then be repeated without rerunning OCR.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps

# Running this file directly puts only ``src`` on sys.path.  Add the repository
# root as well so the application database module is available.
ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from app.backend.vino_api.database import EventDatabase
from stage4.features import normalize_text, text_tokens
from stage5.features import ocr_field_evidence


IMAGE_DIR = ROOT / "data/test/field_66_review/images"
CATALOG_PATH = ROOT / "data/artifacts/stage1/catalog_manifest.jsonl"
DATABASE_PATH = ROOT / "data/app/stage6/events.sqlite3"
OUTPUT_DIR = ROOT / "data/artifacts/stage8/ocr_benchmark_65"


@dataclass(frozen=True)
class RapidConfig:
    name: str
    detector: str
    recognizer: str = "cyrillic"
    limit_side_len: int = 736
    text_score: float = 0.35
    det_thresh: float = 0.30
    box_thresh: float = 0.50
    views: tuple[str, ...] = ("medium",)
    preprocess: str = "autocontrast"


RAPID_CONFIGS = (
    RapidConfig("rapid_current", "tiny"),
    RapidConfig("rapid_tiny_1024_low", "tiny", limit_side_len=1024, text_score=0.25, box_thresh=0.45),
    RapidConfig("rapid_tiny_1280_low", "tiny", limit_side_len=1280, text_score=0.25, box_thresh=0.45),
    RapidConfig("rapid_small_736", "small"),
    RapidConfig("rapid_small_1024_low", "small", limit_side_len=1024, text_score=0.25, box_thresh=0.45),
    RapidConfig("rapid_small_1280_low", "small", limit_side_len=1280, text_score=0.25, box_thresh=0.45),
    RapidConfig("rapid_small_1600_low", "small", limit_side_len=1600, text_score=0.25, box_thresh=0.45),
    RapidConfig("rapid_medium_1024_low", "medium", limit_side_len=1024, text_score=0.25, box_thresh=0.45),
    RapidConfig("rapid_medium_1280_low", "medium", limit_side_len=1280, text_score=0.25, box_thresh=0.45),
    RapidConfig("rapid_medium_1600_low", "medium", limit_side_len=1600, text_score=0.25, box_thresh=0.45),
    RapidConfig("rapid_small_1280_eslav", "small", "eslav", 1280, 0.25, 0.30, 0.45),
    RapidConfig("rapid_medium_1280_eslav", "medium", "eslav", 1280, 0.25, 0.30, 0.45),
    RapidConfig("rapid_small_1280_very_low", "small", limit_side_len=1280, text_score=0.15, box_thresh=0.38),
    RapidConfig("rapid_small_1280_boost", "small", limit_side_len=1280, text_score=0.25, box_thresh=0.42, preprocess="boost"),
    RapidConfig("rapid_small_1280_multiview", "small", limit_side_len=1280, text_score=0.22, box_thresh=0.42, views=("medium", "detail")),
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def build_manifest() -> list[dict[str, Any]]:
    catalog = {str(row["slug"]): row for row in read_jsonl(CATALOG_PATH)}
    image_paths = sorted(path for path in IMAGE_DIR.iterdir() if path.is_file())
    database = EventDatabase(DATABASE_PATH)
    records = {row["original_name"]: row for row in database.latest_review_records([path.name for path in image_paths])}
    manifest: list[dict[str, Any]] = []
    for path in image_paths:
        record = records.get(path.name)
        feedback = (record or {}).get("feedback") or {}
        verdict = feedback.get("verdict")
        correct_slug = feedback.get("correct_slug")
        if verdict == "correct" and record:
            correct_slug = record.get("candidate_slug") or record.get("predicted_slug")
        target = catalog.get(str(correct_slug)) if correct_slug else None
        manifest.append(
            {
                "name": path.name,
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "verdict": verdict,
                "correct_slug": correct_slug,
                "target": target,
                "baseline_ocr_lines": ((record or {}).get("result") or {}).get("ocr_lines") or [],
            }
        )
    write_json(OUTPUT_DIR / "manifest.json", manifest)
    return manifest


def load_manifest() -> list[dict[str, Any]]:
    path = OUTPUT_DIR / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else build_manifest()


def crop_view(image: Image.Image, name: str) -> Image.Image:
    width, height = image.size
    if name == "full":
        return image.copy()
    if name == "medium":
        return image.crop((round(width * 0.10), round(height * 0.10), round(width * 0.90), round(height * 0.99)))
    if name == "detail":
        return image.crop((round(width * 0.20), round(height * 0.22), round(width * 0.80), round(height * 0.94)))
    raise ValueError(f"Unknown view: {name}")


def preprocess(image: Image.Image, name: str) -> np.ndarray:
    rgb = image.convert("RGB")
    if name == "none":
        return np.asarray(rgb)
    if name == "autocontrast":
        return np.asarray(ImageOps.autocontrast(rgb, cutoff=1))
    if name == "boost":
        array = np.asarray(rgb)
        lab = cv2.cvtColor(array, cv2.COLOR_RGB2LAB)
        light, a, b = cv2.split(lab)
        light = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(light)
        boosted = cv2.cvtColor(cv2.merge((light, a, b)), cv2.COLOR_LAB2RGB)
        return np.asarray(Image.fromarray(boosted).filter(ImageFilter.UnsharpMask(radius=1.4, percent=135, threshold=3)))
    raise ValueError(f"Unknown preprocessing: {name}")


def deduplicate_lines(lines: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for line in sorted(lines, key=lambda item: float(item.get("score") or 0.0), reverse=True):
        normalized = normalize_text(str(line.get("normalized") or line.get("text") or ""))
        if not normalized:
            continue
        duplicate = False
        for existing in selected:
            previous = str(existing["normalized"])
            ratio = SequenceMatcher(None, normalized, previous).ratio()
            if ratio >= 0.90 or (len(normalized) >= 4 and (normalized in previous or previous in normalized)):
                duplicate = True
                break
        if not duplicate:
            selected.append({**line, "normalized": normalized})
    return selected


def create_rapid_engine(config: RapidConfig) -> Any:
    from rapidocr import LangDet, LangRec, ModelType, OCRVersion, RapidOCR

    model_types = {"tiny": ModelType.TINY, "small": ModelType.SMALL, "medium": ModelType.MEDIUM}
    recognizers = {"cyrillic": LangRec.CYRILLIC, "eslav": LangRec.ESLAV}
    return RapidOCR(
        params={
            "Global.use_cls": False,
            "Global.text_score": config.text_score,
            "Global.log_level": "error",
            "Det.lang_type": LangDet.CH,
            "Det.model_type": model_types[config.detector],
            "Det.ocr_version": OCRVersion.PPOCRV6,
            "Det.limit_side_len": config.limit_side_len,
            "Det.thresh": config.det_thresh,
            "Det.box_thresh": config.box_thresh,
            "Rec.lang_type": recognizers[config.recognizer],
            "Rec.model_type": ModelType.MOBILE,
            "Rec.ocr_version": OCRVersion.PPOCRV5,
            "EngineConfig.onnxruntime.intra_op_num_threads": 4,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        }
    )


def rapid_lines(engine: Any, path: Path, config: RapidConfig) -> tuple[list[dict[str, Any]], dict[str, float]]:
    started = time.perf_counter()
    with Image.open(path) as source:
        source.load()
        image = ImageOps.exif_transpose(source).convert("RGB")
    combined: list[dict[str, Any]] = []
    view_times: dict[str, float] = {}
    for view in config.views:
        view_started = time.perf_counter()
        result = engine(preprocess(crop_view(image, view), config.preprocess), use_cls=False)
        view_times[view] = (time.perf_counter() - view_started) * 1000
        # RapidOCR may return NumPy arrays.  Their truth value is ambiguous, so
        # avoid the convenient-looking ``array or []`` form here.
        texts = list(result.txts) if result.txts is not None else []
        scores = list(result.scores) if result.scores is not None else []
        boxes = list(result.boxes) if result.boxes is not None else []
        for index, text in enumerate(texts):
            normalized = normalize_text(str(text))
            if normalized:
                box = np.asarray(boxes[index]).round(1).tolist() if index < len(boxes) else None
                combined.append(
                    {
                        "text": str(text),
                        "normalized": normalized,
                        "score": float(scores[index]) if index < len(scores) else 1.0,
                        "view": view,
                        "box": box,
                    }
                )
    return deduplicate_lines(combined), {"total": (time.perf_counter() - started) * 1000, **view_times}


def run_rapid(config_names: list[str] | None, limit: int | None, force: bool) -> None:
    manifest = load_manifest()
    if limit:
        manifest = manifest[:limit]
    configs = [config for config in RAPID_CONFIGS if not config_names or config.name in config_names]
    unknown = set(config_names or []) - {config.name for config in configs}
    if unknown:
        raise SystemExit(f"Unknown configs: {', '.join(sorted(unknown))}")
    for config in configs:
        output = OUTPUT_DIR / "raw" / f"{config.name}.jsonl"
        completed = {row["name"] for row in read_jsonl(output)} if output.exists() and not force else set()
        if force and output.exists():
            output.unlink()
        print(f"[{config.name}] {len(completed)}/{len(manifest)} already complete", flush=True)
        engine = create_rapid_engine(config)
        for index, item in enumerate(manifest, start=1):
            if item["name"] in completed:
                continue
            started = time.perf_counter()
            try:
                lines, timing = rapid_lines(engine, ROOT / item["path"], config)
                row = {
                    "engine": "rapidocr",
                    "config": config.name,
                    "config_details": asdict(config),
                    "name": item["name"],
                    "lines": lines,
                    "timing_ms": timing,
                    "error": None,
                }
            except Exception as error:  # pragma: no cover - long-running artifact path
                row = {
                    "engine": "rapidocr",
                    "config": config.name,
                    "config_details": asdict(config),
                    "name": item["name"],
                    "lines": [],
                    "timing_ms": {"total": (time.perf_counter() - started) * 1000},
                    "error": f"{type(error).__name__}: {error}",
                }
            append_jsonl(output, row)
            print(f"[{config.name}] {index}/{len(manifest)} lines={len(row['lines'])} ms={row['timing_ms']['total']:.0f}", flush=True)


def build_ensemble(name: str, members: list[str], weights: list[float] | None, force: bool) -> None:
    output = OUTPUT_DIR / "raw" / f"{name}.jsonl"
    if output.exists() and not force:
        raise SystemExit(f"Output already exists: {output}; pass --force to replace it")
    sources = {member: read_jsonl(OUTPUT_DIR / "raw" / f"{member}.jsonl") for member in members}
    weights = weights or [1.0] * len(members)
    if len(weights) != len(members):
        raise SystemExit("--weights must contain exactly one value per --members entry")
    incomplete = [member for member, rows in sources.items() if len(rows) != len(load_manifest())]
    if incomplete:
        raise SystemExit(f"Ensemble members are incomplete: {', '.join(incomplete)}")
    by_member = {member: {row["name"]: row for row in rows} for member, rows in sources.items()}
    if output.exists():
        output.unlink()
    for item in load_manifest():
        source_rows = [by_member[member][item["name"]] for member in members]
        lines = deduplicate_lines(
            {
                **line,
                "source": member,
                "score": (1.0 if line.get("score") is None else line.get("score")) * weight,
            }
            for member, source, weight in zip(members, source_rows, weights, strict=True)
            for line in source.get("lines") or []
        )
        append_jsonl(
            output,
            {
                "engine": "ensemble",
                "config": name,
                "config_details": {"members": members, "weights": weights, "deduplication": "normalized_similarity_0.90"},
                "name": item["name"],
                "lines": lines,
                # Conservative serial timing.  CPU/GPU branches could later be
                # executed in parallel, where wall time approaches the maximum.
                "timing_ms": {
                    "total": sum(float((row.get("timing_ms") or {}).get("total") or 0.0) for row in source_rows),
                    "parallel_estimate": max(float((row.get("timing_ms") or {}).get("total") or 0.0) for row in source_rows),
                },
                "error": next((row.get("error") for row in source_rows if row.get("error")), None),
            },
        )
    print(f"Wrote {name}: {' + '.join(members)}")


def token_recall(lines: list[dict[str, Any]], target: dict[str, Any]) -> float:
    query = text_tokens(" ".join(str(line.get("normalized") or line.get("text") or "") for line in lines))
    expected = text_tokens(
        " ".join(
            [
                str(target.get("title") or ""),
                str(target.get("manufacturer") or ""),
                " ".join(str(value) for value in target.get("grapes") or []),
                str(target.get("category") or ""),
            ]
        )
    )
    if not query or not expected:
        return 0.0
    weights = [min(len(token), 10) for token in expected]
    matches = [max(SequenceMatcher(None, token, candidate).ratio() for candidate in query) for token in expected]
    return sum(weight * (score >= 0.72) for weight, score in zip(weights, matches, strict=True)) / sum(weights)


def coverage_metrics(lines: list[dict[str, Any]]) -> dict[str, float]:
    normalized_lines = [normalize_text(str(line.get("normalized") or line.get("text") or "")) for line in lines]
    tokens = [token for line in normalized_lines for token in text_tokens(line)]
    unique = set(tokens)
    scores = [float(line.get("score")) for line in lines if line.get("score") is not None]
    return {
        "line_count": float(len(normalized_lines)),
        "char_count": float(sum(len(line.replace(" ", "")) for line in normalized_lines)),
        "token_count": float(len(tokens)),
        "unique_token_count": float(len(unique)),
        "long_token_count": float(sum(len(token) >= 4 for token in unique)),
        "mean_confidence": float(statistics.fmean(scores)) if scores else math.nan,
    }


def aggregate(values: list[float]) -> float:
    finite = [value for value in values if not math.isnan(value)]
    return float(statistics.fmean(finite)) if finite else math.nan


def prepare_reference_tokens(
    references: list[dict[str, Any]],
) -> tuple[list[list[str]], list[np.ndarray], list[set[str]], list[str]]:
    token_lists = [
        text_tokens(
            " ".join(
                [
                    str(reference.get("metadata_text") or ""),
                    " ".join(str(line.get("normalized") or "") for line in reference.get("ocr_lines", [])),
                ]
            )
        )
        for reference in references
    ]
    vocabulary = sorted({token for tokens in token_lists for token in tokens if not token.isdigit()})
    lookup = {token: index for index, token in enumerate(vocabulary)}
    indexes = [np.asarray([lookup[token] for token in tokens if not token.isdigit()], dtype=np.int32) for tokens in token_lists]
    numbers = [{token for token in tokens if token.isdigit()} for tokens in token_lists]
    return token_lists, indexes, numbers, vocabulary


def fast_ocr_catalog_scores(
    query_lines: list[dict[str, Any]],
    reference_indexes: list[np.ndarray],
    reference_numbers: list[set[str]],
    vocabulary: list[str],
) -> np.ndarray:
    """Vectorized equivalent of the production OCR similarity for ranking.

    RapidFuzz computes the expensive token-to-vocabulary matrix in C++; the
    aggregation intentionally mirrors ``stage4.features.ocr_similarity``.
    """
    from rapidfuzz import fuzz, process

    query: list[tuple[str, float]] = []
    for line in query_lines:
        confidence = float(1.0 if line.get("score") is None else line.get("score") or 0.0)
        query.extend((token, confidence) for token in text_tokens(str(line.get("normalized") or "")))
    if not query:
        return np.zeros(len(reference_indexes), dtype=np.float32)
    words = [token for token, _ in query if not token.isdigit()]
    word_matrix = (
        process.cdist(words, vocabulary, scorer=fuzz.ratio, score_cutoff=55.0, dtype=np.uint8) / 100.0
        if words and vocabulary
        else np.empty((0, len(vocabulary)), dtype=np.float32)
    )
    scores = np.zeros(len(reference_indexes), dtype=np.float32)
    for reference_id, indexes in enumerate(reference_indexes):
        matches: list[tuple[float, float]] = []
        word_row = 0
        for token, confidence in query:
            if token.isdigit():
                similarity = 1.0 if token in reference_numbers[reference_id] else 0.0
            else:
                similarity = float(word_matrix[word_row, indexes].max()) if len(indexes) else 0.0
                word_row += 1
            if similarity >= 0.55:
                matches.append((similarity, confidence * min(len(token) / 7.0, 1.0)))
        if not matches:
            continue
        matches.sort(key=lambda item: item[0] * item[1], reverse=True)
        selected = matches[:6]
        total_weight = sum(weight for _, weight in selected)
        if total_weight > 0:
            evidence = min(len(selected) / 3.0, 1.0)
            scores[reference_id] = sum(similarity * weight for similarity, weight in selected) / total_weight * (0.55 + 0.45 * evidence)
    return scores


def score_outputs() -> list[dict[str, Any]]:
    manifest = {item["name"]: item for item in load_manifest()}
    references = read_jsonl(ROOT / "data/artifacts/stage4/index/reference_text.jsonl")
    reference_slugs = [str(row["slug"]) for row in references]
    reference_lookup = {slug: index for index, slug in enumerate(reference_slugs)}
    _, reference_indexes, reference_numbers, vocabulary = prepare_reference_tokens(references)
    summaries: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for path in sorted((OUTPUT_DIR / "raw").glob("*.jsonl")):
        rows = read_jsonl(path)
        if not rows:
            continue
        config = str(rows[0]["config"])
        for row in rows:
            item = manifest[row["name"]]
            metrics = coverage_metrics(row.get("lines") or [])
            target = item.get("target")
            fields = ocr_field_evidence(row.get("lines") or [], target) if target else {}
            retrieval: dict[str, float] = {}
            target_index = reference_lookup.get(str(item.get("correct_slug"))) if target else None
            if target_index is not None and len(rows) == len(manifest):
                scoring_lines = [
                    {**line, "score": 1.0 if line.get("score") is None else line.get("score")}
                    for line in (row.get("lines") or [])
                ]
                scores = fast_ocr_catalog_scores(
                    scoring_lines, reference_indexes, reference_numbers, vocabulary
                )
                order = np.argsort(-scores)
                rank = int(np.flatnonzero(order == target_index)[0]) + 1
                best_other = float(np.max(np.delete(scores, target_index))) if len(scores) > 1 else 0.0
                retrieval = {
                    "ocr_catalog_rank": float(rank),
                    "ocr_catalog_top1": float(rank == 1),
                    "ocr_catalog_top5": float(rank <= 5),
                    "ocr_catalog_mrr": 1.0 / rank,
                    "ocr_target_score": float(scores[target_index]),
                    "ocr_target_margin": float(scores[target_index] - best_other),
                }
            details.append(
                {
                    "config": config,
                    "engine": row.get("engine"),
                    "name": row["name"],
                    "labeled": bool(target),
                    "latency_ms": float((row.get("timing_ms") or {}).get("total") or 0.0),
                    "error": row.get("error"),
                    **metrics,
                    **fields,
                    **retrieval,
                    "metadata_token_recall": token_recall(row.get("lines") or [], target) if target else math.nan,
                }
            )
        own = [row for row in details if row["config"] == config]
        labeled = [row for row in own if row["labeled"]]
        coverage = aggregate([row["char_count"] for row in own])
        title = aggregate([row.get("title_evidence", math.nan) for row in labeled])
        manufacturer = aggregate([row.get("manufacturer_evidence", math.nan) for row in labeled])
        grape = aggregate([row.get("grape_evidence", math.nan) for row in labeled])
        metadata_recall = aggregate([row["metadata_token_recall"] for row in labeled])
        quality = 0.35 * title + 0.30 * manufacturer + 0.20 * grape + 0.15 * metadata_recall
        summaries.append(
            {
                "config": config,
                "engine": rows[0].get("engine"),
                "count": len(own),
                "labeled_count": len(labeled),
                "errors": sum(bool(row["error"]) for row in own),
                "avg_lines": aggregate([row["line_count"] for row in own]),
                "avg_chars": coverage,
                "avg_unique_tokens": aggregate([row["unique_token_count"] for row in own]),
                "avg_confidence": aggregate([row["mean_confidence"] for row in own]),
                "title_evidence": title,
                "manufacturer_evidence": manufacturer,
                "grape_evidence": grape,
                "year_hit_rate": aggregate([row.get("year_evidence", math.nan) for row in labeled]),
                "metadata_token_recall": metadata_recall,
                "quality_score": quality,
                "ocr_catalog_top1": aggregate([row.get("ocr_catalog_top1", math.nan) for row in labeled]),
                "ocr_catalog_top5": aggregate([row.get("ocr_catalog_top5", math.nan) for row in labeled]),
                "ocr_catalog_mrr": aggregate([row.get("ocr_catalog_mrr", math.nan) for row in labeled]),
                "ocr_target_margin": aggregate([row.get("ocr_target_margin", math.nan) for row in labeled]),
                "avg_latency_ms": aggregate([row["latency_ms"] for row in own]),
                "p95_latency_ms": float(np.percentile([row["latency_ms"] for row in own], 95)),
            }
        )
    if summaries:
        baseline = next((row for row in summaries if row["config"] == "rapid_current"), None)
        for row in summaries:
            row["coverage_gain"] = row["avg_chars"] / baseline["avg_chars"] - 1 if baseline and baseline["avg_chars"] else math.nan
            row["quality_gain"] = row["quality_score"] - baseline["quality_score"] if baseline else math.nan
        summaries.sort(key=lambda row: (row["quality_score"], row["avg_chars"]), reverse=True)
    write_json(OUTPUT_DIR / "summary.json", summaries)
    write_json(OUTPUT_DIR / "details.json", details)
    if summaries:
        with (OUTPUT_DIR / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
        lines = [
            "# OCR benchmark: 65 field photos",
            "",
            "Quality is measured on the 57 photos with an exact catalog label; coverage uses all 65 photos.",
            "",
            "| Rank | Config | Count | Chars | Coverage vs baseline | Quality | Δ quality | OCR top-1 | OCR top-5 | Title | Manufacturer | Grapes | Metadata recall | Avg ms | P95 ms |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for rank, row in enumerate(summaries, 1):
            lines.append(
                f"| {rank} | `{row['config']}` | {row['count']} | {row['avg_chars']:.1f} | {row['coverage_gain']:+.1%} | "
                f"{row['quality_score']:.3f} | {row['quality_gain']:+.3f} | {row['ocr_catalog_top1']:.1%} | {row['ocr_catalog_top5']:.1%} | {row['title_evidence']:.3f} | "
                f"{row['manufacturer_evidence']:.3f} | {row['grape_evidence']:.3f} | {row['metadata_token_recall']:.3f} | "
                f"{row['avg_latency_ms']:.0f} | {row['p95_latency_ms']:.0f} |"
            )
        (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("manifest")
    rapid = subparsers.add_parser("rapid")
    rapid.add_argument("--configs", nargs="*")
    rapid.add_argument("--limit", type=int)
    rapid.add_argument("--force", action="store_true")
    subparsers.add_parser("score")
    subparsers.add_parser("list")
    ensemble = subparsers.add_parser("ensemble")
    ensemble.add_argument("--name", required=True)
    ensemble.add_argument("--members", nargs="+", required=True)
    ensemble.add_argument("--weights", nargs="+", type=float)
    ensemble.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "manifest":
        manifest = build_manifest()
        print(f"Wrote {len(manifest)} rows; {sum(bool(row['target']) for row in manifest)} have exact catalog labels")
    elif args.command == "rapid":
        run_rapid(args.configs, args.limit, args.force)
    elif args.command == "score":
        summaries = score_outputs()
        print(json.dumps(summaries[:5], ensure_ascii=False, indent=2))
    elif args.command == "list":
        for config in RAPID_CONFIGS:
            print(config.name)
    elif args.command == "ensemble":
        build_ensemble(args.name, args.members, args.weights, args.force)


if __name__ == "__main__":
    main()
