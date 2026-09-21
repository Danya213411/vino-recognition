from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from stage1.catalog import read_jsonl, write_json, write_jsonl
from stage1.evaluation import evaluate_predictions
from stage3.baseline import ranked
from stage3.embeddings import Dinov2Embedder, query_views
from stage3.index import DEFAULT_OUTPUT_DIR as STAGE3_INDEX_DIR
from stage3.index import VisualIndex
from stage4.features import (
    extract_sift,
    initialize_ocr_worker,
    ocr_similarity,
    run_ocr_worker,
    sift_similarity,
)
from stage4.index import DEFAULT_OUTPUT_DIR as STAGE4_INDEX_DIR
from stage4.index import HybridIndex


DEFAULT_SYNTHETIC_MANIFEST = Path("data/synthetic/stage2/synthetic_manifest.jsonl")
DEFAULT_CATALOG_MANIFEST = Path("data/artifacts/stage1/catalog_manifest.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/artifacts/stage4/baseline")
DEFAULT_NEAR_DUPLICATES = Path("data/artifacts/stage1/near_duplicate_groups.json")
DEFAULT_EXACT_DUPLICATES = Path("data/artifacts/stage1/exact_duplicate_groups.json")


def normalize_rows(values: np.ndarray) -> np.ndarray:
    minimum = values.min(axis=1, keepdims=True)
    maximum = values.max(axis=1, keepdims=True)
    return (values - minimum) / np.maximum(maximum - minimum, 1e-7)


def calibration_mask(records: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [
            int(hashlib.sha1(str(record["sample_id"]).encode()).hexdigest()[:8], 16) % 5 == 0
            for record in records
        ],
        dtype=bool,
    )


def metric_tuple(
    scores: np.ndarray,
    candidates: np.ndarray,
    expected: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, float, float]:
    order = np.argsort(-scores, axis=1)
    ranked_candidates = np.take_along_axis(candidates, order, axis=1)
    selected = ranked_candidates[mask]
    truth = expected[mask]
    if not len(selected):
        return 0.0, 0.0, 0.0
    matches = selected == truth[:, None]
    top1 = float(matches[:, 0].mean())
    top5 = float(matches[:, :5].any(axis=1).mean())
    ranks = np.where(matches.any(axis=1), matches.argmax(axis=1) + 1, 0)
    mrr = float(np.where(ranks > 0, 1.0 / np.maximum(ranks, 1), 0.0).mean())
    return top1, top5, mrr


def choose_weights(
    embedding: np.ndarray,
    sift: np.ndarray,
    ocr: np.ndarray,
    candidates: np.ndarray,
    expected: np.ndarray,
    calibration: np.ndarray,
) -> tuple[float, float, dict[str, Any]]:
    best_local = (metric_tuple(embedding, candidates, expected, calibration), 0.0)
    for sift_weight in np.arange(0.05, 0.65, 0.05):
        scores = (1.0 - sift_weight) * embedding + sift_weight * sift
        metrics = metric_tuple(scores, candidates, expected, calibration)
        if (*metrics[:2], -sift_weight) > (*best_local[0][:2], -best_local[1]):
            best_local = (metrics, float(sift_weight))

    best_hybrid = (metric_tuple(embedding, candidates, expected, calibration), 0.0, 0.0)
    for sift_weight in np.arange(0.0, 0.65, 0.05):
        for ocr_weight in np.arange(0.05, 0.55, 0.05):
            if sift_weight + ocr_weight > 0.80:
                continue
            scores = (
                (1.0 - sift_weight - ocr_weight) * embedding
                + sift_weight * sift
                + ocr_weight * ocr
            )
            metrics = metric_tuple(scores, candidates, expected, calibration)
            if (*metrics[:2], -(sift_weight + ocr_weight)) > (
                *best_hybrid[0][:2],
                -(best_hybrid[1] + best_hybrid[2]),
            ):
                best_hybrid = (metrics, float(sift_weight), float(ocr_weight))

    return best_local[1], best_hybrid[2], {
        "embedding_sift": {
            "sift_weight": best_local[1],
            "calibration_top1": best_local[0][0],
            "calibration_top5": best_local[0][1],
        },
        "embedding_sift_ocr": {
            "sift_weight": best_hybrid[1],
            "ocr_weight": best_hybrid[2],
            "embedding_weight": 1.0 - best_hybrid[1] - best_hybrid[2],
            "calibration_top1": best_hybrid[0][0],
            "calibration_top5": best_hybrid[0][1],
        },
    }


def group_mask(records: list[dict[str, Any]], group_path: Path) -> np.ndarray:
    groups = json.loads(group_path.read_text(encoding="utf-8"))
    slugs = {str(item["slug"]) for group in groups for item in group["items"]}
    return np.asarray([str(record["expected_slug"]) in slugs for record in records], dtype=bool)


def named_metrics(
    scores: np.ndarray,
    candidates: np.ndarray,
    expected: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    top1, top5, mrr = metric_tuple(scores, candidates, expected, mask)
    return {
        "sample_count": int(mask.sum()),
        "top1_accuracy": top1,
        "top5_recall": top5,
        "mean_reciprocal_rank": mrr,
    }


def run_query_ocr(paths: list[Path], workers: int) -> list[dict[str, Any]]:
    arguments = [(str(path), False) for path in paths]
    if workers <= 1:
        initialize_ocr_worker()
        return [run_ocr_worker(argument) for argument in arguments]
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize_ocr_worker) as executor:
        return list(executor.map(run_ocr_worker, arguments, chunksize=8))


def build_signals(
    records: list[dict[str, Any]],
    paths: list[Path],
    visual_index: VisualIndex,
    hybrid_index: HybridIndex,
    output_dir: Path,
    device: str,
    batch_size: int,
    candidate_count: int,
    workers: int,
) -> tuple[dict[str, np.ndarray], dict[str, float], list[dict[str, Any]]]:
    if [str(slug) for slug in visual_index.slugs] != [str(slug) for slug in hybrid_index.slugs]:
        raise ValueError("Stage 3 and stage 4 indexes use a different slug order")

    embedder = Dinov2Embedder(visual_index.model_name, device, True)
    embedder.warmup()
    dino_started = time.perf_counter()
    full, _ = embedder.encode_paths(paths, query_views, 0, batch_size)
    medium, _ = embedder.encode_paths(paths, query_views, 1, batch_size)
    detail, _ = embedder.encode_paths(paths, query_views, 2, batch_size)
    full_view_scores = full @ visual_index.full.T
    medium_view_scores = medium @ visual_index.label.T
    detail_view_scores = detail @ visual_index.label.T
    visual_scores = 0.55 * full_view_scores + 0.45 * np.maximum(
        medium_view_scores, detail_view_scores
    )
    candidates, embedding_scores = ranked(visual_scores, candidate_count)
    view_scores = np.stack(
        [
            np.take_along_axis(scores, candidates, axis=1)
            for scores in (full_view_scores, medium_view_scores, detail_view_scores)
        ],
        axis=1,
    )
    dino_ms = (time.perf_counter() - dino_started) * 1000
    del embedder

    sift_started = time.perf_counter()
    sift_scores = np.zeros_like(embedding_scores, dtype=np.float32)
    sift_inliers = np.zeros_like(candidates, dtype=np.int16)
    for row, path in enumerate(paths):
        query_points, query_descriptors = extract_sift(path, reference=False)
        for column, candidate in enumerate(candidates[row]):
            reference_points, reference_descriptors = hybrid_index.features(int(candidate))
            score, details = sift_similarity(
                query_points, query_descriptors, reference_points, reference_descriptors
            )
            sift_scores[row, column] = score
            sift_inliers[row, column] = int(details["inliers"])
        if (row + 1) % 250 == 0:
            print(f"SIFT queries: {row + 1}/{len(paths)}")
    sift_ms = (time.perf_counter() - sift_started) * 1000

    ocr_cache = output_dir / "query_ocr.jsonl"
    ocr_started = time.perf_counter()
    cached = read_jsonl(ocr_cache) if ocr_cache.is_file() else []
    if [item.get("sample_id") for item in cached] == [item["sample_id"] for item in records]:
        query_ocr = cached
    else:
        raw_ocr = run_query_ocr(paths, workers)
        query_ocr = [
            {
                "sample_id": record["sample_id"],
                "image": path.as_posix(),
                **result,
            }
            for record, path, result in zip(records, paths, raw_ocr, strict=True)
        ]
        write_jsonl(ocr_cache, query_ocr)
    ocr_scores = np.zeros_like(embedding_scores, dtype=np.float32)
    for row, result in enumerate(query_ocr):
        for column, candidate in enumerate(candidates[row]):
            ocr_scores[row, column] = ocr_similarity(
                result["lines"], hybrid_index.text_records[int(candidate)]
            )
    ocr_ms = (time.perf_counter() - ocr_started) * 1000

    signals = {
        "candidates": candidates.astype(np.int32),
        "embedding": embedding_scores.astype(np.float32),
        "sift": sift_scores,
        "sift_inliers": sift_inliers,
        "ocr": ocr_scores,
        "view_scores": view_scores.astype(np.float32),
    }
    np.savez(output_dir / "signals.npz", **signals)
    timings = {"dino_ms": dino_ms, "sift_ms": sift_ms, "ocr_ms": ocr_ms}
    write_json(output_dir / "signal_timings.json", timings)
    return signals, timings, query_ocr


def predictions_for_scores(
    records: list[dict[str, Any]],
    paths: list[Path],
    slugs: np.ndarray,
    candidates: np.ndarray,
    scores: np.ndarray,
    latency_ms: float,
) -> list[dict[str, Any]]:
    order = np.argsort(-scores, axis=1)[:, :5]
    indices = np.take_along_axis(candidates, order, axis=1)
    values = np.take_along_axis(scores, order, axis=1)
    return [
        {
            "sample_id": record["sample_id"],
            "image": paths[row].as_posix(),
            "expected_slug": record["expected_slug"],
            "profile": record.get("profile"),
            "latency_ms": latency_ms,
            "predictions": [
                {"slug": str(slugs[index]), "score": float(values[row, rank])}
                for rank, index in enumerate(indices[row])
            ],
        }
        for row, record in enumerate(records)
    ]


def run_baseline(
    synthetic_manifest: Path,
    catalog_manifest: Path,
    visual_index_path: Path,
    hybrid_index_dir: Path,
    output_dir: Path,
    device: str = "cuda",
    batch_size: int = 16,
    candidate_count: int = 30,
    workers: int = 2,
    reuse_signals: bool = False,
    limit: int | None = None,
    near_duplicates_path: Path = DEFAULT_NEAR_DUPLICATES,
    exact_duplicates_path: Path = DEFAULT_EXACT_DUPLICATES,
) -> dict[str, Any]:
    records = read_jsonl(synthetic_manifest)
    if limit is not None:
        records = records[:limit]
    root = synthetic_manifest.parent
    paths = []
    for record in records:
        raw = Path(str(record["image"]))
        paths.append(raw if raw.is_absolute() or raw.is_file() else root / raw)
    output_dir.mkdir(parents=True, exist_ok=True)

    visual_index = VisualIndex.load(visual_index_path)
    hybrid_index = HybridIndex.load(hybrid_index_dir)
    signals_path = output_dir / "signals.npz"
    if reuse_signals and signals_path.is_file():
        with np.load(signals_path, allow_pickle=False) as payload:
            signals = {key: payload[key] for key in payload.files}
        timings = json.loads((output_dir / "signal_timings.json").read_text(encoding="utf-8"))
        query_ocr = read_jsonl(output_dir / "query_ocr.jsonl")
    else:
        signals, timings, query_ocr = build_signals(
            records,
            paths,
            visual_index,
            hybrid_index,
            output_dir,
            device,
            batch_size,
            candidate_count,
            workers,
        )

    slug_to_index = {str(slug): index for index, slug in enumerate(visual_index.slugs)}
    expected = np.asarray([slug_to_index[str(record["expected_slug"])] for record in records])
    candidates = signals["candidates"]
    calibration = calibration_mask(records)
    heldout = ~calibration
    embedding = normalize_rows(signals["embedding"])
    sift = signals["sift"].astype(np.float32)
    ocr = signals["ocr"].astype(np.float32)

    local_weight, _, tuning = choose_weights(
        embedding, sift, ocr, candidates, expected, calibration
    )
    hybrid_sift_weight = float(tuning["embedding_sift_ocr"]["sift_weight"])
    ocr_weight = float(tuning["embedding_sift_ocr"]["ocr_weight"])
    score_variants = {
        "embedding": embedding,
        "embedding_sift": (1.0 - local_weight) * embedding + local_weight * sift,
        "embedding_sift_ocr": (
            (1.0 - hybrid_sift_weight - ocr_weight) * embedding
            + hybrid_sift_weight * sift
            + ocr_weight * ocr
        ),
    }

    throughput_latency = sum(timings.values()) / len(records)
    mean_ocr_engine_ms = sum(float(item["latency_ms"]) for item in query_ocr) / len(query_ocr)
    estimated_serial_latency = (
        timings["dino_ms"] / len(records)
        + timings["sift_ms"] / len(records)
        + mean_ocr_engine_ms
    )
    evaluations: dict[str, Any] = {}
    for name, scores in score_variants.items():
        prediction_path = output_dir / f"predictions_{name}.jsonl"
        write_jsonl(
            prediction_path,
            predictions_for_scores(
                records, paths, visual_index.slugs, candidates, scores, estimated_serial_latency
            ),
        )
        evaluations[name] = evaluate_predictions(
            prediction_path, catalog_manifest, output_dir / f"evaluation_{name}"
        )

    candidate_recall = float((candidates == expected[:, None]).any(axis=1).mean())
    report = {
        "schema_version": 1,
        "sample_count": len(records),
        "calibration_count": int(calibration.sum()),
        "heldout_count": int(heldout.sum()),
        "candidate_count": candidate_count,
        "candidate_recall": candidate_recall,
        "weights": tuning,
        "timing_ms": {
            **timings,
            "batch_throughput_mean_per_image": throughput_latency,
            "estimated_serial_mean_per_image": estimated_serial_latency,
            "mean_ocr_engine_ms": mean_ocr_engine_ms,
        },
        "results": {},
    }
    for name, scores in score_variants.items():
        evaluation = evaluations[name]
        report["results"][name] = {
            "all": {
                "top1_accuracy": evaluation["summary"]["top1_accuracy"],
                "top5_recall": evaluation["summary"]["top5_recall"],
                "mean_reciprocal_rank": evaluation["summary"]["mean_reciprocal_rank"],
            },
            "calibration": dict(
                zip(
                    ("top1_accuracy", "top5_recall", "mean_reciprocal_rank"),
                    metric_tuple(scores, candidates, expected, calibration),
                    strict=True,
                )
            ),
            "heldout": dict(
                zip(
                    ("top1_accuracy", "top5_recall", "mean_reciprocal_rank"),
                    metric_tuple(scores, candidates, expected, heldout),
                    strict=True,
                )
            ),
            "profiles": evaluation["breakdowns"]["augmentation_profile"],
        }
    if limit is None and near_duplicates_path.is_file() and exact_duplicates_path.is_file():
        subset_masks = {
            "near_duplicates": group_mask(records, near_duplicates_path),
            "exact_duplicates": group_mask(records, exact_duplicates_path),
        }
        report["challenging_groups"] = {}
        for subset_name, subset_mask in subset_masks.items():
            report["challenging_groups"][subset_name] = {
                name: {
                    "all": named_metrics(scores, candidates, expected, subset_mask),
                    "heldout": named_metrics(
                        scores, candidates, expected, subset_mask & heldout
                    ),
                }
                for name, scores in score_variants.items()
            }
    write_json(output_dir / "stage4_report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate embedding + SIFT + OCR reranking.")
    parser.add_argument("--synthetic-manifest", type=Path, default=DEFAULT_SYNTHETIC_MANIFEST)
    parser.add_argument("--catalog-manifest", type=Path, default=DEFAULT_CATALOG_MANIFEST)
    parser.add_argument("--visual-index", type=Path, default=STAGE3_INDEX_DIR / "index.npz")
    parser.add_argument("--hybrid-index-dir", type=Path, default=STAGE4_INDEX_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--candidate-count", type=int, default=30)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--reuse-signals", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--near-duplicates", type=Path, default=DEFAULT_NEAR_DUPLICATES)
    parser.add_argument("--exact-duplicates", type=Path, default=DEFAULT_EXACT_DUPLICATES)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = run_baseline(
        args.synthetic_manifest,
        args.catalog_manifest,
        args.visual_index,
        args.hybrid_index_dir,
        args.output_dir,
        args.device,
        args.batch_size,
        args.candidate_count,
        max(args.workers, 1),
        args.reuse_signals,
        args.limit,
        args.near_duplicates,
        args.exact_duplicates,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
