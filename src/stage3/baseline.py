from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from stage1.catalog import read_jsonl, write_json, write_jsonl
from stage1.evaluation import evaluate_predictions
from stage3.embeddings import Dinov2Embedder, query_views
from stage3.index import DEFAULT_MANIFEST, DEFAULT_OUTPUT_DIR, VisualIndex


DEFAULT_SYNTHETIC_MANIFEST = Path("data/synthetic/stage2/synthetic_manifest.jsonl")
DEFAULT_OUTPUT = Path("data/artifacts/stage3/baseline")
PATCH_WEIGHTS = (0.20, 0.35, 0.50)


def ranked(scores: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    top_k = min(top_k, scores.shape[1])
    candidates = np.argpartition(-scores, top_k - 1, axis=1)[:, :top_k]
    values = np.take_along_axis(scores, candidates, axis=1)
    order = np.argsort(-values, axis=1)
    return np.take_along_axis(candidates, order, axis=1), np.take_along_axis(values, order, axis=1)


def ranked_candidates(
    candidates: np.ndarray,
    scores: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    top_k = min(top_k, scores.shape[1])
    order = np.argsort(-scores, axis=1)[:, :top_k]
    return np.take_along_axis(candidates, order, axis=1), np.take_along_axis(scores, order, axis=1)


def prediction_records(
    records: list[dict[str, Any]],
    paths: list[Path],
    slugs: np.ndarray,
    indices: np.ndarray,
    values: np.ndarray,
    latency_ms: float,
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for row, record in enumerate(records):
        predictions.append(
            {
                "sample_id": record["sample_id"],
                "image": paths[row].as_posix(),
                "expected_slug": record["expected_slug"],
                "profile": record.get("profile"),
                "predictions": [
                    {"slug": str(slugs[item]), "score": float(values[row, rank])}
                    for rank, item in enumerate(indices[row])
                ],
                "latency_ms": latency_ms,
            }
        )
    return predictions


def run_baseline(
    synthetic_manifest: Path,
    catalog_manifest: Path,
    index_path: Path,
    output_dir: Path,
    model_name: str | None = None,
    device: str = "auto",
    batch_size: int = 16,
    top_k: int = 5,
    limit: int | None = None,
    local_files_only: bool = False,
    candidate_count: int = 30,
) -> dict[str, Any]:
    records = read_jsonl(synthetic_manifest)
    if limit is not None:
        records = records[:limit]
    if not records:
        raise ValueError(f"Synthetic manifest is empty: {synthetic_manifest}")

    root = synthetic_manifest.parent
    paths: list[Path] = []
    for record in records:
        raw_path = Path(str(record["image"]))
        path = raw_path if raw_path.is_absolute() or raw_path.is_file() else root / raw_path
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.append(path)

    index = VisualIndex.load(index_path)
    selected_model = model_name or index.model_name
    embedder = Dinov2Embedder(selected_model, device, local_files_only)
    if embedder.embedding_size != index.full.shape[1]:
        raise ValueError(
            f"Model/index mismatch: {selected_model} produces {embedder.embedding_size} features, "
            f"but {index_path} contains {index.full.shape[1]}"
        )
    embedder.warmup()
    started = time.perf_counter()

    full_vectors, full_ms = embedder.encode_paths(paths, query_views, 0, batch_size)
    medium_vectors, medium_patches, medium_ms = embedder.encode_paths_with_patches(
        paths, query_views, 1, batch_size, index.patch_grid
    )
    detail_vectors, detail_patches, detail_ms = embedder.encode_paths_with_patches(
        paths, query_views, 2, batch_size, index.patch_grid
    )
    query_labels = np.stack((medium_vectors, detail_vectors), axis=1)
    query_patches = np.stack((medium_patches, detail_patches), axis=1)

    full_scores, _, multiscale_scores = index.cls_scores(full_vectors, query_labels)
    full_indices, full_values = ranked(full_scores, top_k)
    multiscale_indices, multiscale_values = ranked(multiscale_scores, top_k)

    candidate_count = min(max(candidate_count, top_k), len(index.slugs))
    candidate_indices, candidate_cls = ranked(multiscale_scores, candidate_count)
    patch_started = time.perf_counter()
    patch_scores = index.patch_scores(query_patches, candidate_indices, str(embedder.device))
    patch_ms = (time.perf_counter() - patch_started) * 1000

    variants: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "cls_full": (full_indices, full_values),
        "cls_multiscale": (multiscale_indices, multiscale_values),
        "patch_only": ranked_candidates(candidate_indices, patch_scores, top_k),
    }
    for weight in PATCH_WEIGHTS:
        blended = (1.0 - weight) * candidate_cls + weight * patch_scores
        variants[f"cls_patch_{int(weight * 100):02d}"] = ranked_candidates(
            candidate_indices, blended, top_k
        )

    total_ms = (time.perf_counter() - started) * 1000
    mean_latency_ms = total_ms / len(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_paths: dict[str, Path] = {}
    for name, (indices, values) in variants.items():
        path = output_dir / f"predictions_{name}.jsonl"
        write_jsonl(
            path,
            prediction_records(
                records, paths, index.slugs, indices, values, mean_latency_ms
            ),
        )
        prediction_paths[name] = path

    evaluations = {
        name: evaluate_predictions(path, catalog_manifest, output_dir / f"evaluation_{name}")
        for name, path in prediction_paths.items()
    }
    report: dict[str, Any] = {
        "schema_version": 2,
        "sample_count": len(records),
        "index": index_path.as_posix(),
        "model": selected_model,
        "configuration": {
            "views": ["full", "medium", "detail"],
            "patch_grid": index.patch_grid,
            "candidate_count": candidate_count,
            "patch_weights": list(PATCH_WEIGHTS),
        },
        "timing_ms": {
            "full_embedding": full_ms,
            "medium_embedding": medium_ms,
            "detail_embedding": detail_ms,
            "patch_reranking": patch_ms,
            "end_to_end": total_ms,
            "mean_per_image": mean_latency_ms,
        },
        "runtime": embedder.runtime_info(),
        "results": {
            name: {
                "top1_accuracy": evaluation["summary"]["top1_accuracy"],
                "top5_recall": evaluation["summary"]["top5_recall"],
                "mean_reciprocal_rank": evaluation["summary"]["mean_reciprocal_rank"],
                "mean_top1_margin": evaluation["confidence"]["mean_top1_margin"],
                "profiles": evaluation["breakdowns"]["augmentation_profile"],
            }
            for name, evaluation in evaluations.items()
        },
    }
    write_json(output_dir / "baseline_report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate multi-scale DINOv2 retrieval.")
    parser.add_argument("--synthetic-manifest", type=Path, default=DEFAULT_SYNTHETIC_MANIFEST)
    parser.add_argument("--catalog-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--index", type=Path, default=DEFAULT_OUTPUT_DIR / "index.npz")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-count", type=int, default=30)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = run_baseline(
        args.synthetic_manifest,
        args.catalog_manifest,
        args.index,
        args.output_dir,
        args.model,
        args.device,
        args.batch_size,
        args.top_k,
        args.limit,
        args.local_files_only,
        args.candidate_count,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
