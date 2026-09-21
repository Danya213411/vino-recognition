from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from stage1.catalog import write_jsonl
from stage3.baseline import ranked, ranked_candidates
from stage3.embeddings import Dinov2Embedder, query_views
from stage3.index import DEFAULT_OUTPUT_DIR as STAGE3_INDEX_DIR
from stage3.index import VisualIndex
from stage3.recognize import collect_paths
from stage4.baseline import DEFAULT_OUTPUT_DIR as STAGE4_BASELINE_DIR
from stage4.baseline import normalize_rows
from stage4.features import create_ocr_engine, extract_sift, ocr_similarity, run_ocr, sift_similarity
from stage4.index import DEFAULT_OUTPUT_DIR as STAGE4_INDEX_DIR
from stage4.index import HybridIndex


DEFAULT_WEIGHTS = {"embedding_weight": 0.45, "sift_weight": 0.15, "ocr_weight": 0.40}
DEFAULT_OCR_CANDIDATE_COUNT = 10


def select_candidate_union(
    visual_scores: np.ndarray,
    ocr_scores: np.ndarray,
    has_ocr: np.ndarray,
    visual_count: int,
    ocr_count: int,
) -> tuple[np.ndarray, list[set[int]], list[set[int]]]:
    """Keep visual retrieval broad while allowing strong text matches into reranking."""
    pool_count = min(visual_count + ocr_count, visual_scores.shape[1])
    visual_pool, _ = ranked(visual_scores, pool_count)
    selected_rows: list[list[int]] = []
    visual_candidate_sets: list[set[int]] = []
    ocr_candidate_sets: list[set[int]] = []
    for row in range(len(visual_scores)):
        visual_candidates = [int(value) for value in visual_pool[row, :visual_count]]
        selected = visual_candidates.copy()
        seen = set(selected)
        ocr_selected: set[int] = set()
        if bool(has_ocr[row]) and ocr_count:
            for value in np.argsort(-ocr_scores[row]):
                candidate = int(value)
                if float(ocr_scores[row, candidate]) <= 0.0:
                    break
                if candidate not in seen:
                    selected.append(candidate)
                    seen.add(candidate)
                    ocr_selected.add(candidate)
                if len(selected) >= pool_count:
                    break
        for value in visual_pool[row]:
            candidate = int(value)
            if candidate not in seen:
                selected.append(candidate)
                seen.add(candidate)
            if len(selected) >= pool_count:
                break
        selected_rows.append(selected)
        visual_candidate_sets.append(set(visual_candidates))
        ocr_candidate_sets.append(ocr_selected)
    return np.asarray(selected_rows, dtype=np.int64), visual_candidate_sets, ocr_candidate_sets


def load_weights(report_path: Path) -> dict[str, float]:
    if not report_path.is_file():
        return DEFAULT_WEIGHTS.copy()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    values = report["weights"]["embedding_sift_ocr"]
    return {key: float(values[key]) for key in DEFAULT_WEIGHTS}


def recognize_paths(
    paths: list[Path],
    visual_index: VisualIndex,
    hybrid_index: HybridIndex,
    embedder: Dinov2Embedder,
    ocr_engine: Any,
    weights: dict[str, float],
    top_k: int = 5,
    candidate_count: int = 30,
    batch_size: int = 8,
    ocr_candidate_count: int = DEFAULT_OCR_CANDIDATE_COUNT,
) -> list[dict[str, Any]]:
    if [str(slug) for slug in visual_index.slugs] != [str(slug) for slug in hybrid_index.slugs]:
        raise ValueError("Stage 3 and stage 4 indexes use a different slug order")
    if abs(sum(weights.values()) - 1.0) > 1e-6 or min(weights.values()) < 0:
        raise ValueError("Hybrid weights must be non-negative and sum to 1")

    total_started = time.perf_counter()
    dino_started = time.perf_counter()
    full, _ = embedder.encode_paths(paths, query_views, 0, batch_size)
    medium, _ = embedder.encode_paths(paths, query_views, 1, batch_size)
    detail, _ = embedder.encode_paths(paths, query_views, 2, batch_size)
    full_view_scores = full @ visual_index.full.T
    medium_view_scores = medium @ visual_index.label.T
    detail_view_scores = detail @ visual_index.label.T
    view_scores = np.stack((full_view_scores, medium_view_scores, detail_view_scores), axis=1)
    all_scores = 0.55 * full_view_scores + 0.45 * np.maximum(
        medium_view_scores, detail_view_scores
    )
    dino_ms = (time.perf_counter() - dino_started) * 1000
    candidate_count = min(max(candidate_count, top_k), len(visual_index.slugs))
    ocr_candidate_count = min(max(ocr_candidate_count, 0), len(visual_index.slugs) - candidate_count)

    ocr_started = time.perf_counter()
    query_ocr = [run_ocr(path, False, ocr_engine) for path in paths]
    global_ocr_scores = np.zeros(
        (len(paths), len(hybrid_index.text_records)), dtype=np.float32
    )
    for row, result in enumerate(query_ocr):
        for column, reference in enumerate(hybrid_index.text_records):
            global_ocr_scores[row, column] = ocr_similarity(result["lines"], reference)
    ocr_ms = (time.perf_counter() - ocr_started) * 1000

    candidates, visual_candidate_sets, ocr_candidate_sets = select_candidate_union(
        all_scores,
        global_ocr_scores,
        np.asarray([bool(result["lines"]) for result in query_ocr]),
        candidate_count,
        ocr_candidate_count,
    )
    raw_embedding = np.take_along_axis(all_scores, candidates, axis=1)
    embedding_scores = normalize_rows(raw_embedding)

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
    sift_ms = (time.perf_counter() - sift_started) * 1000

    ocr_scores = np.take_along_axis(global_ocr_scores, candidates, axis=1)

    final_scores = (
        weights["embedding_weight"] * embedding_scores
        + weights["sift_weight"] * sift_scores
        + weights["ocr_weight"] * ocr_scores
    )
    indices, scores = ranked_candidates(candidates, final_scores, top_k)
    order = np.argsort(-final_scores, axis=1)[:, :top_k]
    ranked_embedding = np.take_along_axis(embedding_scores, order, axis=1)
    ranked_sift = np.take_along_axis(sift_scores, order, axis=1)
    ranked_inliers = np.take_along_axis(sift_inliers, order, axis=1)
    ranked_ocr = np.take_along_axis(ocr_scores, order, axis=1)
    total_ms = (time.perf_counter() - total_started) * 1000
    count = max(len(paths), 1)

    results = []
    for row, path in enumerate(paths):
        predictions = [
            {
                "slug": str(visual_index.slugs[index]),
                "score": float(scores[row, rank]),
                "embedding_score": float(ranked_embedding[row, rank]),
                "sift_score": float(ranked_sift[row, rank]),
                "sift_inliers": int(ranked_inliers[row, rank]),
                "ocr_score": float(ranked_ocr[row, rank]),
                "candidate_sources": [
                    source
                    for source, selected in (
                        ("visual", int(index) in visual_candidate_sets[row]),
                        ("ocr", int(index) in ocr_candidate_sets[row]),
                    )
                    if selected
                ],
            }
            for rank, index in enumerate(indices[row])
        ]
        winner = int(indices[row, 0])
        candidate_view_scores = view_scores[row][:, candidates[row]]
        winner_column = int(np.flatnonzero(candidates[row] == winner)[0])
        per_view_top1 = np.argmax(candidate_view_scores, axis=1)
        view_top_k = min(5, candidate_view_scores.shape[1])
        per_view_top5 = np.argpartition(
            -candidate_view_scores, view_top_k - 1, axis=1
        )[:, :view_top_k]
        unique, counts = np.unique(per_view_top1, return_counts=True)
        del unique
        winner_view_scores = candidate_view_scores[:, winner_column]
        crop_consistency = {
            "top1_slugs": [
                str(visual_index.slugs[candidates[row, column]]) for column in per_view_top1
            ],
            "top1_agreement": float(counts.max() / len(per_view_top1)),
            "winner_top1_fraction": float((per_view_top1 == winner_column).mean()),
            "winner_top5_fraction": float(
                (per_view_top5 == winner_column).any(axis=1).mean()
            ),
            "winner_score_mean": float(winner_view_scores.mean()),
            "winner_score_std": float(winner_view_scores.std()),
        }
        results.append(
            {
                "image": path.as_posix(),
                "slug": predictions[0]["slug"],
                "predictions": predictions,
                "ocr_lines": query_ocr[row]["lines"],
                "crop_consistency": crop_consistency,
                "weights": weights,
                "timing_ms": {
                    "dino": dino_ms / count,
                    "sift": sift_ms / count,
                    "ocr": ocr_ms / count,
                    "total": total_ms / count,
                },
            }
        )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recognize wine with DINOv2 + SIFT + OCR.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--visual-index", type=Path, default=STAGE3_INDEX_DIR / "index.npz")
    parser.add_argument("--hybrid-index-dir", type=Path, default=STAGE4_INDEX_DIR)
    parser.add_argument(
        "--weights-report", type=Path, default=STAGE4_BASELINE_DIR / "stage4_report.json"
    )
    parser.add_argument("--model")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-count", type=int, default=30)
    parser.add_argument("--ocr-candidate-count", type=int, default=DEFAULT_OCR_CANDIDATE_COUNT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    paths = collect_paths(args.path)
    if not paths:
        raise ValueError(f"No supported images found: {args.path}")
    visual_index = VisualIndex.load(args.visual_index)
    hybrid_index = HybridIndex.load(args.hybrid_index_dir)
    embedder = Dinov2Embedder(args.model or visual_index.model_name, args.device, args.local_files_only)
    if embedder.embedding_size != visual_index.full.shape[1]:
        raise ValueError("DINOv2 model and visual index embedding sizes differ")
    embedder.warmup()
    ocr_engine = create_ocr_engine()
    results = recognize_paths(
        paths,
        visual_index,
        hybrid_index,
        embedder,
        ocr_engine,
        load_weights(args.weights_report),
        args.top_k,
        args.candidate_count,
        args.batch_size,
        args.ocr_candidate_count,
    )
    if args.output:
        write_jsonl(args.output, results)
    if len(results) == 1:
        print(json.dumps(results[0], ensure_ascii=False, indent=2))
    else:
        print(f"recognized={len(results)}; output={args.output or '<stdout omitted>'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
