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
from stage3.index import DEFAULT_OUTPUT_DIR, VisualIndex


def recognize_paths(
    paths: list[Path],
    index: VisualIndex,
    embedder: Dinov2Embedder,
    top_k: int = 5,
    batch_size: int = 8,
    candidate_count: int = 30,
    patch_weight: float = 0.0,
) -> list[dict[str, Any]]:
    if not 0.0 <= patch_weight <= 1.0:
        raise ValueError("patch_weight must be between 0 and 1")
    embedder.warmup()
    started = time.perf_counter()
    full_vectors, _ = embedder.encode_paths(paths, query_views, 0, batch_size)
    medium_vectors, medium_patches, _ = embedder.encode_paths_with_patches(
        paths, query_views, 1, batch_size, index.patch_grid
    )
    detail_vectors, detail_patches, _ = embedder.encode_paths_with_patches(
        paths, query_views, 2, batch_size, index.patch_grid
    )
    query_labels = np.stack((medium_vectors, detail_vectors), axis=1)
    query_patches = np.stack((medium_patches, detail_patches), axis=1)
    _, _, cls_scores = index.cls_scores(full_vectors, query_labels)
    candidate_count = min(max(candidate_count, top_k), len(index.slugs))
    candidate_indices, candidate_cls = ranked(cls_scores, candidate_count)
    if patch_weight > 0.0:
        local_scores = index.patch_scores(
            query_patches, candidate_indices, str(embedder.device), batch_size=max(1, batch_size)
        )
        final_scores = (1.0 - patch_weight) * candidate_cls + patch_weight * local_scores
    else:
        final_scores = candidate_cls
    indices, scores = ranked_candidates(candidate_indices, final_scores, top_k)
    elapsed_ms = (time.perf_counter() - started) * 1000
    latency_ms = elapsed_ms / len(paths) if paths else 0.0

    results: list[dict[str, Any]] = []
    for row, path in enumerate(paths):
        predictions = [
            {"slug": str(index.slugs[item]), "score": float(scores[row, rank])}
            for rank, item in enumerate(indices[row])
        ]
        results.append(
            {
                "image": path.as_posix(),
                "slug": predictions[0]["slug"],
                "predictions": predictions,
                "latency_ms": latency_ms,
                "model": embedder.model_name,
                "patch_weight": patch_weight,
            }
        )
    return results


def collect_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return [item for item in sorted(path.iterdir()) if item.suffix.lower() in extensions]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recognize wine with multi-scale DINOv2 retrieval.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--index", type=Path, default=DEFAULT_OUTPUT_DIR / "index.npz")
    parser.add_argument("--model")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-count", type=int, default=30)
    parser.add_argument("--patch-weight", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    paths = collect_paths(args.path)
    if not paths:
        raise ValueError(f"No supported images found: {args.path}")
    index = VisualIndex.load(args.index)
    embedder = Dinov2Embedder(args.model or index.model_name, args.device, args.local_files_only)
    if embedder.embedding_size != index.full.shape[1]:
        raise ValueError(
            f"Model/index mismatch: {embedder.model_name} produces {embedder.embedding_size} features, "
            f"but {args.index} contains {index.full.shape[1]}"
        )
    results = recognize_paths(
        paths,
        index,
        embedder,
        args.top_k,
        args.batch_size,
        args.candidate_count,
        args.patch_weight,
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
