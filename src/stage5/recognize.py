from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from stage1.catalog import read_jsonl, write_jsonl
from stage3.embeddings import Dinov2Embedder
from stage3.index import DEFAULT_OUTPUT_DIR as STAGE3_INDEX_DIR
from stage3.index import VisualIndex
from stage3.recognize import collect_paths
from stage4.features import create_ocr_engine
from stage4.index import DEFAULT_OUTPUT_DIR as STAGE4_INDEX_DIR
from stage4.index import HybridIndex
from stage4.recognize import recognize_paths as recognize_hybrid_paths
from stage5.calibrate import DEFAULT_CATALOG_MANIFEST, DEFAULT_OUTPUT_DIR
from stage5.features import FEATURE_NAMES, runtime_confidence_features
from stage5.model import ConfidenceModel
from stage5.semantic_filters import rerank_predictions


def is_multimodal_verified(
    prediction: dict[str, Any], evidence: dict[str, float]
) -> bool:
    return bool(
        float(prediction["ocr_score"]) >= 0.92
        and float(prediction["sift_score"]) >= 0.72
        and int(prediction["sift_inliers"]) >= 12
        and float(evidence["title_evidence"]) >= 0.82
    )


def recognize_confident_paths(
    paths: list[Path],
    visual_index: VisualIndex,
    hybrid_index: HybridIndex,
    embedder: Dinov2Embedder,
    ocr_engine: Any,
    catalog_by_slug: dict[str, dict[str, Any]],
    model_payload: dict[str, Any],
    top_k: int = 5,
    candidate_count: int = 30,
    batch_size: int = 8,
    ocr_candidate_count: int = 10,
) -> list[dict[str, Any]]:
    model = ConfidenceModel.from_dict(model_payload)
    if model.feature_names != FEATURE_NAMES:
        raise ValueError("Confidence model feature schema differs from runtime")
    weights = {key: float(value) for key, value in model_payload["hybrid_weights"].items()}
    raw_results = recognize_hybrid_paths(
        paths,
        visual_index,
        hybrid_index,
        embedder,
        ocr_engine,
        weights,
        top_k,
        candidate_count,
        batch_size,
        # Dual OCR has a second structured retrieval channel.  Give it the
        # same extra candidate budget used by the offline dual-OCR benchmark
        # (10 Rapid candidates + 10 GLM-field candidates).
        ocr_candidate_count + (10 if getattr(ocr_engine, "dual_enabled", False) else 0),
        catalog_by_slug,
    )
    results = []
    for raw in raw_results:
        raw["predictions"] = rerank_predictions(
            raw["predictions"], raw.get("ocr_lines", []), catalog_by_slug
        )
        raw["slug"] = raw["predictions"][0]["slug"]
        candidate_slug = str(raw["slug"])
        vector, evidence = runtime_confidence_features(raw, catalog_by_slug[candidate_slug])
        calibrated_confidence = float(model.predict_proba(vector[None, :])[0])
        winner = raw["predictions"][0]
        verified_match = is_multimodal_verified(winner, evidence)
        confidence = max(calibrated_confidence, 0.95) if verified_match else calibrated_confidence
        decision = "match" if verified_match else model.decision(confidence)
        alternatives = raw["predictions"] if decision == "alternatives" else []
        results.append(
            {
                "image": raw["image"],
                "decision": decision,
                "slug": candidate_slug if decision == "match" else None,
                "confidence": confidence,
                "calibrated_confidence": calibrated_confidence,
                "confidence_source": (
                    "multimodal_verification" if verified_match else "calibrated_model"
                ),
                "verified_match": verified_match,
                "candidate_slug": candidate_slug,
                "alternatives": alternatives,
                "thresholds": {
                    "accept": model.accept_threshold,
                    "review": model.review_threshold,
                },
                "evidence": evidence,
                "predictions": raw["predictions"],
                "ocr_lines": raw["ocr_lines"],
                "manufacturer_match": raw.get("manufacturer_match"),
                "crop_consistency": raw["crop_consistency"],
                "timing_ms": raw["timing_ms"],
            }
        )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recognize wine with calibrated match/alternatives/not-found confidence."
    )
    parser.add_argument("path", type=Path)
    parser.add_argument("--visual-index", type=Path, default=STAGE3_INDEX_DIR / "index.npz")
    parser.add_argument("--hybrid-index-dir", type=Path, default=STAGE4_INDEX_DIR)
    parser.add_argument("--catalog-manifest", type=Path, default=DEFAULT_CATALOG_MANIFEST)
    parser.add_argument(
        "--confidence-model", type=Path, default=DEFAULT_OUTPUT_DIR / "confidence_model.json"
    )
    parser.add_argument("--model")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-count", type=int, default=30)
    parser.add_argument("--ocr-candidate-count", type=int, default=10)
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
    catalog = read_jsonl(args.catalog_manifest)
    catalog_by_slug = {str(record["slug"]): record for record in catalog}
    model_payload = json.loads(args.confidence_model.read_text(encoding="utf-8"))
    results = recognize_confident_paths(
        paths,
        visual_index,
        hybrid_index,
        embedder,
        ocr_engine,
        catalog_by_slug,
        model_payload,
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
