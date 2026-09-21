from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from stage1.catalog import read_jsonl, write_json, write_jsonl
from stage3.embeddings import Dinov2Embedder, query_views
from stage3.index import DEFAULT_OUTPUT_DIR as STAGE3_INDEX_DIR
from stage3.index import VisualIndex
from stage4.baseline import (
    DEFAULT_OUTPUT_DIR as STAGE4_BASELINE_DIR,
    build_signals,
    calibration_mask,
    normalize_rows,
)
from stage4.index import DEFAULT_OUTPUT_DIR as STAGE4_INDEX_DIR
from stage4.index import HybridIndex
from stage4.recognize import load_weights
from stage5.features import FEATURE_NAMES, confidence_features
from stage5.model import (
    ConfidenceModel,
    binary_auc,
    choose_thresholds,
    expected_calibration_error,
    fit_logistic,
)


DEFAULT_KNOWN_MANIFEST = Path("data/synthetic/stage2/synthetic_manifest.jsonl")
DEFAULT_UNKNOWN_MANIFEST = Path("data/unknown/stage5/unknown_manifest.jsonl")
DEFAULT_CATALOG_MANIFEST = Path("data/artifacts/stage1/catalog_manifest.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/artifacts/stage5")


def manifest_paths(manifest: Path, records: list[dict[str, Any]]) -> list[Path]:
    root = manifest.parent
    result = []
    for record in records:
        raw = Path(str(record["image"]))
        result.append(raw if raw.is_absolute() or raw.is_file() else root / raw)
    return result


def candidate_view_scores(
    paths: list[Path],
    candidates: np.ndarray,
    index: VisualIndex,
    device: str,
    batch_size: int,
) -> np.ndarray:
    embedder = Dinov2Embedder(index.model_name, device, True)
    embedder.warmup()
    full, _ = embedder.encode_paths(paths, query_views, 0, batch_size)
    medium, _ = embedder.encode_paths(paths, query_views, 1, batch_size)
    detail, _ = embedder.encode_paths(paths, query_views, 2, batch_size)
    full_references = index.full[candidates]
    label_references = index.label[candidates]
    return np.stack(
        (
            np.einsum("nd,nkd->nk", full, full_references),
            np.einsum("nd,nkd->nk", medium, label_references),
            np.einsum("nd,nkd->nk", detail, label_references),
        ),
        axis=1,
    ).astype(np.float32)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def known_view_scores(
    paths: list[Path],
    candidates: np.ndarray,
    index: VisualIndex,
    cache_path: Path,
    device: str,
    batch_size: int,
) -> np.ndarray:
    if cache_path.is_file():
        cached = load_npz(cache_path)
        if np.array_equal(cached.get("candidates"), candidates):
            return cached["view_scores"].astype(np.float32)
    values = candidate_view_scores(paths, candidates, index, device, batch_size)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, candidates=candidates, view_scores=values)
    return values


def build_examples(
    records: list[dict[str, Any]],
    signals: dict[str, np.ndarray],
    query_ocr: list[dict[str, Any]],
    catalog_records: list[dict[str, Any]],
    slugs: np.ndarray,
    weights: dict[str, float],
    known: bool,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    embedding = normalize_rows(signals["embedding"].astype(np.float32))
    hybrid = (
        weights["embedding_weight"] * embedding
        + weights["sift_weight"] * signals["sift"]
        + weights["ocr_weight"] * signals["ocr"]
    )
    examples = []
    features = []
    for row, record in enumerate(records):
        order = np.argsort(-hybrid[row])
        winner_column = int(order[0])
        winner_index = int(signals["candidates"][row, winner_column])
        winner_slug = str(slugs[winner_index])
        vector, verified_column, details = confidence_features(
            hybrid[row],
            embedding[row],
            signals["sift"][row],
            signals["sift_inliers"][row],
            signals["ocr"][row],
            signals["view_scores"][row],
            query_ocr[row]["lines"],
            catalog_records[winner_index],
        )
        if verified_column != winner_column:
            raise AssertionError("Confidence feature winner differs from hybrid winner")
        expected_slug = str(record.get("expected_slug") or "")
        safe = bool(known and winner_slug == expected_slug)
        if known:
            split = "calibration" if calibration_mask([record])[0] else "heldout"
        else:
            split = str(record["split"])
        top5 = [
            {
                "slug": str(slugs[int(signals["candidates"][row, column])]),
                "score": float(hybrid[row, column]),
            }
            for column in order[:5]
        ]
        examples.append(
            {
                "sample_id": record["sample_id"],
                "image": record["image"],
                "known": known,
                "safe": safe,
                "split": split,
                "expected_slug": record.get("expected_slug"),
                "source_slug": record.get("source_slug"),
                "predicted_slug": winner_slug,
                "profile": record.get("profile"),
                "features": details,
                "top5": top5,
            }
        )
        features.append(vector)
    return examples, np.stack(features)


def evaluation_metrics(
    examples: list[dict[str, Any]],
    confidence: np.ndarray,
    model: ConfidenceModel,
    selected: np.ndarray,
) -> dict[str, Any]:
    known = np.asarray([bool(item["known"]) for item in examples]) & selected
    safe = np.asarray([bool(item["safe"]) for item in examples]) & selected
    unknown = ~np.asarray([bool(item["known"]) for item in examples]) & selected
    incorrect_known = known & ~safe
    accepted = (confidence >= model.accept_threshold) & selected
    alternatives = (
        (confidence >= model.review_threshold)
        & (confidence < model.accept_threshold)
        & selected
    )
    not_found = (confidence < model.review_threshold) & selected

    def rate(mask: np.ndarray, population: np.ndarray) -> float | None:
        return float(mask[population].mean()) if population.any() else None

    accepted_count = int(accepted.sum())
    unknown_source_acceptance: dict[str, bool] = {}
    for index, item in enumerate(examples):
        if unknown[index]:
            source = str(item.get("source_slug") or item["sample_id"])
            unknown_source_acceptance[source] = (
                unknown_source_acceptance.get(source, False) or bool(accepted[index])
            )

    def wilson(successes: int, total: int, z: float = 1.96) -> list[float] | None:
        if total == 0:
            return None
        observed = successes / total
        denominator = 1.0 + z * z / total
        center = (observed + z * z / (2.0 * total)) / denominator
        radius = (
            z
            * np.sqrt(observed * (1.0 - observed) / total + z * z / (4.0 * total * total))
            / denominator
        )
        return [float(max(0.0, center - radius)), float(min(1.0, center + radius))]

    unknown_accepts = int(accepted[unknown].sum())
    unknown_total = int(unknown.sum())
    source_accepts = sum(unknown_source_acceptance.values())
    source_total = len(unknown_source_acceptance)
    return {
        "sample_count": int(selected.sum()),
        "known_count": int(known.sum()),
        "correct_known_count": int(safe.sum()),
        "incorrect_known_count": int(incorrect_known.sum()),
        "unknown_count": int(unknown.sum()),
        "false_accept_rate_unknown": rate(accepted, unknown),
        "false_accept_rate_unknown_95ci": wilson(unknown_accepts, unknown_total),
        "false_accept_rate_unknown_sources": source_accepts / source_total
        if source_total
        else None,
        "false_accept_rate_unknown_sources_95ci": wilson(source_accepts, source_total),
        "wrong_known_accept_rate": rate(accepted, incorrect_known),
        "false_reject_rate_correct_known": rate(~accepted, safe),
        "known_match_coverage": rate(accepted, known),
        "accepted_precision": float(np.asarray([item["safe"] for item in examples])[accepted].mean())
        if accepted_count
        else None,
        "decisions": {
            "match": accepted_count,
            "alternatives": int(alternatives.sum()),
            "not_found": int(not_found.sum()),
        },
        "roc_auc_safe_vs_unsafe": binary_auc(safe[selected], confidence[selected]),
        "brier_score": float(np.mean((confidence[selected] - safe[selected].astype(float)) ** 2)),
        "expected_calibration_error": expected_calibration_error(
            safe[selected], confidence[selected]
        ),
    }


def run_calibration(
    known_manifest: Path,
    unknown_manifest: Path,
    catalog_manifest: Path,
    visual_index_path: Path,
    hybrid_index_dir: Path,
    stage4_signals_path: Path,
    stage4_ocr_path: Path,
    stage4_report_path: Path,
    output_dir: Path,
    device: str = "cuda",
    batch_size: int = 16,
    workers: int = 3,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    visual_index = VisualIndex.load(visual_index_path)
    hybrid_index = HybridIndex.load(hybrid_index_dir)
    catalog_records = read_jsonl(catalog_manifest)
    catalog_slugs = [str(record["slug"]) for record in catalog_records]
    if catalog_slugs != [str(slug) for slug in visual_index.slugs]:
        raise ValueError("Catalog and visual index slug order differs")
    if catalog_slugs != [str(slug) for slug in hybrid_index.slugs]:
        raise ValueError("Catalog and hybrid index slug order differs")
    weights = load_weights(stage4_report_path)

    known_records = read_jsonl(known_manifest)
    known_paths = manifest_paths(known_manifest, known_records)
    known_signals = load_npz(stage4_signals_path)
    if len(known_records) != len(known_signals["candidates"]):
        raise ValueError("Known manifest and stage 4 signals have different lengths")
    known_signals["view_scores"] = known_view_scores(
        known_paths,
        known_signals["candidates"],
        visual_index,
        output_dir / "known_view_scores.npz",
        device,
        batch_size,
    )
    known_ocr = read_jsonl(stage4_ocr_path)

    unknown_records = read_jsonl(unknown_manifest)
    unknown_paths = manifest_paths(unknown_manifest, unknown_records)
    unknown_dir = output_dir / "unknown_signals"
    unknown_signal_path = unknown_dir / "signals.npz"
    if unknown_signal_path.is_file():
        unknown_signals = load_npz(unknown_signal_path)
        unknown_ocr = read_jsonl(unknown_dir / "query_ocr.jsonl")
    else:
        unknown_dir.mkdir(parents=True, exist_ok=True)
        unknown_signals, _, unknown_ocr = build_signals(
            unknown_records,
            unknown_paths,
            visual_index,
            hybrid_index,
            unknown_dir,
            device,
            batch_size,
            30,
            workers,
        )
    if "view_scores" not in unknown_signals:
        unknown_signals["view_scores"] = candidate_view_scores(
            unknown_paths,
            unknown_signals["candidates"],
            visual_index,
            device,
            batch_size,
        )

    known_examples, known_features = build_examples(
        known_records,
        known_signals,
        known_ocr,
        catalog_records,
        visual_index.slugs,
        weights,
        True,
    )
    unknown_examples, unknown_features = build_examples(
        unknown_records,
        unknown_signals,
        unknown_ocr,
        catalog_records,
        visual_index.slugs,
        weights,
        False,
    )
    examples = known_examples + unknown_examples
    features = np.concatenate((known_features, unknown_features))
    calibration = np.asarray([item["split"] == "calibration" for item in examples])
    heldout = ~calibration
    safe = np.asarray([bool(item["safe"]) for item in examples])
    known = np.asarray([bool(item["known"]) for item in examples])

    model = fit_logistic(features[calibration], safe[calibration], FEATURE_NAMES)
    initial_confidence = model.predict_proba(features)
    accept_threshold, review_threshold = choose_thresholds(
        initial_confidence[calibration], safe[calibration], known[calibration]
    )
    model = ConfidenceModel(
        model.feature_names,
        model.mean,
        model.scale,
        model.coefficients,
        model.intercept,
        accept_threshold,
        review_threshold,
    )
    confidence = model.predict_proba(features)
    for item, score in zip(examples, confidence, strict=True):
        item["confidence"] = float(score)
        item["decision"] = model.decision(float(score))
    write_jsonl(output_dir / "confidence_examples.jsonl", examples)

    model_payload = {
        **model.to_dict(),
        "hybrid_weights": weights,
        "visual_model": visual_index.model_name,
        "visual_index_sha256": hashlib.sha256(visual_index_path.read_bytes()).hexdigest(),
        "target_unknown_far": 0.025,
        "target_wrong_known_accept_rate": 0.05,
    }
    write_json(output_dir / "confidence_model.json", model_payload)
    report = {
        "schema_version": 1,
        "known_sample_count": len(known_examples),
        "unknown_sample_count": len(unknown_examples),
        "feature_names": list(FEATURE_NAMES),
        "thresholds": {
            "accept": accept_threshold,
            "review": review_threshold,
            "selection": "lowest threshold satisfying calibration unknown FAR <= 2.5% and wrong-known accept <= 5%",
        },
        "model": {
            "type": "L2-regularized logistic regression",
            "coefficients": {
                name: float(value) for name, value in zip(FEATURE_NAMES, model.coefficients, strict=True)
            },
            "intercept": model.intercept,
        },
        "calibration": evaluation_metrics(examples, confidence, model, calibration),
        "heldout": evaluation_metrics(examples, confidence, model, heldout),
        "all": evaluation_metrics(examples, confidence, model, np.ones(len(examples), dtype=bool)),
    }
    write_json(output_dir / "stage5_report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate wine-match confidence and OOD rejection.")
    parser.add_argument("--known-manifest", type=Path, default=DEFAULT_KNOWN_MANIFEST)
    parser.add_argument("--unknown-manifest", type=Path, default=DEFAULT_UNKNOWN_MANIFEST)
    parser.add_argument("--catalog-manifest", type=Path, default=DEFAULT_CATALOG_MANIFEST)
    parser.add_argument("--visual-index", type=Path, default=STAGE3_INDEX_DIR / "index.npz")
    parser.add_argument("--hybrid-index-dir", type=Path, default=STAGE4_INDEX_DIR)
    parser.add_argument(
        "--stage4-signals", type=Path, default=STAGE4_BASELINE_DIR / "signals.npz"
    )
    parser.add_argument(
        "--stage4-ocr", type=Path, default=STAGE4_BASELINE_DIR / "query_ocr.jsonl"
    )
    parser.add_argument(
        "--stage4-report", type=Path, default=STAGE4_BASELINE_DIR / "stage4_report.json"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=3)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = run_calibration(
        args.known_manifest,
        args.unknown_manifest,
        args.catalog_manifest,
        args.visual_index,
        args.hybrid_index_dir,
        args.stage4_signals,
        args.stage4_ocr,
        args.stage4_report,
        args.output_dir,
        args.device,
        args.batch_size,
        args.workers,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
