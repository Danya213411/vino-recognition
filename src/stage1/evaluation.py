from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean, median
from typing import Any

from stage1.catalog import read_jsonl, write_json, write_jsonl


DEFAULT_MANIFEST = Path("data/artifacts/stage1/catalog_manifest.jsonl")


def percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(probability * len(ordered)) - 1)
    return ordered[index]


def normalize_predictions(record: dict[str, Any], line_number: int) -> list[dict[str, Any]]:
    raw_predictions = record.get("predictions")
    if not isinstance(raw_predictions, list):
        raise ValueError(f"Record {line_number}: 'predictions' must be an array")

    fallback_scores = record.get("scores")
    if fallback_scores is not None and not isinstance(fallback_scores, list):
        raise ValueError(f"Record {line_number}: 'scores' must be an array when present")

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, prediction in enumerate(raw_predictions):
        if isinstance(prediction, str):
            slug = prediction
            score = fallback_scores[index] if fallback_scores and index < len(fallback_scores) else None
        elif isinstance(prediction, dict):
            slug = prediction.get("slug")
            score = prediction.get("score")
        else:
            raise ValueError(f"Record {line_number}: prediction {index + 1} has an invalid type")

        if not isinstance(slug, str) or not slug.strip():
            raise ValueError(f"Record {line_number}: prediction {index + 1} has no slug")
        slug = slug.strip()
        if slug in seen:
            continue
        seen.add(slug)

        if score is not None and not isinstance(score, (int, float)):
            raise ValueError(f"Record {line_number}: score for '{slug}' is not numeric")
        if score is not None and not math.isfinite(float(score)):
            raise ValueError(f"Record {line_number}: score for '{slug}' must be finite")
        normalized.append({"slug": slug, "score": float(score) if score is not None else None})
    return normalized


def breakdown(
    samples: list[dict[str, Any]],
    manifest: dict[str, dict[str, Any]],
    field: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        expected = manifest.get(sample["expected_slug"])
        if expected:
            grouped[str(expected.get(field) or "<missing>")].append(sample)

    result: list[dict[str, Any]] = []
    for value, group in grouped.items():
        count = len(group)
        result.append(
            {
                "value": value,
                "sample_count": count,
                "top1_accuracy": sum(sample["rank"] == 1 for sample in group) / count,
                "top5_recall": sum(sample["rank"] is not None and sample["rank"] <= 5 for sample in group) / count,
            }
        )
    return sorted(result, key=lambda item: (-item["sample_count"], item["value"]))


def sample_field_breakdown(samples: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        value = sample.get(field)
        if value is not None:
            grouped[str(value)].append(sample)

    result: list[dict[str, Any]] = []
    for value, group in grouped.items():
        count = len(group)
        result.append(
            {
                "value": value,
                "sample_count": count,
                "top1_accuracy": sum(sample["rank"] == 1 for sample in group) / count,
                "top5_recall": sum(
                    sample["rank"] is not None and sample["rank"] <= 5 for sample in group
                )
                / count,
            }
        )
    return sorted(result, key=lambda item: (-item["sample_count"], item["value"]))


def evaluate_predictions(
    predictions_path: Path,
    manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    manifest_records = read_jsonl(manifest_path)
    manifest = {record["slug"]: record for record in manifest_records}
    if len(manifest) != len(manifest_records):
        raise ValueError("Manifest contains duplicate slugs")

    raw_records = read_jsonl(predictions_path)
    if not raw_records:
        raise ValueError(f"Predictions file is empty: {predictions_path}")
    samples: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    invalid_prediction_slugs: Counter[str] = Counter()
    latencies: list[float] = []
    margins: list[float] = []

    for line_number, record in enumerate(raw_records, start=1):
        sample_id = str(record.get("sample_id") or record.get("image") or f"sample-{line_number:06d}")
        expected_slug = record.get("expected_slug")
        if not isinstance(expected_slug, str) or not expected_slug.strip():
            raise ValueError(f"Record {line_number}: 'expected_slug' must be a non-empty string")
        expected_slug = expected_slug.strip()
        if expected_slug not in manifest:
            raise ValueError(f"Record {line_number}: expected slug is absent from manifest: {expected_slug}")

        predictions = normalize_predictions(record, line_number)
        for prediction in predictions:
            if prediction["slug"] not in manifest:
                invalid_prediction_slugs[prediction["slug"]] += 1

        rank = next(
            (index for index, prediction in enumerate(predictions, start=1) if prediction["slug"] == expected_slug),
            None,
        )
        latency = record.get("latency_ms")
        if latency is not None:
            if not isinstance(latency, (int, float)) or latency < 0 or not math.isfinite(float(latency)):
                raise ValueError(f"Record {line_number}: latency_ms must be a non-negative number")
            latencies.append(float(latency))

        margin = None
        if len(predictions) >= 2:
            first_score = predictions[0]["score"]
            second_score = predictions[1]["score"]
            if first_score is not None and second_score is not None:
                margin = first_score - second_score
                margins.append(margin)

        normalized_sample = {
            "sample_id": sample_id,
            "expected_slug": expected_slug,
            "profile": record.get("profile"),
            "predictions": predictions,
            "rank": rank,
            "latency_ms": float(latency) if latency is not None else None,
            "top1_margin": margin,
        }
        samples.append(normalized_sample)

        top1_slug = predictions[0]["slug"] if predictions else None
        if rank != 1:
            expected = manifest[expected_slug]
            predicted = manifest.get(top1_slug) if top1_slug else None
            errors.append(
                {
                    "sample_id": sample_id,
                    "expected_slug": expected_slug,
                    "expected_title": expected.get("title"),
                    "predicted_slug": top1_slug,
                    "predicted_title": predicted.get("title") if predicted else None,
                    "expected_rank": rank,
                    "same_manufacturer": bool(predicted) and expected.get("manufacturer") == predicted.get("manufacturer"),
                    "same_category": bool(predicted) and expected.get("category") == predicted.get("category"),
                    "same_title": bool(predicted) and expected.get("title") == predicted.get("title"),
                    "top1_margin": margin,
                    "latency_ms": normalized_sample["latency_ms"],
                    "predictions": predictions[:5],
                }
            )

    sample_count = len(samples)
    top1_correct = sum(sample["rank"] == 1 for sample in samples)
    top5_correct = sum(sample["rank"] is not None and sample["rank"] <= 5 for sample in samples)
    reciprocal_ranks = [1 / sample["rank"] if sample["rank"] else 0.0 for sample in samples]
    missing_from_predictions = sum(sample["rank"] is None for sample in samples)

    report: dict[str, Any] = {
        "schema_version": 1,
        "inputs": {
            "predictions_path": predictions_path.as_posix(),
            "manifest_path": manifest_path.as_posix(),
        },
        "summary": {
            "sample_count": sample_count,
            "top1_correct": top1_correct,
            "top1_accuracy": top1_correct / sample_count if sample_count else None,
            "top5_correct": top5_correct,
            "top5_recall": top5_correct / sample_count if sample_count else None,
            "mean_reciprocal_rank": fmean(reciprocal_ranks) if reciprocal_ranks else None,
            "expected_slug_missing_from_predictions": missing_from_predictions,
            "error_count": len(errors),
            "invalid_prediction_slug_count": sum(invalid_prediction_slugs.values()),
            "unique_invalid_prediction_slug_count": len(invalid_prediction_slugs),
        },
        "confidence": {
            "margin_sample_count": len(margins),
            "mean_top1_margin": fmean(margins) if margins else None,
            "median_top1_margin": median(margins) if margins else None,
            "p05_top1_margin": percentile(margins, 0.05),
        },
        "latency_ms": {
            "sample_count": len(latencies),
            "mean": fmean(latencies) if latencies else None,
            "median": median(latencies) if latencies else None,
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
            "max": max(latencies) if latencies else None,
        },
        "invalid_prediction_slugs": dict(sorted(invalid_prediction_slugs.items())),
        "breakdowns": {
            "augmentation_profile": sample_field_breakdown(samples, "profile"),
            "manufacturer": breakdown(samples, manifest, "manufacturer"),
            "region": breakdown(samples, manifest, "region"),
            "category": breakdown(samples, manifest, "category"),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "evaluation_report.json", report)
    write_jsonl(output_dir / "evaluation_errors.jsonl", errors)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate ranked wine predictions.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=Path("data/artifacts/evaluation"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = evaluate_predictions(args.predictions, args.manifest, args.output_dir)
    summary = report["summary"]
    latency = report["latency_ms"]
    top1 = summary["top1_accuracy"]
    top5 = summary["top5_recall"]
    print(
        f"samples={summary['sample_count']}; "
        f"top1={top1:.4f}; top5={top5:.4f}; "
        f"p95_ms={latency['p95']}; errors={summary['error_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
