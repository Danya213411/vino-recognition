"""Evaluate RapidOCR + GLM-OCR on all 65 field photos and export a review matrix.

RapidOCR remains the calibrated catalog-retrieval anchor.  GLM-OCR contributes
structured title/manufacturer/grape/year evidence, injects additional catalog
candidates, and strengthens the manufacturer gate.  OCR evidence used by the
hybrid score is ``max(rapid_score, glm_field_score)`` so existing strong Rapid
matches are never numerically weakened.

The expensive OCR inference was already completed by ``benchmark_ocr_65.py``;
this evaluator deliberately consumes those raw, per-photo outputs so reranking
experiments are reproducible without generating different OCR text each run.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
from rapidfuzz import fuzz, process

ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT, ROOT / "src", ROOT / "app" / "backend"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from app.backend.export_review_matrix import export as export_review_matrix
from stage1.catalog import read_jsonl, write_json, write_jsonl
from stage3.baseline import ranked
from stage3.embeddings import Dinov2Embedder, query_views
from stage3.index import VisualIndex
from stage4.baseline import normalize_rows
from stage4.features import extract_sift, normalize_text, ocr_similarity, sift_similarity, text_tokens
from stage4.index import HybridIndex
from stage4.manufacturer import apply_manufacturer_gate, manufacturer_values, resolve_taxonomy_value
from stage4.recognize import select_candidate_union
from stage5.features import runtime_confidence_features
from stage5.model import ConfidenceModel
from stage5.recognize import is_multimodal_verified


MANIFEST = ROOT / "data/artifacts/stage8/ocr_benchmark_65/manifest.json"
RAPID_RAW = ROOT / "data/artifacts/stage8/ocr_benchmark_65/raw/rapid_current.jsonl"
GLM_RAW = ROOT / "data/artifacts/stage8/ocr_benchmark_65/raw/glm_detail.jsonl"
CATALOG = ROOT / "data/artifacts/stage1/catalog_manifest.jsonl"
VISUAL_INDEX = ROOT / "data/artifacts/stage3/dinov2-small/index.npz"
HYBRID_INDEX = ROOT / "data/artifacts/stage4/index"
MODEL = ROOT / "data/artifacts/stage5/confidence_model.json"
CURRENT_RESULTS = ROOT / "data/artifacts/stage7/catalog_blurless_v2/field_66_robust_predictions.jsonl"
OUTPUT_DIR = ROOT / "data/artifacts/stage8/dual_ocr_65"
MANUAL_LABELS = OUTPUT_DIR / "manual_labels.json"
PREDICTIONS = OUTPUT_DIR / "predictions.jsonl"
REPORT_JSON = OUTPUT_DIR / "report.json"
REPORT_MD = OUTPUT_DIR / "report.md"
MATRIX = ROOT / "artifacts/review-matrix-dual-ocr-top1-errors.png"

YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")
WEIGHTS = {"embedding_weight": 0.45, "sift_weight": 0.05, "ocr_weight": 0.50}
VISUAL_COUNT = 30
RAPID_CANDIDATE_COUNT = 10
GLM_CANDIDATE_COUNT = 10


def rows_by(path: Path, key: str) -> dict[str, dict[str, Any]]:
    return {str(row[key]): row for row in read_jsonl(path)}


def merge_ocr_lines(
    rapid_lines: list[dict[str, Any]], glm_lines: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    positions: dict[str, int] = {}
    for source, lines in (("RapidOCR", rapid_lines), ("GLM-OCR", glm_lines)):
        for raw in lines:
            normalized = normalize_text(str(raw.get("normalized") or raw.get("text") or ""))
            if not normalized:
                continue
            if normalized in positions:
                current = merged[positions[normalized]]
                current["source"] = "RapidOCR+GLM-OCR"
                current["score"] = max(float(current.get("score") or 0.0), float(raw.get("score") or 1.0))
                continue
            positions[normalized] = len(merged)
            merged.append(
                {
                    **raw,
                    "normalized": normalized,
                    "score": float(raw.get("score") or 1.0),
                    "source": source,
                }
            )
    return merged


def prepare_field_index(
    catalog: list[dict[str, Any]],
) -> tuple[list[tuple[tuple[np.ndarray, np.ndarray], ...]], list[set[str]], list[str]]:
    record_tokens: list[list[list[str]]] = []
    vocabulary: set[str] = set()
    years: list[set[str]] = []
    for record in catalog:
        fields = [
            text_tokens(str(record.get("title") or "")),
            text_tokens(str(record.get("manufacturer") or "")),
            text_tokens(" ".join(str(value) for value in record.get("grapes") or [])),
        ]
        record_tokens.append(fields)
        for tokens in fields:
            vocabulary.update(tokens)
        years.append(
            set(
                YEAR_PATTERN.findall(
                    " ".join(
                        [
                            str(record.get("title") or ""),
                            str(record.get("image_alt") or ""),
                            str(record.get("slug") or ""),
                        ]
                    )
                )
            )
        )
    vocabulary_list = sorted(vocabulary)
    lookup = {token: index for index, token in enumerate(vocabulary_list)}
    indexed: list[tuple[tuple[np.ndarray, np.ndarray], ...]] = []
    for fields in record_tokens:
        indexed.append(
            tuple(
                (
                    np.asarray([lookup[token] for token in tokens], dtype=np.int32),
                    np.asarray([min(len(token), 10) for token in tokens], dtype=np.float32),
                )
                for tokens in fields
            )
        )
    return indexed, years, vocabulary_list


def glm_field_scores(
    lines: list[dict[str, Any]],
    field_index: list[tuple[tuple[np.ndarray, np.ndarray], ...]],
    years: list[set[str]],
    vocabulary: list[str],
) -> np.ndarray:
    query = text_tokens(" ".join(str(line.get("normalized") or line.get("text") or "") for line in lines))
    query_years = {token for token in query if YEAR_PATTERN.fullmatch(token)}
    similarities = (
        process.cdist(vocabulary, query, scorer=fuzz.ratio, dtype=np.uint8) / 100.0
        if query
        else np.zeros((len(vocabulary), 0), dtype=np.float32)
    )
    output = np.zeros(len(field_index), dtype=np.float32)
    for index, fields in enumerate(field_index):
        values: list[float] = []
        for token_indexes, weights in fields:
            if not len(token_indexes) or not query:
                values.append(0.0)
                continue
            best = similarities[token_indexes].max(axis=1)
            values.append(float(np.dot(best, weights) / weights.sum()))
        year = 1.0 if years[index].intersection(query_years) else 0.0
        output[index] = 0.45 * values[0] + 0.30 * values[1] + 0.20 * values[2] + 0.05 * year
    return output


def extend_candidates(
    base: np.ndarray,
    glm_scores: np.ndarray,
    visual_scores: np.ndarray,
    count: int,
) -> tuple[np.ndarray, list[set[int]]]:
    width = min(base.shape[1] + count, base.shape[1] + max(glm_scores.shape[1] - base.shape[1], 0))
    rows: list[list[int]] = []
    selected_sets: list[set[int]] = []
    for row in range(len(base)):
        selected = [int(value) for value in base[row]]
        seen = set(selected)
        glm_selected: set[int] = set()
        for value in np.argsort(-glm_scores[row]):
            candidate = int(value)
            if candidate not in seen:
                selected.append(candidate)
                seen.add(candidate)
                glm_selected.add(candidate)
            if len(glm_selected) >= count:
                break
        for value in np.argsort(-visual_scores[row]):
            candidate = int(value)
            if candidate not in seen:
                selected.append(candidate)
                seen.add(candidate)
            if len(selected) >= width:
                break
        rows.append(selected[:width])
        selected_sets.append(glm_selected)
    return np.asarray(rows, dtype=np.int64), selected_sets


def public_wine(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: record.get(key)
        for key in (
            "slug", "title", "manufacturer", "region", "category", "color",
            "description", "grapes", "alcohol", "temperature", "source_url",
        )
    } | {"image_url": str((ROOT / "data/parser" / str(record["image_local"])).resolve())}


def rank_metrics(rows: list[dict[str, Any]], rank_key: str) -> dict[str, Any]:
    labeled = [row for row in rows if row.get("expected_slug")]
    ranks = [row.get(rank_key) for row in labeled]
    return {
        "count": len(labeled),
        "top1_correct": sum(rank == 1 for rank in ranks),
        "top5_correct": sum(rank is not None and rank <= 5 for rank in ranks),
        "top1": sum(rank == 1 for rank in ranks) / len(labeled),
        "top5": sum(rank is not None and rank <= 5 for rank in ranks) / len(labeled),
        "mrr": float(mean(1 / rank if rank else 0.0 for rank in ranks)),
    }


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manual_labels = {}
    if MANUAL_LABELS.is_file():
        manual_labels = json.loads(MANUAL_LABELS.read_text(encoding="utf-8"))
    rapid = rows_by(RAPID_RAW, "name")
    glm = rows_by(GLM_RAW, "name")
    previous = rows_by(CURRENT_RESULTS, "file")
    catalog_rows = read_jsonl(CATALOG)
    catalog_by_slug = {str(row["slug"]): row for row in catalog_rows}
    visual_index = VisualIndex.load(VISUAL_INDEX)
    hybrid_index = HybridIndex.load(HYBRID_INDEX)
    if [str(value) for value in visual_index.slugs] != [str(value) for value in hybrid_index.slugs]:
        raise ValueError("Visual and hybrid index slug order differs")
    ordered_catalog = [catalog_by_slug[str(slug)] for slug in visual_index.slugs]
    paths = [ROOT / item["path"] for item in manifest]
    names = [str(item["name"]) for item in manifest]
    if set(names) - rapid.keys() or set(names) - glm.keys():
        raise ValueError("OCR benchmark outputs do not cover all 65 photos")

    embedder = Dinov2Embedder(visual_index.model_name, "cuda", True)
    embedder.warmup()
    started = time.perf_counter()
    full, _ = embedder.encode_paths(paths, query_views, 0, 8)
    medium, _ = embedder.encode_paths(paths, query_views, 1, 8)
    detail, _ = embedder.encode_paths(paths, query_views, 2, 8)
    full_view_scores = full @ visual_index.full.T
    medium_view_scores = medium @ visual_index.label.T
    detail_view_scores = detail @ visual_index.label.T
    view_scores = np.stack((full_view_scores, medium_view_scores, detail_view_scores), axis=1)
    visual_scores = 0.55 * full_view_scores + 0.45 * np.maximum(medium_view_scores, detail_view_scores)
    dino_ms = (time.perf_counter() - started) * 1000
    print(f"DINO complete: {dino_ms / 1000:.1f}s", flush=True)

    rapid_started = time.perf_counter()
    rapid_scores = np.zeros_like(visual_scores, dtype=np.float32)
    for row, name in enumerate(names):
        lines = rapid[name]["lines"]
        for column, reference in enumerate(hybrid_index.text_records):
            rapid_scores[row, column] = ocr_similarity(lines, reference)
        if (row + 1) % 5 == 0 or row + 1 == len(names):
            print(f"Rapid catalog scoring: {row + 1}/{len(names)}", flush=True)
    rapid_scoring_ms = (time.perf_counter() - rapid_started) * 1000

    field_index, catalog_years, vocabulary = prepare_field_index(ordered_catalog)
    glm_started = time.perf_counter()
    structured_scores = np.stack(
        [glm_field_scores(glm[name]["lines"], field_index, catalog_years, vocabulary) for name in names]
    )
    glm_scoring_ms = (time.perf_counter() - glm_started) * 1000
    print(f"GLM structured scoring complete: {glm_scoring_ms / 1000:.1f}s", flush=True)

    base_candidates, visual_sets, rapid_sets = select_candidate_union(
        visual_scores,
        rapid_scores,
        np.asarray([bool(rapid[name]["lines"]) for name in names]),
        VISUAL_COUNT,
        RAPID_CANDIDATE_COUNT,
    )
    candidates, glm_sets = extend_candidates(
        base_candidates, structured_scores, visual_scores, GLM_CANDIDATE_COUNT
    )
    combined_lines = [merge_ocr_lines(rapid[name]["lines"], glm[name]["lines"]) for name in names]
    manufacturers = manufacturer_values(catalog_by_slug)
    rapid_matches = [resolve_taxonomy_value(rapid[name]["lines"], manufacturers).to_dict() for name in names]
    manufacturer_matches = [resolve_taxonomy_value(lines, manufacturers).to_dict() for lines in combined_lines]
    candidates, manufacturer_sets = apply_manufacturer_gate(
        candidates,
        visual_scores,
        rapid_scores,
        visual_index.slugs,
        catalog_by_slug,
        manufacturer_matches,
    )

    embedding_scores = normalize_rows(np.take_along_axis(visual_scores, candidates, axis=1))
    rapid_selected = np.take_along_axis(rapid_scores, candidates, axis=1)
    glm_selected = np.take_along_axis(structured_scores, candidates, axis=1)
    effective_ocr = np.maximum(rapid_selected, glm_selected)

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
        print(f"SIFT: {row + 1}/{len(paths)}", flush=True)
    sift_ms = (time.perf_counter() - sift_started) * 1000

    final_scores = (
        WEIGHTS["embedding_weight"] * embedding_scores
        + WEIGHTS["sift_weight"] * sift_scores
        + WEIGHTS["ocr_weight"] * effective_ocr
    )
    payload = json.loads(MODEL.read_text(encoding="utf-8"))
    confidence_model = ConfidenceModel.from_dict(payload)
    results: list[dict[str, Any]] = []
    for row, item in enumerate(manifest):
        allowed = manufacturer_sets[row]
        valid = np.asarray(
            [column for column, candidate in enumerate(candidates[row]) if not allowed or int(candidate) in allowed],
            dtype=np.int64,
        )
        order = valid[np.argsort(-final_scores[row, valid])][:5]
        prediction_rows: list[dict[str, Any]] = []
        for column in order:
            candidate_index = int(candidates[row, column])
            slug = str(visual_index.slugs[candidate_index])
            prediction_rows.append(
                {
                    "slug": slug,
                    "score": float(final_scores[row, column]),
                    "embedding_score": float(embedding_scores[row, column]),
                    "sift_score": float(sift_scores[row, column]),
                    "sift_inliers": int(sift_inliers[row, column]),
                    "ocr_score": float(effective_ocr[row, column]),
                    "rapid_ocr_score": float(rapid_selected[row, column]),
                    "glm_field_score": float(glm_selected[row, column]),
                    "candidate_sources": [
                        source
                        for source, selected in (
                            ("visual", candidate_index in visual_sets[row]),
                            ("rapid_ocr", candidate_index in rapid_sets[row]),
                            ("glm_fields", candidate_index in glm_sets[row]),
                            ("manufacturer", candidate_index in manufacturer_sets[row]),
                        )
                        if selected
                    ],
                    "wine": public_wine(catalog_by_slug[slug]),
                }
            )

        winner = int(candidates[row, order[0]])
        winner_column = int(np.flatnonzero(candidates[row] == winner)[0])
        candidate_view_scores = view_scores[row][:, candidates[row]].copy()
        if allowed:
            allowed_mask = np.asarray([int(candidate) in allowed for candidate in candidates[row]])
            candidate_view_scores[:, ~allowed_mask] = -np.inf
        per_view_top1 = np.argmax(candidate_view_scores, axis=1)
        per_view_top5 = np.argpartition(-candidate_view_scores, 4, axis=1)[:, :5]
        _, counts = np.unique(per_view_top1, return_counts=True)
        winner_view_scores = candidate_view_scores[:, winner_column]
        crop_consistency = {
            "top1_slugs": [
                str(visual_index.slugs[candidates[row, column]]) for column in per_view_top1
            ],
            "top1_agreement": float(counts.max() / 3.0),
            "winner_top1_fraction": float((per_view_top1 == winner_column).mean()),
            "winner_top5_fraction": float((per_view_top5 == winner_column).any(axis=1).mean()),
            "winner_score_mean": float(winner_view_scores.mean()),
            "winner_score_std": float(winner_view_scores.std()),
        }
        rapid_ms = float((rapid[item["name"]].get("timing_ms") or {}).get("total") or 0.0)
        glm_ms = float((glm[item["name"]].get("timing_ms") or {}).get("total") or 0.0)
        timing = {
            "dino": dino_ms / len(paths),
            "sift": sift_ms / len(paths),
            "rapid_ocr": rapid_ms,
            "glm_ocr": glm_ms,
            "ocr_scoring": (rapid_scoring_ms + glm_scoring_ms) / len(paths),
        }
        timing["serial_total"] = sum(timing.values())
        timing["parallel_total"] = (
            max(timing["dino"], timing["rapid_ocr"], timing["glm_ocr"])
            + timing["ocr_scoring"]
            + timing["sift"]
        )
        # Keep the historical field for the matrix/export API. It deliberately
        # remains the conservative, fully serial estimate.
        timing["total"] = timing["serial_total"]
        raw_result = {
            "image": str(paths[row]),
            "slug": prediction_rows[0]["slug"],
            "predictions": prediction_rows,
            "ocr_lines": combined_lines[row],
            "manufacturer_match": manufacturer_matches[row],
            "manufacturer_match_rapid": rapid_matches[row],
            "crop_consistency": crop_consistency,
            "weights": WEIGHTS,
            "timing_ms": timing,
        }
        vector, evidence = runtime_confidence_features(
            raw_result, catalog_by_slug[prediction_rows[0]["slug"]]
        )
        calibrated = float(confidence_model.predict_proba(vector[None, :])[0])
        verified = is_multimodal_verified(prediction_rows[0], evidence)
        confidence = max(calibrated, 0.95) if verified else calibrated
        decision = "match" if verified else confidence_model.decision(confidence)
        expected = manual_labels.get(item["name"], item.get("correct_slug"))
        slugs = [prediction["slug"] for prediction in prediction_rows]
        expected_rank = slugs.index(expected) + 1 if expected and expected in slugs else None
        previous_row = previous.get(item["name"], {})
        results.append(
            {
                "file": item["name"],
                "image_path": str(paths[row].resolve()),
                "expected_slug": expected,
                "expected_rank": expected_rank,
                "previous_rank": previous_row.get("expected_rank"),
                "result": {
                    **raw_result,
                    "candidate_slug": prediction_rows[0]["slug"],
                    "confidence": confidence,
                    "calibrated_confidence": calibrated,
                    "verified_match": verified,
                    "decision": decision,
                    "evidence": evidence,
                },
            }
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(PREDICTIONS, results)
    current_metrics = rank_metrics(results, "previous_rank")
    dual_metrics = rank_metrics(results, "expected_rank")
    gains = [row["file"] for row in results if row["expected_slug"] and row["previous_rank"] != 1 and row["expected_rank"] == 1]
    losses = [row["file"] for row in results if row["expected_slug"] and row["previous_rank"] == 1 and row["expected_rank"] != 1]
    top5_gains = [row["file"] for row in results if row["expected_slug"] and row["previous_rank"] is None and row["expected_rank"] is not None]
    top5_losses = [row["file"] for row in results if row["expected_slug"] and row["previous_rank"] is not None and row["expected_rank"] is None]
    serial_timings = [float(row["result"]["timing_ms"]["serial_total"]) for row in results]
    parallel_timings = [float(row["result"]["timing_ms"]["parallel_total"]) for row in results]
    report = {
        "schema_version": 1,
        "photos": len(results),
        "labeled": dual_metrics["count"],
        "mechanism": {
            "rapid": "catalog retrieval anchor and native confidence",
            "glm": "structured title/manufacturer/grape/year evidence and 10 candidate injections",
            "deduplication": "exact normalized line, preserving source",
            "effective_ocr_score": "max(rapid_ocr_score, glm_field_score)",
            "weights": WEIGHTS,
        },
        "current": current_metrics,
        "dual_ocr": dual_metrics,
        "top1_gains": gains,
        "top1_losses": losses,
        "top5_gains": top5_gains,
        "top5_losses": top5_losses,
        "timing_ms": {
            "serial": {
                "mean": mean(serial_timings),
                "median": median(serial_timings),
                "p95": float(np.percentile(serial_timings, 95)),
                "max": max(serial_timings),
                "over_10s": sum(value > 10_000 for value in serial_timings),
            },
            "parallel_estimate": {
                "mean": mean(parallel_timings),
                "median": median(parallel_timings),
                "p95": float(np.percentile(parallel_timings, 95)),
                "max": max(parallel_timings),
                "over_10s": sum(value > 10_000 for value in parallel_timings),
            },
        },
    }
    write_json(REPORT_JSON, report)
    REPORT_MD.write_text(
        "\n".join(
            [
                "# Dual OCR: RapidOCR + GLM-OCR на 65 полевых кадрах",
                "",
                f"Точный slug известен для **{dual_metrics['count']}** из **{len(results)}** кадров.",
                "",
                "| Метрика | Текущий пайплайн | Dual OCR | Изменение |",
                "|---|---:|---:|---:|",
                f"| Top-1 | {current_metrics['top1_correct']}/{current_metrics['count']} ({current_metrics['top1']:.1%}) | {dual_metrics['top1_correct']}/{dual_metrics['count']} ({dual_metrics['top1']:.1%}) | {dual_metrics['top1_correct'] - current_metrics['top1_correct']:+d} |",
                f"| Top-5 | {current_metrics['top5_correct']}/{current_metrics['count']} ({current_metrics['top5']:.1%}) | {dual_metrics['top5_correct']}/{dual_metrics['count']} ({dual_metrics['top5']:.1%}) | {dual_metrics['top5_correct'] - current_metrics['top5_correct']:+d} |",
                f"| MRR | {current_metrics['mrr']:.3f} | {dual_metrics['mrr']:.3f} | {dual_metrics['mrr'] - current_metrics['mrr']:+.3f} |",
                "",
                f"- Исправления Top-1: **{len(gains)}**.",
                f"- Регрессии Top-1: **{len(losses)}**.",
                f"- Исправления Top-5: **{len(top5_gains)}**.",
                f"- Регрессии Top-5: **{len(top5_losses)}**.",
                f"- Последовательный расчёт: в среднем **{mean(serial_timings) / 1000:.2f} с**, P95 **{np.percentile(serial_timings, 95) / 1000:.2f} с**.",
                f"- Оценка при параллельном запуске DINO + двух OCR: в среднем **{mean(parallel_timings) / 1000:.2f} с**, P95 **{np.percentile(parallel_timings, 95) / 1000:.2f} с**; кадров дольше 10 с: **{sum(value > 10_000 for value in parallel_timings)}/{len(parallel_timings)}**.",
                "",
                "Полные кандидаты, строки обеих OCR и evidence сохранены в `predictions.jsonl`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    items = []
    for row in results:
        if not row["expected_slug"] or row["expected_rank"] == 1:
            continue
        items.append(
            {
                "name": row["file"],
                "input_image_url": row["image_path"],
                "recognition": {
                    "result": row["result"],
                    "feedback": {
                        "verdict": "incorrect" if row["expected_rank"] else "not_in_top5",
                        "correct_slug": row["expected_slug"],
                    },
                    "total_ms": row["result"]["timing_ms"]["total"],
                },
            }
        )
    export_review_matrix(
        MATRIX,
        items,
        title="Матрица ошибок Top-1 · Dual OCR",
        subtitle=(
            "RapidOCR ищет кандидатов, GLM-OCR читает производитель / название / сорт / год. "
            "Показаны только оставшиеся ошибки Top-1."
        ),
        weights_label="DINO 45%  ·  MAX(Rapid, GLM) 50%  ·  SIFT 5%",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"matrix={MATRIX}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
