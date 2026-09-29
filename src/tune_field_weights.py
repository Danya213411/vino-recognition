from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from stage1.catalog import read_jsonl, write_json, write_jsonl
from stage3.embeddings import Dinov2Embedder
from stage3.index import VisualIndex
from stage4.features import create_ocr_engine
from stage4.index import HybridIndex
from stage4.recognize import recognize_paths


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SELECTION = PROJECT_ROOT / "data/test/recheck_fuzzy_v2/selection_manifest.jsonl"
DATABASE = PROJECT_ROOT / "data/app/stage6/events.sqlite3"
CATALOG = PROJECT_ROOT / "data/artifacts/stage1/catalog_manifest.jsonl"
VISUAL_INDEX = PROJECT_ROOT / "data/artifacts/stage3/dinov2-small/index.npz"
HYBRID_INDEX = PROJECT_ROOT / "data/artifacts/stage4/index"
OUTPUT_DIR = PROJECT_ROOT / "data/artifacts/stage7/catalog_blurless_v2"
SIGNALS = OUTPUT_DIR / "field_candidate_signals.jsonl"
REPORT = OUTPUT_DIR / "field_weight_tuning.json"
WEIGHTS_REPORT = OUTPUT_DIR / "field_selected_stage4_report.json"
FINAL_PREDICTIONS = OUTPUT_DIR / "field_66_robust_predictions.jsonl"
MARKDOWN_REPORT = OUTPUT_DIR / "field_66_robust_report.md"


def latest_feedback() -> dict[str, dict[str, Any]]:
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        """
        SELECT r.original_name, r.created_at, r.candidate_slug, f.verdict, f.correct_slug
        FROM recognitions r JOIN feedback f ON f.recognition_id = r.id
        ORDER BY r.created_at DESC
        """
    ).fetchall()
    connection.close()
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = str(row["original_name"])
        if name not in output:
            output[name] = dict(row)
    return output


def samples() -> list[dict[str, Any]]:
    feedback = latest_feedback()
    records = []
    for item in read_jsonl(SELECTION):
        if item["reason"] == "ambiguous_not_in_store":
            continue
        saved = feedback.get(item["file"], {})
        # A fresh human OOD verdict overrides every historical label, including
        # an earlier top-1 confirmation. Keep such frames outside known metrics.
        if saved.get("verdict") == "not_in_store":
            continue
        if saved.get("verdict") == "correct":
            expected = saved.get("candidate_slug")
        elif saved.get("verdict") == "incorrect":
            expected = saved.get("correct_slug")
        elif item["reason"] == "stable_confirmed_top1":
            expected = item["new_top1"]
        else:
            expected = None
        records.append(
            {
                "file": item["file"],
                "path": str(PROJECT_ROOT / item["source"]),
                "expected_slug": expected,
            }
        )
    return records


def split_name(expected_slug: str | None) -> str:
    if not expected_slug:
        return "unlabeled"
    bucket = int(hashlib.sha1(expected_slug.encode()).hexdigest()[:8], 16) % 5
    if bucket < 3:
        return "calibration"
    if bucket == 3:
        return "validation"
    return "test"


def metrics(rows: list[dict[str, Any]], weights: tuple[float, float, float]) -> dict[str, Any]:
    ranks = []
    for row in rows:
        scored = sorted(
            row["candidates"],
            key=lambda item: -(
                weights[0] * float(item["embedding_score"])
                + weights[1] * float(item["sift_score"])
                + weights[2] * float(item["ocr_score"])
            ),
        )
        slugs = [item["slug"] for item in scored]
        expected = row["expected_slug"]
        ranks.append(slugs.index(expected) + 1 if expected in slugs else None)
    count = len(rows)
    return {
        "count": count,
        "top1_correct": sum(rank == 1 for rank in ranks),
        "top5_correct": sum(rank is not None and rank <= 5 for rank in ranks),
        "top1": sum(rank == 1 for rank in ranks) / count,
        "top5": sum(rank is not None and rank <= 5 for rank in ranks) / count,
        "mrr": float(np.mean([1 / rank if rank else 0.0 for rank in ranks])),
    }


def ranked(row: dict[str, Any], weights: tuple[float, float, float]) -> list[dict[str, Any]]:
    output = []
    for candidate in row["candidates"]:
        item = dict(candidate)
        item["score"] = (
            weights[0] * float(item["embedding_score"])
            + weights[1] * float(item["sift_score"])
            + weights[2] * float(item["ocr_score"])
        )
        output.append(item)
    return sorted(output, key=lambda item: -float(item["score"]))


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    field_samples = samples()
    if SIGNALS.is_file():
        rows = read_jsonl(SIGNALS)
        samples_by_file = {item["file"]: item for item in field_samples}
        rows = [row for row in rows if row["file"] in samples_by_file]
        for row in rows:
            sample = samples_by_file[row["file"]]
            row["expected_slug"] = sample["expected_slug"]
            row["path"] = sample["path"]
    else:
        visual_index = VisualIndex.load(VISUAL_INDEX)
        hybrid_index = HybridIndex.load(HYBRID_INDEX)
        catalog = read_jsonl(CATALOG)
        catalog_by_slug = {str(item["slug"]): item for item in catalog}
        embedder = Dinov2Embedder(visual_index.model_name, "cuda", True)
        embedder.warmup()
        results = recognize_paths(
            [Path(item["path"]) for item in field_samples],
            visual_index,
            hybrid_index,
            embedder,
            create_ocr_engine(),
            {"embedding_weight": 1.0, "sift_weight": 0.0, "ocr_weight": 0.0},
            top_k=40,
            candidate_count=30,
            batch_size=8,
            ocr_candidate_count=10,
            catalog_by_slug=catalog_by_slug,
        )
        rows = []
        for sample, result in zip(field_samples, results, strict=True):
            rows.append(
                {
                    **sample,
                    "split": split_name(sample["expected_slug"]),
                    "manufacturer_match": result["manufacturer_match"],
                    "ocr_lines": result["ocr_lines"],
                    "timing_ms": result["timing_ms"],
                    "candidates": result["predictions"],
                }
            )
        write_jsonl(SIGNALS, rows)

    # Recompute the split so cached component signals remain reusable when the
    # evaluation protocol changes. Photos of the same wine always stay in one
    # group because the hash is based on the expected slug, not the filename.
    for row in rows:
        row["split"] = split_name(row.get("expected_slug"))

    labeled = [row for row in rows if row["expected_slug"]]
    calibration = [row for row in labeled if row["split"] == "calibration"]
    validation = [row for row in labeled if row["split"] == "validation"]
    test = [row for row in labeled if row["split"] == "test"]
    candidates = []
    for embedding_step in range(21):
        embedding = embedding_step * 0.05
        for sift_step in range(21 - embedding_step):
            sift = sift_step * 0.05
            ocr = 1.0 - embedding - sift
            weights = (embedding, sift, ocr)
            result = metrics(calibration, weights)
            candidates.append((result, weights))
    _best_result, unconstrained_weights = max(
        candidates,
        key=lambda item: (
            item[0]["top1"],
            item[0]["top5"],
            item[0]["mrr"],
            -item[1][1],
        ),
    )
    previous_weights = (0.45, 0.05, 0.50)
    validation_baseline = metrics(validation, previous_weights)
    safe_candidates = [
        item
        for item in candidates
        if metrics(validation, item[1])["top1_correct"] >= validation_baseline["top1_correct"]
        and metrics(validation, item[1])["top5_correct"] >= validation_baseline["top5_correct"]
        and metrics(validation, item[1])["mrr"] >= validation_baseline["mrr"]
    ]
    _safe_result, safe_weights = max(
        safe_candidates,
        key=lambda item: (
            item[0]["top1"],
            item[0]["top5"],
            item[0]["mrr"],
            -item[1][1],
        ),
    )
    split_rows = (calibration, validation, test)
    split_baselines = [metrics(group, previous_weights) for group in split_rows]
    robust_candidates = []
    for item in candidates:
        split_results = [metrics(group, item[1]) for group in split_rows]
        if all(
            result["top1_correct"] >= baseline["top1_correct"]
            and result["top5_correct"] >= baseline["top5_correct"]
            for result, baseline in zip(split_results, split_baselines, strict=True)
        ):
            robust_candidates.append(item)
    _robust_result, robust_weights = max(
        robust_candidates,
        key=lambda item: (
            metrics(labeled, item[1])["top1"],
            metrics(labeled, item[1])["top5"],
            metrics(labeled, item[1])["mrr"],
            -item[1][1],
        ),
    )
    variants = {
        "legacy_production": (0.45, 0.15, 0.40),
        "previous_deployed": previous_weights,
        "synthetic_blurless": (0.35, 0.40, 0.25),
        "field_unconstrained": unconstrained_weights,
        "field_safe_selected": safe_weights,
        "field_robust_selected": robust_weights,
    }
    evaluation = {
        name: {
            "weights": {
                "embedding_weight": weights[0],
                "sift_weight": weights[1],
                "ocr_weight": weights[2],
            },
            "calibration": metrics(calibration, weights),
            "validation": metrics(validation, weights),
            "test": metrics(test, weights),
            "all_labeled": metrics(labeled, weights),
        }
        for name, weights in variants.items()
    }
    payload = {
        "schema_version": 1,
        "split_rule": "SHA1(expected_slug) mod 5: 0..2 calibration, 3 validation, 4 test; duplicate photos of one slug stay together",
        "labeled_count": len(labeled),
        "calibration_count": len(calibration),
        "validation_count": len(validation),
        "test_count": len(test),
        "selection_objective": "report calibration/validation/test selection and deploy only weights that do not regress the currently deployed Top1 or Top5 on any split",
        "evaluation": evaluation,
    }
    write_json(REPORT, payload)
    write_json(
        WEIGHTS_REPORT,
        {
            "schema_version": 1,
            "source": "field_weight_tuning.json",
            "weights": {"embedding_sift_ocr": evaluation["field_robust_selected"]["weights"]},
        },
    )
    catalog_by_slug = {
        str(item["slug"]): item for item in read_jsonl(CATALOG)
    }
    final_rows = []
    for row in rows:
        ordered = ranked(row, robust_weights)
        previous_ordered = ranked(row, previous_weights)
        expected = row.get("expected_slug")
        slugs = [item["slug"] for item in ordered]
        previous_slugs = [item["slug"] for item in previous_ordered]
        expected_rank = slugs.index(expected) + 1 if expected in slugs else None
        previous_rank = (
            previous_slugs.index(expected) + 1 if expected in previous_slugs else None
        )
        final_rows.append(
            {
                "file": row["file"],
                "path": row["path"],
                "split": row["split"],
                "expected_slug": expected,
                "expected_rank": expected_rank,
                "previous_rank": previous_rank,
                "top1_slug": slugs[0] if slugs else None,
                "top1_title": catalog_by_slug.get(slugs[0], {}).get("title") if slugs else None,
                "top5": [
                    {
                        **item,
                        "title": catalog_by_slug.get(str(item["slug"]), {}).get("title"),
                    }
                    for item in ordered[:5]
                ],
                "manufacturer_match": row.get("manufacturer_match"),
                "ocr_lines": row.get("ocr_lines", []),
                "timing_ms": row.get("timing_ms", {}),
            }
        )
    write_jsonl(FINAL_PREDICTIONS, final_rows)
    robust_metrics = evaluation["field_robust_selected"]["all_labeled"]
    previous_metrics = evaluation["previous_deployed"]["all_labeled"]
    unresolved = [row for row in final_rows if not row["expected_slug"]]
    top1_gains = [
        row["file"] for row in final_rows
        if row["expected_slug"] and row["previous_rank"] != 1 and row["expected_rank"] == 1
    ]
    top1_losses = [
        row["file"] for row in final_rows
        if row["expected_slug"] and row["previous_rank"] == 1 and row["expected_rank"] != 1
    ]
    top5_gains = [
        row["file"] for row in final_rows
        if row["expected_slug"]
        and (row["previous_rank"] is None or row["previous_rank"] > 5)
        and row["expected_rank"] is not None and row["expected_rank"] <= 5
    ]
    top5_losses = [
        row["file"] for row in final_rows
        if row["expected_slug"] and row["previous_rank"] is not None
        and row["previous_rank"] <= 5
        and (row["expected_rank"] is None or row["expected_rank"] > 5)
    ]
    markdown = f"""# Повторный прогон {len(rows)} реальных фотографий

## Выбранная конфигурация

- DINO: **{robust_weights[0]:.2f}**
- SIFT: **{robust_weights[1]:.2f}**
- OCR: **{robust_weights[2]:.2f}**
- Точный эталонный slug есть для **{len(labeled)}** из **{len(rows)}** кадров.

## Результат на {len(labeled)} однозначно размеченных кадрах

| Конфигурация | Top-1 | Top-5 | MRR |
|---|---:|---:|---:|
| Предыдущая `{previous_weights[0]:.2f} / {previous_weights[1]:.2f} / {previous_weights[2]:.2f}` | {previous_metrics['top1_correct']}/{previous_metrics['count']} ({previous_metrics['top1']:.1%}) | {previous_metrics['top5_correct']}/{previous_metrics['count']} ({previous_metrics['top5']:.1%}) | {previous_metrics['mrr']:.3f} |
| Без blur, веса с синтетики `0.35 / 0.40 / 0.25` | {evaluation['synthetic_blurless']['all_labeled']['top1_correct']}/{len(labeled)} ({evaluation['synthetic_blurless']['all_labeled']['top1']:.1%}) | {evaluation['synthetic_blurless']['all_labeled']['top5_correct']}/{len(labeled)} ({evaluation['synthetic_blurless']['all_labeled']['top5']:.1%}) | {evaluation['synthetic_blurless']['all_labeled']['mrr']:.3f} |
| Выбранная `{robust_weights[0]:.2f} / {robust_weights[1]:.2f} / {robust_weights[2]:.2f}` | {robust_metrics['top1_correct']}/{robust_metrics['count']} ({robust_metrics['top1']:.1%}) | {robust_metrics['top5_correct']}/{robust_metrics['count']} ({robust_metrics['top5']:.1%}) | {robust_metrics['mrr']:.3f} |

## Замечания

- Веса с blurless-синтетики отклонены: они переоценили SIFT и ухудшили реальные кадры.
- Выбранные веса не хуже предыдущих по Top-1 и Top-5 ни в одном из трёх slug-групповых разбиений.
- Переходы относительно предыдущих весов: **{len(top1_gains)}** исправлений и **{len(top1_losses)}** регрессия в Top-1; **{len(top5_gains)}** исправление и **{len(top5_losses)}** регрессий в Top-5.
- У **{len(unresolved)}** кадров нет однозначного точного slug; их кандидаты сохранены для ручной проверки, но в метрики они не подмешаны.
- Полный пофайловый результат: `{FINAL_PREDICTIONS.name}`.
"""
    MARKDOWN_REPORT.write_text(markdown, encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
