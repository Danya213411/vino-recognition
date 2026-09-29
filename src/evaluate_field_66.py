from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from statistics import median
from typing import Any

from vino_api.config import Settings
from vino_api.engine import RecognitionEngine


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SELECTION_MANIFEST = PROJECT_ROOT / "data/test/recheck_fuzzy_v2/selection_manifest.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data/artifacts/stage7/catalog_blurless_v2"
PREDICTIONS = OUTPUT_DIR / "field_66_predictions.jsonl"
REPORT = OUTPUT_DIR / "field_66_report.md"
DATABASE = PROJECT_ROOT / "data/app/stage6/events.sqlite3"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def rank_of(slugs: list[str], expected: str | None) -> int | None:
    if not expected or expected not in slugs:
        return None
    return slugs.index(expected) + 1


def latest_feedback() -> dict[str, dict[str, Any]]:
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        """
        SELECT r.original_name, r.created_at, f.verdict, f.correct_slug
        FROM recognitions r
        JOIN feedback f ON f.recognition_id = r.id
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


def main() -> int:
    records = [
        record
        for record in read_jsonl(SELECTION_MANIFEST)
        if record["reason"] != "ambiguous_not_in_store"
    ]
    if len(records) != 66:
        raise ValueError(f"Expected 66 unambiguous field photos, got {len(records)}")

    settings = Settings(device="cuda", local_files_only=True)
    engine = RecognitionEngine(settings)
    feedback = latest_feedback()
    catalog_slugs = list(engine.catalog_by_slug)
    added_slugs = set(catalog_slugs[2025:])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cached = {
        row["file"]: row for row in read_jsonl(PREDICTIONS)
    } if PREDICTIONS.is_file() else {}

    results: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        if record["reason"] == "stable_confirmed_top1":
            expected_slug = record["new_top1"]
            previous_rank = 1
        else:
            item = feedback.get(record["file"], {})
            expected_slug = (
                item.get("correct_slug")
                if item.get("verdict") == "incorrect"
                else record.get("previous_correct_slug")
            )
            previous_rank = 5 if expected_slug else None
        source_path = PROJECT_ROOT / record["source"]
        cached_row = cached.get(record["file"])
        if cached_row:
            predictions = cached_row["predictions"]
            slugs = [str(item["slug"]) for item in predictions]
            row = {
                **cached_row,
                "expected_slug": expected_slug,
                "previous_rank_group": "top1" if previous_rank == 1 else ("top5" if expected_slug else "miss"),
                "new_rank": rank_of(slugs, expected_slug),
            }
            results.append(row)
            continue
        result = engine.recognize(source_path)
        slugs = [str(item["slug"]) for item in result["predictions"]]
        row = {
            "file": record["file"],
            "image": record["source"],
            "expected_slug": expected_slug,
            "previous_rank_group": "top1" if previous_rank == 1 else ("top5" if expected_slug else "miss"),
            "new_rank": rank_of(slugs, expected_slug),
            "decision": result["decision"],
            "confidence": result["confidence"],
            "manufacturer_match": result.get("manufacturer_match"),
            "ocr_lines": result.get("ocr_lines", []),
            "timing_ms": result["timing_ms"],
            "predictions": result["predictions"],
            "added_catalog_candidates": [slug for slug in slugs if slug in added_slugs],
        }
        results.append(row)
        with PREDICTIONS.open("w", encoding="utf-8", newline="\n") as stream:
            for item in results:
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(
            f"{index:02d}/66 {record['file']} -> {slugs[0]} "
            f"rank={row['new_rank']} {result['timing_ms']['total']:.0f}ms",
            flush=True,
        )

    with PREDICTIONS.open("w", encoding="utf-8", newline="\n") as stream:
        for item in results:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")

    exact = [row for row in results if row["expected_slug"]]
    unresolved = [row for row in results if not row["expected_slug"]]
    old_top1 = sum(row["previous_rank_group"] == "top1" for row in exact)
    old_top5 = len(exact)
    new_top1 = sum(row["new_rank"] == 1 for row in exact)
    new_top5 = sum(row["new_rank"] is not None for row in exact)
    timings = [float(row["timing_ms"]["total"]) for row in results]
    decisions: dict[str, int] = {}
    for row in results:
        decisions[row["decision"]] = decisions.get(row["decision"], 0) + 1

    lines = [
        "# Полевой прогон после обновления каталога и удаления blur",
        "",
        "## Конфигурация",
        "",
        f"- Каталог: **{len(catalog_slugs)}** вин (+{len(added_slugs)} Литавщук).",
        "- Синтетическая calibration: без standalone blur и без blur в combined.",
        f"- Гибридные веса: embedding **{engine.model_payload['hybrid_weights']['embedding_weight']:.2f}**, "
        f"SIFT **{engine.model_payload['hybrid_weights']['sift_weight']:.2f}**, "
        f"OCR **{engine.model_payload['hybrid_weights']['ocr_weight']:.2f}**.",
        "",
        "## 52 кадра с точным slug",
        "",
        "| Метрика | До | После | Изменение |",
        "|---|---:|---:|---:|",
        f"| Top-1 | {old_top1}/52 ({old_top1/52:.1%}) | {new_top1}/52 ({new_top1/52:.1%}) | {new_top1-old_top1:+d} |",
        f"| Top-5 | {old_top5}/52 ({old_top5/52:.1%}) | {new_top5}/52 ({new_top5/52:.1%}) | {new_top5-old_top5:+d} |",
        "",
        "Оставшиеся 14 кадров ранее были отмечены `not_in_catalog`, поэтому для них нет",
        "точного slug и они не включены в accuracy автоматически. Новые кандидаты перечислены ниже.",
        "",
        "## 14 ранее неразмеченных промахов",
        "",
        "| Файл | Новый Top-1 | Новые карточки Литавщука в Top-5 | OCR |",
        "|---|---|---|---|",
    ]
    for row in unresolved:
        top1 = row["predictions"][0]
        ocr = "; ".join(str(item["text"]) for item in row["ocr_lines"][:5]) or "—"
        added = ", ".join(f"`{slug}`" for slug in row["added_catalog_candidates"]) or "—"
        lines.append(f"| `{row['file']}` | `{top1['slug']}` | {added} | {ocr} |")
    lines.extend(
        [
            "",
            "## Решения confidence",
            "",
            ", ".join(f"`{key}`: {value}" for key, value in sorted(decisions.items())),
            "",
            "## Скорость",
            "",
            f"- Среднее: **{sum(timings)/len(timings)/1000:.2f} с**.",
            f"- Медиана: **{median(timings)/1000:.2f} с**.",
            f"- P95: **{sorted(timings)[round(0.95*(len(timings)-1))]/1000:.2f} с**.",
            f"- Максимум: **{max(timings)/1000:.2f} с**.",
            f"- Сумма времени распознаваний: **{sum(timings)/60000:.1f} мин**.",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"report={REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
