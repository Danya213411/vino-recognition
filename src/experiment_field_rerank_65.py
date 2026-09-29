"""Offline field-aware rerank experiments for the 65-photo evaluation set.

This script intentionally does not touch the API or its runtime artifacts. It
uses saved dual-OCR output, builds extra catalog candidates from title,
manufacturer, grape, category, region and color fields, then compares a few
rerank policies against the current Top-1/Top-5 baseline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz, process

from stage1.catalog import read_jsonl, write_json
from stage4.manufacturer import fuzzy_key


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "data/artifacts/stage8/dual_ocr_65/predictions.jsonl"
CATALOG = ROOT / "data/artifacts/stage1/catalog_manifest.jsonl"
OUTPUT = ROOT / "data/artifacts/stage8/dual_ocr_65/field_rerank_experiment.json"

FIELD_WEIGHTS = {
    "title": 0.40,
    "manufacturer": 0.25,
    "grapes": 0.17,
    "category": 0.10,
    "region": 0.04,
    "color": 0.04,
}


def field_value(record: dict[str, Any], field: str) -> str:
    value = record.get(field)
    if field == "grapes":
        return fuzzy_key(" ".join(str(item) for item in value or []))
    return fuzzy_key(str(value or ""))


def line_score(lines: list[str], target: str) -> float:
    if not target or not lines:
        return 0.0
    best = 0.0
    target_tokens = set(target.split())
    for line in lines:
        ratio = max(fuzz.token_set_ratio(line, target), fuzz.partial_ratio(line, target)) / 100.0
        line_tokens = set(line.split())
        overlap = len(line_tokens & target_tokens) / max(1, min(len(line_tokens), len(target_tokens)))
        best = max(best, 0.75 * ratio + 0.25 * overlap)
    return best


def rank_metrics(rows: list[dict[str, Any]], key: str) -> dict[str, float | int]:
    ranks = [row[key] for row in rows]
    return {
        "top1": sum(rank == 1 for rank in ranks),
        "top5": sum(rank is not None and rank <= 5 for rank in ranks),
        "top1_rate": sum(rank == 1 for rank in ranks) / len(ranks),
        "top5_rate": sum(rank is not None and rank <= 5 for rank in ranks) / len(ranks),
    }


def main() -> int:
    catalog = read_jsonl(CATALOG)
    slug_to_index = {str(record["slug"]): index for index, record in enumerate(catalog)}
    rows = [json.loads(line) for line in PREDICTIONS.read_text(encoding="utf-8").splitlines()]
    values = {
        field: [field_value(record, field) for record in catalog]
        for field in FIELD_WEIGHTS
    }
    prepared: list[dict[str, Any]] = []

    for row in rows:
        lines = [
            fuzzy_key(str(item.get("normalized") or item.get("text") or ""))
            for item in row["result"].get("ocr_lines", [])
        ]
        lines = [line for line in lines if line]
        existing = {str(item["slug"]): float(item.get("score") or 0.0) for item in row["result"]["predictions"]}
        pool = {slug_to_index[slug] for slug in existing if slug in slug_to_index}
        field_scores: dict[int, dict[str, float]] = {}

        # Retrieve candidates independently per field. This prevents a long,
        # noisy GLM paragraph from hiding a strong one-word match such as
        # CHARDONNAY, МУСКАТЕЛЬ or ПОЗДНИЙ СБОР.
        for field in FIELD_WEIGHTS:
            queries = [(line, index) for index, line in enumerate(lines)]
            for line, _ in queries:
                for _, score, index in process.extract(
                    line, values[field], scorer=fuzz.WRatio, limit=20
                ):
                    pool.add(index)

        for index in pool:
            field_scores[index] = {
                field: line_score(lines, values[field][index])
                for field in FIELD_WEIGHTS
            }

        prepared.append({"row": row, "existing": existing, "field_scores": field_scores})

    policies = {
        "current": lambda item, index: item["existing"].get(str(catalog[index]["slug"]), 0.0),
        "field_only": lambda item, index: sum(
            FIELD_WEIGHTS[field] * item["field_scores"][index][field]
            for field in FIELD_WEIGHTS
        ),
        "balanced": lambda item, index: 0.55 * item["existing"].get(str(catalog[index]["slug"]), 0.0)
        + 0.45 * sum(FIELD_WEIGHTS[field] * item["field_scores"][index][field] for field in FIELD_WEIGHTS),
        "field_priority": lambda item, index: 0.35 * item["existing"].get(str(catalog[index]["slug"]), 0.0)
        + 0.65 * sum(FIELD_WEIGHTS[field] * item["field_scores"][index][field] for field in FIELD_WEIGHTS),
    }

    output: dict[str, Any] = {"photos": len(rows), "policies": {}, "rows": []}
    policy_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in policies}
    for item in prepared:
        row = item["row"]
        candidates = list(item["field_scores"])
        for policy_name, policy in policies.items():
            order = sorted(candidates, key=lambda index: policy(item, index), reverse=True)
            expected = row.get("expected_slug")
            rank = next(
                (position + 1 for position, index in enumerate(order) if catalog[index]["slug"] == expected),
                None,
            )
            policy_rows[policy_name].append({"file": row["file"], "rank": rank})

    for policy_name, policy_result in policy_rows.items():
        output["policies"][policy_name] = rank_metrics(policy_result, "rank")

    for item in prepared:
        row = item["row"]
        expected = row.get("expected_slug")
        candidate_rows = []
        for index, scores in item["field_scores"].items():
            structured = sum(FIELD_WEIGHTS[field] * scores[field] for field in FIELD_WEIGHTS)
            candidate_rows.append(
                {
                    "slug": catalog[index]["slug"],
                    "title": catalog[index].get("title"),
                    "structured_score": round(structured, 4),
                    "current_score": item["existing"].get(str(catalog[index]["slug"]), 0.0),
                }
            )
        candidate_rows.sort(key=lambda candidate: candidate["structured_score"], reverse=True)
        output["rows"].append(
            {
                "file": row["file"],
                "expected_slug": expected,
                "current_candidate": row["result"]["candidate_slug"],
                "structured_top5": candidate_rows[:5],
            }
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT, output)
    print(json.dumps(output["policies"], ensure_ascii=False, indent=2))
    print(f"saved={OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
