"""Offline experiment: high-confidence OCR anchors for the 65-photo set.

This is deliberately isolated from ``app/``.  It measures how much accuracy
can be recovered when a distinctive producer/line phrase is treated as a
catalog constraint before the visual score is used.  The anchor table is an
experiment, not production configuration; its purpose is to quantify the
ceiling and expose which misses are genuinely ambiguous.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from stage1.catalog import read_jsonl, write_json


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "data/artifacts/stage8/dual_ocr_65/predictions.jsonl"
CATALOG = ROOT / "data/artifacts/stage1/catalog_manifest.jsonl"
OUTPUT = ROOT / "data/artifacts/stage8/dual_ocr_65/anchor_rerank_experiment.json"


# Patterns use normalized OCR text.  Each rule is intentionally distinctive
# enough to be explainable in a review; no app/runtime code imports this file.
ANCHORS: list[tuple[str, str, str]] = [
    (r"(?=.*DENISOV)(?=.*СТРЕЛКА)(?=.*РУБИН)(?=.*КРАСНАЯ)", "denisov_rubin_klaret_krasnaya_strelka", "Denisov + Красная стрелка + Рубин"),
    (r"(?=.*ТАБИЯ)(?=.*1945)(?=.*КАБЕРНЕ)(?=.*МЕРЛО)(?=.*РУБИН)", "pobeda", "Tabia + blend tokens from Победа"),
    (r"(?=.*МАССАНДРА)(?=.*М[УУ]СКАТЕЛ)(?=.*БЕЛЫЙ)", "massandra-muskatel-belyy-belye-sorta-vinograda-beloe-sladkoe-16", "Massandra + Мускатель + белый"),
    (r"(?=.*GOLUBITSKOE)(?=.*CHARDONNAY)", "golubitskoe-estate-chardonnay", "Golubitskoe + Chardonnay"),
    (r"(?=.*АРАТТ)(?=.*ЦИТРОН)(?=.*СОВИНЬОН БЛАН)", "czitronnyj-magaracha-sovinon-blan", "Aratti + Цитрон + Совиньон блан"),
    (r"(?=.*INKERMAN)(?=.*WINEMAKER)(?=.*RIESLING)", "winemaker-selection", "Inkerman + Winemaker's Selection + Riesling"),
    (r"(?=.*GOLUBITS)(?=.*MERLOT)", "golubitskoe-estate-merlo-krasnoe-suhoe-135", "Golubitskoe + Merlot"),
    (r"(?=.*НОВЫЙ СВ[ЕЪ]Т)(?=.*ПОЛУСЛАДКОЕ РОЗОВОЕ)", "novyy-svet-dom-shampanskih-vin-novyy-svet-polusladkoe-roze-shardone-rozovoe-115", "Новый Свет + полусладкое розовое"),
    (r"(?=.*LITAVSHCHUK)(?=.*ПОЗДНИЙ)(?=.*СБОР)", "pozdnij-sbor-krasnoe", "Litavshchuk + Поздний сбор"),
]


def metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    ranks = [row["rank"] for row in rows]
    return {
        "count": len(ranks),
        "top1": sum(rank == 1 for rank in ranks),
        "top5": sum(rank <= 5 for rank in ranks),
        "top1_rate": sum(rank == 1 for rank in ranks) / len(ranks),
        "top5_rate": sum(rank <= 5 for rank in ranks) / len(ranks),
    }


def main() -> int:
    catalog = read_jsonl(CATALOG)
    by_slug = {str(row["slug"]): row for row in catalog}
    rows = [json.loads(line) for line in PREDICTIONS.read_text(encoding="utf-8").splitlines()]
    compiled = [(re.compile(pattern, re.IGNORECASE), slug, label) for pattern, slug, label in ANCHORS]

    evaluated: list[dict[str, Any]] = []
    for row in rows:
        result = row["result"]
        ocr = " ".join(str(item.get("normalized") or item.get("text") or "") for item in result.get("ocr_lines", []))
        ocr = re.sub(r"\s+", " ", ocr).strip()
        current = [item["slug"] for item in result["predictions"]]
        selected = current[0]
        matched_rule = None
        for pattern, slug, label in compiled:
            if pattern.search(ocr) and slug in by_slug:
                selected = slug
                matched_rule = label
                break
        rank = next((index + 1 for index, slug in enumerate(current) if slug == row["expected_slug"]), 99)
        final_rank = 1 if selected == row["expected_slug"] else (rank if rank <= 5 else 99)
        evaluated.append({
            "file": row["file"],
            "expected_slug": row["expected_slug"],
            "baseline_slug": current[0],
            "selected_slug": selected,
            "baseline_rank": rank,
            "selected_rank": final_rank,
            "matched_rule": matched_rule,
            "ocr": ocr,
        })

    baseline = [{"rank": row["baseline_rank"]} for row in evaluated]
    after = [{"rank": row["selected_rank"]} for row in evaluated]
    output = {
        "photos": len(rows),
        "warning": "This is an offline, label-aware anchor experiment; it is not applied to app/.",
        "baseline": metrics(baseline),
        "anchor_rerank": metrics(after),
        "rules": [{"pattern": pattern, "slug": slug, "description": label} for pattern, slug, label in ANCHORS],
        "rows": evaluated,
    }
    write_json(OUTPUT, output)
    print(json.dumps({"baseline": output["baseline"], "anchor_rerank": output["anchor_rerank"]}, ensure_ascii=False, indent=2))
    print(f"saved={OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
