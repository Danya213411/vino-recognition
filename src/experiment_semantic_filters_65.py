"""Offline, catalog-wide semantic attribute filtering experiment.

The experiment derives broad wine attributes from OCR (colour, sweetness and
sparkling style) and softly reranks catalog candidates.  It never references
an individual bottle or slug, so the same logic can be applied to the whole
catalog.  Nothing here is imported by ``app/``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz, process

from stage1.catalog import read_jsonl, write_json
from stage4.manufacturer import fuzzy_key


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "data/artifacts/stage8/dual_ocr_65/predictions.jsonl"
CATALOG = ROOT / "data/artifacts/stage1/catalog_manifest.jsonl"
OUTPUT = ROOT / "data/artifacts/stage8/dual_ocr_65/semantic_filter_experiment.json"

FIELDS = ("title", "manufacturer", "grapes", "category", "region", "color")


def norm(value: Any) -> str:
    return fuzzy_key(str(value or "")).lower()


def field_value(record: dict[str, Any], field: str) -> str:
    value = record.get(field)
    if field == "grapes":
        return norm(" ".join(str(item) for item in value or []))
    return norm(value)


def ocr_text(row: dict[str, Any]) -> str:
    return norm(" ".join(str(item.get("normalized") or item.get("text") or "") for item in row["result"].get("ocr_lines", [])))


def cues(text: str) -> dict[str, set[str]]:
    """Extract intentionally broad, language/label-style cues from OCR."""
    return {
        "colour": {
            key
            for key, patterns in {
                "red": (r"\bkrasn", r"\brubin", r"\bred", r"\brouge"),
                "white": (r"\bbel", r"\bwhite", r"\bblanc", r"\bbeloe"),
                "rose": (r"\broz", r"\brose", r"\bros[ée]"),
                "orange": (r"\boranzh", r"\borange", r"\byantar"),
            }.items()
            if any(re.search(pattern, text) for pattern in patterns)
        },
        "sweetness": {
            key
            for key, patterns in {
                "dry": (r"\bsuh", r"\bdry", r"\bsuhoe"),
                "semi_dry": (r"\bpolusuh", r"\bsemi.?dry"),
                "sweet": (r"\bslad", r"\bsweet"),
                "semi_sweet": (r"\bpoluslad", r"\bsemi.?sweet"),
                "brut": (r"\bbryut", r"\bbrut"),
            }.items()
            if any(re.search(pattern, text) for pattern in patterns)
        },
        "sparkling": {"sparkling"} if re.search(r"shamp|igrist|sparkling|brut|bryut", text) else set(),
    }


def candidate_colour(category: str, color: str) -> set[str]:
    text = f"{category} {color}"
    result = set()
    if re.search(r"krasn|rubin|granat|red", text): result.add("red")
    if re.search(r"bel|solom|zolot|white|beloe", text): result.add("white")
    if re.search(r"roz|pink|rose", text): result.add("rose")
    if re.search(r"oranzh|yantar|orange", text): result.add("orange")
    return result


def candidate_sweetness(category: str) -> set[str]:
    text = norm(category)
    result = set()
    if "poluslad" in text: result.update(("semi_sweet", "sweet"))
    elif "slad" in text: result.add("sweet")
    if "polusuh" in text: result.update(("semi_dry", "dry"))
    elif "suh" in text: result.add("dry")
    if "bryut" in text or "brut" in text: result.add("brut")
    return result


def attribute_bonus(query: dict[str, set[str]], record: dict[str, Any]) -> float:
    """Return a soft compatibility score; no cue means no penalty."""
    bonus = 0.0
    cc = candidate_colour(norm(record.get("category")), norm(record.get("color")))
    cs = candidate_sweetness(norm(record.get("category")))
    if query["colour"]:
        bonus += 0.08 if query["colour"] & cc else -0.04
    if query["sweetness"]:
        # A dry label may be read as either dry or semi-dry; same for sweet.
        compatible = query["sweetness"] & cs
        if {"dry", "semi_dry"} & query["sweetness"]:
            compatible |= {"dry", "semi_dry"} & cs
        if {"sweet", "semi_sweet"} & query["sweetness"]:
            compatible |= {"sweet", "semi_sweet"} & cs
        bonus += 0.08 if compatible else -0.04
    if query["sparkling"]:
        bonus += 0.05 if ("brut" in cs or "sparkling" in norm(record.get("category"))) else -0.025
    return bonus


def metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    ranks = [row["rank"] for row in rows]
    return {"count": len(ranks), "top1": sum(x == 1 for x in ranks), "top5": sum(x <= 5 for x in ranks), "top1_rate": sum(x == 1 for x in ranks) / len(ranks), "top5_rate": sum(x <= 5 for x in ranks) / len(ranks)}


def main() -> int:
    catalog = read_jsonl(CATALOG)
    rows = [json.loads(line) for line in PREDICTIONS.read_text(encoding="utf-8").splitlines()]
    values = {field: [field_value(record, field) for record in catalog] for field in FIELDS}
    slug_to_index = {str(record["slug"]): index for index, record in enumerate(catalog)}
    evaluated: list[dict[str, Any]] = []

    for row in rows:
        text = ocr_text(row)
        query = cues(text)
        existing = {str(item["slug"]): float(item.get("score") or 0.0) for item in row["result"]["predictions"]}
        pool = {slug_to_index[slug] for slug in existing if slug in slug_to_index}
        # Generic OCR retrieval, independent of any product-specific rule.
        for field in FIELDS:
            for token in (part for part in text.split() if len(part) >= 4):
                for _, _, index in process.extract(token, values[field], scorer=fuzz.WRatio, limit=12):
                    pool.add(index)

        scored = []
        for index in pool:
            record = catalog[index]
            base = existing.get(str(record["slug"]), 0.0)
            field_match = max((fuzz.token_set_ratio(text, values[field][index]) for field in FIELDS), default=0.0) / 100.0
            # Keep the learned score dominant for retrieved Top-5 items.  A
            # catalog item injected by the generic OCR retrieval may compete
            # only when its field text is genuinely close to the OCR text.
            # This conservative pass only reranks the model's retrieved
            # candidates.  Candidate expansion is measured separately; an
            # injected item must not jump ahead on a weak generic cue.
            score = base + attribute_bonus(query, record) if base > 0 else -1.0
            scored.append((score, index))
        scored.sort(reverse=True)
        order = [index for _, index in scored]
        expected = slug_to_index[row["expected_slug"]]
        rank = next((position + 1 for position, index in enumerate(order) if index == expected), 999)
        baseline = next((position + 1 for position, item in enumerate(row["result"]["predictions"]) if item["slug"] == row["expected_slug"]), 999)
        evaluated.append({"file": row["file"], "expected_slug": row["expected_slug"], "baseline_rank": baseline, "semantic_rank": rank, "cues": {key: sorted(value) for key, value in query.items()}, "ocr": text, "semantic_top5": [catalog[index]["slug"] for index in order[:5]]})

    output = {"photos": len(rows), "method": "generic OCR attributes: colour, sweetness, sparkling; fuzzy catalog retrieval; soft compatibility bonuses", "baseline": metrics([{"rank": r["baseline_rank"]} for r in evaluated]), "semantic_filter": metrics([{"rank": r["semantic_rank"]} for r in evaluated]), "rows": evaluated}
    write_json(OUTPUT, output)
    print(json.dumps({"baseline": output["baseline"], "semantic_filter": output["semantic_filter"]}, ensure_ascii=False, indent=2))
    print(f"saved={OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
