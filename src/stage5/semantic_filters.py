"""Catalog-wide semantic reranking from OCR attributes.

This module deliberately knows nothing about individual wines.  It extracts
only broad label attributes and applies small tie-break bonuses to the
already-retrieved candidates.  Visual/OCR scores remain the primary signal.
"""

from __future__ import annotations

import re
from typing import Any

from stage4.manufacturer import fuzzy_key


def _norm(value: Any) -> str:
    return fuzzy_key(str(value or "")).lower()


def _has(text: str, pattern: str) -> bool:
    return bool(re.search(pattern, text))


def extract_attribute_cues(ocr_lines: list[dict[str, Any]]) -> dict[str, set[str]]:
    text = _norm(" ".join(str(line.get("normalized") or line.get("text") or "") for line in ocr_lines))
    return {
        "colour": {
            key
            for key, patterns in {
                "red": (r"\bkrasn", r"\brubin", r"\bred", r"\brouge"),
                "white": (r"\bbel", r"\bwhite", r"\bblanc", r"\bbeloe"),
                "rose": (r"\broz", r"\brose", r"\bros[ée]"),
                "orange": (r"\boranzh", r"\borange", r"\byantar"),
            }.items()
            if any(_has(text, pattern) for pattern in patterns)
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
            if any(_has(text, pattern) for pattern in patterns)
        },
        "sparkling": {"sparkling"}
        if _has(text, r"\bshamp|\bigrist|\bsparkling|\bbrut|\bbryut")
        else set(),
    }


def _candidate_colour(record: dict[str, Any]) -> set[str]:
    text = f"{_norm(record.get('category'))} {_norm(record.get('color'))}"
    result: set[str] = set()
    if _has(text, r"krasn|rubin|granat|red"): result.add("red")
    if _has(text, r"bel|solom|zolot|white"): result.add("white")
    if _has(text, r"roz|pink|rose"): result.add("rose")
    if _has(text, r"oranzh|yantar|orange"): result.add("orange")
    return result


def _candidate_sweetness(record: dict[str, Any]) -> set[str]:
    text = _norm(record.get("category"))
    result: set[str] = set()
    if "poluslad" in text: result.update(("semi_sweet", "sweet"))
    elif "slad" in text: result.add("sweet")
    if "polusuh" in text: result.update(("semi_dry", "dry"))
    elif "suh" in text: result.add("dry")
    if "bryut" in text or "brut" in text: result.add("brut")
    return result


def _attribute_bonus(cues: dict[str, set[str]], record: dict[str, Any]) -> float:
    bonus = 0.0
    candidate_colour = _candidate_colour(record)
    candidate_sweetness = _candidate_sweetness(record)
    if cues["colour"]:
        bonus += 0.08 if cues["colour"] & candidate_colour else -0.04
    if cues["sweetness"]:
        compatible = cues["sweetness"] & candidate_sweetness
        if {"dry", "semi_dry"} & cues["sweetness"]:
            compatible |= {"dry", "semi_dry"} & candidate_sweetness
        if {"sweet", "semi_sweet"} & cues["sweetness"]:
            compatible |= {"sweet", "semi_sweet"} & candidate_sweetness
        bonus += 0.08 if compatible else -0.04
    if cues["sparkling"]:
        bonus += 0.05 if "brut" in candidate_sweetness or "sparkling" in _norm(record.get("category")) else -0.025
    return bonus


def rerank_predictions(
    predictions: list[dict[str, Any]],
    ocr_lines: list[dict[str, Any]],
    catalog_by_slug: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Rerank only current candidates; never injects or removes catalog items."""
    if len(predictions) < 2:
        return predictions
    cues = extract_attribute_cues(ocr_lines)
    if not any(cues.values()):
        return predictions
    scored = []
    for position, prediction in enumerate(predictions):
        record = catalog_by_slug.get(str(prediction["slug"]))
        score = float(prediction.get("score") or 0.0)
        if record:
            score += _attribute_bonus(cues, record)
        scored.append((score, -position, prediction))
    return [prediction for _, _, prediction in sorted(scored, reverse=True)]
