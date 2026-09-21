from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

import numpy as np

from stage4.features import normalize_text, text_tokens


YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")
FEATURE_NAMES = (
    "hybrid_score",
    "top1_margin",
    "embedding_score",
    "sift_score",
    "sift_inliers_log",
    "ocr_score",
    "ocr_line_count",
    "title_evidence",
    "manufacturer_evidence",
    "grape_evidence",
    "year_evidence",
    "view_top1_agreement",
    "winner_view_top1_fraction",
    "winner_view_top5_fraction",
    "winner_view_score_mean",
    "winner_view_consistency",
)


def query_text(lines: list[dict[str, Any]]) -> str:
    return normalize_text(" ".join(str(line.get("normalized") or "") for line in lines))


def fuzzy_field_score(query: str, value: str) -> float:
    query_items = text_tokens(query)
    target_items = text_tokens(value)
    if not query_items or not target_items:
        return 0.0
    weighted = []
    for target in target_items:
        similarity = max(SequenceMatcher(None, target, item).ratio() for item in query_items)
        weighted.append((similarity, min(len(target), 10)))
    numerator = sum(score * weight for score, weight in weighted)
    denominator = sum(weight for _, weight in weighted)
    return float(numerator / max(denominator, 1))


def ocr_field_evidence(
    lines: list[dict[str, Any]],
    catalog_record: dict[str, Any],
) -> dict[str, float]:
    text = query_text(lines)
    grapes = catalog_record.get("grapes") or []
    grape_text = " ".join(str(item) for item in grapes) if isinstance(grapes, list) else str(grapes)
    target_years = set(
        YEAR_PATTERN.findall(
            " ".join(
                (
                    str(catalog_record.get("title") or ""),
                    str(catalog_record.get("image_alt") or ""),
                    str(catalog_record.get("slug") or ""),
                )
            )
        )
    )
    query_years = set(YEAR_PATTERN.findall(text))
    year_score = 1.0 if target_years and target_years.intersection(query_years) else 0.0
    return {
        "title_evidence": fuzzy_field_score(text, str(catalog_record.get("title") or "")),
        "manufacturer_evidence": fuzzy_field_score(
            text, str(catalog_record.get("manufacturer") or "")
        ),
        "grape_evidence": fuzzy_field_score(text, grape_text),
        "year_evidence": year_score,
    }


def crop_diagnostics(view_scores: np.ndarray, winner_column: int) -> dict[str, float]:
    if view_scores.ndim != 2 or view_scores.shape[0] != 3:
        raise ValueError("view_scores must have shape (3, candidate_count)")
    top1 = np.argmax(view_scores, axis=1)
    top_k = min(5, view_scores.shape[1])
    top5 = np.argpartition(-view_scores, top_k - 1, axis=1)[:, :top_k]
    _, counts = np.unique(top1, return_counts=True)
    winner_scores = view_scores[:, winner_column]
    return {
        "view_top1_agreement": float(counts.max() / 3.0),
        "winner_view_top1_fraction": float((top1 == winner_column).mean()),
        "winner_view_top5_fraction": float((top5 == winner_column).any(axis=1).mean()),
        "winner_view_score_mean": float(winner_scores.mean()),
        "winner_view_consistency": float(1.0 - min(winner_scores.std() / 0.15, 1.0)),
    }


def confidence_features(
    hybrid_scores: np.ndarray,
    embedding_scores: np.ndarray,
    sift_scores: np.ndarray,
    sift_inliers: np.ndarray,
    ocr_scores: np.ndarray,
    view_scores: np.ndarray,
    query_lines: list[dict[str, Any]],
    catalog_record: dict[str, Any],
) -> tuple[np.ndarray, int, dict[str, float]]:
    order = np.argsort(-hybrid_scores)
    winner_column = int(order[0])
    runner_column = int(order[1]) if len(order) > 1 else winner_column
    fields = ocr_field_evidence(query_lines, catalog_record)
    crops = crop_diagnostics(view_scores, winner_column)
    details = {
        "hybrid_score": float(hybrid_scores[winner_column]),
        "top1_margin": float(
            hybrid_scores[winner_column] - hybrid_scores[runner_column]
            if len(order) > 1
            else hybrid_scores[winner_column]
        ),
        "embedding_score": float(embedding_scores[winner_column]),
        "sift_score": float(sift_scores[winner_column]),
        "sift_inliers_log": float(np.log1p(sift_inliers[winner_column]) / np.log(65.0)),
        "ocr_score": float(ocr_scores[winner_column]),
        "ocr_line_count": float(min(len(query_lines) / 6.0, 1.0)),
        **fields,
        **crops,
    }
    return np.asarray([details[name] for name in FEATURE_NAMES], dtype=np.float64), winner_column, details


def runtime_confidence_features(
    recognition: dict[str, Any],
    catalog_record: dict[str, Any],
) -> tuple[np.ndarray, dict[str, float]]:
    predictions = recognition["predictions"]
    if not predictions:
        raise ValueError("Recognition result has no predictions")
    winner = predictions[0]
    runner_score = float(predictions[1]["score"]) if len(predictions) > 1 else 0.0
    crops = recognition["crop_consistency"]
    fields = ocr_field_evidence(recognition["ocr_lines"], catalog_record)
    details = {
        "hybrid_score": float(winner["score"]),
        "top1_margin": float(winner["score"]) - runner_score,
        "embedding_score": float(winner["embedding_score"]),
        "sift_score": float(winner["sift_score"]),
        "sift_inliers_log": float(np.log1p(winner["sift_inliers"]) / np.log(65.0)),
        "ocr_score": float(winner["ocr_score"]),
        "ocr_line_count": float(min(len(recognition["ocr_lines"]) / 6.0, 1.0)),
        **fields,
        "view_top1_agreement": float(crops["top1_agreement"]),
        "winner_view_top1_fraction": float(crops["winner_top1_fraction"]),
        "winner_view_top5_fraction": float(crops["winner_top5_fraction"]),
        "winner_view_score_mean": float(crops["winner_score_mean"]),
        "winner_view_consistency": float(
            1.0 - min(float(crops["winner_score_std"]) / 0.15, 1.0)
        ),
    }
    return np.asarray([details[name] for name in FEATURE_NAMES], dtype=np.float64), details
