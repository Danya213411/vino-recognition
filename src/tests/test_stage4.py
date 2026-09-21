from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from stage4.baseline import group_mask, normalize_rows
from stage4.features import normalize_text, ocr_similarity, sift_similarity, text_tokens
from stage4.recognize import select_candidate_union


def test_text_normalization_handles_cyrillic_latin_and_numbers() -> None:
    assert normalize_text("  Вино Ёлка — Cabernet 2024! ") == "вино елка cabernet 2024"
    assert text_tokens("A вино 7 2024") == ["вино", "2024"]


def test_ocr_similarity_prefers_matching_metadata() -> None:
    query = [
        {"normalized": "марченко прибой 2024", "score": 0.95},
        {"normalized": "рислинг", "score": 0.85},
    ]
    matching = {"metadata_text": "прибоЙ марченко рислинг 2024", "ocr_lines": []}
    unrelated = {"metadata_text": "фанагория каберне 2021", "ocr_lines": []}
    assert ocr_similarity(query, matching) > ocr_similarity(query, unrelated)


def test_sift_similarity_rewards_consistent_geometry() -> None:
    points = np.asarray(
        [(x, y) for y in range(0, 80, 10) for x in range(0, 80, 10)], dtype=np.float32
    )
    descriptors = np.eye(128, dtype=np.float32)[: len(points)]
    matrix = np.asarray([[1.05, 0.04, 8.0], [-0.03, 1.02, 5.0]], dtype=np.float32)
    transformed = cv2.transform(points[:, None, :], matrix)[:, 0, :]
    score, details = sift_similarity(points, descriptors, transformed, descriptors)
    assert score > 0.8
    assert details["inliers"] == len(points)


def test_normalize_rows_and_group_mask(tmp_path: Path) -> None:
    values = np.asarray([[2.0, 4.0, 6.0], [3.0, 3.0, 3.0]], dtype=np.float32)
    assert np.allclose(normalize_rows(values)[0], [0.0, 0.5, 1.0])
    assert np.allclose(normalize_rows(values)[1], [0.0, 0.0, 0.0])

    groups = tmp_path / "groups.json"
    groups.write_text('[{"items":[{"slug":"wanted"}]}]', encoding="utf-8")
    mask = group_mask([{"expected_slug": "wanted"}, {"expected_slug": "other"}], groups)
    assert mask.tolist() == [True, False]


def test_candidate_union_recovers_strong_ocr_match_outside_visual_topk() -> None:
    visual = np.asarray([[0.99, 0.90, 0.80, 0.70, 0.60]], dtype=np.float32)
    ocr = np.asarray([[0.0, 0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    candidates, visual_sets, ocr_sets = select_candidate_union(
        visual, ocr, np.asarray([True]), visual_count=2, ocr_count=1
    )
    assert candidates.tolist() == [[0, 1, 3]]
    assert visual_sets == [{0, 1}]
    assert ocr_sets == [{3}]
