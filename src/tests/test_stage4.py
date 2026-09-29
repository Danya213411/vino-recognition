from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from stage4.baseline import group_mask, normalize_rows
from stage4.features import normalize_text, ocr_similarity, sift_similarity, text_tokens
from stage4.manufacturer import (
    apply_manufacturer_gate,
    fuzzy_key,
    normalized_edit_similarity,
    resolve_taxonomy_value,
)
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


def test_fuzzy_manufacturer_resolver_handles_scripts_and_ocr_typos() -> None:
    values = ["Табия", "Винодельня Покровская", "Denisov Winery"]
    exact = resolve_taxonomy_value(
        [{"text": "ТАБИЯ", "score": 0.98}], values
    )
    typo = resolve_taxonomy_value(
        [{"text": "TAБИA", "score": 0.92}], values
    )
    english = resolve_taxonomy_value(
        [{"text": "ДЕНИСОВ", "score": 0.95}], values
    )
    assert fuzzy_key("ТАБИЯ") == "tabiya"
    assert normalized_edit_similarity("tabiya", "tabia") > 0.8
    assert exact.value == "Табия" and exact.mode == "hard"
    assert typo.value == "Табия" and typo.mode == "hard"
    assert english.value == "Denisov Winery" and english.mode == "hard"


def test_fuzzy_manufacturer_resolver_handles_compounds_prefixes_and_homoglyphs() -> None:
    values = [
        "Литавщук. Litavshchuk vineyards & winery",
        "Инкерманский ЗМВ",
        "Массандра",
    ]
    compound = resolve_taxonomy_value([{"text": "LITAVSHCHUK", "score": 0.98}], values)
    prefix = resolve_taxonomy_value([{"text": "INKERMAN", "score": 0.96}], values)
    homoglyph = resolve_taxonomy_value([{"text": "MACCAHAPA", "score": 0.94}], values)
    assert compound.value == "Литавщук. Litavshchuk vineyards & winery"
    assert prefix.value == "Инкерманский ЗМВ"
    assert homoglyph.value == "Массандра"


def test_manufacturer_gate_injects_catalog_wines_outside_visual_pool() -> None:
    slugs = np.asarray(["other-1", "tabiya-1", "other-2", "tabiya-2", "other-3"])
    catalog = {
        slug: {"manufacturer": "Табия" if slug.startswith("tabiya") else "Другой"}
        for slug in slugs
    }
    candidates = np.asarray([[0, 2, 4]], dtype=np.int64)
    visual = np.asarray([[0.9, 0.2, 0.8, 0.3, 0.7]], dtype=np.float32)
    ocr = np.asarray([[0.0, 0.9, 0.0, 0.8, 0.0]], dtype=np.float32)
    matches = [{"value": "Табия", "mode": "hard", "candidate_count": 0}]
    gated, allowed = apply_manufacturer_gate(
        candidates, visual, ocr, slugs, catalog, matches
    )
    assert set(gated[0, :2]) == {1, 3}
    assert allowed == [{1, 3}]
    assert matches[0]["candidate_count"] == 2


def test_manufacturer_resolver_rejects_regions_substrings_and_ties() -> None:
    region = resolve_taxonomy_value(
        [{"text": "ЗГУ КУБАНЬ АНАПА", "score": 0.99}],
        ["Кубань-Вино", "АРАТТИ"],
    )
    substring = resolve_taxonomy_value(
        [
            {"text": "FANAGORIA", "score": 0.98},
            {"text": "NOBLESSE OBLIGE", "score": 0.99},
        ],
        ["ESSE", "Фанагория"],
    )
    canonical_group = resolve_taxonomy_value(
        [{"text": "GOLUBITSKOE", "score": 0.99}],
        ["Golubitskoe Estate", "Поместье Голубицкое"],
    )
    assert region.mode == "none"
    assert substring.value == "Фанагория"
    assert canonical_group.mode == "hard"
    assert set(canonical_group.values) == {"Golubitskoe Estate", "Поместье Голубицкое"}


def test_manufacturer_gate_accepts_all_canonical_spellings() -> None:
    slugs = np.asarray(["golubitskoe-en", "other", "golubitskoe-ru"])
    catalog = {
        "golubitskoe-en": {"manufacturer": "Golubitskoe Estate"},
        "other": {"manufacturer": "Другой"},
        "golubitskoe-ru": {"manufacturer": "Поместье Голубицкое"},
    }
    candidates = np.asarray([[1, 0]], dtype=np.int64)
    visual = np.asarray([[0.8, 0.9, 0.7]], dtype=np.float32)
    ocr = np.asarray([[0.9, 0.0, 0.8]], dtype=np.float32)
    matches = [{
        "value": "Golubitskoe Estate",
        "values": ["Golubitskoe Estate", "Поместье Голубицкое"],
        "mode": "hard",
        "candidate_count": 0,
    }]
    gated, allowed = apply_manufacturer_gate(
        candidates, visual, ocr, slugs, catalog, matches
    )
    assert set(gated[0]) == {0, 2}
    assert allowed == [{0, 2}]
