from __future__ import annotations

import numpy as np

from stage5.features import (
    FEATURE_NAMES,
    crop_diagnostics,
    ocr_field_evidence,
    runtime_confidence_features,
)
from stage5.model import ConfidenceModel, choose_thresholds, fit_logistic
from stage5.recognize import is_multimodal_verified
from stage5.unknown_dataset import parse_cards


def test_live_catalog_parser_selects_bottle_image() -> None:
    page = """
    <a href="/wines/test-wine" class="wine-item wine-item_s">
      <img src="/svg/public-rating.svg">
      <img src="https://api.vino-svoe.ru/v1/img/str-api/540/540/resize/uploads/test.webp">
      <h2 class="wine-item__title">Тестовое вино</h2>
      <span class="wine-item__manufacturer">Винодельня</span>
    </a>
    """
    records = parse_cards(page, 3)
    assert records == [
        {
            "slug": "test-wine",
            "title": "Тестовое вино",
            "manufacturer": "Винодельня",
            "source_url": "https://vino-svoe.ru/wines/test-wine",
            "image_source_url": "https://api.vino-svoe.ru/v1/img/str-api/540/540/resize/uploads/test.webp",
            "catalog_page": 3,
        }
    ]


def test_ocr_field_evidence_checks_year_and_metadata() -> None:
    lines = [{"normalized": "винодельня рислинг резерв 2024", "score": 0.9}]
    record = {
        "title": "Рислинг Резерв 2024",
        "manufacturer": "Винодельня",
        "grapes": ["Рислинг"],
        "slug": "risling-rezerv-2024",
    }
    evidence = ocr_field_evidence(lines, record)
    assert evidence["year_evidence"] == 1.0
    assert evidence["title_evidence"] > 0.9
    assert evidence["manufacturer_evidence"] == 1.0
    assert evidence["grape_evidence"] == 1.0


def test_crop_diagnostics_reward_agreement() -> None:
    scores = np.asarray(
        [[0.9, 0.2, 0.1], [0.8, 0.3, 0.1], [0.7, 0.4, 0.2]], dtype=np.float32
    )
    details = crop_diagnostics(scores, 0)
    assert details["view_top1_agreement"] == 1.0
    assert details["winner_view_top1_fraction"] == 1.0
    assert details["winner_view_top5_fraction"] == 1.0


def test_confidence_model_and_thresholds() -> None:
    features = np.asarray([[0.0], [0.1], [0.2], [0.8], [0.9], [1.0]])
    labels = np.asarray([False, False, False, True, True, True])
    model = fit_logistic(features, labels, ("signal",))
    probabilities = model.predict_proba(features)
    assert np.all(np.diff(probabilities) > 0)

    known = np.asarray([False, False, True, True, True, True])
    accept, review = choose_thresholds(probabilities, labels, known, 0.5, 0.5)
    calibrated = ConfidenceModel(
        model.feature_names,
        model.mean,
        model.scale,
        model.coefficients,
        model.intercept,
        accept,
        review,
    )
    assert calibrated.decision(1.0) == "match"
    assert calibrated.decision(0.0) == "not_found"


def test_runtime_feature_schema() -> None:
    recognition = {
        "predictions": [
            {
                "score": 0.9,
                "embedding_score": 0.8,
                "sift_score": 0.7,
                "sift_inliers": 12,
                "ocr_score": 0.6,
            },
            {"score": 0.5},
        ],
        "ocr_lines": [{"normalized": "рислинг 2024", "score": 0.9}],
        "crop_consistency": {
            "top1_agreement": 1.0,
            "winner_top1_fraction": 1.0,
            "winner_top5_fraction": 1.0,
            "winner_score_mean": 0.75,
            "winner_score_std": 0.02,
        },
    }
    vector, details = runtime_confidence_features(
        recognition, {"title": "Рислинг 2024", "manufacturer": "", "grapes": ["Рислинг"]}
    )
    assert vector.shape == (len(FEATURE_NAMES),)
    assert details["top1_margin"] == 0.4


def test_multimodal_verification_requires_text_and_geometry() -> None:
    prediction = {
        "ocr_score": 1.0,
        "sift_score": 1.0,
        "sift_inliers": 65,
    }
    evidence = {"title_evidence": 1.0}
    assert is_multimodal_verified(prediction, evidence)
    assert not is_multimodal_verified({**prediction, "sift_inliers": 5}, evidence)
    assert not is_multimodal_verified(prediction, {"title_evidence": 0.5})
