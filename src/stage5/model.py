from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


@dataclass(frozen=True)
class ConfidenceModel:
    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    accept_threshold: float
    review_threshold: float

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        standardized = (features - self.mean) / self.scale
        return sigmoid(standardized @ self.coefficients + self.intercept)

    def decision(self, confidence: float) -> str:
        if confidence >= self.accept_threshold:
            return "match"
        if confidence >= self.review_threshold:
            return "alternatives"
        return "not_found"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "feature_names": list(self.feature_names),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coefficients": self.coefficients.tolist(),
            "intercept": self.intercept,
            "accept_threshold": self.accept_threshold,
            "review_threshold": self.review_threshold,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ConfidenceModel":
        return cls(
            tuple(str(value) for value in payload["feature_names"]),
            np.asarray(payload["mean"], dtype=np.float64),
            np.asarray(payload["scale"], dtype=np.float64),
            np.asarray(payload["coefficients"], dtype=np.float64),
            float(payload["intercept"]),
            float(payload["accept_threshold"]),
            float(payload["review_threshold"]),
        )


def fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: tuple[str, ...],
    l2: float = 0.08,
    iterations: int = 100,
) -> ConfidenceModel:
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-6] = 1.0
    standardized = (features - mean) / scale
    design = np.column_stack((np.ones(len(standardized)), standardized))
    beta = np.zeros(design.shape[1], dtype=np.float64)
    regularizer = np.eye(design.shape[1], dtype=np.float64) * l2
    regularizer[0, 0] = 0.0
    for _ in range(iterations):
        probability = sigmoid(design @ beta)
        gradient = design.T @ (probability - labels) + regularizer @ beta
        curvature = np.clip(probability * (1.0 - probability), 1e-5, None)
        hessian = design.T @ (design * curvature[:, None]) + regularizer
        step = np.linalg.solve(hessian, gradient)
        beta -= step
        if float(np.max(np.abs(step))) < 1e-8:
            break
    return ConfidenceModel(feature_names, mean, scale, beta[1:], float(beta[0]), 1.0, 0.0)


def choose_thresholds(
    confidence: np.ndarray,
    safe: np.ndarray,
    known: np.ndarray,
    target_unknown_far: float = 0.025,
    target_wrong_accept: float = 0.05,
) -> tuple[float, float]:
    candidates = np.unique(
        np.concatenate(([0.0], confidence, np.nextafter(confidence, 1.0), [1.0]))
    )
    correct_known = known & safe
    incorrect_known = known & ~safe
    unknown = ~known
    feasible: list[tuple[float, float, float]] = []
    for threshold in candidates:
        accepted = confidence >= threshold
        unknown_far = float(accepted[unknown].mean()) if unknown.any() else 0.0
        wrong_accept = float(accepted[incorrect_known].mean()) if incorrect_known.any() else 0.0
        recall = float(accepted[correct_known].mean()) if correct_known.any() else 0.0
        if unknown_far <= target_unknown_far and wrong_accept <= target_wrong_accept:
            feasible.append((recall, -threshold, float(threshold)))
    accept_threshold = max(feasible)[2] if feasible else 1.0

    positive_scores = np.sort(confidence[correct_known])
    if len(positive_scores):
        review_threshold = float(np.quantile(positive_scores, 0.05, method="higher"))
    else:
        review_threshold = accept_threshold
    return accept_threshold, min(review_threshold, accept_threshold)


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = scores[labels]
    negatives = scores[~labels]
    if not len(positives) or not len(negatives):
        return 0.0
    comparisons = (positives[:, None] > negatives[None, :]).mean()
    ties = (positives[:, None] == negatives[None, :]).mean()
    return float(comparisons + 0.5 * ties)


def expected_calibration_error(labels: np.ndarray, scores: np.ndarray, bins: int = 10) -> float:
    result = 0.0
    for lower in np.linspace(0.0, 1.0, bins, endpoint=False):
        upper = lower + 1.0 / bins
        selected = (scores >= lower) & (scores < upper if upper < 1.0 else scores <= upper)
        if selected.any():
            result += float(selected.mean()) * abs(
                float(labels[selected].mean()) - float(scores[selected].mean())
            )
    return result
