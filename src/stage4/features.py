from __future__ import annotations

import re
import time
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

from stage3.embeddings import open_rgb, query_views, reference_views


TOKEN_PATTERN = re.compile(r"[0-9a-zа-яё]+", re.IGNORECASE)
_OCR_ENGINE: Any | None = None


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold().replace("ё", "е")
    return " ".join(TOKEN_PATTERN.findall(normalized))


def text_tokens(value: str) -> list[str]:
    return [token for token in normalize_text(value).split() if len(token) >= 2]


def metadata_text(record: dict[str, Any]) -> str:
    values: list[str] = []
    for field in ("title", "manufacturer", "category", "region", "alcohol"):
        value = record.get(field)
        if value is not None:
            values.append(str(value))
    for field in ("grapes",):
        value = record.get(field)
        if isinstance(value, list):
            values.extend(str(item) for item in value)
    return normalize_text(" ".join(values))


def pil_to_gray(image: Image.Image, target_long_side: int = 720) -> np.ndarray:
    image = image.convert("RGB")
    scale = target_long_side / max(image.size)
    if scale > 1.0:
        image = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def extract_sift(
    path: Path,
    reference: bool,
    nfeatures: int = 450,
) -> tuple[np.ndarray, np.ndarray]:
    image = reference_views(path)[1] if reference else open_rgb(path)
    gray = pil_to_gray(image)
    sift = cv2.SIFT_create(
        nfeatures=nfeatures,
        contrastThreshold=0.018,
        edgeThreshold=12,
        sigma=1.3,
    )
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    if descriptors is None or not keypoints:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 128), dtype=np.float32)
    points = np.asarray([point.pt for point in keypoints], dtype=np.float32)
    descriptors = descriptors.astype(np.float32)
    descriptors /= descriptors.sum(axis=1, keepdims=True) + 1e-7
    descriptors = np.sqrt(descriptors)
    return points, descriptors


def sift_similarity(
    query_points: np.ndarray,
    query_descriptors: np.ndarray,
    reference_points: np.ndarray,
    reference_descriptors: np.ndarray,
    ratio_threshold: float = 0.78,
) -> tuple[float, dict[str, Any]]:
    if len(query_descriptors) < 2 or len(reference_descriptors) < 2:
        return 0.0, {"good_matches": 0, "inliers": 0, "inlier_ratio": 0.0}

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = matcher.knnMatch(reference_descriptors, query_descriptors, k=2)
    good = [first for first, second in pairs if first.distance < ratio_threshold * second.distance]
    if len(good) < 4:
        score = min(len(good) / 10.0, 1.0) * 0.18
        return score, {"good_matches": len(good), "inliers": 0, "inlier_ratio": 0.0}

    source = np.float32([reference_points[match.queryIdx] for match in good]).reshape(-1, 1, 2)
    target = np.float32([query_points[match.trainIdx] for match in good]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(source, target, cv2.RANSAC, 5.0)
    inliers = int(mask.sum()) if mask is not None else 0
    inlier_ratio = inliers / len(good) if good else 0.0

    coverage = 0.0
    if mask is not None and inliers >= 4:
        inlier_points = source[mask.ravel().astype(bool), 0]
        all_width = max(float(np.ptp(reference_points[:, 0])), 1.0)
        all_height = max(float(np.ptp(reference_points[:, 1])), 1.0)
        width = float(np.ptp(inlier_points[:, 0]))
        height = float(np.ptp(inlier_points[:, 1]))
        coverage = min(1.0, (width * height) / (all_width * all_height + 1e-7))

    score = (
        0.55 * min(inliers / 14.0, 1.0)
        + 0.30 * min(inlier_ratio / 0.65, 1.0)
        + 0.15 * min(coverage / 0.22, 1.0)
    )
    if inliers < 4:
        score *= 0.25
    return float(np.clip(score, 0.0, 1.0)), {
        "good_matches": len(good),
        "inliers": inliers,
        "inlier_ratio": inlier_ratio,
        "coverage": coverage,
    }


def create_ocr_engine() -> Any:
    from rapidocr import LangDet, LangRec, ModelType, OCRVersion, RapidOCR

    return RapidOCR(
        params={
            "Global.use_cls": False,
            "Global.text_score": 0.35,
            "Global.log_level": "error",
            "Det.lang_type": LangDet.CH,
            "Det.model_type": ModelType.TINY,
            "Det.ocr_version": OCRVersion.PPOCRV6,
            "Rec.lang_type": LangRec.CYRILLIC,
            "Rec.model_type": ModelType.MOBILE,
            "Rec.ocr_version": OCRVersion.PPOCRV5,
            "EngineConfig.onnxruntime.intra_op_num_threads": 2,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        }
    )


def prepare_ocr_image(path: Path, reference: bool) -> np.ndarray:
    image = reference_views(path)[1] if reference else query_views(path)[1]
    # The catalog labels are already clean crops.  Upscaling them beyond 540 px
    # makes the tiny detector slower and, in practice, slightly less accurate on
    # narrow Cyrillic glyphs.  Query images keep a little more detail because of
    # perspective distortion and blur.
    target = 540 if reference else 700
    scale = target / max(image.size)
    if scale > 1.0:
        image = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
    image = ImageOps.autocontrast(image.convert("RGB"), cutoff=1)
    return np.asarray(image)


def initialize_ocr_worker() -> None:
    global _OCR_ENGINE
    _OCR_ENGINE = create_ocr_engine()


def run_ocr(path: Path, reference: bool, engine: Any) -> dict[str, Any]:
    started = time.perf_counter()
    result = engine(prepare_ocr_image(path, reference), use_cls=False)
    lines = []
    if result.txts and result.scores:
        for text, score in zip(result.txts, result.scores, strict=True):
            normalized = normalize_text(text)
            if normalized:
                lines.append({"text": text, "normalized": normalized, "score": float(score)})
    return {"lines": lines, "latency_ms": (time.perf_counter() - started) * 1000}


def run_ocr_worker(arguments: tuple[str, bool]) -> dict[str, Any]:
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        _OCR_ENGINE = create_ocr_engine()
    path_string, reference = arguments
    return run_ocr(Path(path_string), reference, _OCR_ENGINE)


def ocr_similarity(query_lines: list[dict[str, Any]], reference: dict[str, Any]) -> float:
    candidate_tokens = text_tokens(
        " ".join(
            [
                str(reference.get("metadata_text") or ""),
                " ".join(str(line.get("normalized") or "") for line in reference.get("ocr_lines", [])),
            ]
        )
    )
    if not candidate_tokens or not query_lines:
        return 0.0

    matches: list[tuple[float, float]] = []
    candidate_numbers = {token for token in candidate_tokens if token.isdigit()}
    for line in query_lines:
        confidence = float(line.get("score") or 0.0)
        for token in text_tokens(str(line.get("normalized") or "")):
            if token.isdigit():
                similarity = 1.0 if token in candidate_numbers else 0.0
            else:
                similarity = max(
                    SequenceMatcher(None, token, candidate).ratio() for candidate in candidate_tokens
                )
            if similarity >= 0.55:
                weight = confidence * min(len(token) / 7.0, 1.0)
                matches.append((similarity, weight))
    if not matches:
        return 0.0
    matches.sort(key=lambda item: item[0] * item[1], reverse=True)
    selected = matches[:6]
    total_weight = sum(weight for _, weight in selected)
    if total_weight <= 0:
        return 0.0
    score = sum(similarity * weight for similarity, weight in selected) / total_weight
    evidence = min(len(selected) / 3.0, 1.0)
    return float(score * (0.55 + 0.45 * evidence))
