from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from pathlib import Path
from typing import Any

import psutil

from stage1.catalog import read_jsonl
from stage3.embeddings import Dinov2Embedder
from stage3.index import VisualIndex
from stage4.features import create_ocr_engine
from stage4.index import HybridIndex
from stage5.dual_ocr import DualOCREngine
from stage5.recognize import recognize_confident_paths
from vino_api.config import Settings


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path, project_root: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return path.as_posix()


class RecognitionEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.visual_index = VisualIndex.load(settings.visual_index)
        self.hybrid_index = HybridIndex.load(settings.hybrid_index_dir)
        self.catalog = read_jsonl(settings.catalog_manifest)
        self.catalog_by_slug = {str(record["slug"]): record for record in self.catalog}
        self.model_payload = json.loads(settings.confidence_model.read_text(encoding="utf-8"))
        self.embedder = Dinov2Embedder(
            self.visual_index.model_name, settings.device, settings.local_files_only
        )
        if self.embedder.embedding_size != self.visual_index.full.shape[1]:
            raise ValueError("DINOv2 model and visual index embedding sizes differ")
        rapid_ocr = create_ocr_engine()
        self.ocr_engine = DualOCREngine(
            rapid_ocr,
            device=str(self.embedder.device.type),
            local_files_only=settings.local_files_only,
        )
        self.semaphore = threading.BoundedSemaphore(max(settings.max_parallel, 1))
        self.embedder.warmup()
        self.pipeline = self._pipeline_manifest()

    def _pipeline_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "name": "stage5-confident-hybrid",
            "visual_model": self.embedder.runtime_info(),
            "signals": {
                "visual": "DINOv2 CLS, three image scales",
                "local_features": "OpenCV SIFT + geometric inliers",
                "ocr": "RapidOCR PP-OCRv6 detector + PP-OCRv5 Cyrillic recognizer + GLM-OCR Text Recognition",
                "confidence": "calibrated logistic model, 16 evidence features",
            },
            "candidate_retrieval": {
                "strategy": "visual+ocr union with fuzzy manufacturer gate",
                "semantic_rerank": {
                    "enabled": True,
                    "attributes": ["colour", "sweetness", "sparkling"],
                    "policy": "small compatibility bonus on retrieved candidates; no catalog items removed",
                },
                "visual_top_k": self.settings.candidate_count,
                "global_ocr_top_k": self.settings.ocr_candidate_count,
                "manufacturer_source": "https://vino-svoe.ru/wines filters / local catalog intersection",
                "manufacturer_match": {
                    "normalization": (
                        "Cyrillic/Latin transliteration + normalized edit distance + "
                        "compound aliases + canonical groups + safe prefixes + OCR homoglyphs"
                    ),
                    "hard_threshold": 0.86,
                    "strong_fuzzy_threshold": 0.82,
                    "minimum_margin": 0.08,
                },
            },
            "multimodal_verification": {
                "ocr_score_min": 0.92,
                "sift_score_min": 0.72,
                "sift_inliers_min": 12,
                "title_evidence_min": 0.82,
                "decision_confidence_floor": 0.95,
            },
            "parameters": {
                "top_k": self.settings.top_k,
                "candidate_count": self.settings.candidate_count,
                "ocr_candidate_count": self.settings.ocr_candidate_count,
                "dual_ocr_extra_candidate_count": 10,
                "hybrid_weights": self.model_payload["hybrid_weights"],
                "thresholds": {
                    "accept": self.model_payload["accept_threshold"],
                    "review": self.model_payload["review_threshold"],
                },
                "max_parallel": self.settings.max_parallel,
                "local_files_only": self.settings.local_files_only,
            },
            "artifacts": {
                "visual_index": {
                    "path": display_path(self.settings.visual_index, self.settings.project_root),
                    "sha256": sha256_file(self.settings.visual_index),
                },
                "confidence_model": {
                    "path": display_path(self.settings.confidence_model, self.settings.project_root),
                    "sha256": sha256_file(self.settings.confidence_model),
                },
                "catalog_manifest": {
                    "path": display_path(self.settings.catalog_manifest, self.settings.project_root),
                    "sha256": sha256_file(self.settings.catalog_manifest),
                    "items": len(self.catalog),
                },
            },
        }

    def recognize(self, image_path: Path) -> dict[str, Any]:
        started = time.perf_counter()
        with self.semaphore:
            queue_ms = (time.perf_counter() - started) * 1000
            process = psutil.Process()
            rss_before = process.memory_info().rss
            result = recognize_confident_paths(
                [image_path],
                self.visual_index,
                self.hybrid_index,
                self.embedder,
                self.ocr_engine,
                self.catalog_by_slug,
                self.model_payload,
                top_k=self.settings.top_k,
                candidate_count=self.settings.candidate_count,
                batch_size=1,
                ocr_candidate_count=self.settings.ocr_candidate_count,
            )[0]
            rss_after = process.memory_info().rss
        total_ms = (time.perf_counter() - started) * 1000
        result["timing_ms"] = {
            **result["timing_ms"],
            "queue": queue_ms,
            "api_total": total_ms,
        }
        result["memory"] = {
            "process_rss_bytes_before": rss_before,
            "process_rss_bytes_after": rss_after,
        }
        if self.embedder.device.type == "cuda":
            result["memory"]["cuda_peak_allocated_bytes"] = int(
                self.embedder.torch.cuda.max_memory_allocated(self.embedder.device)
            )
        result["predictions"] = self._decorate_predictions(result["predictions"])
        result["wine"] = self.public_wine(result["candidate_slug"])
        return result

    def _decorate_predictions(self, predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not predictions:
            return []
        exponentials = [math.exp((float(item["score"]) - float(predictions[0]["score"])) / 0.08) for item in predictions]
        denominator = sum(exponentials)
        return [
            {
                **item,
                "relative_support": exponentials[index] / denominator,
                "wine": self.public_wine(str(item["slug"])),
            }
            for index, item in enumerate(predictions)
        ]

    def public_wine(self, slug: str) -> dict[str, Any]:
        record = self.catalog_by_slug[slug]
        return {
            key: record.get(key)
            for key in (
                "slug",
                "title",
                "manufacturer",
                "region",
                "category",
                "color",
                "description",
                "grapes",
                "dishes",
                "alcohol",
                "rating",
                "public_rating",
                "temperature",
                "source_url",
                "color_gradient",
            )
        } | {"image_url": f"/api/v1/catalog/{slug}/image"}

    def catalog_search(self, query: str, limit: int) -> list[dict[str, Any]]:
        normalized = query.casefold().strip()
        records = self.catalog
        if normalized:
            records = [
                item
                for item in records
                if normalized in " ".join(
                    str(item.get(key) or "")
                    for key in ("title", "manufacturer", "region", "slug")
                ).casefold()
            ]
        return [self.public_wine(str(item["slug"])) for item in records[:limit]]

    def catalog_image(self, slug: str) -> Path:
        record = self.catalog_by_slug[slug]
        path = (self.settings.project_root / "data/parser" / str(record["image_local"])).resolve()
        parser_root = (self.settings.project_root / "data/parser").resolve()
        if parser_root not in path.parents or not path.is_file():
            raise FileNotFoundError(path)
        return path
