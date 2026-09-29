"""Runtime RapidOCR + GLM-OCR adapter.

GLM is loaded lazily on the first query so the API can start quickly.  If the
optional VLM is unavailable (for example on a CPU-only host), RapidOCR remains
the safe fallback and the request still completes.
"""

from __future__ import annotations

import re
import threading
import time
from typing import Any

import numpy as np
from PIL import Image
from rapidfuzz import fuzz

from stage4.features import normalize_text, text_tokens


GLM_MODEL = "zai-org/GLM-OCR"


def _clean_glm_lines(text: str) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in text.replace("<|endoftext|>", "").splitlines():
        value = re.sub(r"^\s*(?:```\w*|```|[-*#>]+)\s*", "", raw).strip()
        normalized = normalize_text(value)
        if len(normalized) < 2 or normalized in seen:
            continue
        seen.add(normalized)
        lines.append({"text": value, "normalized": normalized, "score": 1.0, "view": "detail", "box": None})
    return lines


def _merge_lines(rapid: list[dict[str, Any]], glm: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    positions: dict[str, int] = {}
    for source, lines in (("RapidOCR", rapid), ("GLM-OCR", glm)):
        for line in lines:
            normalized = normalize_text(str(line.get("normalized") or line.get("text") or ""))
            if not normalized:
                continue
            if normalized in positions:
                current = merged[positions[normalized]]
                current["source"] = "RapidOCR+GLM-OCR"
                current["score"] = max(float(current.get("score") or 0.0), float(line.get("score") or 1.0))
                continue
            positions[normalized] = len(merged)
            merged.append({**line, "normalized": normalized, "score": float(line.get("score") or 1.0), "source": source})
    return merged


class DualOCREngine:
    dual_enabled = True

    def __init__(self, rapid_engine: Any, device: str = "cuda", local_files_only: bool = True) -> None:
        self.rapid_engine = rapid_engine
        self.device_name = device
        self.local_files_only = local_files_only
        self._model: Any | None = None
        self._processor: Any | None = None
        self._load_error: str | None = None
        self._lock = threading.Lock()

    @property
    def model_ready(self) -> bool:
        return self._model is not None and self._processor is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def _ensure_glm(self) -> bool:
        if self.model_ready:
            return True
        if self._load_error:
            return False
        with self._lock:
            if self.model_ready:
                return True
            try:
                import torch
                from transformers import AutoModelForMultimodalLM, AutoProcessor

                if self.device_name != "cuda" or not torch.cuda.is_available():
                    raise RuntimeError("GLM-OCR runtime requires CUDA")
                self._processor = AutoProcessor.from_pretrained(GLM_MODEL, local_files_only=self.local_files_only)
                self._model = AutoModelForMultimodalLM.from_pretrained(
                    GLM_MODEL,
                    dtype=torch.bfloat16,
                    local_files_only=self.local_files_only,
                ).to("cuda").eval()
                return True
            except Exception as error:  # pragma: no cover - depends on local model/GPU
                self._load_error = f"{type(error).__name__}: {error}"
                self._model = None
                self._processor = None
                return False

    def _infer_glm(self, image: Image.Image) -> str:
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "Text Recognition:"}]}]
        inputs = self._processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(self._model.device)
        inputs.pop("token_type_ids", None)
        import torch

        with torch.inference_mode():
            output = self._model.generate(**inputs, max_new_tokens=512, do_sample=False)
        generated = output[0][inputs["input_ids"].shape[-1] :]
        return self._processor.decode(generated, skip_special_tokens=True)

    def run(self, image: np.ndarray, reference: bool = False) -> dict[str, Any]:
        started = time.perf_counter()
        rapid_result = self.rapid_engine(image, use_cls=False)
        rapid_lines = []
        if rapid_result.txts is not None and rapid_result.scores is not None:
            for text, score in zip(rapid_result.txts, rapid_result.scores, strict=True):
                normalized = normalize_text(str(text))
                if normalized:
                    rapid_lines.append({"text": str(text), "normalized": normalized, "score": float(score)})
        # Catalog reference OCR is already clean and must remain cheap.
        if reference or not self._ensure_glm():
            return {"lines": rapid_lines, "rapid_lines": rapid_lines, "glm_lines": [], "latency_ms": (time.perf_counter() - started) * 1000}
        pil_image = Image.fromarray(image).convert("RGB")
        pil_image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
        glm_lines = _clean_glm_lines(self._infer_glm(pil_image))
        return {
            "lines": _merge_lines(rapid_lines, glm_lines),
            "rapid_lines": rapid_lines,
            "glm_lines": glm_lines,
            "latency_ms": (time.perf_counter() - started) * 1000,
        }

    def structured_scores(
        self,
        lines: list[dict[str, Any]],
        catalog_by_slug: dict[str, dict[str, Any]],
        slugs: Any,
    ) -> np.ndarray:
        """Score title/manufacturer/grape/year evidence for GLM text.

        This mirrors the offline dual-OCR field score and is used only to
        widen/reweight the existing visual+Rapid candidate pool.
        """
        query = text_tokens(" ".join(str(line.get("normalized") or line.get("text") or "") for line in lines))
        result = np.zeros(len(slugs), dtype=np.float32)
        if not query:
            return result
        for index, slug in enumerate(slugs):
            record = catalog_by_slug.get(str(slug), {})
            values = [
                text_tokens(str(record.get("title") or "")),
                text_tokens(str(record.get("manufacturer") or "")),
                text_tokens(" ".join(str(value) for value in record.get("grapes") or [])),
            ]
            field_scores = []
            for field_tokens in values:
                if not field_tokens:
                    field_scores.append(0.0)
                    continue
                matches = [max(fuzz.ratio(token, candidate) for token in query) / 100.0 for candidate in field_tokens]
                weights = [min(len(token), 10) for token in field_tokens]
                field_scores.append(float(np.average(matches, weights=weights)))
            result[index] = 0.45 * field_scores[0] + 0.30 * field_scores[1] + 0.20 * field_scores[2]
        return result
