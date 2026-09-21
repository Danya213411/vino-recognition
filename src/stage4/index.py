from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from stage1.catalog import read_jsonl, write_json, write_jsonl
from stage4.features import (
    extract_sift,
    initialize_ocr_worker,
    metadata_text,
    run_ocr_worker,
)


DEFAULT_MANIFEST = Path("data/artifacts/stage1/catalog_manifest.jsonl")
DEFAULT_SOURCE_ROOT = Path("data/parser")
DEFAULT_OUTPUT_DIR = Path("data/artifacts/stage4/index")


@dataclass(frozen=True)
class HybridIndex:
    slugs: np.ndarray
    points: np.ndarray
    descriptors: np.ndarray
    offsets: np.ndarray
    text_records: list[dict[str, Any]]

    @classmethod
    def load(cls, directory: Path) -> "HybridIndex":
        with np.load(directory / "sift_index.npz", allow_pickle=False) as payload:
            slugs = payload["slugs"]
            points = payload["points"].astype(np.float32)
            descriptors = payload["descriptors"].astype(np.float32)
            offsets = payload["offsets"]
        text_records = read_jsonl(directory / "reference_text.jsonl")
        if [str(slug) for slug in slugs] != [str(record["slug"]) for record in text_records]:
            raise ValueError("SIFT and OCR index slug order differs")
        return cls(slugs, points, descriptors, offsets, text_records)

    def features(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        start, stop = int(self.offsets[index]), int(self.offsets[index + 1])
        return self.points[start:stop], self.descriptors[start:stop]


def run_reference_ocr(paths: list[Path], workers: int) -> list[dict[str, Any]]:
    arguments = [(str(path), True) for path in paths]
    if workers <= 1:
        initialize_ocr_worker()
        return [run_ocr_worker(argument) for argument in arguments]
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize_ocr_worker) as executor:
        return list(executor.map(run_ocr_worker, arguments, chunksize=8))


def build_index(
    manifest_path: Path,
    source_root: Path,
    output_dir: Path,
    workers: int = 2,
    limit: int | None = None,
) -> dict[str, Any]:
    records = read_jsonl(manifest_path)
    if limit is not None:
        records = records[:limit]
    if not records:
        raise ValueError(f"Catalog manifest is empty: {manifest_path}")

    paths = [source_root / str(record["image_local"]) for record in records]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])

    started = time.perf_counter()
    all_points: list[np.ndarray] = []
    all_descriptors: list[np.ndarray] = []
    offsets = [0]
    empty_sift = 0
    for index, path in enumerate(paths, start=1):
        points, descriptors = extract_sift(path, reference=True)
        if len(descriptors) == 0:
            empty_sift += 1
        all_points.append(points)
        all_descriptors.append(descriptors.astype(np.float16))
        offsets.append(offsets[-1] + len(descriptors))
        if index % 250 == 0:
            print(f"SIFT references: {index}/{len(paths)}")
    sift_ms = (time.perf_counter() - started) * 1000

    ocr_started = time.perf_counter()
    ocr_results = run_reference_ocr(paths, workers)
    ocr_ms = (time.perf_counter() - ocr_started) * 1000

    output_dir.mkdir(parents=True, exist_ok=True)
    points_array = (
        np.concatenate(all_points).astype(np.float32)
        if offsets[-1]
        else np.empty((0, 2), dtype=np.float32)
    )
    descriptors_array = (
        np.concatenate(all_descriptors).astype(np.float16)
        if offsets[-1]
        else np.empty((0, 128), dtype=np.float16)
    )
    np.savez(
        output_dir / "sift_index.npz",
        slugs=np.asarray([str(record["slug"]) for record in records]),
        points=points_array,
        descriptors=descriptors_array,
        offsets=np.asarray(offsets, dtype=np.int64),
    )

    text_records = []
    for record, ocr_result in zip(records, ocr_results, strict=True):
        text_records.append(
            {
                "slug": record["slug"],
                "metadata_text": metadata_text(record),
                "ocr_lines": ocr_result["lines"],
                "ocr_latency_ms": ocr_result["latency_ms"],
            }
        )
    write_jsonl(output_dir / "reference_text.jsonl", text_records)

    report = {
        "schema_version": 1,
        "item_count": len(records),
        "sift": {
            "descriptor_count": int(offsets[-1]),
            "empty_item_count": empty_sift,
            "descriptor_dtype": "float16",
            "build_ms": sift_ms,
        },
        "ocr": {
            "engine": "RapidOCR PP-OCRv6 tiny detector + PP-OCRv5 Cyrillic recognizer",
            "workers": workers,
            "nonempty_item_count": sum(bool(result["lines"]) for result in ocr_results),
            "line_count": sum(len(result["lines"]) for result in ocr_results),
            "build_ms": ocr_ms,
            "mean_image_ms": sum(result["latency_ms"] for result in ocr_results) / len(records),
        },
        "files": {
            "sift_index": (output_dir / "sift_index.npz").as_posix(),
            "reference_text": (output_dir / "reference_text.jsonl").as_posix(),
        },
    }
    write_json(output_dir / "index_report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build SIFT and OCR reference indexes.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = build_index(
        args.manifest,
        args.source_root,
        args.output_dir,
        max(args.workers, 1),
        args.limit,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
