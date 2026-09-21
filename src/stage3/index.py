from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from stage1.catalog import read_jsonl, write_json
from stage3.embeddings import Dinov2Embedder, MODEL_NAME, reference_views


DEFAULT_MANIFEST = Path("data/artifacts/stage1/catalog_manifest.jsonl")
DEFAULT_SOURCE_ROOT = Path("data/parser")
DEFAULT_OUTPUT_DIR = Path("data/artifacts/stage3/dinov2-small")


@dataclass(frozen=True)
class VisualIndex:
    slugs: np.ndarray
    image_paths: np.ndarray
    full: np.ndarray
    label: np.ndarray
    label_patches: np.ndarray | None = None
    model_name: str = MODEL_NAME
    patch_grid: int = 0

    @classmethod
    def load(cls, path: Path) -> "VisualIndex":
        with np.load(path, allow_pickle=False) as payload:
            return cls(
                slugs=payload["slugs"],
                image_paths=payload["image_paths"],
                full=payload["full"].astype(np.float32),
                label=payload["label"].astype(np.float32),
                label_patches=payload["label_patches"] if "label_patches" in payload else None,
                model_name=str(payload["model_name"].item()) if "model_name" in payload else MODEL_NAME,
                patch_grid=int(payload["patch_grid"].item()) if "patch_grid" in payload else 0,
            )

    def cls_scores(
        self,
        query_full: np.ndarray,
        query_labels: np.ndarray,
        full_weight: float = 0.55,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if query_labels.ndim == 2:
            query_labels = query_labels[:, None, :]
        full_scores = query_full @ self.full.T
        per_view = np.stack([view @ self.label.T for view in query_labels.transpose(1, 0, 2)])
        label_scores = per_view.max(axis=0)
        combined = full_weight * full_scores + (1.0 - full_weight) * label_scores
        return full_scores, label_scores, combined

    def search(
        self,
        query_full: np.ndarray,
        query_label: np.ndarray,
        top_k: int = 5,
        full_weight: float = 0.60,
    ) -> tuple[np.ndarray, np.ndarray]:
        _, _, scores = self.cls_scores(query_full, query_label, full_weight)
        top_k = min(top_k, len(self.slugs))
        candidates = np.argpartition(-scores, top_k - 1, axis=1)[:, :top_k]
        candidate_scores = np.take_along_axis(scores, candidates, axis=1)
        order = np.argsort(-candidate_scores, axis=1)
        return np.take_along_axis(candidates, order, axis=1), np.take_along_axis(
            candidate_scores, order, axis=1
        )

    def patch_scores(
        self,
        query_patches: np.ndarray,
        candidates: np.ndarray,
        device: str,
        batch_size: int = 8,
    ) -> np.ndarray:
        if self.label_patches is None:
            raise ValueError("Index does not contain patch tokens; rebuild it with the current code")
        import torch

        if query_patches.ndim == 3:
            query_patches = query_patches[:, None, :, :]
        result = np.empty(candidates.shape, dtype=np.float32)
        torch_device = torch.device(device)
        dtype = torch.float16 if torch_device.type == "cuda" else torch.float32
        started = 0
        while started < len(candidates):
            stopped = min(started + batch_size, len(candidates))
            query = torch.from_numpy(query_patches[started:stopped]).to(
                device=torch_device, dtype=dtype
            )
            reference = torch.from_numpy(self.label_patches[candidates[started:stopped]]).to(
                device=torch_device, dtype=dtype
            )
            with torch.inference_mode():
                similarities = torch.einsum("bvtd,bksd->bvkts", query, reference)
                maxsim = similarities.max(dim=-1).values.mean(dim=-1).max(dim=1).values
            result[started:stopped] = maxsim.float().cpu().numpy()
            started = stopped
        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)
        return result


def build_index(
    manifest_path: Path,
    source_root: Path,
    output_dir: Path,
    model_name: str = MODEL_NAME,
    device: str = "auto",
    batch_size: int = 16,
    limit: int | None = None,
    local_files_only: bool = False,
    patch_grid: int = 7,
) -> dict[str, Any]:
    records = read_jsonl(manifest_path)
    if limit is not None:
        records = records[:limit]
    if not records:
        raise ValueError(f"Catalog manifest is empty: {manifest_path}")

    slugs: list[str] = []
    image_paths: list[str] = []
    paths: list[Path] = []
    for record in records:
        image_path = source_root / str(record["image_local"])
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        slugs.append(str(record["slug"]))
        image_paths.append(image_path.as_posix())
        paths.append(image_path)

    embedder = Dinov2Embedder(model_name, device, local_files_only)
    full_vectors, full_ms = embedder.encode_paths(paths, reference_views, 0, batch_size)
    label_vectors, label_patches, label_ms = embedder.encode_paths_with_patches(
        paths, reference_views, 1, batch_size, patch_grid
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "index.npz"
    np.savez(
        index_path,
        slugs=np.asarray(slugs),
        image_paths=np.asarray(image_paths),
        full=full_vectors,
        label=label_vectors,
        label_patches=label_patches,
        model_name=np.asarray(model_name),
        patch_grid=np.asarray(patch_grid),
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "manifest": manifest_path.as_posix(),
        "source_root": source_root.as_posix(),
        "index": index_path.as_posix(),
        "item_count": len(slugs),
        "view_count": 2,
        "patch_grid": patch_grid,
        "patch_tokens_per_reference": patch_grid * patch_grid,
        "embedding_ms": {"full": full_ms, "label": label_ms, "total": full_ms + label_ms},
        "runtime": embedder.runtime_info(),
    }
    write_json(output_dir / "index_report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a DINOv2 wine reference index.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--patch-grid", type=int, default=7)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = build_index(
        args.manifest,
        args.source_root,
        args.output_dir,
        args.model,
        args.device,
        args.batch_size,
        args.limit,
        args.local_files_only,
        args.patch_grid,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
