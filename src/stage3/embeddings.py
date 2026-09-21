from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps


MODEL_NAME = "facebook/dinov2-small"
IMAGE_SIZE = 518
IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


def open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as source:
        source.load()
        if source.mode in {"RGBA", "LA"} or "transparency" in source.info:
            rgba = source.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (242, 242, 242, 255))
            return Image.alpha_composite(background, rgba).convert("RGB")
        return source.convert("RGB")


def subject_bbox(path: Path, image: Image.Image) -> tuple[int, int, int, int]:
    with Image.open(path) as source:
        source.load()
        if source.mode in {"RGBA", "LA"} or "transparency" in source.info:
            alpha = np.asarray(source.convert("RGBA"), dtype=np.uint8)[..., 3]
            ys, xs = np.where(alpha > 12)
            if xs.size:
                return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)
    return 0, 0, image.width, image.height


def reference_views(path: Path) -> tuple[Image.Image, Image.Image, Image.Image]:
    image = open_rgb(path)
    left, top, right, bottom = subject_bbox(path, image)
    subject = image.crop((left, top, right, bottom))
    height = subject.height
    label = subject.crop((0, round(height * 0.24), subject.width, round(height * 0.94)))
    detail = subject.crop((0, round(height * 0.36), subject.width, round(height * 0.86)))
    return image, label, detail


def query_views(path: Path) -> tuple[Image.Image, Image.Image, Image.Image]:
    image = open_rgb(path)
    width, height = image.size
    medium = image.crop(
        (
            round(width * 0.10),
            round(height * 0.10),
            round(width * 0.90),
            round(height * 0.99),
        )
    )
    detail = image.crop(
        (
            round(width * 0.22),
            round(height * 0.30),
            round(width * 0.78),
            round(height * 0.91),
        )
    )
    return image, medium, detail


def prepare_image(image: Image.Image, size: int = IMAGE_SIZE) -> np.ndarray:
    contained = ImageOps.contain(image, (size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (242, 242, 242))
    offset = ((size - contained.width) // 2, (size - contained.height) // 2)
    canvas.paste(contained, offset)
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(array, (2, 0, 1))


class Dinov2Embedder:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: str = "auto",
        local_files_only: bool = False,
    ) -> None:
        import torch
        from transformers import AutoModel

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false")

        self.torch = torch
        self.device = torch.device(device)
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
        self.model.eval().to(self.device)
        self.embedding_size = int(self.model.config.hidden_size)
        self.patch_size = int(getattr(self.model.config, "patch_size", 14))
        self.register_tokens = int(getattr(self.model.config, "num_register_tokens", 0))
        configured_size = getattr(self.model.config, "image_size", IMAGE_SIZE)
        self.image_size = int(configured_size[0] if isinstance(configured_size, (list, tuple)) else configured_size)
        self.revision = getattr(self.model.config, "_commit_hash", None)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def _encode(
        self,
        images: Iterable[Image.Image],
        batch_size: int,
        patch_grid: int | None,
    ) -> tuple[np.ndarray, np.ndarray | None, float]:
        image_list = list(images)
        if not image_list:
            empty = np.empty((0, self.embedding_size), dtype=np.float32)
            patches = (
                np.empty((0, patch_grid * patch_grid, self.embedding_size), dtype=np.float16)
                if patch_grid
                else None
            )
            return empty, patches, 0.0

        vectors: list[np.ndarray] = []
        patch_vectors: list[np.ndarray] = []
        started = time.perf_counter()
        for offset in range(0, len(image_list), batch_size):
            batch = np.stack(
                [prepare_image(image, self.image_size) for image in image_list[offset : offset + batch_size]]
            )
            tensor = self.torch.from_numpy(batch).to(self.device)
            with self.torch.inference_mode():
                if self.device.type == "cuda":
                    with self.torch.autocast(device_type="cuda", dtype=self.torch.float16):
                        hidden = self.model(pixel_values=tensor).last_hidden_state
                else:
                    hidden = self.model(pixel_values=tensor).last_hidden_state
                cls_output = self.torch.nn.functional.normalize(hidden[:, 0].float(), dim=1)
                if patch_grid:
                    patch_output = hidden[:, 1 + self.register_tokens :].float()
                    side = int(round(patch_output.shape[1] ** 0.5))
                    if side * side != patch_output.shape[1]:
                        raise ValueError(
                            f"Patch token count is not square: {patch_output.shape[1]} for {self.model_name}"
                        )
                    patch_output = patch_output.transpose(1, 2).reshape(
                        patch_output.shape[0], self.embedding_size, side, side
                    )
                    patch_output = self.torch.nn.functional.adaptive_avg_pool2d(
                        patch_output, (patch_grid, patch_grid)
                    )
                    patch_output = patch_output.flatten(2).transpose(1, 2)
                    patch_output = self.torch.nn.functional.normalize(patch_output, dim=2)
                    patch_vectors.append(patch_output.cpu().numpy().astype(np.float16))
            vectors.append(cls_output.cpu().numpy())
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)
        elapsed_ms = (time.perf_counter() - started) * 1000
        patches = np.concatenate(patch_vectors) if patch_vectors else None
        return np.concatenate(vectors).astype(np.float32, copy=False), patches, elapsed_ms

    def encode(self, images: Iterable[Image.Image], batch_size: int = 16) -> tuple[np.ndarray, float]:
        vectors, _, elapsed_ms = self._encode(images, batch_size, None)
        return vectors, elapsed_ms

    def encode_with_patches(
        self,
        images: Iterable[Image.Image],
        batch_size: int = 16,
        patch_grid: int = 7,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        vectors, patches, elapsed_ms = self._encode(images, batch_size, patch_grid)
        if patches is None:
            raise AssertionError("Patch extraction returned no patches")
        return vectors, patches, elapsed_ms

    def encode_paths(
        self,
        paths: Sequence[Path],
        view_factory: Callable[[Path], tuple[Image.Image, ...]],
        view_index: int,
        batch_size: int = 16,
    ) -> tuple[np.ndarray, float]:
        vectors: list[np.ndarray] = []
        started = time.perf_counter()
        for offset in range(0, len(paths), batch_size):
            images = [view_factory(path)[view_index] for path in paths[offset : offset + batch_size]]
            batch_vectors, _ = self.encode(images, batch_size)
            vectors.append(batch_vectors)
        elapsed_ms = (time.perf_counter() - started) * 1000
        if not vectors:
            return np.empty((0, self.embedding_size), dtype=np.float32), elapsed_ms
        return np.concatenate(vectors).astype(np.float32, copy=False), elapsed_ms

    def encode_paths_with_patches(
        self,
        paths: Sequence[Path],
        view_factory: Callable[[Path], tuple[Image.Image, ...]],
        view_index: int,
        batch_size: int = 16,
        patch_grid: int = 7,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        vectors: list[np.ndarray] = []
        patches: list[np.ndarray] = []
        started = time.perf_counter()
        for offset in range(0, len(paths), batch_size):
            images = [view_factory(path)[view_index] for path in paths[offset : offset + batch_size]]
            batch_vectors, batch_patches, _ = self.encode_with_patches(
                images, batch_size, patch_grid
            )
            vectors.append(batch_vectors)
            patches.append(batch_patches)
        elapsed_ms = (time.perf_counter() - started) * 1000
        if not vectors:
            return (
                np.empty((0, self.embedding_size), dtype=np.float32),
                np.empty((0, patch_grid * patch_grid, self.embedding_size), dtype=np.float16),
                elapsed_ms,
            )
        return (
            np.concatenate(vectors).astype(np.float32, copy=False),
            np.concatenate(patches).astype(np.float16, copy=False),
            elapsed_ms,
        )

    def warmup(self) -> None:
        self.encode([Image.new("RGB", (self.image_size, self.image_size), (242, 242, 242))], 1)

    def runtime_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "model": self.model_name,
            "model_revision": self.revision,
            "device": str(self.device),
            "embedding_size": self.embedding_size,
            "image_size": self.image_size,
            "patch_size": self.patch_size,
        }
        if self.device.type == "cuda":
            info.update(
                {
                    "gpu_name": self.torch.cuda.get_device_name(self.device),
                    "peak_vram_bytes": int(self.torch.cuda.max_memory_allocated(self.device)),
                }
            )
        return info
