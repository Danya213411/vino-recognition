from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from stage1.catalog import read_jsonl, write_json, write_jsonl


DEFAULT_MANIFEST = Path("data/artifacts/stage1/catalog_manifest.jsonl")
DEFAULT_SOURCE_ROOT = Path("data/parser")
DEFAULT_OUTPUT = Path("data/synthetic/stage2")
DEFAULT_SEED = 20260920
PROFILES = (
    "background",
    "perspective",
    "glare",
    "crop",
    "compression",
    "combined",
)


def stable_seed(global_seed: int, slug: str, variant_index: int) -> int:
    payload = f"{global_seed}:{slug}:{variant_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def read_image(path: Path) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def infer_alpha(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    border = max(2, min(height, width) // 24)
    border_pixels = np.concatenate(
        (
            image[:border].reshape(-1, 3),
            image[-border:].reshape(-1, 3),
            image[:, :border].reshape(-1, 3),
            image[:, -border:].reshape(-1, 3),
        ),
        axis=0,
    ).astype(np.float32)
    background = np.median(border_pixels, axis=0)
    distance = np.linalg.norm(image.astype(np.float32) - background, axis=2)
    alpha = np.clip((distance - 7.0) * 10.0, 0, 255).astype(np.uint8)
    binary = (alpha > 18).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        alpha = np.where(labels == largest, alpha, 0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel, iterations=2)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 1.1)
    coverage = float(np.count_nonzero(alpha > 16)) / alpha.size
    if coverage < 0.04 or coverage > 0.98:
        alpha = np.full((height, width), 255, dtype=np.uint8)
    return alpha


def load_foreground(path: Path) -> tuple[np.ndarray, str]:
    source = read_image(path)
    if source.ndim == 2:
        source = cv2.cvtColor(source, cv2.COLOR_GRAY2BGRA)
        mask_source = "opaque_grayscale"
    elif source.shape[2] == 4:
        mask_source = "source_alpha"
    else:
        alpha = infer_alpha(source[:, :, :3])
        source = np.dstack((source[:, :, :3], alpha))
        mask_source = "inferred_border"

    alpha = source[:, :, 3]
    points = cv2.findNonZero((alpha > 8).astype(np.uint8))
    if points is not None:
        x, y, width, height = cv2.boundingRect(points)
        source = source[y : y + height, x : x + width]
    return source, mask_source


def shelf_background(size: int, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
    palettes = (
        ((39, 46, 58), (99, 109, 122)),
        ((61, 46, 37), (137, 105, 74)),
        ((48, 52, 47), (116, 122, 105)),
        ((70, 57, 61), (143, 116, 119)),
        ((46, 58, 70), (121, 137, 148)),
    )
    palette_index = int(rng.integers(0, len(palettes)))
    top, bottom = palettes[palette_index]
    gradient = np.linspace(0.0, 1.0, size, dtype=np.float32)[:, None, None]
    background = (
        np.asarray(top, dtype=np.float32)[None, None, :] * (1.0 - gradient)
        + np.asarray(bottom, dtype=np.float32)[None, None, :] * gradient
    )
    background = np.repeat(background, size, axis=1)
    noise = rng.normal(0.0, 7.0, background.shape).astype(np.float32)
    background = np.clip(background + noise, 0, 255).astype(np.uint8)

    bottle_count = int(rng.integers(8, 17))
    for _ in range(bottle_count):
        bottle_height = int(rng.uniform(0.20, 0.43) * size)
        bottle_width = int(rng.uniform(0.035, 0.075) * size)
        center_x = int(rng.uniform(-0.05, 1.05) * size)
        base_y = int(rng.choice((0.43, 0.72, 1.02)) * size)
        color = tuple(int(value) for value in rng.integers(18, 105, size=3))
        cv2.rectangle(
            background,
            (center_x - bottle_width // 2, base_y - bottle_height),
            (center_x + bottle_width // 2, base_y),
            color,
            thickness=-1,
        )
        cv2.ellipse(
            background,
            (center_x, base_y - bottle_height),
            (bottle_width // 2, max(2, bottle_width // 5)),
            0,
            180,
            360,
            color,
            thickness=-1,
        )

    shelf_color = tuple(int(value) for value in rng.integers(45, 105, size=3))
    for ratio in (0.44, 0.73):
        y = int(size * ratio)
        cv2.rectangle(background, (0, y), (size, y + max(4, size // 55)), shelf_color, -1)
    background = cv2.GaussianBlur(background, (0, 0), float(rng.uniform(2.5, 5.5)))
    return background, {"kind": "procedural_shelf", "palette": palette_index, "bottle_count": bottle_count}


def profile_parameters(profile: str, rng: np.random.Generator) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        "yaw_limit_degrees": 4.0,
        "pitch_limit_degrees": 3.0,
        "roll_limit_degrees": 2.5,
        "target_height": float(rng.uniform(0.74, 0.90)),
        "shift_x": float(rng.uniform(-0.07, 0.07)),
        "shift_y": float(rng.uniform(-0.025, 0.035)),
        "glare": False,
        "blur": "none",
        "low_resolution_scale": 1.0,
        "occlusion": False,
        "jpeg_quality": None,
    }
    if profile == "perspective":
        parameters["yaw_limit_degrees"] = 24.0
        parameters["pitch_limit_degrees"] = 11.0
        parameters["roll_limit_degrees"] = 8.0
    elif profile == "glare":
        parameters["glare"] = True
        parameters["yaw_limit_degrees"] = 7.0
        parameters["pitch_limit_degrees"] = 5.0
        parameters["roll_limit_degrees"] = 4.0
    elif profile == "crop":
        parameters["target_height"] = float(rng.uniform(1.00, 1.24))
        parameters["shift_x"] = float(rng.uniform(-0.17, 0.17))
        parameters["shift_y"] = float(rng.uniform(-0.12, 0.10))
    elif profile == "compression":
        parameters["low_resolution_scale"] = float(rng.uniform(0.38, 0.65))
        parameters["jpeg_quality"] = int(rng.integers(32, 56))
    elif profile == "combined":
        parameters["yaw_limit_degrees"] = 18.0
        parameters["pitch_limit_degrees"] = 9.0
        parameters["roll_limit_degrees"] = 7.0
        parameters["target_height"] = float(rng.uniform(0.85, 1.08))
        parameters["shift_x"] = float(rng.uniform(-0.13, 0.13))
        parameters["glare"] = bool(rng.random() < 0.75)
        parameters["low_resolution_scale"] = float(rng.uniform(0.48, 0.78))
        parameters["occlusion"] = bool(rng.random() < 0.55)
        parameters["jpeg_quality"] = int(rng.integers(42, 68))
    return parameters


def warp_bottle(
    foreground: np.ndarray,
    size: int,
    parameters: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    height, width = foreground.shape[:2]
    target_height = size * parameters["target_height"]
    center_x = size * (0.5 + parameters["shift_x"])
    center_y = size * (0.5 + parameters["shift_y"])
    source = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    aspect_ratio = width / height
    object_corners = np.float64(
        [
            [-aspect_ratio / 2, -0.5, 0.0],
            [aspect_ratio / 2, -0.5, 0.0],
            [aspect_ratio / 2, 0.5, 0.0],
            [-aspect_ratio / 2, 0.5, 0.0],
        ]
    )

    def rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
        yaw_radians, pitch_radians, roll_radians = np.deg2rad((yaw, pitch, roll))
        cos_y, sin_y = np.cos(yaw_radians), np.sin(yaw_radians)
        cos_p, sin_p = np.cos(pitch_radians), np.sin(pitch_radians)
        cos_r, sin_r = np.cos(roll_radians), np.sin(roll_radians)
        rotate_y = np.array(((cos_y, 0, sin_y), (0, 1, 0), (-sin_y, 0, cos_y)))
        rotate_x = np.array(((1, 0, 0), (0, cos_p, -sin_p), (0, sin_p, cos_p)))
        rotate_z = np.array(((cos_r, -sin_r, 0), (sin_r, cos_r, 0), (0, 0, 1)))
        return rotate_z @ rotate_x @ rotate_y

    def project(yaw: float, pitch: float, roll: float) -> np.ndarray:
        rotated = (rotation_matrix(yaw, pitch, roll) @ object_corners.T).T
        camera_distance = 2.8
        denominator = camera_distance - rotated[:, 2]
        projected = rotated[:, :2] * (camera_distance / denominator[:, None])
        projected_height = float(projected[:, 1].max() - projected[:, 1].min())
        scale = target_height / max(projected_height, 1e-6)
        projected *= scale
        projected[:, 0] += center_x - float(projected[:, 0].mean())
        projected[:, 1] += center_y - float(projected[:, 1].mean())
        return projected.astype(np.float32)

    def quality(destination: np.ndarray, warped: np.ndarray) -> dict[str, Any]:
        contour = destination.reshape((-1, 1, 2))
        edge_lengths = [
            float(np.linalg.norm(destination[(index + 1) % 4] - destination[index]))
            for index in range(4)
        ]
        top_bottom_ratio = edge_lengths[0] / max(edge_lengths[2], 1e-6)
        side_ratio = edge_lengths[1] / max(edge_lengths[3], 1e-6)
        alpha = warped[:, :, 3]
        points = cv2.findNonZero((alpha > 8).astype(np.uint8))
        if points is None:
            bbox_width = bbox_height = 0
        else:
            _, _, bbox_width, bbox_height = cv2.boundingRect(points)
        corner_min = float(destination.min() / size)
        corner_max = float(destination.max() / size)
        visible_fraction = float(np.count_nonzero(alpha > 8)) / alpha.size
        checks = {
            "convex": bool(cv2.isContourConvex(contour)),
            "edge_ratios": 0.48 <= top_bottom_ratio <= 2.1 and 0.68 <= side_ratio <= 1.48,
            "corner_bounds": corner_min >= -0.28 and corner_max <= 1.28,
            "visible_height": bbox_height / size >= 0.50,
            "visible_width": bbox_width / size >= 0.055,
            "visible_area": visible_fraction >= 0.018,
        }
        return {
            "passed": all(checks.values()),
            "checks": checks,
            "top_bottom_edge_ratio": top_bottom_ratio,
            "side_edge_ratio": side_ratio,
            "visible_area_fraction": visible_fraction,
            "visible_bbox_ratio": [bbox_width / size, bbox_height / size],
            "corner_range": [corner_min, corner_max],
        }

    selected: tuple[np.ndarray, np.ndarray, dict[str, Any], float, float, float] | None = None
    for attempt in range(6):
        reduction = 0.78**attempt
        yaw = float(rng.uniform(-parameters["yaw_limit_degrees"], parameters["yaw_limit_degrees"]) * reduction)
        pitch = float(rng.uniform(-parameters["pitch_limit_degrees"], parameters["pitch_limit_degrees"]) * reduction)
        roll = float(rng.uniform(-parameters["roll_limit_degrees"], parameters["roll_limit_degrees"]) * reduction)
        destination = project(yaw, pitch, roll)
        matrix = cv2.getPerspectiveTransform(source, destination)
        warped = cv2.warpPerspective(
            foreground,
            matrix,
            (size, size),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        qa = quality(destination, warped)
        if qa["passed"]:
            selected = destination, warped, qa, yaw, pitch, roll
            qa["attempt"] = attempt + 1
            break

    if selected is None:
        yaw = pitch = roll = 0.0
        destination = project(yaw, pitch, roll)
        matrix = cv2.getPerspectiveTransform(source, destination)
        warped = cv2.warpPerspective(
            foreground,
            matrix,
            (size, size),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        qa = quality(destination, warped)
        qa["attempt"] = 7
        qa["fallback"] = True
        selected = destination, warped, qa, yaw, pitch, roll

    destination, warped, qa, yaw, pitch, roll = selected
    metadata = {
        "target_height_ratio": parameters["target_height"],
        "shift_x": parameters["shift_x"],
        "shift_y": parameters["shift_y"],
        "yaw_degrees": yaw,
        "pitch_degrees": pitch,
        "roll_degrees": roll,
        "destination_corners": [
            [round(float(x / size), 5), round(float(y / size), 5)] for x, y in destination
        ],
        "qa": qa,
    }
    return warped, metadata


def composite_bottle(background: np.ndarray, bottle: np.ndarray) -> np.ndarray:
    alpha = bottle[:, :, 3].astype(np.float32) / 255.0
    shadow = cv2.GaussianBlur(alpha, (0, 0), 7.0)
    shadow = cv2.warpAffine(
        shadow,
        np.float32([[1, 0, 6], [0, 1, 8]]),
        (background.shape[1], background.shape[0]),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    result = background.astype(np.float32) * (1.0 - shadow[:, :, None] * 0.28)
    result = result * (1.0 - alpha[:, :, None]) + bottle[:, :, :3].astype(np.float32) * alpha[:, :, None]
    return np.clip(result, 0, 255).astype(np.uint8)


def add_glare(
    image: np.ndarray,
    bottle_alpha: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    size = image.shape[0]
    mask = np.zeros((size, size), dtype=np.uint8)
    center = (int(rng.uniform(0.36, 0.65) * size), int(rng.uniform(0.34, 0.68) * size))
    axes = (int(rng.uniform(0.025, 0.07) * size), int(rng.uniform(0.20, 0.43) * size))
    angle = float(rng.uniform(-28, 28))
    cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), float(rng.uniform(7, 14)))
    mask = mask.astype(np.float32) / 255.0 * (bottle_alpha.astype(np.float32) / 255.0)
    strength = float(rng.uniform(0.30, 0.62))
    result = image.astype(np.float32) * (1.0 - mask[:, :, None] * strength) + 255.0 * mask[:, :, None] * strength
    return np.clip(result, 0, 255).astype(np.uint8), {
        "center": [center[0] / size, center[1] / size],
        "axes": [axes[0] / size, axes[1] / size],
        "angle": angle,
        "strength": strength,
    }


def apply_blur(image: np.ndarray, kind: str, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
    if kind == "gaussian":
        sigma = float(rng.uniform(1.1, 2.4))
        return cv2.GaussianBlur(image, (0, 0), sigma), {"kind": kind, "sigma": sigma}
    if kind == "motion":
        length = int(rng.integers(5, 13))
        if length % 2 == 0:
            length += 1
        kernel = np.zeros((length, length), dtype=np.float32)
        kernel[length // 2, :] = 1.0 / length
        angle = float(rng.uniform(-25, 25))
        rotation = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1.0)
        kernel = cv2.warpAffine(kernel, rotation, (length, length))
        kernel /= max(float(kernel.sum()), 1e-6)
        return cv2.filter2D(image, -1, kernel), {"kind": kind, "length": length, "angle": angle}
    return image, {"kind": "none"}


def add_occlusion(image: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
    size = image.shape[0]
    overlay = image.copy()
    side = "left" if rng.random() < 0.5 else "right"
    width = int(rng.uniform(0.07, 0.15) * size)
    x1 = 0 if side == "left" else size - width
    color = tuple(int(value) for value in rng.integers(15, 80, size=3))
    cv2.rectangle(overlay, (x1, int(size * 0.18)), (x1 + width, size), color, -1)
    overlay = cv2.GaussianBlur(overlay, (0, 0), 2.5)
    alpha = float(rng.uniform(0.72, 0.92))
    result = cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0)
    return result, {"side": side, "width_ratio": width / size, "opacity": alpha}


def render_sample(
    source_path: Path,
    profile: str,
    size: int,
    seed: int,
    default_jpeg_quality: int,
) -> tuple[np.ndarray, dict[str, Any], int, str]:
    rng = np.random.default_rng(seed)
    foreground, mask_source = load_foreground(source_path)
    background, background_metadata = shelf_background(size, rng)
    parameters = profile_parameters(profile, rng)
    bottle, geometry_metadata = warp_bottle(foreground, size, parameters, rng)
    result = composite_bottle(background, bottle)

    transformations: dict[str, Any] = {
        "background": background_metadata,
        "geometry": geometry_metadata,
        "foreground_mask": mask_source,
    }
    if parameters["glare"]:
        result, glare_metadata = add_glare(result, bottle[:, :, 3], rng)
        transformations["glare"] = glare_metadata

    result, blur_metadata = apply_blur(result, parameters["blur"], rng)
    transformations["blur"] = blur_metadata

    if parameters["occlusion"]:
        result, occlusion_metadata = add_occlusion(result, rng)
        transformations["occlusion"] = occlusion_metadata

    brightness = float(rng.uniform(0.78, 1.18))
    contrast = float(rng.uniform(0.84, 1.20))
    result = np.clip((result.astype(np.float32) - 127.5) * contrast + 127.5, 0, 255)
    result = np.clip(result * brightness, 0, 255).astype(np.uint8)
    transformations["photometric"] = {"brightness": brightness, "contrast": contrast}

    low_resolution_scale = float(parameters["low_resolution_scale"])
    if low_resolution_scale < 1.0:
        reduced_size = max(96, int(size * low_resolution_scale))
        result = cv2.resize(result, (reduced_size, reduced_size), interpolation=cv2.INTER_AREA)
        result = cv2.resize(result, (size, size), interpolation=cv2.INTER_CUBIC)
        transformations["low_resolution"] = {
            "scale": low_resolution_scale,
            "intermediate_size": reduced_size,
        }

    jpeg_quality = int(parameters["jpeg_quality"] or default_jpeg_quality)
    return result, transformations, jpeg_quality, mask_source


def safe_filename(catalog_index: int, slug: str, variant_index: int) -> str:
    digest = hashlib.sha256(slug.encode("utf-8")).hexdigest()[:10]
    shortened = slug[:42].rstrip("-")
    return f"{catalog_index:04d}-{shortened}-{digest}-v{variant_index:02d}.jpg"


def encode_jpeg(path: Path, image: np.ndarray, quality: int) -> str:
    success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise ValueError(f"Cannot encode JPEG: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(path)
    return hashlib.sha256(encoded.tobytes()).hexdigest()


def report_markdown(report: dict[str, Any]) -> str:
    profile_rows = "\n".join(
        f"| `{profile}` | {count} |" for profile, count in report["profiles"].items()
    )
    return f"""# Синтетическая validation-выборка

## Сводка

| Показатель | Значение |
|---|---:|
| Исходных slug | {report['summary']['source_slug_count']} |
| Сгенерированных кадров | {report['summary']['sample_count']} |
| Размер кадра | {report['config']['image_size']}×{report['config']['image_size']} |
| Вариантов на slug | {report['config']['variants_per_wine']} |
| Seed | {report['config']['seed']} |
| Общий размер | {report['summary']['total_bytes']} байт |

## Профили

| Профиль | Кадров |
|---|---:|
{profile_rows}

Все производные одного `slug` относятся только к split `validation`. Обучающая выборка этим генератором не создаётся, поэтому утечки между train и validation нет.

`synthetic_manifest.jsonl` хранит правильный `expected_slug`, профиль, SHA-256 и точные параметры преобразований каждого изображения.
"""


def generate_dataset(
    manifest_path: Path,
    source_root: Path,
    output_dir: Path,
    *,
    seed: int = DEFAULT_SEED,
    variants_per_wine: int = 1,
    image_size: int = 512,
    jpeg_quality: int = 88,
    limit: int | None = None,
) -> dict[str, Any]:
    if variants_per_wine < 1:
        raise ValueError("variants_per_wine must be at least 1")
    if image_size < 128:
        raise ValueError("image_size must be at least 128")
    if not 20 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be between 20 and 100")

    cv2.setNumThreads(1)
    records = read_jsonl(manifest_path)
    if limit is not None:
        records = records[:limit]
    if not records:
        raise ValueError(f"Manifest has no records: {manifest_path}")

    images_dir = output_dir / "images"
    output_records: list[dict[str, Any]] = []
    profile_counts: Counter[str] = Counter()
    mask_counts: Counter[str] = Counter()
    total_bytes = 0

    for record_position, record in enumerate(records):
        slug = str(record["slug"])
        catalog_index = int(record.get("catalog_index", record_position))
        source_image = str(record["image_local"])
        source_path = source_root / source_image
        if not source_path.exists():
            raise FileNotFoundError(f"Source image is missing for {slug}: {source_path}")

        for variant_index in range(variants_per_wine):
            profile = PROFILES[(catalog_index + variant_index) % len(PROFILES)]
            sample_seed = stable_seed(seed, slug, variant_index)
            image, transformations, sample_quality, mask_source = render_sample(
                source_path,
                profile,
                image_size,
                sample_seed,
                jpeg_quality,
            )
            filename = safe_filename(catalog_index, slug, variant_index)
            output_path = images_dir / filename
            output_sha256 = encode_jpeg(output_path, image, sample_quality)
            image_bytes = output_path.stat().st_size
            total_bytes += image_bytes
            profile_counts[profile] += 1
            mask_counts[mask_source] += 1
            output_records.append(
                {
                    "schema_version": 1,
                    "sample_id": f"synthetic-{catalog_index:04d}-v{variant_index:02d}",
                    "split": "validation",
                    "profile": profile,
                    "expected_slug": slug,
                    "image": output_path.as_posix(),
                    "source_image": source_image,
                    "source_sha256": record.get("image", {}).get("sha256"),
                    "seed": sample_seed,
                    "variant_index": variant_index,
                    "width": image_size,
                    "height": image_size,
                    "jpeg_quality": sample_quality,
                    "bytes": image_bytes,
                    "sha256": output_sha256,
                    "transformations": transformations,
                }
            )

    report: dict[str, Any] = {
        "schema_version": 1,
        "config": {
            "manifest_path": manifest_path.as_posix(),
            "source_root": source_root.as_posix(),
            "output_dir": output_dir.as_posix(),
            "seed": seed,
            "variants_per_wine": variants_per_wine,
            "image_size": image_size,
            "default_jpeg_quality": jpeg_quality,
            "profiles": list(PROFILES),
            "limit": limit,
        },
        "summary": {
            "source_slug_count": len({record["slug"] for record in records}),
            "sample_count": len(output_records),
            "total_bytes": total_bytes,
            "split": "validation",
            "train_validation_source_overlap": 0,
        },
        "profiles": {profile: profile_counts.get(profile, 0) for profile in PROFILES},
        "foreground_masks": dict(sorted(mask_counts.items())),
    }
    write_jsonl(output_dir / "synthetic_manifest.jsonl", output_records)
    write_json(output_dir / "generation_report.json", report)
    (output_dir / "generation_report.md").write_text(
        report_markdown(report), encoding="utf-8", newline="\n"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a deterministic synthetic field-photo dataset.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--variants-per-wine", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--jpeg-quality", type=int, default=88)
    parser.add_argument("--limit", type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = generate_dataset(
        args.manifest,
        args.source_root,
        args.output_dir,
        seed=args.seed,
        variants_per_wine=args.variants_per_wine,
        image_size=args.image_size,
        jpeg_quality=args.jpeg_quality,
        limit=args.limit,
    )
    print(
        f"Generated {report['summary']['sample_count']} validation images from "
        f"{report['summary']['source_slug_count']} slugs; output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
