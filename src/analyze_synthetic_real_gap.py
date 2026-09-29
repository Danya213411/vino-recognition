from __future__ import annotations

import csv
import json
import math
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_MANIFEST = PROJECT_ROOT / "data/synthetic/stage2/synthetic_manifest.jsonl"
CATALOG_MANIFEST = PROJECT_ROOT / "data/artifacts/stage1/catalog_manifest.jsonl"
SELECTION_MANIFEST = PROJECT_ROOT / "data/test/recheck_fuzzy_v2/selection_manifest.jsonl"
DATABASE = PROJECT_ROOT / "data/app/stage6/events.sqlite3"
BASELINE_REPORT = (
    PROJECT_ROOT
    / "data/artifacts/stage3/baseline-small-quality/evaluation_cls_multiscale/evaluation_report.json"
)
BASELINE_PREDICTIONS = (
    PROJECT_ROOT / "data/artifacts/stage3/baseline-small-quality/predictions_cls_multiscale.jsonl"
)
TRANSFORM_SENSITIVITY = (
    PROJECT_ROOT / "data/artifacts/stage7/synthetic_real_gap/transform_sensitivity.json"
)
OUTPUT_DIR = PROJECT_ROOT / "data/artifacts/stage7/synthetic_real_gap"
PROFILE_ORDER = (
    "background",
    "perspective",
    "glare",
    "crop",
    "compression",
    "blur",
    "combined",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def decode_image(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def resize_long_side(image: np.ndarray, target: int = 512) -> np.ndarray:
    height, width = image.shape[:2]
    scale = target / max(height, width)
    return cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
    )


def crop_ratio(
    image: np.ndarray,
    left: float,
    top: float,
    right: float,
    bottom: float,
) -> np.ndarray:
    height, width = image.shape[:2]
    return image[
        round(height * top) : round(height * bottom),
        round(width * left) : round(width * right),
    ]


def entropy(gray: np.ndarray) -> float:
    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    probabilities = histogram[histogram > 0] / gray.size
    return float(-(probabilities * np.log2(probabilities)).sum())


def region_metrics(image: np.ndarray) -> dict[str, float]:
    normalized = resize_long_side(image)
    gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    hsv = cv2.cvtColor(normalized, cv2.COLOR_BGR2HSV)
    return {
        "laplacian_var": float(laplacian.var()),
        "tenengrad": float(np.mean(sobel_x * sobel_x + sobel_y * sobel_y)),
        "edge_density": float(np.mean(cv2.Canny(gray, 70, 140) > 0)),
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "saturation": float(hsv[:, :, 1].mean()),
        "entropy": entropy(gray),
    }


def image_metrics(path: Path) -> dict[str, float | int]:
    image = decode_image(path)
    height, width = image.shape[:2]
    full = region_metrics(image)
    medium = region_metrics(crop_ratio(image, 0.10, 0.10, 0.90, 0.99))
    detail = region_metrics(crop_ratio(image, 0.22, 0.30, 0.78, 0.91))
    result: dict[str, float | int] = {
        "width": width,
        "height": height,
        "megapixels": width * height / 1_000_000,
        "aspect_ratio": width / height,
        "file_kib": path.stat().st_size / 1024,
    }
    for prefix, values in (("full", full), ("medium", medium), ("detail", detail)):
        result.update({f"{prefix}_{key}": value for key, value in values.items()})
    return result


def quantile(values: Iterable[float], q: float) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    return float(np.quantile(array, q)) if array.size else math.nan


def describe(rows: list[dict[str, Any]], fields: Iterable[str]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for field in fields:
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        output[field] = {
            "mean": float(np.mean(values)),
            "p10": quantile(values, 0.10),
            "median": median(values),
            "p90": quantile(values, 0.90),
        }
    return output


def latest_feedback() -> dict[str, dict[str, Any]]:
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        """
        SELECT r.original_name, r.created_at, r.result_json,
               f.verdict, f.correct_slug
        FROM recognitions r
        JOIN feedback f ON f.recognition_id = r.id
        ORDER BY r.created_at DESC
        """
    ).fetchall()
    connection.close()
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = str(row["original_name"])
        if name in result:
            continue
        result[name] = {
            "created_at": row["created_at"],
            "verdict": row["verdict"],
            "correct_slug": row["correct_slug"],
            "result": json.loads(row["result_json"]),
        }
    return result


def real_rows() -> list[dict[str, Any]]:
    feedback = latest_feedback()
    rows: list[dict[str, Any]] = []
    for record in read_jsonl(SELECTION_MANIFEST):
        if record["reason"] == "ambiguous_not_in_store":
            continue
        name = record["file"]
        if record["reason"] == "stable_confirmed_top1":
            outcome = "top1"
        else:
            verdict = feedback.get(name, {}).get("verdict")
            outcome = {
                "correct": "top1",
                "incorrect": "top5",
                "not_in_catalog": "miss",
            }.get(verdict, "miss")
        source = PROJECT_ROOT / record["source"]
        metrics = image_metrics(source)
        result_payload = feedback.get(name, {}).get("result", {})
        predictions = result_payload.get("predictions") or []
        top_prediction = predictions[0] if predictions else {}
        rows.append(
            {
                "dataset": "real",
                "group": outcome,
                "name": name,
                "path": source.relative_to(PROJECT_ROOT).as_posix(),
                "embedding_score": top_prediction.get("embedding_score"),
                "sift_score": top_prediction.get("sift_score"),
                "ocr_score": top_prediction.get("ocr_score"),
                **metrics,
            }
        )
    counts = Counter(row["group"] for row in rows)
    expected = Counter({"top1": 34, "top5": 18, "miss": 14})
    if len(rows) != 66 or counts != expected:
        raise ValueError(f"Unexpected real outcome counts: rows={len(rows)}, counts={counts}")
    return rows


def synthetic_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    transform_parameters: dict[str, list[float]] = defaultdict(list)
    for record in read_jsonl(SYNTHETIC_MANIFEST):
        path = PROJECT_ROOT / record["image"]
        transforms = record["transformations"]
        geometry = transforms["geometry"]
        qa = geometry["qa"]
        lowres = transforms.get("low_resolution") or {}
        blur = transforms.get("blur") or {"kind": "none"}
        scale = float(lowres.get("scale", 1.0))
        bbox_width, bbox_height = qa["visible_bbox_ratio"]
        metrics = image_metrics(path)
        row = {
            "dataset": "synthetic",
            "group": record["profile"],
            "name": record["sample_id"],
            "path": record["image"],
            "jpeg_quality": record["jpeg_quality"],
            "blur_kind": blur["kind"],
            "blur_sigma": blur.get("sigma"),
            "motion_length": blur.get("length"),
            "lowres_scale": scale,
            "effective_bottle_width_px": float(bbox_width) * 512 * scale,
            "effective_bottle_height_px": float(bbox_height) * 512 * scale,
            "visible_area_fraction": qa["visible_area_fraction"],
            "yaw_degrees": abs(float(geometry["yaw_degrees"])),
            "pitch_degrees": abs(float(geometry["pitch_degrees"])),
            "roll_degrees": abs(float(geometry["roll_degrees"])),
            **metrics,
        }
        rows.append(row)
        if blur.get("sigma") is not None:
            transform_parameters[f"{record['profile']}.gaussian_sigma"].append(float(blur["sigma"]))
        if blur.get("length") is not None:
            transform_parameters[f"{record['profile']}.motion_length"].append(float(blur["length"]))
        if lowres:
            transform_parameters[f"{record['profile']}.lowres_scale"].append(scale)
        transform_parameters[f"{record['profile']}.effective_bottle_width_px"].append(
            float(row["effective_bottle_width_px"])
        )
    parameter_summary = {
        name: {
            "count": len(values),
            "min": min(values),
            "median": median(values),
            "max": max(values),
        }
        for name, values in sorted(transform_parameters.items())
    }
    return rows, parameter_summary


def source_catalog_summary() -> dict[str, Any]:
    widths: list[float] = []
    heights: list[float] = []
    bbox_widths: list[float] = []
    bbox_heights: list[float] = []
    for record in read_jsonl(CATALOG_MANIFEST):
        image_meta = record["image"]
        widths.append(float(image_meta["width"]))
        heights.append(float(image_meta["height"]))
        path = PROJECT_ROOT / "data/parser" / record["image_local"]
        encoded = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
        if image is None:
            continue
        if image.ndim == 3 and image.shape[2] == 4:
            points = cv2.findNonZero((image[:, :, 3] > 12).astype(np.uint8))
            if points is not None:
                _, _, width, height = cv2.boundingRect(points)
                bbox_widths.append(float(width))
                bbox_heights.append(float(height))
    return {
        "count": len(widths),
        "alpha_bbox_count": len(bbox_widths),
        "width": {"p10": quantile(widths, 0.1), "median": median(widths), "p90": quantile(widths, 0.9)},
        "height": {"p10": quantile(heights, 0.1), "median": median(heights), "p90": quantile(heights, 0.9)},
        "alpha_bbox_width": {
            "p10": quantile(bbox_widths, 0.1),
            "median": median(bbox_widths),
            "p90": quantile(bbox_widths, 0.9),
        },
        "alpha_bbox_height": {
            "p10": quantile(bbox_heights, 0.1),
            "median": median(bbox_heights),
            "p90": quantile(bbox_heights, 0.9),
        },
    }


def synthetic_transform_effects() -> dict[str, Any]:
    manifest = {row["sample_id"]: row for row in read_jsonl(SYNTHETIC_MANIFEST)}
    predictions = read_jsonl(BASELINE_PREDICTIONS)
    groups: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    profile_scales: dict[str, list[float]] = defaultdict(list)
    for record in manifest.values():
        lowres = record["transformations"].get("low_resolution")
        if lowres:
            profile_scales[record["profile"]].append(float(lowres["scale"]))
    scale_medians = {profile: median(values) for profile, values in profile_scales.items()}
    for prediction in predictions:
        record = manifest[prediction["sample_id"]]
        profile = record["profile"]
        predicted = [item["slug"] for item in prediction["predictions"]]
        expected = prediction["expected_slug"]
        result = (predicted[0] == expected, expected in predicted[:5])
        transforms = record["transformations"]
        blur_kind = transforms.get("blur", {}).get("kind", "none")
        if profile in {"blur", "combined"}:
            groups[f"{profile}.{blur_kind}"].append(result)
        if profile in {"compression", "combined"}:
            scale = float(transforms["low_resolution"]["scale"])
            band = "lower_resolution" if scale < scale_medians[profile] else "higher_resolution"
            groups[f"{profile}.{band}"].append(result)
        if profile == "combined":
            groups[f"combined.occlusion_{'yes' if 'occlusion' in transforms else 'no'}"].append(result)
            groups[f"combined.glare_{'yes' if 'glare' in transforms else 'no'}"].append(result)
    return {
        name: {
            "count": len(values),
            "top1": sum(top1 for top1, _ in values) / len(values),
            "top5": sum(top5 for _, top5 in values) / len(values),
        }
        for name, values in sorted(groups.items())
    }


def accuracy_by_sharpness_tercile(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: row["detail_laplacian_var"])
    buckets = np.array_split(np.asarray(ordered, dtype=object), 3)
    output = []
    for label, bucket in zip(("low", "middle", "high"), buckets, strict=True):
        items = list(bucket)
        top1 = sum(row["group"] == "top1" for row in items)
        top5 = sum(row["group"] in {"top1", "top5"} for row in items)
        output.append(
            {
                "tercile": label,
                "count": len(items),
                "sharpness_min": min(row["detail_laplacian_var"] for row in items),
                "sharpness_max": max(row["detail_laplacian_var"] for row in items),
                "top1": top1 / len(items),
                "top5": top5 / len(items),
            }
        )
    return output


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def svg_chart(
    synthetic_summary: dict[str, Any],
    real_summary: dict[str, Any],
    path: Path,
) -> None:
    groups = list(PROFILE_ORDER) + ["real"]
    medians = [synthetic_summary[group]["detail_laplacian_var"]["median"] for group in PROFILE_ORDER]
    medians.append(real_summary["all"]["detail_laplacian_var"]["median"])
    maximum = max(medians) * 1.12
    width, height = 1100, 510
    margin_left, margin_right, margin_top, margin_bottom = 85, 25, 55, 100
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    bar_width = plot_width / len(groups) * 0.62
    colors = ["#9f3047"] * len(PROFILE_ORDER) + ["#173f35"]
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbf7ed"/>',
        '<text x="55" y="30" font-family="Arial" font-size="20" fill="#302b28">Медианная резкость detail-crop (Laplacian variance, 512 px)</text>',
    ]
    for tick in range(6):
        value = maximum * tick / 5
        y = margin_top + plot_height - plot_height * tick / 5
        elements.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width-margin_right}" y2="{y:.1f}" stroke="#d9d1c3"/>')
        elements.append(f'<text x="{margin_left-10}" y="{y+5:.1f}" text-anchor="end" font-family="Arial" font-size="13" fill="#615a54">{value:.0f}</text>')
    step = plot_width / len(groups)
    for index, (group, value, color) in enumerate(zip(groups, medians, colors, strict=True)):
        x = margin_left + step * index + (step - bar_width) / 2
        bar_height = plot_height * value / maximum
        y = margin_top + plot_height - bar_height
        elements.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" rx="5" fill="{color}"/>')
        elements.append(f'<text x="{x+bar_width/2:.1f}" y="{y-8:.1f}" text-anchor="middle" font-family="Arial" font-size="13" fill="#302b28">{value:.0f}</text>')
        elements.append(f'<text x="{x+bar_width/2:.1f}" y="{height-margin_bottom+24}" text-anchor="middle" font-family="Arial" font-size="13" fill="#302b28" transform="rotate(-25 {x+bar_width/2:.1f} {height-margin_bottom+24})">{group}</text>')
    elements.append('</svg>')
    path.write_text("\n".join(elements), encoding="utf-8")


def fmt(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def report_markdown(summary: dict[str, Any]) -> str:
    synthetic = summary["synthetic"]
    real = summary["real"]
    baseline = summary["synthetic_baseline"]
    profile_rows = []
    for profile in PROFILE_ORDER:
        values = synthetic[profile]
        scores = baseline[profile]
        profile_rows.append(
            "| "
            + " | ".join(
                (
                    f"`{profile}`",
                    str(values["count"]),
                    fmt(values["detail_laplacian_var"]["median"]),
                    fmt(values["effective_bottle_width_px"]["median"]),
                    fmt(scores["top1"] * 100) + "%",
                    fmt(scores["top5"] * 100) + "%",
                )
            )
            + " |"
        )
    real_rows = []
    for group, title in (("all", "Все реальные"), ("top1", "Top-1"), ("top5", "Только Top-2–5"), ("miss", "Промах Top-5")):
        values = real[group]
        real_rows.append(
            f"| {title} | {values['count']} | {fmt(values['megapixels']['median'], 2)} | "
            f"{fmt(values['detail_laplacian_var']['median'])} | {fmt(values['detail_contrast']['median'])} | "
            f"{fmt(values['detail_entropy']['median'], 2)} |"
        )
    tercile_rows = []
    for item in summary["real_sharpness_terciles"]:
        tercile_rows.append(
            f"| `{item['tercile']}` | {item['count']} | {fmt(item['sharpness_min'])}–{fmt(item['sharpness_max'])} | "
            f"{fmt(item['top1'] * 100)}% | {fmt(item['top5'] * 100)}% |"
        )
    source = summary["source_catalog"]
    transforms = summary["transform_parameters"]
    blur_sigma = transforms["blur.gaussian_sigma"]
    blur_motion = transforms["blur.motion_length"]
    combined_scale = transforms["combined.lowres_scale"]
    compression_scale = transforms["compression.lowres_scale"]
    real_median = real["all"]["detail_laplacian_var"]["median"]
    blur_median = synthetic["blur"]["detail_laplacian_var"]["median"]
    combined_median = synthetic["combined"]["detail_laplacian_var"]["median"]
    compression_median = synthetic["compression"]["detail_laplacian_var"]["median"]
    effects = summary["synthetic_transform_effects"]
    sensitivity = summary["real_transform_sensitivity"]["results"]
    ratio_blur = real_median / blur_median
    ratio_combined = real_median / combined_median
    sensitivity_rows = []
    sensitivity_titles = {
        "original": "Без дополнительного искажения",
        "gaussian_sigma_1_7": "Gaussian sigma 1,7",
        "motion_9px": "Motion blur 9 px",
        "compression_median": "Downscale 0,526 + JPEG 44",
        "blur_plus_compression": "Motion 9 + downscale 0,618 + JPEG 55",
    }
    original = sensitivity["original"]
    for name, title in sensitivity_titles.items():
        item = sensitivity[name]
        sensitivity_rows.append(
            f"| {title} | {item['top1_correct']}/52 ({fmt(item['top1_accuracy'] * 100)}%) | "
            f"{item['top5_correct']}/52 ({fmt(item['top5_recall'] * 100)}%) | "
            f"{item['top1_correct'] - original['top1_correct']:+d} | "
            f"{item['top5_correct'] - original['top5_correct']:+d} |"
        )
    return f"""# Синтетика Stage 2 против 66 полевых фотографий

Дата анализа: 21 сентября 2026 года.

## Короткий вывод

**Blur не испортил веса DINOv2:** модель не дообучалась на синтетике. В индекс
помещены чистые изображения каталога из `data/parser`, а все 2025 синтетических
кадров имеют split `validation`. Поэтому синтетика могла повлиять только на
выбор варианта модели/предобработки, веса гибридного ранжирования и калибровку
confidence.

При этом подозрение насчёт слишком жёстких и не вполне реалистичных искажений
верное. Медианная резкость центрального detail-crop реальных кадров равна
**{fmt(real_median)}**, профиля `blur` — **{fmt(blur_median)}** ({fmt(ratio_blur, 2)}× ниже
реальных), а `combined` — **{fmt(combined_median)}** ({fmt(ratio_combined, 2)}× ниже
реальных). `combined` одновременно складывает blur, уменьшение, JPEG,
перспективу и иногда glare/occlusion, поэтому его следует считать стресс-тестом,
а не представителем обычной фотографии из магазина.

![Сравнение медианной резкости](sharpness_comparison.svg)

## Что именно генерировалось

- исходные карточки сами невелики: медиана — **{fmt(source['width']['median'], 0)}×{fmt(source['height']['median'], 0)} px**;
- медиана alpha-bbox бутылки — **{fmt(source['alpha_bbox_width']['median'], 0)}×{fmt(source['alpha_bbox_height']['median'], 0)} px**;
- в синтетическом кадре бутылка имеет эффективную ширину примерно
  **{fmt(synthetic['background']['effective_bottle_width_px']['median'])} px** в обычных профилях;
- `compression` дополнительно уменьшает весь кадр до
  **{fmt(compression_scale['min'] * 100, 0)}–{fmt(compression_scale['max'] * 100, 0)}%**,
  а `combined` — до **{fmt(combined_scale['min'] * 100, 0)}–{fmt(combined_scale['max'] * 100, 0)}%**;
- Gaussian blur использует sigma **{fmt(blur_sigma['min'], 2)}–{fmt(blur_sigma['max'], 2)}**,
  motion blur — ядро **{fmt(blur_motion['min'], 0)}–{fmt(blur_motion['max'], 0)} px**;
- после этого `combined` сохраняется с JPEG quality 42–67.

То есть при медианной ширине бутылки около 100 px режим `combined` временно
сжимает её примерно до 50–80 px. Этикетка занимает лишь часть этой ширины —
для мелких букв это уже физическая потеря информации, которую последующий
upscale не возвращает.

## Синтетические профили и качество DINOv2-small

| Профиль | N | Резкость detail, median | Эффективная ширина бутылки, px | Top-1 | Top-5 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(profile_rows)}

Падение качества согласуется с разрушением деталей: `compression` хуже чистого
background на {fmt((baseline['background']['top1'] - baseline['compression']['top1']) * 100)} п.п.
Top-1, `blur` — на {fmt((baseline['background']['top1'] - baseline['blur']['top1']) * 100)} п.п.,
а `combined` — на {fmt((baseline['background']['top1'] - baseline['combined']['top1']) * 100)} п.п.
Это показывает чувствительность текущего visual retrieval к потере деталей, но
не доказывает, что эти преобразования ухудшили саму модель: её веса неизменны.

Внутри самого профиля `blur` motion blur действительно вреднее Gaussian:

| Подтип синтетики | N | Top-1 | Top-5 |
|---|---:|---:|---:|
| Gaussian blur | {effects['blur.gaussian']['count']} | {fmt(effects['blur.gaussian']['top1'] * 100)}% | {fmt(effects['blur.gaussian']['top5'] * 100)}% |
| Motion blur | {effects['blur.motion']['count']} | {fmt(effects['blur.motion']['top1'] * 100)}% | {fmt(effects['blur.motion']['top5'] * 100)}% |
| Combined + Gaussian | {effects['combined.gaussian']['count']} | {fmt(effects['combined.gaussian']['top1'] * 100)}% | {fmt(effects['combined.gaussian']['top5'] * 100)}% |
| Combined + motion | {effects['combined.motion']['count']} | {fmt(effects['combined.motion']['top1'] * 100)}% | {fmt(effects['combined.motion']['top5'] * 100)}% |

## Реальные 66 кадров

Метрики считаются после приведения длинной стороны к 512 px. `detail` — тот же
центральный crop 22–78% по ширине и 30–91% по высоте, который используется в
multiscale DINO-запросе. Variance of Laplacian — относительная мера резкости:
сравнивать её следует только при одинаковом размере предобработки.

| Группа | N | MP, median | Резкость detail, median | Контраст | Энтропия |
|---|---:|---:|---:|---:|---:|
{chr(10).join(real_rows)}

### Связь резкости с результатом

| Треть выборки по резкости | N | Диапазон | Top-1 | Top-5 |
|---|---:|---:|---:|---:|
{chr(10).join(tercile_rows)}

Резкость сама по себе не объясняет основную часть ошибок. В промахах Top-5
есть отсутствующие в локальном индексе классы Литавщука: ни DINO, ни любая
аугментация не могут выбрать класс, которого нет среди кандидатов. А большая
часть ошибок Top-1 происходит внутри одного производителя между похожими
этикетками — здесь важнее OCR-reranker по названию/сорту/году и более свежие
референсы.

## Контрольный эксперимент на одних и тех же реальных фото

Чтобы не путать различие содержимого двух датасетов с действием фильтра, отдельно
взяты 52 реальных кадра с известным точным slug. К каждому применено одно
искажение после уменьшения длинной стороны до 512 px, затем выполнен чистый
DINOv2-small CLS multiscale retrieval по тому же индексу. OCR, SIFT и
manufacturer gate здесь намеренно выключены.

| Вариант реального кадра | DINO Top-1 | DINO Top-5 | Δ Top-1 | Δ Top-5 |
|---|---:|---:|---:|---:|
{chr(10).join(sensitivity_rows)}

Главный вредитель именно **motion blur**: он убрал 6 правильных Top-1 net
(14 → 8). Gaussian sigma 1,7 изменил Top-1 всего на −1. Downscale+JPEG дал −3.
При этом Top-5 у искажённых вариантов неожиданно вырос: слабое размытие иногда
подавляет фон магазина и приближает фото к гладким каталожным референсам. Это
не делает blur полезной аугментацией для обучения — результат показывает, что
чистый DINO нестабилен и при небольшом фильтре просто переставляет соседей.

Абсолютный DINO-only результат на реальных фото низкий: 14/52 Top-1 и 20/52
Top-5 до искажений. Текущие 34/52 и 52/52 на кадрах с известным slug получаются
уже за счёт OCR, SIFT и manufacturer gate. Следовательно, исправлять только
синтетический blur недостаточно.

## Что в текущей синтетике полезно, а что стоит изменить

1. **Не удалять blur полностью.** Он нужен как отдельный robustness-тест и
   действительно выявляет слабость visual retrieval.
2. **Разделить validation и stress.** `background`, mild perspective/crop/glare
   оставить в реалистичной validation; нынешние `blur` и особенно `combined`
   вынести в отдельный stress split, который не определяет выбор confidence
   threshold.
3. **Подогнать параметры по полевым кадрам.** Для основной validation брать
   эмпирические диапазоны резкости/контраста/JPEG из этих 66 кадров; тяжёлый
   хвост оставить отдельно. Не складывать все тяжёлые искажения в каждом
   `combined`-кадре.
4. **Если начнём дообучение**, генерировать несколько реалистичных вариантов на
   продукт и держать split по slug/продукту. Сейчас один синтетический кадр на
   slug — это слишком мало для обучения и одновременно слишком случайно для
   надёжной оценки каждого класса.
5. **Калибровать confidence на реальных фото.** Именно здесь синтетический
   domain gap уже доказан: confidence-модель обучалась преимущественно на
   синтетических known/unknown кадрах и почти все полевые фото объявляет
   `not_found`, хотя правильный вариант часто находится в Top-5.

## Ограничения анализа

- 66 кадров — небольшая и несбалансированная выборка по производителям;
- резкость измеряет весь центральный crop, а не только пиксели этикетки: без
  детектора бутылки фон тоже влияет на показатель;
- группа `miss` смешивает визуальные ошибки и отсутствующие классы, поэтому её
  нельзя использовать как чистую оценку влияния blur;
- сравнение файлового размера JPEG и WebP не использовалось как мера качества,
  поскольку кодеки разные.

Машиночитаемые результаты находятся рядом: `summary.json` и
`image_metrics.csv`.
"""


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    synthetic, transform_parameters = synthetic_rows()
    real = real_rows()
    metric_fields = (
        "megapixels",
        "full_laplacian_var",
        "medium_laplacian_var",
        "detail_laplacian_var",
        "detail_tenengrad",
        "detail_edge_density",
        "detail_contrast",
        "detail_entropy",
        "effective_bottle_width_px",
    )
    synthetic_groups: dict[str, Any] = {}
    for profile in PROFILE_ORDER:
        group = [row for row in synthetic if row["group"] == profile]
        synthetic_groups[profile] = {"count": len(group), **describe(group, metric_fields)}
    real_groups: dict[str, Any] = {}
    for group_name in ("all", "top1", "top5", "miss"):
        group = real if group_name == "all" else [row for row in real if row["group"] == group_name]
        real_groups[group_name] = {
            "count": len(group),
            **describe(
                group,
                (field for field in metric_fields if field != "effective_bottle_width_px"),
            ),
        }
    baseline_payload = json.loads(BASELINE_REPORT.read_text(encoding="utf-8"))
    baseline = {
        row["value"]: {"top1": row["top1_accuracy"], "top5": row["top5_recall"]}
        for row in baseline_payload["breakdowns"]["augmentation_profile"]
    }
    summary = {
        "schema_version": 1,
        "method": {
            "normalization": "long side resized to 512 px; aspect ratio preserved",
            "detail_crop": "x=22..78%, y=30..91%, matching stage3 query_views",
            "sharpness": "variance of 3x3 Laplacian on grayscale",
        },
        "source_catalog": source_catalog_summary(),
        "transform_parameters": transform_parameters,
        "synthetic_transform_effects": synthetic_transform_effects(),
        "synthetic": synthetic_groups,
        "synthetic_baseline": baseline,
        "real": real_groups,
        "real_sharpness_terciles": accuracy_by_sharpness_tercile(real),
        "real_transform_sensitivity": json.loads(
            TRANSFORM_SENSITIVITY.read_text(encoding="utf-8")
        ),
    }
    write_csv(synthetic + real, OUTPUT_DIR / "image_metrics.csv")
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    svg_chart(synthetic_groups, real_groups, OUTPUT_DIR / "sharpness_comparison.svg")
    (OUTPUT_DIR / "analysis.md").write_text(report_markdown(summary), encoding="utf-8")
    print(f"synthetic={len(synthetic)} real={len(real)} output={OUTPUT_DIR}")


if __name__ == "__main__":
    main()
