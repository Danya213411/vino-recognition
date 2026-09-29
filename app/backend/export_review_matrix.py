"""Export the latest non-Top-1 review cases as one lossless PNG matrix.

The local API must be running before executing this script.
"""

from __future__ import annotations

import argparse
import math
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from PIL import Image, ImageDraw, ImageFont, ImageOps


BASE_URL = "http://127.0.0.1:8000"
CANVAS_WIDTH = 5200
MARGIN = 48
SOURCE_WIDTH = 670
CANDIDATE_WIDTH = 650
OCR_X = MARGIN + SOURCE_WIDTH + 5 * CANDIDATE_WIDTH

BG = "#fbfaf6"
PAPER = "#fffefa"
CREAM = "#f7f1e7"
INK = "#292420"
MUTED = "#756e67"
BURGUNDY = "#9d3949"
BURGUNDY_DARK = "#6d2632"
GREEN = "#5d7b48"
GREEN_BG = "#edf3e6"
AMBER = "#a57324"
AMBER_BG = "#fff4d9"
RED_BG = "#fce8e5"
LINE = "#ded7cc"


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(Path("C:/Windows/Fonts") / name), size=size)


FONTS = {
    "title": font("georgia.ttf", 92),
    "subtitle": font("segoeui.ttf", 30),
    "eyebrow": font("segoeuib.ttf", 25),
    "stat": font("georgia.ttf", 56),
    "stat_label": font("segoeui.ttf", 25),
    "head": font("segoeuib.ttf", 25),
    "rank": font("georgia.ttf", 44),
    "card_title": font("segoeuib.ttf", 27),
    "body": font("segoeui.ttf", 24),
    "body_bold": font("segoeuib.ttf", 24),
    "small": font("segoeui.ttf", 21),
    "small_bold": font("segoeuib.ttf", 21),
    "micro": font("segoeui.ttf", 19),
    "mono": font("consola.ttf", 20),
    "mono_bold": font("consolab.ttf", 20),
}


def text_width(draw: ImageDraw.ImageDraw, value: str, used_font: ImageFont.FreeTypeFont) -> int:
    box = draw.textbbox((0, 0), value, font=used_font)
    return box[2] - box[0]


def wrap(draw: ImageDraw.ImageDraw, value: Any, used_font: ImageFont.FreeTypeFont, width: int) -> list[str]:
    text = str(value or "—").replace("\n", " ").strip() or "—"
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if text_width(draw, candidate, used_font) <= width:
            current = candidate
        elif current:
            lines.append(current)
            current = word
        else:
            fragment = ""
            for char in word:
                if text_width(draw, fragment + char, used_font) <= width:
                    fragment += char
                else:
                    lines.append(fragment)
                    fragment = char
            current = fragment
    if current:
        lines.append(current)
    return lines or ["—"]


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    value: Any,
    used_font: ImageFont.FreeTypeFont,
    fill: str,
    width: int,
    *,
    max_lines: int | None = None,
    gap: int = 6,
) -> int:
    lines = wrap(draw, value, used_font, width)
    if max_lines and len(lines) > max_lines:
        lines = lines[:max_lines]
        last = lines[-1]
        while last and text_width(draw, last + "…", used_font) > width:
            last = last[:-1]
        lines[-1] = last.rstrip() + "…"
    x, y = xy
    line_height = used_font.size + gap
    for line in lines:
        draw.text((x, y), line, font=used_font, fill=fill)
        y += line_height
    return y


def pct(value: Any) -> str:
    return "—" if value is None else f"{round(float(value) * 100)}%"


def number(value: Any) -> str:
    return "—" if value is None else f"{float(value):.3f}"


def absolute_url(value: str) -> str:
    return value if value.startswith(("http://", "https://")) else BASE_URL + value


SESSION = requests.Session()
SESSION.trust_env = False


@lru_cache(maxsize=256)
def load_remote_image(url: str) -> Image.Image:
    local = Path(url)
    if local.is_file():
        image = Image.open(local)
    else:
        response = SESSION.get(absolute_url(url), timeout=30)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
    image = ImageOps.exif_transpose(image)
    if image.mode not in {"RGB", "RGBA"}:
        image = image.convert("RGBA")
    return image.copy()


def paste_contained(canvas: Image.Image, source: Image.Image, box: tuple[int, int, int, int], background: str) -> None:
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    tile = Image.new("RGB", (width, height), background)
    rgba = source.convert("RGBA")
    rgba.thumbnail((width - 20, height - 20), Image.Resampling.LANCZOS)
    x = (width - rgba.width) // 2
    y = (height - rgba.height) // 2
    tile.paste(rgba, (x, y), rgba)
    canvas.paste(tile, (left, top))


def correct_slug(item: dict[str, Any]) -> str | None:
    feedback = item["recognition"].get("feedback") or {}
    return feedback.get("correct_slug")


def correct_rank(item: dict[str, Any]) -> int | None:
    expected = correct_slug(item)
    if not expected:
        return None
    predictions = item["recognition"]["result"]["predictions"]
    return next((index + 1 for index, candidate in enumerate(predictions) if candidate["slug"] == expected), None)


def row_height(item: dict[str, Any]) -> int:
    line_count = len(item["recognition"]["result"].get("ocr_lines") or [])
    ocr_rows = math.ceil(line_count / 2)
    return max(1080, 670 + ocr_rows * 62)


def rounded(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], radius: int, fill: str, outline: str | None = None, width: int = 1) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def draw_metric_grid(draw: ImageDraw.ImageDraw, x: int, y: int, width: int, candidate: dict[str, Any]) -> None:
    values = [
        ("Итог", number(candidate.get("score"))),
        ("DINO", pct(candidate.get("embedding_score"))),
        ("SIFT", pct(candidate.get("sift_score"))),
        ("OCR", pct(candidate.get("ocr_score"))),
    ]
    if "rapid_ocr_score" in candidate or "glm_field_score" in candidate:
        values.extend(
            [
                ("Rapid", pct(candidate.get("rapid_ocr_score"))),
                ("GLM fields", pct(candidate.get("glm_field_score"))),
            ]
        )
    values.append(("Inliers", str(candidate.get("sift_inliers", "—"))))
    col_width = width // 2
    for index, (label, value) in enumerate(values):
        col = index % 2
        row = index // 2
        px = x + col * col_width
        py = y + row * 44
        draw.text((px, py), label, font=FONTS["micro"], fill=MUTED)
        value_width = text_width(draw, value, FONTS["mono_bold"])
        draw.text((px + col_width - value_width - 8, py), value, font=FONTS["mono_bold"], fill=INK)


def draw_source(canvas: Image.Image, draw: ImageDraw.ImageDraw, item: dict[str, Any], top: int, height: int) -> None:
    x = MARGIN + 18
    width = SOURCE_WIDTH - 36
    analysis = item["recognition"]["result"]
    rank = correct_rank(item)
    verdict = item["recognition"].get("feedback", {}).get("verdict")
    rounded(draw, (x, top + 18, x + width, top + height - 18), 28, CREAM)
    image_box = (x + 22, top + 42, x + width - 22, top + 590)
    rounded(draw, image_box, 22, PAPER)
    try:
        paste_contained(canvas, load_remote_image(item["input_image_url"]), image_box, PAPER)
    except Exception:
        draw.text((image_box[0] + 28, image_box[1] + 28), "Фото не загрузилось", font=FONTS["body_bold"], fill=BURGUNDY)
    cursor = draw_wrapped(draw, (x + 24, top + 612), item["name"], FONTS["small_bold"], INK, width - 48, max_lines=2)
    label = f"ПРАВИЛЬНЫЙ #{rank}" if rank else "НЕТ В TOP‑5"
    label_fill = AMBER_BG if rank else RED_BG
    label_text = AMBER if rank else BURGUNDY
    label_width = text_width(draw, label, FONTS["small_bold"]) + 34
    rounded(draw, (x + 24, cursor + 12, x + 24 + label_width, cursor + 54), 21, label_fill)
    draw.text((x + 41, cursor + 19), label, font=FONTS["small_bold"], fill=label_text)
    meta_top = cursor + 82
    meta = [
        ("Confidence", pct(analysis.get("confidence"))),
        ("Время", f"{item['recognition']['total_ms'] / 1000:.2f} с"),
        ("Решение", analysis.get("decision", "—")),
        ("Разметка", verdict or "—"),
    ]
    for label, value in meta:
        draw.text((x + 26, meta_top), label, font=FONTS["small"], fill=MUTED)
        value_width = text_width(draw, str(value), FONTS["small_bold"])
        draw.text((x + width - value_width - 26, meta_top), str(value), font=FONTS["small_bold"], fill=INK)
        meta_top += 42


def draw_candidate(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    candidate: dict[str, Any] | None,
    rank: int,
    expected: str | None,
    x: int,
    top: int,
    height: int,
) -> None:
    inset = 14
    left, right = x + inset, x + CANDIDATE_WIDTH - inset
    correct = bool(candidate and candidate["slug"] == expected)
    fill = GREEN_BG if correct else PAPER
    outline = GREEN if correct else LINE
    rounded(draw, (left, top + 18, right, top + height - 18), 28, fill, outline, 3 if correct else 1)
    draw.text((left + 22, top + 34), f"#{rank}", font=FONTS["rank"], fill=INK)
    if correct:
        badge = "ПРАВИЛЬНЫЙ"
        badge_width = text_width(draw, badge, FONTS["small_bold"]) + 30
        rounded(draw, (right - badge_width - 18, top + 35, right - 18, top + 75), 20, GREEN)
        draw.text((right - badge_width - 3, top + 41), badge, font=FONTS["small_bold"], fill="white")
    if not candidate:
        draw.text((left + 24, top + 130), "Нет кандидата", font=FONTS["body_bold"], fill=MUTED)
        return
    wine = candidate["wine"]
    image_box = (left + 22, top + 96, right - 22, top + 526)
    rounded(draw, image_box, 22, "#ffffff")
    try:
        paste_contained(canvas, load_remote_image(wine["image_url"]), image_box, "#ffffff")
    except Exception:
        draw.text((image_box[0] + 24, image_box[1] + 24), "Нет фото", font=FONTS["body_bold"], fill=BURGUNDY)
    cursor = draw_wrapped(draw, (left + 24, top + 550), wine.get("title"), FONTS["card_title"], INK, right - left - 48, max_lines=3, gap=7)
    cursor += 8
    cursor = draw_wrapped(draw, (left + 24, cursor), wine.get("manufacturer") or "Производитель не указан", FONTS["small"], BURGUNDY_DARK, right - left - 48, max_lines=2)
    cursor = draw_wrapped(draw, (left + 24, cursor + 5), wine.get("category") or wine.get("region") or candidate["slug"], FONTS["small"], MUTED, right - left - 48, max_lines=2)
    metric_top = max(top + 820, cursor + 24)
    draw.line((left + 24, metric_top - 18, right - 24, metric_top - 18), fill=LINE, width=2)
    draw_metric_grid(draw, left + 24, metric_top, right - left - 48, candidate)


def draw_ocr(draw: ImageDraw.ImageDraw, item: dict[str, Any], top: int, height: int) -> None:
    x = OCR_X + 14
    right = CANVAS_WIDTH - MARGIN - 14
    width = right - x
    analysis = item["recognition"]["result"]
    rounded(draw, (x, top + 18, right, top + height - 18), 28, "#fbf7ef", LINE)
    draw.text((x + 26, top + 38), f"OCR · {len(analysis.get('ocr_lines') or [])} строк", font=FONTS["body_bold"], fill=BURGUNDY_DARK)
    lines = analysis.get("ocr_lines") or []
    column_gap = 24
    column_width = (width - 52 - column_gap) // 2
    rows = math.ceil(len(lines) / 2)
    lines_top = top + 86
    for index, line in enumerate(lines):
        col = index // rows if rows else 0
        row = index % rows if rows else 0
        px = x + 26 + col * (column_width + column_gap)
        py = lines_top + row * 62
        raw = str(line.get("text") or "—")
        normalized = str(line.get("normalized") or "—")
        source = str(line.get("source") or "ocr")
        value = f"[{source}] {raw} → {normalized}"
        draw_wrapped(draw, (px, py), value, FONTS["small_bold"], INK, column_width - 85, max_lines=1)
        confidence = pct(line.get("score"))
        conf_width = text_width(draw, confidence, FONTS["mono"])
        draw.text((px + column_width - conf_width, py), confidence, font=FONTS["mono"], fill=GREEN if (line.get("score") or 0) >= 0.8 else AMBER)
        draw.line((px, py + 45, px + column_width, py + 45), fill="#e8e0d4", width=1)
    cursor = lines_top + rows * 62 + 22
    draw.text((x + 26, cursor), "MANUFACTURER GATE", font=FONTS["head"], fill=BURGUNDY_DARK)
    cursor += 46
    manufacturer = analysis.get("manufacturer_match") or {}
    fields = [
        ("Значение", manufacturer.get("value") or "не найдено"),
        ("Режим", manufacturer.get("mode") or "—"),
        ("Текст", manufacturer.get("matched_text") or "—"),
        ("Score", number(manufacturer.get("score"))),
        ("Margin", number(manufacturer.get("margin"))),
    ]
    for label, value in fields:
        draw.text((x + 28, cursor), label, font=FONTS["small"], fill=MUTED)
        draw_wrapped(draw, (x + 240, cursor), value, FONTS["small_bold"], INK, width - 292, max_lines=1)
        cursor += 42
    cursor += 16
    draw.text((x + 26, cursor), "EVIDENCE-ПОЛЯ", font=FONTS["head"], fill=BURGUNDY_DARK)
    cursor += 48
    evidence = analysis.get("evidence") or {}
    selected = [(key, value) for key, value in evidence.items() if any(token in key for token in ("ocr", "title", "manufacturer", "grape", "year"))]
    for index, (key, value) in enumerate(selected):
        col = index % 2
        row = index // 2
        px = x + 28 + col * ((width - 56) // 2)
        py = cursor + row * 42
        draw.text((px, py), key, font=FONTS["micro"], fill=MUTED)
        rendered = number(value)
        draw.text((px + (width - 56) // 2 - text_width(draw, rendered, FONTS["mono_bold"]) - 12, py), rendered, font=FONTS["mono_bold"], fill=INK)


def export(
    output: Path,
    items: list[dict[str, Any]] | None = None,
    *,
    title: str = "Матрица ошибок Top-1",
    subtitle: str = (
        "Только кадры, где правильное вино не оказалось на первом месте. "
        "Исходник → пять кандидатов → полный разбор OCR."
    ),
    weights_label: str = "DINO 45%  ·  OCR 50%  ·  SIFT 5%",
) -> Path:
    if items is None:
        response = SESSION.get(f"{BASE_URL}/api/v1/review-matrix", timeout=30)
        response.raise_for_status()
        items = response.json()["items"]
    heights = [row_height(item) for item in items]
    header_height = 520
    table_header_height = 86
    footer_height = 110
    canvas_height = header_height + table_header_height + sum(heights) + footer_height
    canvas = Image.new("RGB", (CANVAS_WIDTH, canvas_height), BG)
    draw = ImageDraw.Draw(canvas)

    draw.text((MARGIN, 48), "ХАКАТОН · РАСПОЗНАВАНИЕ ВИНА", font=FONTS["eyebrow"], fill=BURGUNDY)
    draw.text((MARGIN, 102), title, font=FONTS["title"], fill=INK)
    draw_wrapped(
        draw,
        (MARGIN, 222),
        subtitle,
        FONTS["subtitle"],
        MUTED,
        2600,
        max_lines=2,
    )
    top5 = sum(1 for item in items if correct_rank(item) is not None)
    misses = len(items) - top5
    stats = [(len(items), "проблемных кадров"), (top5, "правильный ниже Top‑1"), (misses, "правильного нет в Top‑5")]
    stat_x = MARGIN
    for value, label in stats:
        rounded(draw, (stat_x, 342, stat_x + 560, 470), 28, CREAM)
        draw.text((stat_x + 28, 365), str(value), font=FONTS["stat"], fill=BURGUNDY)
        draw_wrapped(draw, (stat_x + 118, 374), label, FONTS["stat_label"], MUTED, 405, max_lines=2)
        stat_x += 590
    draw.text((CANVAS_WIDTH - MARGIN - 1120, 382), weights_label, font=FONTS["body_bold"], fill=BURGUNDY_DARK)

    top = header_height
    headers = ["ФОТО ДЛЯ РАСПОЗНАВАНИЯ", "КАНДИДАТ #1", "КАНДИДАТ #2", "КАНДИДАТ #3", "КАНДИДАТ #4", "КАНДИДАТ #5", "OCR И ИЗВЛЕЧЁННЫЕ ПОЛЯ"]
    xs = [MARGIN, MARGIN + SOURCE_WIDTH] + [MARGIN + SOURCE_WIDTH + index * CANDIDATE_WIDTH for index in range(1, 5)] + [OCR_X]
    widths = [SOURCE_WIDTH] + [CANDIDATE_WIDTH] * 5 + [CANVAS_WIDTH - MARGIN - OCR_X]
    draw.rectangle((MARGIN, top, CANVAS_WIDTH - MARGIN, top + table_header_height), fill="#efe8dc")
    for x, width, label in zip(xs, widths, headers, strict=True):
        draw.text((x + 20, top + 28), label, font=FONTS["head"], fill=MUTED)
        draw.line((x, top, x, top + table_header_height), fill=LINE, width=2)
    draw.line((CANVAS_WIDTH - MARGIN, top, CANVAS_WIDTH - MARGIN, top + table_header_height), fill=LINE, width=2)
    top += table_header_height

    for item, height in zip(items, heights, strict=True):
        expected = correct_slug(item)
        draw_source(canvas, draw, item, top, height)
        predictions = item["recognition"]["result"]["predictions"]
        for index in range(5):
            x = MARGIN + SOURCE_WIDTH + index * CANDIDATE_WIDTH
            draw_candidate(canvas, draw, predictions[index] if index < len(predictions) else None, index + 1, expected, x, top, height)
        draw_ocr(draw, item, top, height)
        draw.line((MARGIN, top + height, CANVAS_WIDTH - MARGIN, top + height), fill=LINE, width=3)
        top += height

    draw.text((MARGIN, top + 40), "Экспортировано из локального сервиса ХАКАТОН · без Top‑1 совпадений", font=FONTS["body"], fill=MUTED)
    draw.text((CANVAS_WIDTH - MARGIN - 760, top + 40), f"Всего строк: {len(items)}", font=FONTS["body_bold"], fill=BURGUNDY_DARK)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True, compress_level=7)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/review-matrix-top1-errors.png"))
    args = parser.parse_args()
    print(export(args.output.resolve()))


if __name__ == "__main__":
    main()
