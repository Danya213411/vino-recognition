"""Export the manually filtered store test set from the stage 6 journal."""

from __future__ import annotations

import csv
import json
import shutil
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "data" / "test" / "given"
TARGET_DIR = PROJECT_ROOT / "data" / "test" / "curated_67"
DATABASE_PATH = PROJECT_ROOT / "data" / "app" / "stage6" / "events.sqlite3"
GROUPS = ("top1_correct", "top5_correct", "top5_miss")


def load_latest_annotations() -> dict[str, sqlite3.Row]:
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            WITH labeled AS (
                SELECT
                    r.id,
                    r.original_name,
                    r.created_at,
                    r.confidence,
                    r.result_json,
                    f.verdict,
                    f.correct_slug,
                    ROW_NUMBER() OVER (
                        PARTITION BY r.original_name
                        ORDER BY f.created_at DESC, r.created_at DESC
                    ) AS row_number
                FROM recognitions AS r
                JOIN feedback AS f ON f.recognition_id = r.id
                WHERE r.mode = 'calib'
            )
            SELECT * FROM labeled WHERE row_number = 1
            """
        ).fetchall()
    finally:
        connection.close()
    return {row["original_name"]: row for row in rows}


def group_for(verdict: str) -> str:
    return {
        "correct": "top1_correct",
        "incorrect": "top5_correct",
        "not_in_catalog": "top5_miss",
    }[verdict]


def build_annotation(source: Path, row: sqlite3.Row) -> dict[str, Any]:
    result = json.loads(row["result_json"])
    predictions = result["predictions"]
    verdict = row["verdict"]
    expected_slug = (
        predictions[0]["slug"]
        if verdict == "correct"
        else row["correct_slug"] if verdict == "incorrect" else None
    )
    expected_prediction = next(
        (item for item in predictions if item["slug"] == expected_slug), None
    )
    group = group_for(verdict)
    return {
        "file": source.name,
        "relative_path": f"images/{group}/{source.name}",
        "group": group,
        "human_verdict": verdict,
        "label_status": "exact" if expected_slug else "needs_exact_wine",
        "expected_slug": expected_slug,
        "expected_title": (
            expected_prediction["wine"]["title"] if expected_prediction else None
        ),
        "model_top1_slug": predictions[0]["slug"],
        "model_top1_title": predictions[0]["wine"]["title"],
        "model_confidence": round(float(row["confidence"]), 6),
        "model_top5_slugs": [item["slug"] for item in predictions],
        "recognition_id": row["id"],
    }


def write_csv(rows: list[dict[str, Any]]) -> None:
    fields = [
        "file",
        "relative_path",
        "group",
        "human_verdict",
        "label_status",
        "expected_slug",
        "expected_title",
        "model_top1_slug",
        "model_top1_title",
        "model_confidence",
        "model_top5_slugs",
        "recognition_id",
    ]
    with (TARGET_DIR / "annotations.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "model_top5_slugs": "|".join(row["model_top5_slugs"])})


def write_jsonl(rows: list[dict[str, Any]]) -> None:
    with (TARGET_DIR / "annotations.jsonl").open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def markdown_cell(value: str | None) -> str:
    return (value or "—").replace("|", "\\|")


def write_manual(rows: list[dict[str, Any]]) -> None:
    counts = Counter(row["group"] for row in rows)
    exact = sum(row["label_status"] == "exact" for row in rows)
    lines = [
        "# Curated store test set — 67 кадров",
        "",
        "Это копия кадров из `data/test/given`, отфильтрованная по ручной разметке",
        "из `/review/`. Исключены 33 кадра с вердиктом `not_in_store`: на них",
        "модель физически не могла найти правильное вино в каталоге магазина.",
        "",
        "## Состав",
        "",
        f"- `images/top1_correct/` — {counts['top1_correct']} кадров, top‑1 подтверждён.",
        f"- `images/top5_correct/` — {counts['top5_correct']} кадров, правильное вино выбрано из позиций 2–5.",
        f"- `images/top5_miss/` — {counts['top5_miss']} кадров, правильного вина не было в top‑5.",
        f"- Точный `expected_slug` известен для {exact} из {len(rows)} кадров.",
        "",
        "Для `top5_miss` интерфейс сохранял только факт промаха, но не точное название",
        "бутылки. Поэтому такие строки имеют `label_status=needs_exact_wine`. Их нельзя",
        "использовать для supervised-обучения конкретного класса, пока мы не укажем",
        "точный `expected_slug`.",
        "",
        "## Файлы разметки",
        "",
        "- `annotations.csv` — удобно открыть в Excel и фильтровать вручную.",
        "- `annotations.jsonl` — тот же манифест для скриптов и метрик.",
        "- `expected_*` — ручной ground truth, если он был выбран явно.",
        "- `model_top1_*` и `model_top5_slugs` — ответ текущего пайплайна на момент разметки.",
        "- `recognition_id` — ссылка на исходную запись в локальном журнале `/admin/`.",
        "",
        "## Правило использования",
        "",
        "Этот набор фиксируем как тестовый. Не добавляем эти 67 кадров в референсный",
        "индекс и не обучаем на них модель: иначе метрики будут завышены из-за утечки.",
        "Подбирать параметры можно на отдельной train/validation-выборке, а эти кадры",
        "использовать только для финального сравнения вариантов пайплайна.",
        "",
        "## Покадровая карта",
        "",
        "| № | Файл | Группа | Что на фото | Top‑1 модели |",
        "|---:|---|---|---|---|",
    ]
    for index, row in enumerate(rows, 1):
        expected = (
            f"{row['expected_title']} (`{row['expected_slug']}`)"
            if row["expected_slug"]
            else "Точное вино ещё нужно указать"
        )
        predicted = f"{row['model_top1_title']} (`{row['model_top1_slug']}`)"
        lines.append(
            f"| {index} | `{markdown_cell(row['file'])}` | `{row['group']}` | "
            f"{markdown_cell(expected)} | {markdown_cell(predicted)} |"
        )
    lines.append("")
    (TARGET_DIR / "MANUAL.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if TARGET_DIR.exists():
        raise SystemExit(f"Target already exists: {TARGET_DIR}")
    source_files = sorted(path for path in SOURCE_DIR.iterdir() if path.is_file())
    annotations = load_latest_annotations()
    missing = [path.name for path in source_files if path.name not in annotations]
    if missing:
        raise SystemExit(f"Missing feedback for {len(missing)} files: {missing}")

    rows: list[dict[str, Any]] = []
    for source in source_files:
        row = annotations[source.name]
        if row["verdict"] == "not_in_store":
            continue
        annotation = build_annotation(source, row)
        rows.append(annotation)

    if len(rows) != 67:
        raise SystemExit(f"Expected 67 curated files, found {len(rows)}")

    for group in GROUPS:
        (TARGET_DIR / "images" / group).mkdir(parents=True, exist_ok=False)
    for row in rows:
        shutil.copy2(SOURCE_DIR / row["file"], TARGET_DIR / row["relative_path"])

    write_csv(rows)
    write_jsonl(rows)
    write_manual(rows)
    print(f"Exported {len(rows)} files to {TARGET_DIR}")


if __name__ == "__main__":
    main()
