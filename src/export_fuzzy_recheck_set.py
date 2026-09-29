"""Build a compact recheck set after the flexible manufacturer resolver run."""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "data/test/curated_67/images"
PREDICTIONS = PROJECT_ROOT / "data/artifacts/stage7/flexible_fuzzy_predictions.jsonl"
DATABASE = PROJECT_ROOT / "data/app/stage6/events.sqlite3"
OUTPUT_ROOT = PROJECT_ROOT / "data/test/recheck_fuzzy_v2"
OUTPUT_IMAGES = OUTPUT_ROOT / "images"
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def latest_manual_rows(names: set[str]) -> dict[str, dict[str, Any]]:
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            WITH latest AS (
                SELECT r.original_name, r.candidate_slug, r.created_at,
                       f.verdict, f.correct_slug,
                       ROW_NUMBER() OVER (
                           PARTITION BY r.original_name ORDER BY r.created_at DESC
                       ) AS row_number
                FROM recognitions r
                LEFT JOIN feedback f ON f.recognition_id = r.id
            )
            SELECT * FROM latest WHERE row_number = 1
            """
        ).fetchall()
    finally:
        connection.close()
    return {row["original_name"]: dict(row) for row in rows if row["original_name"] in names}


def main() -> int:
    source_paths = {
        path.name: path
        for path in SOURCE_ROOT.rglob("*")
        if path.is_file() and path.suffix.casefold() in SUPPORTED_SUFFIXES
    }
    predictions = {Path(row["image"]).name: row for row in read_jsonl(PREDICTIONS)}
    manual = latest_manual_rows(set(source_paths))
    if set(source_paths) != set(predictions) or set(source_paths) != set(manual):
        raise RuntimeError("The curated images, predictions and latest manual rows differ")

    OUTPUT_IMAGES.mkdir(parents=True, exist_ok=True)
    existing = {
        path.name
        for path in OUTPUT_IMAGES.iterdir()
        if path.is_file() and path.suffix.casefold() in SUPPORTED_SUFFIXES
    }
    if existing - set(source_paths):
        raise RuntimeError("The output folder contains unexpected image files")

    manifest: list[dict[str, Any]] = []
    included: list[dict[str, Any]] = []
    for name in sorted(source_paths):
        previous = manual[name]
        prediction = predictions[name]
        new_top1 = str(prediction["predictions"][0]["slug"])
        if previous["verdict"] == "not_in_store":
            reason = "ambiguous_not_in_store"
            include = False
        elif previous["verdict"] == "correct" and previous["candidate_slug"] == new_top1:
            reason = "stable_confirmed_top1"
            include = False
        else:
            reason = "needs_recheck"
            include = True
        row = {
            "file": name,
            "source": source_paths[name].relative_to(PROJECT_ROOT).as_posix(),
            "include": include,
            "reason": reason,
            "previous_verdict": previous["verdict"],
            "previous_correct_slug": previous["correct_slug"],
            "previous_top1": previous["candidate_slug"],
            "new_top1": new_top1,
            "top1_changed": previous["candidate_slug"] != new_top1,
            "manufacturer_match": prediction.get("manufacturer_match"),
        }
        manifest.append(row)
        if include:
            shutil.copy2(source_paths[name], OUTPUT_IMAGES / name)
            included.append(row)

    expected = {"stable_confirmed_top1": 34, "ambiguous_not_in_store": 1, "needs_recheck": 32}
    actual = {reason: sum(row["reason"] == reason for row in manifest) for reason in expected}
    if actual != expected:
        raise RuntimeError(f"Unexpected selection counts: {actual}")

    (OUTPUT_ROOT / "selection_manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest),
        encoding="utf-8",
    )
    (OUTPUT_ROOT / "review_manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in included),
        encoding="utf-8",
    )
    (OUTPUT_ROOT / "README.md").write_text(
        """# Повторная проверка flexible fuzzy v2

В `images/` находятся 32 кадра для повторной ручной проверки.

Из исходных 67 исключены:

- 34 кадра, где ранее подтверждённый Top-1 не изменился после нового fuzzy-resolver;
- один неоднозначный кадр с двумя бутылками, отмеченный `not_in_store`.

`selection_manifest.jsonl` описывает решение для всех 67 кадров,
`review_manifest.jsonl` содержит только выбранные 32.
""",
        encoding="utf-8",
    )
    print(f"review={len(included)}; stable={actual['stable_confirmed_top1']}; ambiguous=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
