"""Compare the curated field benchmark before and after manufacturer gating."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS = PROJECT_ROOT / "data/test/curated_67/annotations.jsonl"
PREDICTIONS = PROJECT_ROOT / "data/artifacts/stage7/manufacturer_gate_predictions.jsonl"
OUTPUT_DIR = PREDICTIONS.parent


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def rank_of(slugs: list[str], expected: str | None) -> int | None:
    if not expected or expected not in slugs:
        return None
    return slugs.index(expected) + 1


def main() -> None:
    baseline = {item["file"]: item for item in read_jsonl(ANNOTATIONS)}
    after = {Path(item["image"]).name: item for item in read_jsonl(PREDICTIONS)}
    if set(baseline) != set(after):
        raise SystemExit("Baseline and manufacturer-gate files differ")

    rows: list[dict[str, Any]] = []
    for name in sorted(baseline):
        old = baseline[name]
        new = after[name]
        before_slugs = list(old["model_top5_slugs"])
        after_slugs = [item["slug"] for item in new["predictions"]]
        expected = old.get("expected_slug")
        match = new.get("manufacturer_match") or {}
        rows.append(
            {
                "file": name,
                "group": old["group"],
                "label_status": old["label_status"],
                "expected_slug": expected or "",
                "before_top1": before_slugs[0],
                "after_top1": after_slugs[0],
                "before_expected_rank": rank_of(before_slugs, expected) or "",
                "after_expected_rank": rank_of(after_slugs, expected) or "",
                "top1_changed": before_slugs[0] != after_slugs[0],
                "manufacturer_gate": match.get("mode") == "hard",
                "manufacturer": match.get("value") or "",
                "manufacturer_score": round(float(match.get("score") or 0.0), 4),
                "manufacturer_candidates": int(match.get("candidate_count") or 0),
                "before_top5": "|".join(before_slugs),
                "after_top5": "|".join(after_slugs),
            }
        )

    exact = [row for row in rows if row["label_status"] == "exact"]
    misses = [row for row in rows if row["group"] == "top5_miss"]
    gated = [row for row in rows if row["manufacturer_gate"]]
    before_top1 = sum(row["before_expected_rank"] == 1 for row in exact)
    after_top1 = sum(row["after_expected_rank"] == 1 for row in exact)
    before_top5 = sum(bool(row["before_expected_rank"]) for row in exact)
    after_top5 = sum(bool(row["after_expected_rank"]) for row in exact)
    maker_counts = Counter(row["manufacturer"] for row in gated)

    csv_path = OUTPUT_DIR / "manufacturer_gate_comparison.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Fuzzy manufacturer gate — полевой прогон 67 кадров",
        "",
        "## Итог",
        "",
        f"- Обработано: **{len(rows)}** кадров.",
        f"- Fuzzy manufacturer gate уверенно сработал: **{len(gated)}** кадров.",
        f"- Top‑1 изменился: **{sum(row['top1_changed'] for row in rows)}** кадров.",
        f"- Из 30 старых `top5_miss` выдача изменилась на **{sum(row['top1_changed'] for row in misses)}** кадрах.",
        "",
        "## Метрики на 37 кадрах с точным expected_slug",
        "",
        "| Метрика | До | После | Изменение |",
        "|---|---:|---:|---:|",
        f"| Top‑1 | {before_top1}/{len(exact)} ({before_top1 / len(exact):.1%}) | {after_top1}/{len(exact)} ({after_top1 / len(exact):.1%}) | {after_top1 - before_top1:+d} |",
        f"| Top‑5 | {before_top5}/{len(exact)} ({before_top5 / len(exact):.1%}) | {after_top5}/{len(exact)} ({after_top5 / len(exact):.1%}) | {after_top5 - before_top5:+d} |",
        "",
        "30 кадров `top5_miss` пока не участвуют в accuracy: для них не указан",
        "точный `expected_slug`. Их изменения перечислены ниже для ручной проверки.",
        "",
        "## Срабатывания по производителям",
        "",
    ]
    lines.extend(f"- {maker}: {count}" for maker, count in maker_counts.most_common())
    lines.extend(
        [
            "",
            "## Изменившиеся кадры",
            "",
            "| Файл | Группа | До | После | Manufacturer gate |",
            "|---|---|---|---|---|",
        ]
    )
    for row in rows:
        if not row["top1_changed"]:
            continue
        gate = (
            f"{row['manufacturer']} ({float(row['manufacturer_score']):.0%})"
            if row["manufacturer_gate"]
            else "—"
        )
        lines.append(
            f"| `{row['file']}` | `{row['group']}` | `{row['before_top1']}` | "
            f"`{row['after_top1']}` | {gate} |"
        )
    lines.append("")
    (OUTPUT_DIR / "manufacturer_gate_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "samples": len(rows),
                "gated": len(gated),
                "changed": sum(row["top1_changed"] for row in rows),
                "exact_top1_before": before_top1,
                "exact_top1_after": after_top1,
                "exact_top5_before": before_top5,
                "exact_top5_after": after_top5,
                "top5_miss_changed": sum(row["top1_changed"] for row in misses),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
