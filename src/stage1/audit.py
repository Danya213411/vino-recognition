from __future__ import annotations

import argparse
import hashlib
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from stage1.catalog import REQUIRED_WINE_FIELDS, is_present, load_catalog, write_json, write_jsonl


DEFAULT_CATALOG = Path("data/parser/wines.json")
DEFAULT_IMAGES = Path("data/parser/images")
DEFAULT_SQLITE = Path("data/parser/wines.sqlite")
DEFAULT_OUTPUT = Path("data/artifacts/stage1")
MANIFEST_SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def label_dhash(image: Image.Image) -> str:
    """Return a 256-bit dHash focused on the likely label area of a bottle."""
    rgba = image.convert("RGBA")
    alpha_bbox = rgba.getchannel("A").getbbox()
    if alpha_bbox:
        rgba = rgba.crop(alpha_bbox)

    width, height = rgba.size
    label_box = (
        int(width * 0.05),
        int(height * 0.30),
        max(int(width * 0.95), 1),
        max(int(height * 0.92), 1),
    )
    label = rgba.crop(label_box)
    background = Image.new("RGBA", label.size, "white")
    background.alpha_composite(label)
    grayscale = background.convert("L").resize((17, 16), Image.Resampling.LANCZOS)
    pixels = grayscale.tobytes()

    value = 0
    for row in range(16):
        offset = row * 17
        for column in range(16):
            value = (value << 1) | int(pixels[offset + column] > pixels[offset + column + 1])
    return f"{value:064x}"


def inspect_image(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        image.load()
        bands = image.getbands()
        return {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "format": image.format,
            "has_alpha": "A" in bands or "transparency" in image.info,
            "label_dhash": label_dhash(image),
        }


def grouped_values(
    records: list[dict[str, Any]], field: str, *, case_insensitive: bool = False
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    observed_values: dict[str, set[str]] = defaultdict(set)
    for record in records:
        value = record.get(field)
        if is_present(value):
            rendered = " ".join(str(value).split())
            key = rendered.casefold() if case_insensitive else rendered
            observed_values[key].add(rendered)
            grouped[key].append(
                {
                    "slug": str(record.get("slug", "")),
                    "title": str(record.get("title", "")),
                }
            )
    return [
        {
            "value": sorted(observed_values[value])[0],
            "observed_values": sorted(observed_values[value]),
            "count": len(items),
            "items": items,
        }
        for value, items in sorted(grouped.items())
        if len(items) > 1
    ]


def exact_image_groups(manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in manifest:
        digest = record.get("image", {}).get("sha256")
        if digest:
            grouped[digest].append(
                {
                    "slug": record["slug"],
                    "title": record["title"],
                    "manufacturer": record["manufacturer"],
                    "region": record.get("region"),
                    "category": record.get("category"),
                    "alcohol": record.get("alcohol"),
                    "image_local": record["image_local"],
                }
            )

    groups: list[dict[str, Any]] = []
    compared_fields = ("title", "manufacturer", "region", "category", "alcohol")
    for digest, items in sorted(grouped.items()):
        if len(items) <= 1:
            continue
        differing_fields = [
            field
            for field in compared_fields
            if len({str(item.get(field)) for item in items}) > 1
        ]
        groups.append(
            {
                "sha256": digest,
                "count": len(items),
                "differing_fields": differing_fields,
                "items": items,
            }
        )
    return groups


class DisjointSet:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def find_near_duplicates(
    manifest: list[dict[str, Any]], threshold: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Find candidate lookalikes inside one manufacturer using label-focused dHash."""
    by_manufacturer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in manifest:
        if record.get("image", {}).get("label_dhash"):
            by_manufacturer[record["manufacturer"]].append(record)

    pairs: list[dict[str, Any]] = []
    all_records = {record["slug"]: record for record in manifest}
    dsu = DisjointSet(list(all_records))

    for manufacturer, records in sorted(by_manufacturer.items()):
        ordered = sorted(records, key=lambda item: item["slug"])
        for index, left in enumerate(ordered):
            left_hash = int(left["image"]["label_dhash"], 16)
            for right in ordered[index + 1 :]:
                if left["image"]["sha256"] == right["image"]["sha256"]:
                    continue
                distance = (left_hash ^ int(right["image"]["label_dhash"], 16)).bit_count()
                if distance <= threshold:
                    dsu.union(left["slug"], right["slug"])
                    pairs.append(
                        {
                            "manufacturer": manufacturer,
                            "distance": distance,
                            "left_slug": left["slug"],
                            "left_title": left["title"],
                            "right_slug": right["slug"],
                            "right_title": right["title"],
                        }
                    )

    component_slugs: dict[str, list[str]] = defaultdict(list)
    paired_slugs = {pair["left_slug"] for pair in pairs} | {pair["right_slug"] for pair in pairs}
    for slug in sorted(paired_slugs):
        component_slugs[dsu.find(slug)].append(slug)

    groups: list[dict[str, Any]] = []
    for group_number, slugs in enumerate(
        sorted(component_slugs.values(), key=lambda values: (-len(values), values)), start=1
    ):
        items = [all_records[slug] for slug in slugs]
        group_pairs = [
            pair
            for pair in pairs
            if pair["left_slug"] in slugs and pair["right_slug"] in slugs
        ]
        groups.append(
            {
                "group_id": f"near-{group_number:04d}",
                "manufacturer": items[0]["manufacturer"],
                "count": len(items),
                "min_distance": min(pair["distance"] for pair in group_pairs),
                "max_distance": max(pair["distance"] for pair in group_pairs),
                "items": [
                    {
                        "slug": item["slug"],
                        "title": item["title"],
                        "image_local": item["image_local"],
                    }
                    for item in items
                ],
            }
        )

    pairs.sort(key=lambda item: (item["distance"], item["manufacturer"], item["left_slug"], item["right_slug"]))
    return pairs, groups


def inspect_sqlite(path: Path, json_slugs: set[str]) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": path.as_posix()}

    connection = sqlite3.connect(path)
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        foreign_key_errors = connection.execute("PRAGMA foreign_key_check").fetchall()
        tables = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
        ]
        row_counts = {
            table: connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            for table in tables
        }
        sqlite_slugs = {
            row[0] for row in connection.execute("SELECT slug FROM wines")
        } if "wines" in tables else set()
        return {
            "exists": True,
            "path": path.as_posix(),
            "integrity_check": integrity,
            "foreign_key_error_count": len(foreign_key_errors),
            "tables": row_counts,
            "json_only_slug_count": len(json_slugs - sqlite_slugs),
            "sqlite_only_slug_count": len(sqlite_slugs - json_slugs),
        }
    finally:
        connection.close()


def markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    images = report["images"]
    duplicates = report["duplicates"]
    sqlite = report["sqlite"]
    issue_counts = report["issues"]["by_severity"]
    return f"""# Аудит первоначального датасета

Аудит построен для снимка каталога от `{report['dataset']['captured_at']}`.

## Сводка

| Показатель | Значение |
|---|---:|
| Записей в JSON | {summary['wine_count']} |
| Уникальных slug | {summary['unique_slug_count']} |
| Позиций заявлено источником | {summary['reported_wine_count']} |
| Покрытие каталога | {summary['coverage_percent']:.2f}% |
| Существующих изображений | {images['existing_count']} |
| Повреждённых изображений | {images['decode_error_count']} |
| Изображений с альфа-каналом | {images['with_alpha_count']} |
| Точных групп-дубликатов изображений | {duplicates['exact_image_group_count']} |
| Кандидатных near-duplicate групп | {duplicates['near_image_group_count']} |
| Групп одинаковых названий | {duplicates['duplicate_title_group_count']} |
| Ошибок аудита | {issue_counts.get('error', 0)} |
| Предупреждений | {issue_counts.get('warning', 0)} |

## Целостность

- отсутствующих локальных изображений: **{images['missing_count']}**;
- изображений без карточки: **{images['orphan_count']}**;
- SQLite integrity check: **{sqlite.get('integrity_check', 'файл отсутствует')}**;
- ошибок внешних ключей SQLite: **{sqlite.get('foreign_key_error_count', '—')}**;
- расхождений slug между JSON и SQLite: **{sqlite.get('json_only_slug_count', '—')} / {sqlite.get('sqlite_only_slug_count', '—')}**.

## Артефакты

- `catalog_manifest.jsonl` — единый manifest карточек и изображений;
- `audit_report.json` — полный машиночитаемый отчёт;
- `quality_issues.jsonl` — ошибки и предупреждения по отдельным позициям;
- `exact_duplicate_groups.json` — одинаковые по SHA-256 изображения;
- `near_duplicate_groups.json` — кандидатные визуально близкие группы;
- `near_duplicate_pairs.jsonl` — пары и perceptual distance.

Near-duplicate группы являются кандидатными: они строятся по label-focused perceptual hash только внутри одного производителя и требуют проверки будущим распознавателем.
"""


def audit_catalog(
    catalog_path: Path,
    images_dir: Path,
    sqlite_path: Path | None,
    output_dir: Path,
    near_duplicate_distance: int = 8,
) -> dict[str, Any]:
    metadata, wines = load_catalog(catalog_path)
    issues: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    referenced_names: set[str] = set()

    def add_issue(severity: str, code: str, message: str, slug: str | None = None) -> None:
        issue: dict[str, Any] = {"severity": severity, "code": code, "message": message}
        if slug:
            issue["slug"] = slug
        issues.append(issue)

    slug_counts = Counter(str(wine.get("slug", "")) for wine in wines if is_present(wine.get("slug")))
    for slug, count in slug_counts.items():
        if count > 1:
            add_issue("error", "duplicate_slug", f"Slug occurs {count} times", slug)

    field_names = sorted({field for wine in wines for field in wine})
    field_coverage: dict[str, dict[str, int]] = {}
    for field in field_names:
        present = sum(is_present(wine.get(field)) for wine in wines)
        field_coverage[field] = {"present": present, "missing": len(wines) - present}

    for index, wine in enumerate(wines):
        slug = str(wine.get("slug", "")).strip()
        for field in REQUIRED_WINE_FIELDS:
            if not is_present(wine.get(field)):
                add_issue("error", "missing_required_field", f"Required field is empty: {field}", slug or None)

        if not is_present(wine.get("manufacturer_slug")):
            add_issue("warning", "missing_manufacturer_slug", "manufacturer_slug is empty", slug or None)

        alcohol = wine.get("alcohol")
        if isinstance(alcohol, (int, float)) and not 5 <= alcohol <= 25:
            add_issue("warning", "suspicious_alcohol", f"Suspicious alcohol value: {alcohol}", slug or None)

        image_local = str(wine.get("image_local", ""))
        image_name = Path(image_local).name
        referenced_names.add(image_name)
        image_path = images_dir / image_name
        image_metadata: dict[str, Any]
        if not image_path.exists():
            image_metadata = {"exists": False}
            add_issue("error", "missing_image", f"Image does not exist: {image_local}", slug or None)
        else:
            try:
                image_metadata = {"exists": True, **inspect_image(image_path)}
            except (UnidentifiedImageError, OSError, ValueError) as error:
                image_metadata = {"exists": True, "decode_error": str(error)}
                add_issue("error", "image_decode_error", str(error), slug or None)

        manifest.append(
            {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "catalog_index": index,
                **wine,
                "slug": slug,
                "image_local": image_local.replace("\\", "/"),
                "image": image_metadata,
            }
        )

    image_files = sorted(path for path in images_dir.iterdir() if path.is_file())
    orphan_names = sorted(path.name for path in image_files if path.name not in referenced_names)
    for name in orphan_names:
        add_issue("warning", "orphan_image", f"Image is not referenced by the catalog: {name}")

    exact_groups = exact_image_groups(manifest)
    near_pairs, near_groups = find_near_duplicates(manifest, near_duplicate_distance)
    duplicate_titles = grouped_values(wines, "title", case_insensitive=True)
    duplicate_source_images = grouped_values(wines, "image_source_url")

    json_slugs = {record["slug"] for record in manifest if record["slug"]}
    sqlite_report = inspect_sqlite(sqlite_path, json_slugs) if sqlite_path else {"exists": False}
    if sqlite_report.get("integrity_check") not in (None, "ok"):
        add_issue("error", "sqlite_integrity", str(sqlite_report["integrity_check"]))
    if sqlite_report.get("foreign_key_error_count", 0):
        add_issue("error", "sqlite_foreign_keys", "SQLite contains foreign-key violations")
    if sqlite_report.get("json_only_slug_count", 0) or sqlite_report.get("sqlite_only_slug_count", 0):
        add_issue("error", "sqlite_json_mismatch", "JSON and SQLite slug sets differ")

    reported_count = int(metadata.get("total_items_reported") or len(wines))
    if reported_count != len(wines):
        add_issue(
            "warning",
            "incomplete_source_coverage",
            f"Collected {len(wines)} of {reported_count} source items",
        )

    issues.sort(key=lambda item: (item["severity"], item["code"], item.get("slug", ""), item["message"]))
    issue_counts = Counter(issue["severity"] for issue in issues)
    existing_images = [record for record in manifest if record["image"].get("exists")]
    decoded_images = [record for record in existing_images if not record["image"].get("decode_error")]
    region_counts = Counter(str(wine.get("region", "")) for wine in wines)
    category_counts = Counter(str(wine.get("category", "")) for wine in wines)
    dimension_counts = Counter(
        f"{record['image']['width']}x{record['image']['height']}" for record in decoded_images
    )
    mode_counts = Counter(str(record["image"].get("mode")) for record in decoded_images)

    report: dict[str, Any] = {
        "schema_version": 1,
        "dataset": {
            "catalog_path": catalog_path.as_posix(),
            "images_dir": images_dir.as_posix(),
            "source_name": metadata.get("source_name"),
            "source_url": metadata.get("source_url"),
            "captured_at": metadata.get("captured_at"),
            "collection_failures": metadata.get("failures", []),
        },
        "summary": {
            "wine_count": len(wines),
            "unique_slug_count": len(json_slugs),
            "reported_wine_count": reported_count,
            "coverage_percent": (len(wines) / reported_count * 100) if reported_count else 0.0,
            "manufacturer_count": len({wine.get("manufacturer") for wine in wines}),
            "region_count": len(region_counts),
            "category_count": len(category_counts),
        },
        "field_coverage": field_coverage,
        "distributions": {
            "regions": dict(sorted(region_counts.items(), key=lambda item: (-item[1], item[0]))),
            "categories": dict(sorted(category_counts.items(), key=lambda item: (-item[1], item[0]))),
        },
        "images": {
            "referenced_count": len(referenced_names),
            "existing_count": len(existing_images),
            "missing_count": len(manifest) - len(existing_images),
            "decoded_count": len(decoded_images),
            "decode_error_count": len(existing_images) - len(decoded_images),
            "orphan_count": len(orphan_names),
            "orphan_names": orphan_names,
            "with_alpha_count": sum(bool(record["image"].get("has_alpha")) for record in decoded_images),
            "total_bytes": sum(int(record["image"].get("bytes", 0)) for record in decoded_images),
            "unique_dimension_count": len(dimension_counts),
            "dimensions": dict(sorted(dimension_counts.items(), key=lambda item: (-item[1], item[0]))),
            "modes": dict(sorted(mode_counts.items(), key=lambda item: (-item[1], item[0]))),
        },
        "duplicates": {
            "duplicate_title_group_count": len(duplicate_titles),
            "duplicate_source_image_group_count": len(duplicate_source_images),
            "exact_image_group_count": len(exact_groups),
            "exact_image_file_count": sum(group["count"] for group in exact_groups),
            "exact_image_conflicting_metadata_group_count": sum(
                bool(group["differing_fields"]) for group in exact_groups
            ),
            "near_image_distance_threshold": near_duplicate_distance,
            "near_image_pair_count": len(near_pairs),
            "near_image_group_count": len(near_groups),
        },
        "sqlite": sqlite_report,
        "issues": {
            "total": len(issues),
            "by_severity": dict(sorted(issue_counts.items())),
            "by_code": dict(sorted(Counter(issue["code"] for issue in issues).items())),
        },
    }

    write_jsonl(output_dir / "catalog_manifest.jsonl", manifest)
    write_json(output_dir / "audit_report.json", report)
    write_jsonl(output_dir / "quality_issues.jsonl", issues)
    write_json(output_dir / "exact_duplicate_groups.json", exact_groups)
    write_json(output_dir / "duplicate_title_groups.json", duplicate_titles)
    write_json(output_dir / "near_duplicate_groups.json", near_groups)
    write_jsonl(output_dir / "near_duplicate_pairs.jsonl", near_pairs)
    (output_dir / "audit_report.md").write_text(markdown_report(report), encoding="utf-8", newline="\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit the source wine catalog and build a manifest.")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--sqlite", type=Path, default=DEFAULT_SQLITE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--near-distance", type=int, default=8)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = audit_catalog(
        catalog_path=args.catalog,
        images_dir=args.images,
        sqlite_path=args.sqlite,
        output_dir=args.output_dir,
        near_duplicate_distance=args.near_distance,
    )
    summary = report["summary"]
    issues = report["issues"]["by_severity"]
    print(
        f"Audited {summary['wine_count']} wines; "
        f"errors={issues.get('error', 0)}, warnings={issues.get('warning', 0)}; "
        f"report={args.output_dir / 'audit_report.md'}"
    )
    return 1 if issues.get("error", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
