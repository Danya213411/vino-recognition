from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import sqlite3
import time
import urllib.request
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = PROJECT_ROOT / "data/parser/wines.json"
DEFAULT_SQLITE = PROJECT_ROOT / "data/parser/wines.sqlite"
DEFAULT_IMAGES = PROJECT_ROOT / "data/parser/images"
DEFAULT_SOURCES = PROJECT_ROOT / "data/unknown/stage5/unknown_sources.jsonl"
DEFAULT_SOURCE_IMAGES = DEFAULT_SOURCES.parent
NUXT_DATA_PATTERN = re.compile(
    r'<script[^>]+id="__NUXT_DATA__"[^>]*>(.*?)</script>', re.IGNORECASE | re.DOTALL
)
USER_AGENT = "VinoResearch/0.1 (catalog refresh)"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def fetch_text(url: str, retries: int = 4) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read().decode("utf-8", errors="replace")
        except Exception:
            if attempt + 1 == retries:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise AssertionError("unreachable")


def decode_nuxt_payload(page_html: str, slug: str) -> dict[str, Any]:
    match = NUXT_DATA_PATTERN.search(page_html)
    if not match:
        raise ValueError("Nuxt payload was not found")
    flattened = json.loads(html.unescape(match.group(1)))
    cache: dict[int, Any] = {}

    def resolve(index: int) -> Any:
        if index < 0:
            return None
        if index in cache:
            return cache[index]
        value = flattened[index]
        if isinstance(value, dict):
            result: dict[str, Any] = {}
            cache[index] = result
            for key, child in value.items():
                result[key] = resolve(child) if isinstance(child, int) and not isinstance(child, bool) else child
            return result
        if isinstance(value, list):
            if value and value[0] in {"ShallowReactive", "Reactive", "Ref", "ShallowRef"}:
                result = resolve(value[1])
                cache[index] = result
                return result
            result_list: list[Any] = []
            cache[index] = result_list
            result_list.extend(
                resolve(child) if isinstance(child, int) and not isinstance(child, bool) else child
                for child in value
            )
            return result_list
        cache[index] = value
        return value

    root = resolve(0)
    data = root.get("data", {}) if isinstance(root, dict) else {}
    wine_page = data.get(f"wine-{slug}") if isinstance(data, dict) else None
    wine = wine_page.get("wine") if isinstance(wine_page, dict) else None
    if not isinstance(wine, dict) or wine.get("slug") != slug:
        raise ValueError(f"Wine payload for {slug!r} was not found")
    return wine


def named(value: Any) -> str | None:
    if isinstance(value, dict):
        raw = value.get("name")
        return str(raw) if raw is not None else None
    return str(value) if value is not None else None


def names(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    result = []
    for value in values:
        item = named(value)
        if item:
            result.append(item)
    return result


def normalize_record(source: dict[str, Any], wine: dict[str, Any]) -> dict[str, Any]:
    category = wine.get("category") if isinstance(wine.get("category"), dict) else {}
    manufacturer = wine.get("manufacturer") if isinstance(wine.get("manufacturer"), dict) else {}
    region = wine.get("region") if isinstance(wine.get("region"), dict) else {}
    image = wine.get("image") if isinstance(wine.get("image"), dict) else {}
    slug = str(source["slug"])
    return {
        "slug": slug,
        "title": str(wine.get("title") or source.get("title") or slug),
        "manufacturer": str(manufacturer.get("name") or source.get("manufacturer") or ""),
        "manufacturer_slug": manufacturer.get("slug"),
        "region": named(region),
        "category": named(category),
        "color": wine.get("color"),
        "public_rating": wine.get("publicRating"),
        "rating": wine.get("rating"),
        "alcohol": wine.get("alcohol"),
        "alcohol_max": wine.get("alcoholMax"),
        "temperature": wine.get("temperature"),
        "description": wine.get("description"),
        "color_gradient": category.get("backgroundGradient"),
        "grapes": names(wine.get("grapes")),
        "dishes": names(wine.get("dishes")),
        "image_source_url": source.get("image_source_url"),
        "image_alt": image.get("altText") or str(wine.get("title") or source.get("title") or slug),
        "source_url": source["source_url"],
        "image_local": f"images/{slug}.webp",
    }


def update_sqlite(path: Path, records: list[dict[str, Any]]) -> None:
    scalar_fields = (
        "slug",
        "title",
        "manufacturer",
        "manufacturer_slug",
        "region",
        "category",
        "color",
        "public_rating",
        "rating",
        "alcohol",
        "alcohol_max",
        "temperature",
        "description",
        "color_gradient",
        "image_local",
        "image_source_url",
        "image_alt",
        "source_url",
    )
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        for record in records:
            placeholders = ", ".join("?" for _ in scalar_fields)
            columns = ", ".join(scalar_fields)
            connection.execute(
                f"INSERT INTO wines ({columns}) VALUES ({placeholders})",
                [record.get(field) for field in scalar_fields],
            )
            wine_id = connection.execute(
                "SELECT id FROM wines WHERE slug = ?", (record["slug"],)
            ).fetchone()[0]
            connection.executemany(
                "INSERT INTO grapes (wine_id, name) VALUES (?, ?)",
                [(wine_id, item) for item in record.get("grapes", [])],
            )
            connection.executemany(
                "INSERT INTO dishes (wine_id, name) VALUES (?, ?)",
                [(wine_id, item) for item in record.get("dishes", [])],
            )
        connection.commit()
    finally:
        connection.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Promote selected live-site wines into the local catalog.")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--sqlite", type=Path, default=DEFAULT_SQLITE)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--manufacturer", default="Литавщук")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    payload = json.loads(args.catalog.read_text(encoding="utf-8"))
    existing = {str(record["slug"]) for record in payload["wines"]}
    selected = [
        record
        for record in read_jsonl(args.sources)
        if args.manufacturer.casefold() in str(record.get("manufacturer") or "").casefold()
        and str(record["slug"]) not in existing
    ]
    promoted = []
    for source in selected:
        wine = decode_nuxt_payload(fetch_text(str(source["source_url"])), str(source["slug"]))
        record = normalize_record(source, wine)
        source_image = args.sources.parent / str(source["source_image"])
        if not source_image.is_file():
            raise FileNotFoundError(source_image)
        promoted.append((record, source_image))
        print(f"{record['slug']}: {record['title']} / {record['category']}")

    if args.dry_run:
        print(json.dumps({"selected": len(selected), "promoted": len(promoted)}, ensure_ascii=False))
        return 0

    records = [record for record, _ in promoted]
    args.images.mkdir(parents=True, exist_ok=True)
    for record, source_image in promoted:
        shutil.copyfile(source_image, args.images / f"{record['slug']}.webp")
    payload["wines"].extend(records)
    payload["collected_items"] = len(payload["wines"])
    payload["total_items_reported"] = max(
        int(payload.get("total_items_reported") or 0), len(payload["wines"])
    )
    with args.catalog.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    update_sqlite(args.sqlite, records)
    print(json.dumps({"promoted": len(records), "catalog_items": len(payload["wines"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
