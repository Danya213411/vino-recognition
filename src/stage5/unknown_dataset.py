from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from stage1.catalog import read_jsonl, write_json, write_jsonl
from stage2.generator import PROFILES, encode_jpeg, render_sample, stable_seed


CATALOG_URL = "https://vino-svoe.ru/wines"
USER_AGENT = "VinoResearch/0.1 (local wine-recognition benchmark)"
DEFAULT_KNOWN_MANIFEST = Path("data/artifacts/stage1/catalog_manifest.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/unknown/stage5")
DEFAULT_SEED = 20260921
CARD_PATTERN = re.compile(
    r'<a href="/wines/([^"?#]+)" class="[^"]*\bwine-item\b[^"]*"[^>]*>(.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)
IMAGE_PATTERN = re.compile(r'<img\b[^>]*\bsrc="([^"]+)"', re.IGNORECASE)
TITLE_PATTERN = re.compile(
    r'<h2\b[^>]*class="[^"]*wine-item__title[^"]*"[^>]*>(.*?)</h2>',
    re.IGNORECASE | re.DOTALL,
)
MANUFACTURER_PATTERN = re.compile(
    r'<span\b[^>]*class="[^"]*wine-item__manufacturer[^"]*"[^>]*>(.*?)</span>',
    re.IGNORECASE | re.DOTALL,
)
TAG_PATTERN = re.compile(r"<[^>]+>")


def plain_text(value: str) -> str:
    return " ".join(html.unescape(TAG_PATTERN.sub(" ", value)).split())


def fetch_bytes(url: str, retries: int = 3) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.read()
        except Exception:
            if attempt + 1 == retries:
                raise
            time.sleep(1.0 + attempt)
    raise AssertionError("unreachable")


def parse_cards(page_html: str, page: int) -> list[dict[str, Any]]:
    records = []
    for match in CARD_PATTERN.finditer(page_html):
        slug, body = match.groups()
        image_matches = IMAGE_PATTERN.findall(body)
        image_value = next(
            (value for value in image_matches if "api.vino-svoe.ru" in html.unescape(value)),
            image_matches[-1] if image_matches else None,
        )
        title_match = TITLE_PATTERN.search(body)
        manufacturer_match = MANUFACTURER_PATTERN.search(body)
        if not image_value or not title_match:
            continue
        image_url = html.unescape(image_value)
        if image_url.startswith("/"):
            image_url = urllib.parse.urljoin(CATALOG_URL, image_url)
        records.append(
            {
                "slug": html.unescape(slug),
                "title": plain_text(title_match.group(1)),
                "manufacturer": plain_text(manufacturer_match.group(1))
                if manufacturer_match
                else "",
                "source_url": f"{CATALOG_URL}/{html.unescape(slug)}",
                "image_source_url": image_url,
                "catalog_page": page,
            }
        )
    return records


def discover_current_catalog(max_pages: int | None = None) -> list[dict[str, Any]]:
    first_html = fetch_bytes(CATALOG_URL).decode("utf-8", errors="replace")
    page_numbers = [int(value) for value in re.findall(r'href="/wines\?page=(\d+)"', first_html)]
    page_count = min(max(page_numbers, default=1), max_pages or 10_000)
    records = parse_cards(first_html, 1)
    for page in range(2, page_count + 1):
        url = f"{CATALOG_URL}?page={page}"
        page_html = fetch_bytes(url).decode("utf-8", errors="replace")
        records.extend(parse_cards(page_html, page))
        if page % 20 == 0 or page == page_count:
            print(f"Catalog pages: {page}/{page_count}", flush=True)
        time.sleep(0.08)
    unique: dict[str, dict[str, Any]] = {}
    for record in records:
        unique.setdefault(record["slug"], record)
    return list(unique.values())


def collect_unknown_dataset(
    known_manifest: Path,
    output_dir: Path,
    variants_per_wine: int = 2,
    seed: int = DEFAULT_SEED,
    max_pages: int | None = None,
) -> dict[str, Any]:
    if variants_per_wine < 1:
        raise ValueError("variants_per_wine must be at least 1")
    known_records = read_jsonl(known_manifest)
    known_slugs = {str(record["slug"]) for record in known_records}
    current_records = discover_current_catalog(max_pages)
    unseen = [record for record in current_records if str(record["slug"]) not in known_slugs]
    if not unseen:
        raise RuntimeError("The live catalog contains no slugs outside the local snapshot")

    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = output_dir / "source_images"
    image_dir = output_dir / "images"
    source_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    output_records: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for source_index, record in enumerate(unseen):
        slug = str(record["slug"])
        source_path = source_dir / f"{slug}.webp"
        try:
            source_path.write_bytes(fetch_bytes(str(record["image_source_url"])))
            content = source_path.read_bytes()
            source_sha = hashlib.sha256(content).hexdigest()
            # The live out-of-catalog set is much smaller than the known set.
            # Use a source-level 50/50 split so the FAR threshold sees enough
            # distinct wine series while both variants of one bottle stay together.
            split = (
                "calibration"
                if int(hashlib.sha1(slug.encode()).hexdigest()[:8], 16) % 2 == 0
                else "heldout"
            )
            source_records.append(
                {
                    **record,
                    "source_image": source_path.relative_to(output_dir).as_posix(),
                    "source_sha256": source_sha,
                    "split": split,
                }
            )
            for variant_index in range(variants_per_wine):
                profile = PROFILES[(source_index * variants_per_wine + variant_index) % len(PROFILES)]
                sample_seed = stable_seed(seed, slug, variant_index)
                image, transformations, jpeg_quality, mask_source = render_sample(
                    source_path, profile, 512, sample_seed, 88
                )
                filename = f"{source_index:04d}-{slug[:42].rstrip('-')}-v{variant_index:02d}.jpg"
                output_path = image_dir / filename
                digest = encode_jpeg(output_path, image, jpeg_quality)
                output_records.append(
                    {
                        "schema_version": 1,
                        "sample_id": f"unknown-{slug}-v{variant_index:02d}",
                        "known": False,
                        "expected_slug": None,
                        "source_slug": slug,
                        "title": record["title"],
                        "manufacturer": record["manufacturer"],
                        "split": split,
                        "profile": profile,
                        "image": output_path.relative_to(output_dir).as_posix(),
                        "sha256": digest,
                        "source_image": source_path.relative_to(output_dir).as_posix(),
                        "source_sha256": source_sha,
                        "source_url": record["source_url"],
                        "image_source_url": record["image_source_url"],
                        "seed": sample_seed,
                        "jpeg_quality": jpeg_quality,
                        "transformations": transformations,
                        "foreground_mask": mask_source,
                    }
                )
        except Exception as error:
            failures.append({"slug": slug, "error": f"{type(error).__name__}: {error}"})
        if (source_index + 1) % 20 == 0:
            print(f"New Russian wines: {source_index + 1}/{len(unseen)}", flush=True)

    manifest_path = output_dir / "unknown_manifest.jsonl"
    write_jsonl(manifest_path, output_records)
    write_jsonl(output_dir / "unknown_sources.jsonl", source_records)
    report = {
        "schema_version": 1,
        "source": CATALOG_URL,
        "definition": "Russian wines present in the live catalog but absent from the local 2025-slug snapshot",
        "known_snapshot_count": len(known_slugs),
        "live_catalog_count": len(current_records),
        "unknown_source_count": len(source_records),
        "sample_count": len(output_records),
        "variants_per_wine": variants_per_wine,
        "calibration_source_count": sum(record["split"] == "calibration" for record in source_records),
        "heldout_source_count": sum(record["split"] == "heldout" for record in source_records),
        "calibration_sample_count": sum(record["split"] == "calibration" for record in output_records),
        "heldout_sample_count": sum(record["split"] == "heldout" for record in output_records),
        "failures": failures,
        "manifest": manifest_path.as_posix(),
    }
    write_json(output_dir / "collection_report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an OOD set from new Russian wines outside the local catalog snapshot."
    )
    parser.add_argument("--known-manifest", type=Path, default=DEFAULT_KNOWN_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--variants", type=int, default=2)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-pages", type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = collect_unknown_dataset(
        args.known_manifest, args.output_dir, args.variants, args.seed, args.max_pages
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
