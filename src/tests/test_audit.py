from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from stage1.audit import audit_catalog


class AuditTests(unittest.TestCase):
    def test_audit_builds_manifest_and_finds_exact_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            images = root / "images"
            images.mkdir()
            first_image = images / "first.webp"
            second_image = images / "second.webp"
            Image.new("RGBA", (40, 80), (120, 30, 10, 255)).save(first_image, "WEBP", lossless=True)
            shutil.copyfile(first_image, second_image)

            wines = [
                self.wine("first", "First", "images/first.webp", 12.0),
                self.wine("second", "Second", "images/second.webp", 108),
            ]
            catalog = root / "wines.json"
            catalog.write_text(
                json.dumps(
                    {
                        "captured_at": "2026-01-01T00:00:00Z",
                        "total_items_reported": 2,
                        "collected_items": 2,
                        "wines": wines,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            output = root / "output"

            report = audit_catalog(catalog, images, None, output)

            self.assertEqual(report["summary"]["wine_count"], 2)
            self.assertEqual(report["images"]["missing_count"], 0)
            self.assertEqual(report["duplicates"]["exact_image_group_count"], 1)
            self.assertEqual(report["issues"]["by_code"]["suspicious_alcohol"], 1)
            self.assertEqual(len((output / "catalog_manifest.jsonl").read_text(encoding="utf-8").splitlines()), 2)

    @staticmethod
    def wine(slug: str, title: str, image_local: str, alcohol: float) -> dict[str, object]:
        return {
            "slug": slug,
            "title": title,
            "manufacturer": "Winery",
            "manufacturer_slug": "winery",
            "region": "Region",
            "category": "Category",
            "description": "Description",
            "image_local": image_local,
            "source_url": f"https://example.test/{slug}",
            "alcohol": alcohol,
        }


if __name__ == "__main__":
    unittest.main()
