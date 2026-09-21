from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from stage2.generator import PROFILES, generate_dataset


class SyntheticDatasetTests(unittest.TestCase):
    def test_generation_is_reproducible_and_covers_all_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_root = root / "source"
            images_dir = source_root / "images"
            images_dir.mkdir(parents=True)
            source_image = images_dir / "bottle.png"
            self.write_bottle(source_image)

            manifest = root / "catalog_manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "catalog_index": 0,
                        "slug": "test-wine",
                        "image_local": "images/bottle.png",
                        "image": {"sha256": "source-hash"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            first_output = root / "first"
            second_output = root / "second"

            first_report = generate_dataset(
                manifest,
                source_root,
                first_output,
                seed=42,
                variants_per_wine=len(PROFILES),
                image_size=160,
            )
            second_report = generate_dataset(
                manifest,
                source_root,
                second_output,
                seed=42,
                variants_per_wine=len(PROFILES),
                image_size=160,
            )

            self.assertEqual(first_report["summary"]["sample_count"], len(PROFILES))
            self.assertEqual(set(first_report["profiles"]), set(PROFILES))
            self.assertTrue(all(count == 1 for count in first_report["profiles"].values()))
            first_records = self.read_jsonl(first_output / "synthetic_manifest.jsonl")
            self.assertTrue(all(record["split"] == "validation" for record in first_records))
            self.assertEqual({record["profile"] for record in first_records}, set(PROFILES))
            self.assertTrue(all(record["sha256"] for record in first_records))
            self.assertTrue(
                all(record["transformations"]["geometry"]["qa"]["passed"] for record in first_records)
            )

            first_hashes = self.image_hashes(first_output / "images")
            second_hashes = self.image_hashes(second_output / "images")
            self.assertEqual(first_hashes, second_hashes)

    @staticmethod
    def write_bottle(path: Path) -> None:
        image = np.zeros((180, 72, 4), dtype=np.uint8)
        cv2.rectangle(image, (22, 4), (50, 176), (35, 80, 50, 255), -1)
        cv2.rectangle(image, (7, 60), (65, 176), (40, 95, 55, 255), -1)
        cv2.rectangle(image, (10, 95), (62, 145), (220, 225, 215, 255), -1)
        cv2.putText(image, "TEST", (13, 123), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20, 255), 1)
        success, encoded = cv2.imencode(".png", image)
        if not success:
            raise AssertionError("Failed to encode test image")
        encoded.tofile(path)

    @staticmethod
    def read_jsonl(path: Path) -> list[dict[str, object]]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    @staticmethod
    def image_hashes(directory: Path) -> dict[str, str]:
        return {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(directory.glob("*.jpg"))
        }


if __name__ == "__main__":
    unittest.main()
