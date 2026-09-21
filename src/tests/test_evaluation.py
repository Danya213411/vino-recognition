from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from stage1.evaluation import evaluate_predictions


class EvaluationTests(unittest.TestCase):
    def test_ranked_metrics_and_error_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest = root / "manifest.jsonl"
            predictions = root / "predictions.jsonl"
            output = root / "output"

            manifest_records = [
                self.manifest("a", "A"),
                self.manifest("b", "B"),
                self.manifest("c", "C"),
            ]
            self.write_jsonl(manifest, manifest_records)
            self.write_jsonl(
                predictions,
                [
                    self.sample("one", "a", [("a", 0.9), ("b", 0.2)], 10),
                    self.sample("two", "b", [("a", 0.7), ("b", 0.6)], 20),
                    self.sample("three", "c", [("c", 0.8), ("a", 0.1)], 30),
                ],
            )

            report = evaluate_predictions(predictions, manifest, output)

            self.assertEqual(report["summary"]["sample_count"], 3)
            self.assertAlmostEqual(report["summary"]["top1_accuracy"], 2 / 3)
            self.assertEqual(report["summary"]["top5_recall"], 1.0)
            self.assertAlmostEqual(report["summary"]["mean_reciprocal_rank"], 5 / 6)
            self.assertEqual(report["latency_ms"]["p95"], 30.0)
            errors = (output / "evaluation_errors.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(errors), 1)

    @staticmethod
    def manifest(slug: str, title: str) -> dict[str, object]:
        return {
            "slug": slug,
            "title": title,
            "manufacturer": "Winery",
            "region": "Region",
            "category": "Category",
        }

    @staticmethod
    def sample(
        sample_id: str,
        expected_slug: str,
        predictions: list[tuple[str, float]],
        latency_ms: float,
    ) -> dict[str, object]:
        return {
            "sample_id": sample_id,
            "expected_slug": expected_slug,
            "predictions": [{"slug": slug, "score": score} for slug, score in predictions],
            "latency_ms": latency_ms,
        }

    @staticmethod
    def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
        path.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
