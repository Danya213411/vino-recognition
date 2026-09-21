from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from stage3.embeddings import prepare_image, query_views
from stage3.index import VisualIndex


class Stage3Tests(unittest.TestCase):
    def test_preprocessing_produces_normalized_model_tensor(self) -> None:
        image = Image.new("RGB", (90, 240), (120, 80, 40))
        tensor = prepare_image(image)
        self.assertEqual(tensor.shape, (3, 518, 518))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(np.isfinite(tensor).all())

    def test_visual_index_returns_sorted_combined_scores(self) -> None:
        index = VisualIndex(
            slugs=np.asarray(["a", "b", "c"]),
            image_paths=np.asarray(["a.webp", "b.webp", "c.webp"]),
            full=np.asarray([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]], dtype=np.float32),
            label=np.asarray([[1.0, 0.0], [0.2, 0.8], [0.0, 1.0]], dtype=np.float32),
        )
        indices, scores = index.search(
            np.asarray([[1.0, 0.0]], dtype=np.float32),
            np.asarray([[0.0, 1.0]], dtype=np.float32),
            top_k=3,
        )
        self.assertEqual(index.slugs[indices[0]].tolist(), ["b", "a", "c"])
        self.assertTrue(np.all(scores[0][:-1] >= scores[0][1:]))

    def test_query_views_accept_real_image_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "query.jpg"
            Image.new("RGB", (200, 100), "white").save(path)
            full, center, detail = query_views(path)
            self.assertEqual(full.size, (200, 100))
            self.assertLess(center.width, full.width)
            self.assertLess(center.height, full.height)
            self.assertLess(detail.width, center.width)
            self.assertLess(detail.height, center.height)

    def test_patch_scores_rank_matching_local_tokens_first(self) -> None:
        index = VisualIndex(
            slugs=np.asarray(["a", "b"]),
            image_paths=np.asarray(["a.webp", "b.webp"]),
            full=np.eye(2, dtype=np.float32),
            label=np.eye(2, dtype=np.float32),
            label_patches=np.asarray(
                [
                    [[1.0, 0.0], [1.0, 0.0]],
                    [[0.0, 1.0], [0.0, 1.0]],
                ],
                dtype=np.float16,
            ),
            patch_grid=1,
        )
        query = np.asarray([[[[0.0, 1.0], [0.0, 1.0]]]], dtype=np.float16)
        candidates = np.asarray([[0, 1]])
        scores = index.patch_scores(query, candidates, "cpu", batch_size=1)
        self.assertGreater(scores[0, 1], scores[0, 0])


if __name__ == "__main__":
    unittest.main()
