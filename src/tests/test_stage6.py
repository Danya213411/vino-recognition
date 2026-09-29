from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from PIL import Image

from vino_api.app import create_app
from vino_api.config import Settings


class FakeEngine:
    def __init__(self, reference: Path) -> None:
        self.reference = reference
        self.catalog = [{"slug": "test-wine"}, {"slug": "other-wine"}]
        self.catalog_by_slug = {item["slug"]: item for item in self.catalog}
        self.embedder = SimpleNamespace(device="cpu")
        self.pipeline = {"name": "fake-pipeline", "parameters": {"top_k": 2}}

    @staticmethod
    def wine(slug: str) -> dict[str, object]:
        return {
            "slug": slug,
            "title": "Тестовое вино" if slug == "test-wine" else "Другое вино",
            "manufacturer": "Тестовая винодельня",
            "region": "Кубань",
            "category": "Красное сухое",
            "description": "Описание",
            "grapes": ["Красностоп"],
            "dishes": ["Сыр"],
            "alcohol": 13,
            "rating": None,
            "public_rating": 5,
            "source_url": "https://example.test/wine",
            "color_gradient": None,
            "image_url": f"/api/v1/catalog/{slug}/image",
        }

    def recognize(self, image_path: Path) -> dict[str, object]:
        assert image_path.is_file()
        predictions = []
        for slug, score, support in (
            ("test-wine", 0.91, 0.75),
            ("other-wine", 0.72, 0.25),
        ):
            predictions.append(
                {
                    "slug": slug,
                    "score": score,
                    "embedding_score": score,
                    "sift_score": 0.8,
                    "sift_inliers": 20,
                    "ocr_score": 0.7,
                    "relative_support": support,
                    "wine": self.wine(slug),
                }
            )
        return {
            "decision": "match",
            "slug": "test-wine",
            "candidate_slug": "test-wine",
            "confidence": 0.94,
            "thresholds": {"accept": 0.86, "review": 0.37},
            "wine": self.wine("test-wine"),
            "predictions": predictions,
            "evidence": {"top1_margin": 0.19},
            "ocr_lines": [{"text": "TEST", "normalized": "test", "score": 0.9}],
            "crop_consistency": {"top1_agreement": 1.0},
            "timing_ms": {"dino": 100.0, "sift": 20.0, "ocr": 30.0, "total": 150.0, "queue": 0.1, "api_total": 160.0},
            "memory": {"process_rss_bytes_after": 1234},
        }

    def catalog_search(self, query: str, limit: int) -> list[dict[str, object]]:
        return [self.wine(slug) for slug in self.catalog_by_slug if query.casefold() in slug][:limit]

    def catalog_image(self, slug: str) -> Path:
        assert slug in self.catalog_by_slug
        return self.reference


def image_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (160, 240), "#8f3d42").save(output, format="JPEG")
    return output.getvalue()


def test_stage6_full_flow(tmp_path: Path) -> None:
    reference = tmp_path / "reference.jpg"
    reference.write_bytes(image_bytes())
    settings = Settings(
        app_data_dir=tmp_path / "state",
        web_dir=tmp_path / "missing-web",
        admin_token="secret-token",
    )
    app = create_app(settings=settings, engine=FakeEngine(reference))
    session_headers = {"X-Session-ID": "test-session-0001"}
    admin_headers = {"Authorization": "Bearer secret-token"}

    with TestClient(app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["catalog_size"] == 2

        response = client.post(
            "/api/v1/analyze",
            headers=session_headers,
            data={"mode": "calib"},
            files={"file": ("bottle.jpg", image_bytes(), "image/jpeg")},
        )
        assert response.status_code == 200
        analysis = response.json()
        assert analysis["decision"] == "match"
        assert analysis["predictions"][0]["slug"] == "test-wine"

        feedback = client.post(
            f"/api/v1/recognitions/{analysis['id']}/feedback",
            headers=session_headers,
            json={"verdict": "incorrect", "correct_slug": "other-wine"},
        )
        assert feedback.status_code == 200

        manifest = client.get(analysis["manifest_url"], headers=session_headers)
        assert manifest.status_code == 200
        assert manifest.json()["recognition"]["feedback"]["correct_slug"] == "other-wine"
        assert "attachment" in manifest.headers["content-disposition"]

        changed_feedback = client.post(
            f"/api/v1/recognitions/{analysis['id']}/feedback",
            headers=session_headers,
            json={"verdict": "not_in_store"},
        )
        assert changed_feedback.status_code == 200
        assert changed_feedback.json()["verdict"] == "not_in_store"

        flat = client.post(
            "/api/v1/recognize",
            headers=session_headers,
            files={"file": ("bottle.jpg", image_bytes(), "image/jpeg")},
        )
        assert flat.status_code == 200
        assert flat.json() == {"slug": "test-wine"}

        case_eval = client.post(
            "/v1/eval/predict",
            files={"image": ("q-000001.jpg", image_bytes(), "image/jpeg")},
        )
        assert case_eval.status_code == 200
        assert case_eval.json() == {"slug": "test-wine"}

        assert client.get("/api/v1/admin/stats").status_code == 401
        stats = client.get("/api/v1/admin/stats", headers=admin_headers)
        assert stats.status_code == 200
        assert stats.json()["recognitions"]["total"] == 3
        assert stats.json()["feedback"]["not_in_store"] == 1
        rows = client.get("/api/v1/admin/recognitions", headers=admin_headers).json()
        assert rows["total"] == 3
        assert any(
            item.get("feedback")
            and item["feedback"]["verdict"] == "not_in_store"
            for item in rows["items"]
        )


def test_stage6_rejects_invalid_image(tmp_path: Path) -> None:
    reference = tmp_path / "reference.jpg"
    reference.write_bytes(image_bytes())
    settings = Settings(app_data_dir=tmp_path / "state", web_dir=tmp_path / "missing-web")
    app = create_app(settings=settings, engine=FakeEngine(reference))
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/analyze",
            headers={"X-Session-ID": "test-session-0002"},
            files={"file": ("fake.jpg", b"not-an-image", "image/jpeg")},
        )
    assert response.status_code == 415


def test_stage6_lists_local_review_set_safely(tmp_path: Path) -> None:
    reference = tmp_path / "reference.jpg"
    reference.write_bytes(image_bytes())
    review_dir = tmp_path / "given"
    review_dir.mkdir()
    (review_dir / "001 bottle.webp").write_bytes(image_bytes())
    nested = review_dir / "top1_correct"
    nested.mkdir()
    (nested / "002 bottle.webp").write_bytes(image_bytes())
    (review_dir / "ignore.txt").write_text("not an image", encoding="utf-8")
    settings = Settings(
        app_data_dir=tmp_path / "state",
        web_dir=tmp_path / "missing-web",
        review_set_dir=review_dir,
    )
    app = create_app(settings=settings, engine=FakeEngine(reference))

    with TestClient(app) as client:
        listing = client.get("/api/v1/review-set")
        assert listing.status_code == 200
        assert listing.json()["total"] == 2
        assert listing.json()["items"][0]["name"] == "001 bottle.webp"
        assert listing.json()["items"][1]["name"] == "top1_correct/002 bottle.webp"

        image = client.get(
            "/api/v1/review-set/image", params={"name": "001 bottle.webp"}
        )
        assert image.status_code == 200
        assert image.headers["content-type"] == "image/webp"
        assert image.content == image_bytes()

        nested_image = client.get(
            "/api/v1/review-set/image", params={"name": "top1_correct/002 bottle.webp"}
        )
        assert nested_image.status_code == 200

        traversal = client.get(
            "/api/v1/review-set/image", params={"name": "../reference.jpg"}
        )
        assert traversal.status_code == 404


def test_stage6_review_matrix_returns_latest_non_top1_result(tmp_path: Path) -> None:
    reference = tmp_path / "reference.jpg"
    reference.write_bytes(image_bytes())
    review_dir = tmp_path / "given"
    review_dir.mkdir()
    (review_dir / "bottle.jpg").write_bytes(image_bytes())
    settings = Settings(
        app_data_dir=tmp_path / "state",
        web_dir=tmp_path / "missing-web",
        review_set_dir=review_dir,
    )
    app = create_app(settings=settings, engine=FakeEngine(reference))
    headers = {"X-Session-ID": "matrix-session"}

    with TestClient(app) as client:
        first = client.post(
            "/api/v1/analyze",
            headers=headers,
            data={"mode": "calib"},
            files={"file": ("bottle.jpg", image_bytes(), "image/jpeg")},
        ).json()
        client.post(
            f"/api/v1/recognitions/{first['id']}/feedback",
            headers=headers,
            json={"verdict": "correct"},
        )
        second = client.post(
            "/api/v1/analyze",
            headers=headers,
            data={"mode": "calib"},
            files={"file": ("bottle.jpg", image_bytes(), "image/jpeg")},
        ).json()
        client.post(
            f"/api/v1/recognitions/{second['id']}/feedback",
            headers=headers,
            json={"verdict": "incorrect", "correct_slug": "other-wine"},
        )

        matrix = client.get("/api/v1/review-matrix")
        assert matrix.status_code == 200
        payload = matrix.json()
        assert payload["total"] == 1
        assert payload["items"][0]["name"] == "bottle.jpg"
        assert payload["items"][0]["recognition"]["id"] == second["id"]
        assert payload["items"][0]["recognition"]["feedback"]["verdict"] == "incorrect"
        assert payload["items"][0]["input_image_url"].startswith(
            "/api/v1/review-set/image"
        )


def test_stage6_admin_detail_pagination_and_deletion(tmp_path: Path) -> None:
    reference = tmp_path / "reference.jpg"
    reference.write_bytes(image_bytes())
    settings = Settings(
        app_data_dir=tmp_path / "state",
        web_dir=tmp_path / "missing-web",
        admin_token="secret-token",
    )
    app = create_app(settings=settings, engine=FakeEngine(reference))
    session_headers = {"X-Session-ID": "test-session-admin"}
    admin_headers = {"Authorization": "Bearer secret-token"}

    with TestClient(app) as client:
        ids = []
        for index in range(3):
            response = client.post(
                "/api/v1/analyze",
                headers=session_headers,
                data={"mode": "user"},
                files={"file": (f"bottle-{index}.jpg", image_bytes(), "image/jpeg")},
            )
            ids.append(response.json()["id"])

        page = client.get(
            "/api/v1/admin/recognitions?limit=2&offset=2", headers=admin_headers
        )
        assert page.status_code == 200
        assert page.json()["total"] == 3
        assert len(page.json()["items"]) == 1

        detail = client.get(
            f"/api/v1/admin/recognitions/{ids[0]}", headers=admin_headers
        )
        assert detail.status_code == 200
        assert detail.json()["id"] == ids[0]

        deleted = client.delete(
            f"/api/v1/admin/recognitions/{ids[0]}", headers=admin_headers
        )
        assert deleted.status_code == 200
        assert deleted.json()["deleted"] is True
        assert client.get(
            f"/api/v1/admin/recognitions/{ids[0]}", headers=admin_headers
        ).status_code == 404

        cleared = client.delete("/api/v1/admin/recognitions", headers=admin_headers)
        assert cleared.status_code == 200
        assert cleared.json()["deleted"] == 2
        assert client.get(
            "/api/v1/admin/recognitions", headers=admin_headers
        ).json()["total"] == 0
