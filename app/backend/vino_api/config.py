from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _path_env(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value).expanduser().resolve() if value else default.resolve()


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    visual_index: Path = _path_env(
        "VINO_VISUAL_INDEX",
        PROJECT_ROOT / "data/artifacts/stage3/dinov2-small/index.npz",
    )
    hybrid_index_dir: Path = _path_env(
        "VINO_HYBRID_INDEX_DIR", PROJECT_ROOT / "data/artifacts/stage4/index"
    )
    catalog_manifest: Path = _path_env(
        "VINO_CATALOG_MANIFEST",
        PROJECT_ROOT / "data/artifacts/stage1/catalog_manifest.jsonl",
    )
    confidence_model: Path = _path_env(
        "VINO_CONFIDENCE_MODEL",
        PROJECT_ROOT / "data/artifacts/stage5/confidence_model.json",
    )
    app_data_dir: Path = _path_env("VINO_APP_DATA_DIR", PROJECT_ROOT / "data/app/stage6")
    review_set_dir: Path = _path_env(
        "VINO_REVIEW_SET_DIR", PROJECT_ROOT / "data/test/recheck_fuzzy_v2/images"
    )
    web_dir: Path = _path_env("VINO_WEB_DIR", PROJECT_ROOT / "app/frontend/out")
    device: str = os.getenv("VINO_DEVICE", "auto")
    local_files_only: bool = os.getenv("VINO_LOCAL_FILES_ONLY", "1") != "0"
    max_upload_bytes: int = int(os.getenv("VINO_MAX_UPLOAD_BYTES", str(15 * 1024 * 1024)))
    max_pixels: int = int(os.getenv("VINO_MAX_PIXELS", "40000000"))
    top_k: int = int(os.getenv("VINO_TOP_K", "5"))
    candidate_count: int = int(os.getenv("VINO_CANDIDATE_COUNT", "30"))
    ocr_candidate_count: int = int(os.getenv("VINO_OCR_CANDIDATE_COUNT", "10"))
    max_parallel: int = int(os.getenv("VINO_MAX_PARALLEL", "1"))
    admin_token: str | None = os.getenv("VINO_ADMIN_TOKEN") or None
    cors_origins: tuple[str, ...] = tuple(
        item.strip()
        for item in os.getenv(
            "VINO_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
        ).split(",")
        if item.strip()
    )

    @property
    def database_path(self) -> Path:
        return self.app_data_dir / "events.sqlite3"

    @property
    def upload_dir(self) -> Path:
        return self.app_data_dir / "uploads"
