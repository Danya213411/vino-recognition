"""Validated, content-addressed storage for submitted photographs."""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError


ALLOWED_FORMATS = {
    "JPEG": ("image/jpeg", ".jpg"),
    "PNG": ("image/png", ".png"),
    "WEBP": ("image/webp", ".webp"),
}


@dataclass(frozen=True)
class StoredImage:
    path: Path
    relative_path: str
    original_name: str
    mime_type: str
    image_format: str
    width: int
    height: int
    bytes: int
    sha256: str

    def manifest(self) -> dict[str, object]:
        return {
            "original_name": self.original_name,
            "mime_type": self.mime_type,
            "format": self.image_format,
            "width": self.width,
            "height": self.height,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "stored_as": self.relative_path,
        }


class UploadStorage:
    def __init__(self, root: Path, max_upload_bytes: int, max_pixels: int) -> None:
        self.root = root.resolve()
        self.max_upload_bytes = max_upload_bytes
        self.max_pixels = max_pixels
        self.root.mkdir(parents=True, exist_ok=True)

    async def save(self, upload: UploadFile, recognition_id: str) -> StoredImage:
        payload = await upload.read(self.max_upload_bytes + 1)
        await upload.close()
        if not payload:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Файл изображения пуст")
        if len(payload) > self.max_upload_bytes:
            raise HTTPException(
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                f"Изображение больше {self.max_upload_bytes // (1024 * 1024)} МБ",
            )
        try:
            with Image.open(io.BytesIO(payload)) as image:
                image_format = str(image.format or "").upper()
                width, height = image.size
                image.verify()
        except (UnidentifiedImageError, OSError, ValueError) as error:
            raise HTTPException(
                status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                "Файл не является корректным JPEG, PNG или WebP",
            ) from error
        if image_format not in ALLOWED_FORMATS:
            raise HTTPException(
                status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                f"Формат {image_format or 'не определён'} не поддерживается",
            )
        if width < 64 or height < 64:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Изображение слишком маленькое")
        if width * height > self.max_pixels:
            raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "Слишком большое разрешение")

        mime_type, suffix = ALLOWED_FORMATS[image_format]
        digest = hashlib.sha256(payload).hexdigest()
        date_dir = self.root / recognition_id[:4]
        date_dir.mkdir(parents=True, exist_ok=True)
        destination = (date_dir / f"{recognition_id}-{digest[:12]}{suffix}").resolve()
        if self.root not in destination.parents:
            raise RuntimeError("Unsafe upload destination")
        destination.write_bytes(payload)
        return StoredImage(
            path=destination,
            relative_path=destination.relative_to(self.root.parent).as_posix(),
            original_name=Path(upload.filename or "photo").name,
            mime_type=mime_type,
            image_format=image_format,
            width=width,
            height=height,
            bytes=len(payload),
            sha256=digest,
        )
