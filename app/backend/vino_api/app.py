import hmac
import json
import re
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from ipaddress import ip_address
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import quote

import uvicorn
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles

from vino_api.config import Settings
from vino_api.database import EventDatabase, utc_now
from vino_api.engine import RecognitionEngine
from vino_api.schemas import FeedbackRequest, FlatRecognition, HealthResponse
from vino_api.storage import StoredImage, UploadStorage


SESSION_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{8,128}$")
REVIEW_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
REVIEW_IMAGE_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}
bearer_scheme = HTTPBearer(auto_error=False)


def create_app(
    settings: Settings | None = None,
    engine: RecognitionEngine | Any | None = None,
    database: EventDatabase | None = None,
) -> FastAPI:
    resolved = settings or Settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.settings = resolved
        application.state.database = database or EventDatabase(resolved.database_path)
        application.state.database.initialize()
        application.state.storage = UploadStorage(
            resolved.upload_dir, resolved.max_upload_bytes, resolved.max_pixels
        )
        application.state.engine = engine or await run_in_threadpool(RecognitionEngine, resolved)
        yield

    application = FastAPI(
        title="ХАКАТОН — распознавание",
        description="DINOv2 + SIFT + OCR с калиброванным отказом от неуверенного ответа.",
        version="0.2.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Authorization", "Content-Type", "X-Session-ID"],
    )

    def db(request: Request) -> EventDatabase:
        return request.app.state.database

    def current_engine(request: Request) -> Any:
        return request.app.state.engine

    def validated_session_id(value: str | None) -> str:
        result = value or "anonymous-local"
        if not SESSION_PATTERN.fullmatch(result):
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Некорректный X-Session-ID")
        return result

    def admin_access(
        request: Request,
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    ) -> None:
        configured = resolved.admin_token
        if configured:
            supplied = credentials.credentials if credentials else ""
            if not hmac.compare_digest(supplied, configured):
                raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Неверный токен администратора")
            return
        host = request.client.host if request.client else ""
        try:
            local = ip_address(host).is_loopback
        except ValueError:
            local = host in {"localhost", "testclient"}
        if not local:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                "Для удалённого доступа задайте VINO_ADMIN_TOKEN",
            )

    def local_access(request: Request) -> None:
        host = request.client.host if request.client else ""
        try:
            local = ip_address(host).is_loopback
        except ValueError:
            local = host in {"localhost", "testclient"}
        if not local:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                "Пакетная тестовая выборка доступна только локально",
            )

    async def analyze_upload(
        request: Request,
        upload: UploadFile,
        mode: str,
        owner_session: str,
    ) -> dict[str, Any]:
        if mode not in {"user", "calib"}:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "mode: user или calib")
        recognition_id = uuid.uuid4().hex
        stored: StoredImage = await request.app.state.storage.save(upload, recognition_id)
        inference_started = datetime.now(UTC).isoformat()
        result = await run_in_threadpool(request.app.state.engine.recognize, stored.path)
        result["image"] = stored.original_name
        created_at = utc_now()
        response = {
            "id": recognition_id,
            "created_at": created_at,
            "inference_started_at": inference_started,
            "mode": mode,
            "decision": result["decision"],
            "slug": result["slug"],
            "candidate_slug": result["candidate_slug"],
            "confidence": result["confidence"],
            "calibrated_confidence": result.get("calibrated_confidence", result["confidence"]),
            "confidence_source": result.get("confidence_source", "calibrated_model"),
            "verified_match": result.get("verified_match", False),
            "thresholds": result["thresholds"],
            "wine": result["wine"],
            "predictions": result["predictions"],
            "evidence": result["evidence"],
            "ocr_lines": result["ocr_lines"],
            "manufacturer_match": result.get("manufacturer_match"),
            "crop_consistency": result["crop_consistency"],
            "timing_ms": result["timing_ms"],
            "memory": result["memory"],
            "manifest_url": f"/api/v1/recognitions/{recognition_id}/manifest",
        }
        client_host = request.client.host if request.client else None
        record = {
            "id": recognition_id,
            "created_at": created_at,
            "session_id": owner_session,
            "mode": mode,
            "client_host": client_host,
            "user_agent": request.headers.get("user-agent"),
            "original_name": stored.original_name,
            "mime_type": stored.mime_type,
            "byte_size": stored.bytes,
            "width": stored.width,
            "height": stored.height,
            "sha256": stored.sha256,
            "image_path": stored.relative_path,
            "decision": result["decision"],
            "predicted_slug": result["slug"],
            "candidate_slug": result["candidate_slug"],
            "confidence": result["confidence"],
            "total_ms": result["timing_ms"]["api_total"],
            "result_json": json.dumps(response, ensure_ascii=False),
            "pipeline_json": json.dumps(request.app.state.engine.pipeline, ensure_ascii=False),
        }
        await run_in_threadpool(request.app.state.database.insert_recognition, record)
        return response

    @application.get("/api/health", response_model=HealthResponse, tags=["system"])
    async def health(request: Request) -> dict[str, Any]:
        loaded = request.app.state.engine
        return {
            "status": "ok",
            "model_ready": True,
            "catalog_size": len(loaded.catalog),
            "queue_limit": resolved.max_parallel,
            "device": str(loaded.embedder.device),
        }

    @application.post("/api/v1/recognize", response_model=FlatRecognition, tags=["recognition"])
    async def recognize_flat(
        request: Request,
        file: Annotated[UploadFile, File(description="JPEG, PNG или WebP")],
        x_session_id: Annotated[str | None, Header(alias="X-Session-ID")] = None,
    ) -> dict[str, str]:
        owner_session = validated_session_id(x_session_id)
        result = await analyze_upload(request, file, "user", owner_session)
        return {"slug": str(result["candidate_slug"])}

    @application.post("/api/v1/analyze", tags=["recognition"])
    async def analyze(
        request: Request,
        file: Annotated[UploadFile, File(description="JPEG, PNG или WebP")],
        mode: Annotated[str, Form()] = "user",
        x_session_id: Annotated[str | None, Header(alias="X-Session-ID")] = None,
    ) -> dict[str, Any]:
        owner_session = validated_session_id(x_session_id)
        return await analyze_upload(request, file, mode, owner_session)

    @application.get("/api/v1/catalog/search", tags=["catalog"])
    async def catalog_search(
        engine_instance: Annotated[Any, Depends(current_engine)],
        q: Annotated[str, Query(max_length=200)] = "",
        limit: Annotated[int, Query(ge=1, le=50)] = 10,
    ) -> dict[str, Any]:
        items = await run_in_threadpool(engine_instance.catalog_search, q, limit)
        return {"items": items}

    @application.get("/api/v1/catalog/{slug}/image", tags=["catalog"])
    async def catalog_image(
        slug: str, engine_instance: Annotated[Any, Depends(current_engine)]
    ) -> FileResponse:
        if slug not in engine_instance.catalog_by_slug:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Вино не найдено")
        try:
            path = engine_instance.catalog_image(slug)
        except FileNotFoundError as error:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Изображение не найдено") from error
        return FileResponse(path, headers={"Cache-Control": "public, max-age=86400"})

    @application.get(
        "/api/v1/review-set",
        tags=["calibration"],
        dependencies=[Depends(local_access)],
    )
    async def review_set() -> dict[str, Any]:
        root = resolved.review_set_dir.resolve()
        if not root.is_dir():
            return {"total": 0, "items": []}
        files = sorted(
            (
                path
                for path in root.rglob("*")
                if path.is_file() and path.suffix.casefold() in REVIEW_IMAGE_SUFFIXES
            ),
            key=lambda path: path.relative_to(root).as_posix().casefold(),
        )
        return {
            "total": len(files),
            "items": [
                {
                    "name": path.relative_to(root).as_posix(),
                    "size": path.stat().st_size,
                    "image_url": f"/api/v1/review-set/image?name={quote(path.relative_to(root).as_posix(), safe='')}",
                }
                for path in files
            ],
        }

    @application.get(
        "/api/v1/review-set/image",
        tags=["calibration"],
        dependencies=[Depends(local_access)],
    )
    async def review_set_image(
        name: Annotated[str, Query(min_length=1, max_length=500)],
    ) -> FileResponse:
        root = resolved.review_set_dir.resolve()
        path = (root / name).resolve()
        if (
            root not in path.parents
            or path.suffix.casefold() not in REVIEW_IMAGE_SUFFIXES
            or not path.is_file()
        ):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Кадр не найден")
        return FileResponse(
            path,
            media_type=REVIEW_IMAGE_MEDIA_TYPES[path.suffix.casefold()],
            headers={"Cache-Control": "private, max-age=3600"},
        )

    @application.get(
        "/api/v1/review-matrix",
        tags=["calibration"],
        dependencies=[Depends(local_access)],
    )
    async def review_matrix(
        database_instance: Annotated[EventDatabase, Depends(db)],
    ) -> dict[str, Any]:
        root = resolved.review_set_dir.resolve()
        if not root.is_dir():
            return {"total": 0, "items": []}
        names = sorted(
            (
                path.relative_to(root).as_posix()
                for path in root.rglob("*")
                if path.is_file() and path.suffix.casefold() in REVIEW_IMAGE_SUFFIXES
            ),
            key=str.casefold,
        )
        records = await run_in_threadpool(database_instance.latest_review_records, names)
        records_by_name = {record["original_name"]: record for record in records}
        items = []
        for name in names:
            record = records_by_name.get(name)
            if not record or record.get("feedback", {}).get("verdict") == "correct":
                continue
            items.append(
                {
                    "name": name,
                    "input_image_url": f"/api/v1/review-set/image?name={quote(name, safe='')}",
                    "recognition": record,
                }
            )
        return {"total": len(items), "items": items}

    def owned_record(
        recognition_id: str,
        owner_session: str,
        request: Request,
        database_instance: EventDatabase,
        credentials: HTTPAuthorizationCredentials | None,
    ) -> dict[str, Any]:
        record = database_instance.get_recognition(recognition_id)
        if not record:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Распознавание не найдено")
        is_owner = record["session_id"] == owner_session
        is_admin = bool(
            resolved.admin_token
            and credentials
            and hmac.compare_digest(credentials.credentials, resolved.admin_token)
        )
        if not is_owner and not is_admin:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Нет доступа к этому результату")
        return record

    @application.post("/api/v1/recognitions/{recognition_id}/feedback", tags=["calibration"])
    async def feedback(
        recognition_id: str,
        payload: FeedbackRequest,
        request: Request,
        database_instance: Annotated[EventDatabase, Depends(db)],
        engine_instance: Annotated[Any, Depends(current_engine)],
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
        x_session_id: Annotated[str | None, Header(alias="X-Session-ID")] = None,
    ) -> dict[str, Any]:
        owner_session = validated_session_id(x_session_id)
        owned_record(recognition_id, owner_session, request, database_instance, credentials)
        if payload.correct_slug and payload.correct_slug not in engine_instance.catalog_by_slug:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "correct_slug отсутствует в каталоге")
        return await run_in_threadpool(
            database_instance.upsert_feedback,
            recognition_id,
            payload.verdict,
            payload.correct_slug,
            payload.note,
        )

    @application.get("/api/v1/recognitions/{recognition_id}/manifest", tags=["calibration"])
    async def manifest(
        recognition_id: str,
        request: Request,
        database_instance: Annotated[EventDatabase, Depends(db)],
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
        x_session_id: Annotated[str | None, Header(alias="X-Session-ID")] = None,
    ) -> JSONResponse:
        owner_session = validated_session_id(x_session_id)
        record = owned_record(recognition_id, owner_session, request, database_instance, credentials)
        payload = {
            "schema_version": 1,
            "exported_at": utc_now(),
            "recognition": record,
        }
        return JSONResponse(
            payload,
            headers={
                "Content-Disposition": f'attachment; filename="vino-manifest-{recognition_id}.json"'
            },
        )

    @application.get(
        "/api/v1/admin/stats", tags=["admin"], dependencies=[Depends(admin_access)]
    )
    async def admin_stats(database_instance: Annotated[EventDatabase, Depends(db)]) -> dict[str, Any]:
        return await run_in_threadpool(database_instance.stats)

    @application.get(
        "/api/v1/admin/recognitions", tags=["admin"], dependencies=[Depends(admin_access)]
    )
    async def admin_recognitions(
        database_instance: Annotated[EventDatabase, Depends(db)],
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
        offset: Annotated[int, Query(ge=0)] = 0,
        mode: Annotated[str | None, Query(pattern="^(user|calib)$")] = None,
        decision: Annotated[
            str | None, Query(pattern="^(match|alternatives|not_found)$")
        ] = None,
    ) -> dict[str, Any]:
        return await run_in_threadpool(
            database_instance.list_recognitions, limit, offset, mode, decision
        )

    @application.get(
        "/api/v1/admin/recognitions/{recognition_id}",
        tags=["admin"],
        dependencies=[Depends(admin_access)],
    )
    async def admin_recognition(
        recognition_id: str, database_instance: Annotated[EventDatabase, Depends(db)]
    ) -> dict[str, Any]:
        record = await run_in_threadpool(database_instance.get_recognition, recognition_id)
        if not record:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Распознавание не найдено")
        return record

    @application.get(
        "/api/v1/admin/recognitions/{recognition_id}/image",
        tags=["admin"],
        dependencies=[Depends(admin_access)],
    )
    async def admin_image(
        recognition_id: str, database_instance: Annotated[EventDatabase, Depends(db)]
    ) -> FileResponse:
        record = database_instance.get_recognition(recognition_id)
        if not record:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Распознавание не найдено")
        path = (resolved.app_data_dir / record["image_path"]).resolve()
        if resolved.app_data_dir.resolve() not in path.parents or not path.is_file():
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Фотография не найдена")
        return FileResponse(path, media_type=record["mime_type"])

    @application.delete(
        "/api/v1/admin/recognitions/{recognition_id}",
        tags=["admin"],
        dependencies=[Depends(admin_access)],
    )
    async def admin_delete_recognition(
        recognition_id: str, database_instance: Annotated[EventDatabase, Depends(db)]
    ) -> dict[str, Any]:
        record = await run_in_threadpool(database_instance.get_recognition, recognition_id)
        if not record:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Распознавание не найдено")
        deleted = await run_in_threadpool(database_instance.delete_recognition, recognition_id)
        path = (resolved.app_data_dir / record["image_path"]).resolve()
        removed_image = False
        if resolved.app_data_dir.resolve() in path.parents and path.is_file():
            await run_in_threadpool(path.unlink)
            removed_image = True
        return {"deleted": deleted, "id": recognition_id, "image_deleted": removed_image}

    @application.delete(
        "/api/v1/admin/recognitions",
        tags=["admin"],
        dependencies=[Depends(admin_access)],
    )
    async def admin_clear_recognitions(
        database_instance: Annotated[EventDatabase, Depends(db)],
    ) -> dict[str, Any]:
        records = await run_in_threadpool(
            database_instance.list_recognitions, 1_000_000, 0, None, None
        )
        deleted = await run_in_threadpool(database_instance.clear_recognitions)
        removed_images = 0
        app_data_root = resolved.app_data_dir.resolve()
        for record in records["items"]:
            path = (resolved.app_data_dir / record["image_path"]).resolve()
            if app_data_root in path.parents and path.is_file():
                await run_in_threadpool(path.unlink)
                removed_images += 1
        return {"deleted": deleted, "images_deleted": removed_images}

    @application.get(
        "/api/v1/admin/export", tags=["admin"], dependencies=[Depends(admin_access)]
    )
    async def admin_export(database_instance: Annotated[EventDatabase, Depends(db)]) -> JSONResponse:
        records = await run_in_threadpool(
            database_instance.list_recognitions, 1_000_000, 0, None, None
        )
        return JSONResponse(
            {
                "schema_version": 1,
                "exported_at": utc_now(),
                "pipeline": application.state.engine.pipeline,
                **records,
            },
            headers={"Content-Disposition": 'attachment; filename="vino-admin-export.json"'},
        )

    if resolved.web_dir.is_dir():
        application.mount("/", StaticFiles(directory=resolved.web_dir, html=True), name="web")
    else:
        @application.get("/", include_in_schema=False)
        async def missing_frontend() -> dict[str, str]:
            return {
                "service": "vino-api",
                "frontend": "not built; run npm install && npm run build in app/frontend",
                "docs": "/api/docs",
            }

    return application


app = create_app()


def main() -> None:
    uvicorn.run(
        "vino_api.app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        access_log=True,
    )


if __name__ == "__main__":
    main()
