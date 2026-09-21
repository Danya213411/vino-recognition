# Приложение Stage 6

`app/` — прикладной слой поверх исследовательского пайплайна `stage1…stage5`:

- `backend/vino_api/` — FastAPI, DINOv2 + SIFT + OCR, confidence-модель, SQLite-журнал и сохранение входных кадров;
- `frontend/` — статически экспортируемый Next.js-интерфейс;
- `start.ps1` — сборка фронтенда и запуск единого сервиса на `http://127.0.0.1:8000`.

## Запуск

Из корня проекта:

```powershell
.\app\start.ps1
```

После первой сборки можно пропустить Next.js build:

```powershell
.\app\start.ps1 -SkipBuild
```

Модель запускается с `VINO_LOCAL_FILES_ONLY=1`: во время демонстрации сетевые загрузки не выполняются. Swagger доступен на `/api/docs`.

## Страницы

- `/` — пользовательский сценарий и полноценная карточка распознанного вина в стилистике каталога;
- `/calib` — top‑5, все DINOv2/SIFT/OCR-сигналы, evidence, кропы, тайминги, полноэкранные эталоны, обратная связь и JSON-манифест;
- `/admin` — журнал по 30 записей на страницу, полноэкранные входные кадры, удаление одного скана и подтверждаемая очистка;
- `/admin/detail/?id=…` — полный разбор отдельного распознавания, включая исходный кадр, кандидатов и конфигурацию pipeline.

Если `VINO_ADMIN_TOKEN` не задан, admin API принимает запросы только с loopback-интерфейса. При удалённом запуске токен обязателен.

## HTTP API

| Метод | Endpoint | Назначение |
|---|---|---|
| `GET` | `/api/health` | готовность, устройство и размер каталога |
| `POST` | `/api/v1/recognize` | плоский проверочный контракт `{"slug":"..."}` |
| `POST` | `/api/v1/analyze` | полный результат для UI, multipart `file` + `mode` |
| `POST` | `/api/v1/recognitions/{id}/feedback` | `correct`, `incorrect` или `not_in_catalog` |
| `GET` | `/api/v1/recognitions/{id}/manifest` | манифест файла, инференса, модели, артефактов, таймингов и feedback |
| `GET` | `/api/v1/admin/stats` | сводка admin |
| `GET` | `/api/v1/admin/recognitions` | журнал запусков |
| `GET` | `/api/v1/admin/recognitions/{id}` | полная запись отдельного запуска |
| `GET` | `/api/v1/admin/recognitions/{id}/image` | исходный кадр запуска |
| `DELETE` | `/api/v1/admin/recognitions/{id}` | удалить запись, feedback и входной кадр |
| `DELETE` | `/api/v1/admin/recognitions` | очистить журнал и входные кадры |
| `GET` | `/api/v1/admin/export` | полная JSON-выгрузка журнала |

Пример контракта кейсодержателя:

```powershell
curl.exe --noproxy "*" -H "X-Session-ID: checker-local" -F "file=@photo.jpg" http://127.0.0.1:8000/api/v1/recognize
```

## Последовательный SLA-бенчмарк

При запущенном API:

```powershell
uv run vino-api-benchmark data/synthetic/stage2/images --limit 50 --output data/artifacts/stage6/api_benchmark_ocr_union.json
```

Команда фиксирует wall-clock p50/p95/max и завершается с ошибкой, если хотя бы один кадр превышает 10 секунд.
