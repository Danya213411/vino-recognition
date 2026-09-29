# Vino

Локальный сервис распознавания российских вин по фотографии. README описывает только установку, запуск, конфигурацию и ограничения. Устройство системы и границы слоёв вынесены в [ARCHITECTURE.md](ARCHITECTURE.md).

## Требования

- Windows 10/11 и PowerShell;
- Python 3.12+;
- [uv](https://docs.astral.sh/uv/);
- Node.js 20+ и npm;
- CUDA/GPU рекомендуется для DINOv2 и OCR, но CPU-режим также поддерживается.

Проверка окружения:

```powershell
python --version
uv --version
node --version
npm --version
```

## Быстрый запуск

Из корня репозитория:

```powershell
cd D:\Document\Programing\olimp\Vino
uv sync
.\app\start.ps1
```

`app/start.ps1` устанавливает npm-зависимости при необходимости, собирает frontend в `app/frontend/out`, включает локальный режим моделей и запускает FastAPI на `http://127.0.0.1:8000`.

Открыть:

- приложение: <http://127.0.0.1:8000>;
- Swagger: <http://127.0.0.1:8000/api/docs>;
- health-check: <http://127.0.0.1:8000/api/health>.

Остановка — `Ctrl+C`.

Если frontend уже собран:

```powershell
.\app\start.ps1 -SkipBuild
```

## Режим разработки

Backend и Next.js запускаются в двух окнах PowerShell.

Окно 1:

```powershell
uv run vino-api
```

Окно 2:

```powershell
cd app/frontend
npm install
$env:NEXT_PUBLIC_API_URL = "http://127.0.0.1:8000"
npm run dev
```

Frontend будет доступен на <http://localhost:3000>.

## Переменные окружения

Шаблон: [app/.env.example](app/.env.example). Он не загружается автоматически, поэтому переменные задаются в окружении PowerShell до запуска.

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `VINO_DEVICE` | `auto` | `auto`, `cuda` или `cpu` |
| `VINO_LOCAL_FILES_ONLY` | `1` | запрет загрузки моделей из сети во время запуска |
| `VINO_ADMIN_TOKEN` | пусто | bearer-токен для admin при удалённом доступе |
| `VINO_MAX_PARALLEL` | `1` | число одновременных inference-запросов |
| `VINO_MAX_UPLOAD_BYTES` | `15728640` | максимальный размер фотографии |
| `VINO_MAX_PIXELS` | `40000000` | максимальное число пикселей |
| `VINO_TOP_K` | `5` | число кандидатов в расширенном ответе |
| `VINO_CANDIDATE_COUNT` | `30` | visual-кандидаты перед rerank |
| `VINO_OCR_CANDIDATE_COUNT` | `10` | кандидаты из глобального OCR-поиска |
| `VINO_REVIEW_SET_DIR` | `data/test/recheck_fuzzy_v2/images` | папка для `/review/` и `/matrix/` |
| `VINO_APP_DATA_DIR` | `data/app/stage6` | SQLite-журнал и загруженные кадры |
| `VINO_WEB_DIR` | `app/frontend/out` | собранный frontend |
| `NEXT_PUBLIC_API_URL` | пусто | URL API только в dev-режиме frontend |

Для удалённого admin:

```powershell
$env:VINO_ADMIN_TOKEN = "replace-with-a-long-random-token"
.\app\start.ps1 -SkipBuild
```

Без токена admin API разрешён только с loopback-интерфейса.

## Основные страницы

- `/` — пользовательская загрузка фото и карточка результата;
- `/calib/` — top-5, диагностические сигналы, feedback и manifest;
- `/review/` — пакетная ручная разметка тестовых кадров;
- `/matrix/` — визуальная матрица ошибок;
- `/admin/` — журнал запусков, пагинация, удаление и очистка;
- `/admin/detail/?id=...` — подробности выбранного запуска.

## API-проверка

Плоский контракт для внешнего проверяющего скрипта:

```powershell
curl.exe --noproxy "*" -H "X-Session-ID: local-check" `
  -F "file=@photo.jpg" `
  http://127.0.0.1:8000/api/v1/recognize
```

Ответ имеет вид:

```json
{"slug":"wine-slug"}
```

Расширенный ответ для calibration/UI:

```powershell
curl.exe --noproxy "*" -H "X-Session-ID: local-check" `
  -F "mode=calib" -F "file=@photo.jpg" `
  http://127.0.0.1:8000/api/v1/analyze
```

Для скрипта кейсодержателя предусмотрен совместимый endpoint:

```text
POST http://127.0.0.1:8000/v1/eval/predict
multipart field: image
response: {"slug":"wine-slug"}
```

В репозитории есть тот же runner — [participant_test.sh](participant_test.sh). Контрольные `queries/` и `queries.tsv` берутся из архива кейсодержателя и не входят в код приложения.

```bash
chmod +x ./participant_test.sh
./participant_test.sh \
  --images-dir ./queries \
  --manifest ./queries.tsv \
  --endpoint 'http://127.0.0.1:8000/v1/eval/predict' \
  --output ./predictions.jsonl
```

Скрипт отправляет кадры строго последовательно, без retry и параллелизма, ограничивает каждый запрос 10 секундами и записывает `query_id`, путь, SHA-256, `predicted_slug` и `latency_ms`. Правильные ответы и итоговый score находятся только у организатора.

Полный список endpoints доступен в Swagger. Для локального API-бенчмарка:

```powershell
uv run vino-api-benchmark data/synthetic/stage2/images `
  --limit 50 `
  --output data/artifacts/stage6/api_benchmark.json
```

## Проверки

```powershell
uv run pytest -q
Push-Location app/frontend
npm run lint
npm run build
Pop-Location
git diff --check
```

## Ограничения

1. Для запуска нужны локальные артефакты: `data/artifacts/stage1/catalog_manifest.jsonl`, visual index stage 3, hybrid index stage 4 и confidence model stage 5. Если они отсутствуют, приложение не сможет инициализировать engine.
2. В local-only режиме модели не скачиваются. Кэш Transformers нужно подготовить заранее, временно разрешив сетевой доступ.
3. Каталог и изображения являются локальным снимком; обновление сайта «Своё Вино» автоматически не выполняется.
4. Основной API слушает только `127.0.0.1:8000`; публикация наружу требует reverse proxy, настройки CORS и `VINO_ADMIN_TOKEN`.
5. CPU-режим значительно медленнее CUDA. Параллелизм по умолчанию равен `1`, чтобы не переполнять память.
6. `/review/` и `/matrix/` предназначены для локального доступа и читают фотографии из `VINO_REVIEW_SET_DIR`.
7. Результат распознавания вероятностный: `match` не означает доказанную идентичность бутылки. Для сомнительных случаев доступны `alternatives`/`not_found` и ручная калибровка.

## Дополнительные материалы

- [ARCHITECTURE.md](ARCHITECTURE.md) — pipeline и границы слоёв;
- [ROADMAP.md](ROADMAP.md) — этапы проекта;
- [docs/STAGE_1.md](docs/STAGE_1.md) — [docs/STAGE_6.md](docs/STAGE_6.md) — исследовательские этапы;
- [docs/OCR_BENCHMARK_65.md](docs/OCR_BENCHMARK_65.md) — сравнение OCR на 65 полевых кадрах.
