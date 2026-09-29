# Архитектура Vino

Документ описывает runtime-пайплайн приложения и границы между слоями. Инструкции установки и запуска находятся в [README.md](README.md).

## Общая схема

```text
Фото пользователя
      │
      ▼
HTTP/FastAPI ── валидация multipart, размера и декодирования
      │
      ▼
Storage ─────── сохранение исходника и SHA-256
      │
      ▼
RecognitionEngine
      │
      ├─ нормализация и views/crops
      ├─ DINOv2 visual retrieval → top-30
      ├─ global OCR retrieval ────┐
      ├─ SIFT + RANSAC ───────────┤ union + rerank
      ├─ RapidOCR ────────────────┘
      └─ confidence + unknown decision
      │
      ├─ карточка вина из локального catalog manifest
      ├─ top-5 и evidence для calib/admin
      └─ SQLite event + feedback + manifest
```

Основной runtime-код находится в `app/backend/vino_api/`. Исследовательские реализации слоёв переиспользуются из `src/stage1`–`src/stage5`, но HTTP-слой не зависит от frontend и может работать отдельно.

## Границы слоёв

### 1. Вход и транспорт

`app/backend/vino_api/app.py` отвечает только за HTTP-контракт и orchestration запроса:

- принимает `multipart/form-data` с полем `file`;
- проверяет `mode=user|calib`, session id и admin-доступ;
- вызывает engine через thread pool, чтобы не блокировать async endpoint;
- возвращает плоский `{"slug":"..."}` для `/api/v1/recognize` или расширенный результат для `/api/v1/analyze`;
- предоставляет совместимый адаптер `/v1/eval/predict` с multipart-полем `image` для скрипта кейсодержателя;
- сохраняет feedback, manifest и административные операции.

Этот слой не вычисляет визуальные признаки и не знает деталей DINO/SIFT/OCR.

### 2. Нормализация фото и storage

`app/backend/vino_api/storage.py` принимает файл, ограничивает размер/число пикселей, декодирует изображение и сохраняет исходный кадр в `VINO_APP_DATA_DIR/uploads`.

Исходник не перезаписывается. Для каждой загрузки сохраняются имя, MIME-тип, размеры, размер в байтах, относительный путь и SHA-256. Нормализация представлений для inference выполняется ниже, внутри CV pipeline; storage остаётся нейтральным к моделям.

### 3. Извлечение признаков

`src/stage3/embeddings.py` и `src/stage4/features.py` извлекают независимые сигналы:

- DINOv2 CLS embedding для нескольких представлений кадра: полный кадр, medium/detail crop и область этикетки;
- SIFT-дескрипторы и локальные соответствия с эталоном;
- OCR-линии RapidOCR с нормализацией текста и оценкой confidence.

Каждый extractor возвращает признаки/оценки, а не карточку вина и не HTTP-ответ. Это позволяет сравнивать модели и повторно использовать уже рассчитанные сигналы в evaluation.

### 4. Поиск по каталогу

Каталог загружается из `data/artifacts/stage1/catalog_manifest.jsonl`. Предварительно рассчитанные артефакты:

- `data/artifacts/stage3/dinov2-small/index.npz` — visual embeddings;
- `data/artifacts/stage4/index/` — SIFT и OCR-индексы;
- `data/artifacts/stage5/confidence_model.json` — веса и пороги решения.

`src/stage3` выполняет быстрый матричный visual retrieval. `src/stage4` добавляет локальную геометрию и текст. В production pipeline visual top-30 объединяется с global OCR top-10. Для производителя используется fuzzy-нормализация: русская/латинская транслитерация, OCR homoglyphs, составные алиасы и безопасное prefix-сопоставление.

После union кандидаты rerank-ятся по комбинации:

```text
итоговый score = visual/DINO + SIFT/geometric + OCR/text evidence
```

Точные веса и пороги не зашиты в frontend: они читаются из confidence model и попадают в manifest конкретного распознавания.

### 5. Выдача решения и карточки

`src/stage5/recognize.py` формирует решение и top-k:

- `match` — достаточно уверенное совпадение;
- `alternatives` — кандидаты есть, но уверенность недостаточна для единственного ответа;
- `not_found` — объект не подтверждён или не проходит unknown-порог.

`app/backend/vino_api/engine.py` добавляет к slug публичные поля из catalog manifest: название, производитель, регион, категорию, цвет, сорта, блюда, описание, крепость, температуру и URL изображения. Frontend получает готовую карточку и не читает внутренние индексы напрямую.

### 6. Confidence, evidence и manifest

Вместе с результатом engine возвращает:

- top-5 predictions;
- visual/OCR/SIFT scores;
- число геометрических inliers;
- OCR-строки и manufacturer match;
- crop consistency;
- thresholds и confidence source;
- timing по стадиям и память процесса.

Манифест `/api/v1/recognitions/{id}/manifest` фиксирует конфигурацию pipeline, хэши ключевых артефактов и полный результат. Это позволяет понять, почему два запуска дали разные ответы, не полагаясь на состояние frontend.

### 7. Хранилище событий и feedback

`app/backend/vino_api/database.py` хранит в SQLite:

- входной кадр и session id;
- время запуска и общий latency;
- выбранный slug и serialised result;
- pipeline manifest;
- feedback `correct`, `incorrect`, `not_in_catalog`, `not_in_store`.

`/calib/`, `/review/` и `/matrix/` используют feedback для ручной проверки. `/admin/` читает тот же журнал с пагинацией и предоставляет удаление отдельной записи или всей выборки.

### 8. Frontend

`app/frontend/` — Next.js с static export. UI разделён по ответственности:

- `features/recognition` — пользовательский сценарий;
- `features/calibration` — диагностический и пакетный review;
- `features/admin` — журнал и detail;
- `entities/wine` — отображение карточки вина;
- `shared/api` — единственный клиент HTTP-контрактов.

В production-like запуске Next.js не является отдельным сервером: `npm run build` создаёт `app/frontend/out`, после чего FastAPI раздаёт статические файлы через `StaticFiles`. В dev-режиме Next.js работает на `:3000`, а API — на `:8000`; адрес API задаётся `NEXT_PUBLIC_API_URL`.

## Контракты между слоями

```text
HTTP upload
  → StoredImage
  → recognize(image_path)
  → RecognitionResult
  → API response / database record
```

На границе engine/API результат содержит `candidate_slug`, `decision`, `confidence`, `predictions`, `evidence`, `ocr_lines`, `timing_ms`, `memory` и `wine`. Плоский endpoint намеренно выбрасывает диагностические поля и возвращает только `slug`, чтобы оставаться совместимым с проверяющим скриптом.

## Offline и внешние границы

Runtime не обращается к сайту «Своё Вино» во время распознавания. Каталог, изображения, индексы и confidence-модель — локальные артефакты. Сайт используется только на этапах сбора/обновления каталога и формирования справочника производителей.

GLM-OCR присутствует в исследовательском benchmark и dual-OCR анализе. Текущий API runtime использует RapidOCR из `create_ocr_engine()`; подключение GLM-OCR как второго inference-ветвления должно оставаться отдельным адаптером, не смешивая его сырые генеративные строки с калиброванным RapidOCR score без повторной валидации.

## Ограничения и точки расширения

- Каталог является снимком и требует отдельного процесса обновления и пересборки индексов.
- Индекс сейчас загружается в память процесса; горизонтальное масштабирование потребует отдельной стратегии хранения и прогрева.
- По умолчанию `VINO_MAX_PARALLEL=1`, так как DINO/SIFT/OCR потребляют заметную память.
- Решение вероятностное: UI обязан показывать отказ/альтернативы, а не превращать любой top-1 в доказанный факт.
- Новые модели следует подключать через отдельный extractor/adapter и сравнивать по тем же validation-наборам, не меняя HTTP-контракт.
