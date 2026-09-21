# Этап 6. API и проверочный интерфейс

Этап собран как одно локальное приложение в [`app/`](../app/README.md). Python-бэкенд загружает существующий Stage 5 pipeline в память один раз, ограничивает GPU-инференс одним параллельным запросом и раздаёт статическую сборку Next.js с того же origin.

## Контракты

Проверочный endpoint сохраняет обязательный плоский формат:

```json
{"slug":"wine-slug"}
```

Расширенный endpoint отделён: он возвращает решение `match / alternatives / not_found`, top‑5, калиброванную уверенность, DINOv2/SIFT/OCR evidence, crop-consistency, wall/inference timings, память и карточки каталога.

## Валидация и эксплуатация

- только декодируемые JPEG, PNG и WebP;
- лимит по умолчанию: 15 МБ и 40 мегапикселей;
- имя файла не участвует в пути хранения;
- снимки адресуются UUID + SHA‑256;
- события и feedback лежат отдельно от каталога в `data/app/stage6/events.sqlite3`;
- admin по умолчанию доступен только с loopback, для удалённого доступа требуется `VINO_ADMIN_TOKEN`;
- удаление одной записи и очистка журнала удаляют также связанные feedback и файлы входных кадров;
- frontend хранит случайный псевдонимный session ID, а не имя или email пользователя;
- модель запускается с `local_files_only`, без сетевой загрузки на демонстрации.

## Манифест

Индивидуальный манифест содержит метаданные и SHA‑256 входного файла, полный результат, OCR-строки, scores каждого кандидата, evidence confidence-модели, все тайминги, RSS/VRAM, параметры pipeline, пороги, веса и SHA‑256 ключевых артефактов, а также сохранённую обратную связь.

## Проверка

- backend unit/integration: `uv run pytest -q` (21 тест, включая detail/pagination/delete/clear);
- frontend lint: `npm run lint` в `app/frontend`;
- static build: `npm run build` в `app/frontend`;
- API SLA: `uv run vino-api-benchmark ...`;
- Swagger: `http://127.0.0.1:8000/api/docs`.

После первого полевого теста retrieval изменён: visual top‑30 объединяется с global OCR top‑10 до SIFT rerank. Это исправляет случаи, когда правильная бутылка имеет точный текстовый и геометрический матч, но DINO ставит её немного ниже визуального cutoff. Консервативный `multimodal_verification` принимается только при OCR ≥ 0,92, SIFT ≥ 0,72, не менее 12 inliers и title evidence ≥ 0,82.

Актуальный последовательный прогон 10 synthetic field-кадров на RTX 4060 дал p50 **1 500 мс**, p95/max **2 387 мс**; 10/10 запросов уложились в лимит 10 секунд. Машиночитаемый отчёт сохранён в `data/artifacts/stage6/api_benchmark_ocr_union.json`.
