# Этап 1: аудит и оценка

На первом этапе исходный каталог остаётся неизменным. Все производные файлы создаются в `data/artifacts/`.

## Установка окружения

```powershell
uv sync
```

## Аудит каталога

```powershell
uv run vino-audit
```

По умолчанию команда читает:

- `data/parser/wines.json`;
- `data/parser/wines.sqlite`;
- `data/parser/images/`.

Результаты сохраняются в `data/artifacts/stage1/`:

| Файл | Назначение |
|---|---|
| `catalog_manifest.jsonl` | по одной нормализованной записи на каждый `slug` |
| `audit_report.json` | полный машиночитаемый отчёт |
| `audit_report.md` | краткая сводка для человека |
| `quality_issues.jsonl` | ошибки и предупреждения по записям |
| `exact_duplicate_groups.json` | группы одинаковых по SHA-256 файлов |
| `duplicate_title_groups.json` | повторяющиеся названия |
| `near_duplicate_groups.json` | кандидатные визуально близкие группы |
| `near_duplicate_pairs.jsonl` | пары с perceptual distance |

Порог поиска визуально близких изображений можно изменить:

```powershell
uv run vino-audit --near-distance 6
```

Near-duplicate поиск на этом этапе является эвристикой. Он сравнивает область предполагаемой этикетки только внутри одного производителя. Окончательно сложные группы будут подтверждаться ошибками распознавателя на следующих этапах.

## Формат evaluation-набора

Evaluator принимает JSONL: одна строка соответствует одной фотографии.

```json
{"sample_id":"photo-001","expected_slug":"cabernet-franc-pinot-noir-2022","predictions":[{"slug":"cabernet-franc-pinot-noir-2021","score":0.91},{"slug":"cabernet-franc-pinot-noir-2022","score":0.89}],"latency_ms":184.2}
```

Обязательные поля:

- `expected_slug` — правильный `slug` из manifest;
- `predictions` — ранжированный массив кандидатов.

Рекомендуемые поля:

- `sample_id` или `image` — идентификатор фотографии;
- `score` у каждого кандидата — нужен для расчёта margin;
- `latency_ms` — полное время обработки фотографии.

Допустим и компактный вариант:

```json
{"image":"queries/photo-001.jpg","expected_slug":"wine-a","predictions":["wine-a","wine-b"],"scores":[0.92,0.71],"latency_ms":140}
```

Повторяющийся `slug` внутри одного списка учитывается только один раз. Неизвестные predicted-slug попадают в отчёт, а неизвестный `expected_slug` считается ошибкой входных данных.

## Запуск evaluator

```powershell
uv run vino-evaluate `
  --predictions data/queries/public/predictions.jsonl `
  --output-dir data/artifacts/evaluation/public
```

Готовый пример входного файла находится в `data/examples/predictions.example.jsonl`. Его можно проверить командой:

```powershell
uv run vino-evaluate `
  --predictions data/examples/predictions.example.jsonl `
  --output-dir data/artifacts/evaluation/example
```

Результаты:

- `evaluation_report.json` — общие метрики, confidence, latency и breakdown;
- `evaluation_errors.jsonl` — все ошибки top-1 вместе с top-5 кандидатами.

Основные метрики:

- `top1_accuracy`;
- `top5_recall`;
- `mean_reciprocal_rank`;
- `mean_top1_margin` и `p05_top1_margin`;
- `p50`, `p95`, `p99` времени ответа;
- количество правильных `slug`, отсутствующих в списке кандидатов.

## Тесты

```powershell
uv run python -m unittest discover -s src/tests -v
```

Тесты проверяют генерацию manifest, обнаружение точных дубликатов и расчёт ранжированных метрик.
