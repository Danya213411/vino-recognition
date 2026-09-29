# Полевой прогон после обновления каталога и удаления blur

## Конфигурация

- Каталог: **2035** вин (+10 Литавщук).
- Синтетическая calibration: без standalone blur и без blur в combined.
- Гибридные веса: embedding **0.35**, SIFT **0.40**, OCR **0.25**.

## 52 кадра с точным slug

| Метрика | До | После | Изменение |
|---|---:|---:|---:|
| Top-1 | 34/52 (65.4%) | 30/52 (57.7%) | -4 |
| Top-5 | 52/52 (100.0%) | 48/52 (92.3%) | -4 |

Оставшиеся 14 кадров ранее были отмечены `not_in_catalog`, поэтому для них нет
точного slug и они не включены в accuracy автоматически. Новые кандидаты перечислены ниже.

## 14 ранее неразмеченных промахов

| Файл | Новый Top-1 | Новые карточки Литавщука в Top-5 | OCR |
|---|---|---|---|
| `82.33_27-08-2026_10-03-52.webp` | `leto-risling-2024-polusuhoe-vino` | — | IBE |
| `87.88_28-08-2026_16-56-20.webp` | `massandra-portveyn-belyy-gurzuf-kokur-belyy-beloe-sladkoe-135` | — | ОБЪЕДИ; MACCAHAPA; ПРОИЗВОДСТ; PA; MAC |
| `91.43_02-09-2026_12-08-50.webp` | `merlo-2` | `merlo-2`, `saperavi`, `kaberne-sovinon-3`, `sovinon-blan-2` | Семейная винодельня; Литавщуков; lepлo; сухое красное |
| `93.97_28-08-2026_16-02-48.webp` | `golubitskoe-rose` | — | GOLUBITSKOE; —ESTATE; CHARDONNAY; TAMAGNE; 2024 |
| `95.32_02-09-2026_12-09-23.webp` | `sovinon-blan` | `sovinon-blan`, `merlo-2`, `risling-2`, `sovinon-blan-2`, `pinoy-noir` | Семейная винодельня; Литавщуков; Собиньон; полусладкое; белое |
| `96.44_05-09-2026_15-33-59.webp` | `sovinon-blan` | `sovinon-blan`, `risling-2`, `saperavi`, `merlo-2`, `muskat` | Семейная винодельня; Литавщуков; aH; полусухое белое; фермерское вино |
| `96.55_30-08-2026_19-34-31.webp` | `novyj-svet-risling-kyuve-de-prestizh` | — | 18; 78; НовЫЙ Свъть; Дом Шампанских Вин; ВЫДЕРЖАННОЕ |
| `96.64_26-08-2026_16-52-37.webp` | `massandra-kagor-gurzuf-saperavi-krasnoe-sladkoe-16` | — | АВТОРСКОЕ ВИНО; КАБЕРНЕ; СОВИНЬОН; ЛЮБЛО |
| `96.65_02-09-2026_12-08-51.webp` | `kaberne-sovinon-3` | `kaberne-sovinon-3`, `saperavi`, `merlo-2`, `sovinon-blan` | Семейная винодельня; Литавщуков; биньон; ское вино; 02,5%5 |
| `96.76_07-09-2026_14-57-04.webp` | `pinoy-noir` | `pinoy-noir`, `muskat` | LITAVSHCHUK; VINEYARDS &WINERY; АНАПА, ВИННАЯ ДЕРЕВНЯ; ПОЗДНИЙ |
| `96.79_20-08-2026_17-48-31.webp` | `roze-2` | — | ТАБИЯ; ВИНОДЕЛЬНЯ; 2025; 202; БEЛOE |
| `96.83_20-08-2026_17-48-13.webp` | `roze-2` | — | ТАБИЯ; ВИНОДЕЛЬНЯ; 20; 2025; PO3 |
| `96.8_25-08-2026_22-20-38.webp` | `sovinon-blan` | `sovinon-blan`, `saperavi`, `risling-2`, `merlo-2`, `domashnyaya-gorka` | Семейная винодельня; Литавщуков; Сaвиньон; полусухое белое; фермерское вино |
| `97.23_05-09-2026_15-34-36.webp` | `sovinon-blan` | `sovinon-blan`, `kaberne-sovinon-3`, `sovinon-blan-2`, `saperavi` | 1600; hen; lapgene; Литавщуков; полусухое белое |

## Решения confidence

`alternatives`: 17, `match`: 5, `not_found`: 44

## Скорость

- Среднее: **6.98 с**.
- Медиана: **6.40 с**.
- P95: **10.69 с**.
- Максимум: **16.22 с**.
- Сумма времени распознаваний: **7.7 мин**.
