# Fuzzy manufacturer gate — полевой прогон 67 кадров

## Итог

- Обработано: **67** кадров.
- Fuzzy manufacturer gate уверенно сработал: **43** кадров.
- Top‑1 изменился: **25** кадров.
- Из 30 старых `top5_miss` выдача изменилась на **19** кадрах.

## Метрики на 37 кадрах с точным expected_slug

| Метрика |             До |         После | Изменение |
| ------- | -------------: | ------------: | --------: |
| Top‑1   |  27/37 (73.0%) | 25/37 (67.6%) |        -2 |
| Top‑5   | 37/37 (100.0%) | 33/37 (89.2%) |        -4 |

30 кадров `top5_miss` пока не участвуют в accuracy: для них не указан
точный `expected_slug`. Их изменения перечислены ниже для ручной проверки.

## Срабатывания по производителям

- Табия: 13
- Denisov Winery: 7
- ESSE: 5
- АРАТТИ: 4
- Поместье Голубицкое: 4
- Усадьба Дивноморское: 4
- Кубань-Вино: 2
- Реликта: 1
- Новый Свет. Дом шампанских вин: 1
- Усадьба Мезыбь: 1
- Ведерниковъ: 1

## Изменившиеся кадры

| Файл                             | Группа         | До                                                                               | После                                            | Manufacturer gate                    |
| -------------------------------- | -------------- | -------------------------------------------------------------------------------- | ------------------------------------------------ | ------------------------------------ |
| `70.45_04-09-2026_13-35-37.webp` | `top5_miss`    | `zmv-koktebel-portveyn-krasnoe-kreplyonoe-kaberne-sovinon-sladkoe-17`            | `esse-merlo-krasnoe-suhoe-135`                   | ESSE (100%)                          |
| `87.09_22-08-2026_20-57-16.webp` | `top5_miss`    | `vinodelnya-pokrovskaya-pokrovskoe-krasnoe-kaberne-sovinon-suhoe-14`             | `rubin-golodrigi`                                | Табия (100%)                         |
| `87.54_03-09-2026_13-27-53.webp` | `top5_correct` | `artvin-odesskiy-chernyy-rezerv-krasnoe-suhoe-13`                                | `relikta-relikta-shardone-beloe-suhoe-12`        | Реликта (99%)                        |
| `92.96_19-08-2026_21-07-12.webp` | `top1_correct` | `aratti-kaberne-sovinon-2020-krasnoe-suhoe`                                      | `makitra-selection`                              | Кубань-Вино (100%)                   |
| `93.31_05-09-2026_13-39-37.webp` | `top1_correct` | `golubitskoe-estate-noble-selection-red-blend-kaberne-sovinon-krasnoe-suhoe-136` | `golubitskoe-rose`                               | Поместье Голубицкое (100%)           |
| `93.97_28-08-2026_16-02-48.webp` | `top5_miss`    | `kuban-vino-shato-tamane-rezerv-merlo-limited-edishn-2018-krasnoe-suhoe-125`     | `golubitskoe-rose`                               | Поместье Голубицкое (100%)           |
| `94.02_24-08-2026_17-46-57.webp` | `top5_miss`    | `vinodelnya-batrak-prikumskoe-vitale-kraft-kaberne-sovinon-krasnoe-suhoe-135`    | `rubin-golodrigi`                                | Табия (100%)                         |
| `94.55_02-09-2026_16-53-08.webp` | `top1_correct` | `denisov-winery-pino-nuar-shardone-ekstra-bryut-beloe-125`                       | `denisov_pazori_risling`                         | Denisov Winery (99%)                 |
| `95.03_04-09-2026_11-39-35.webp` | `top5_correct` | `vinodelnya-batrak-perfekt-klassik-muskat-belyy-beloe-suhoe-125`                 | `esse-kaberne-fran-krasnoe-suhoe-13`             | ESSE (100%)                          |
| `95.63_24-08-2026_17-47-43.webp` | `top5_miss`    | `vinodelnya-batrak-perfekt-klassik-saperavi-krasnoe-suhoe-135`                   | `rubin-golodrigi`                                | Табия (99%)                          |
| `95.88_02-09-2026_14-41-59.webp` | `top5_miss`    | `vinodelnya-batrak-perfekt-klassik-muskat-belyy-beloe-suhoe-125`                 | `esse-rose-sira-rozovoe-suhoe-12`                | ESSE (98%)                           |
| `95.8_19-08-2026_19-33-23.webp`  | `top5_miss`    | `leto-muskat-oranzh-2024-suhoe-vino`                                             | `czitronnyj-magaracha`                           | Табия (100%)                         |
| `96.07_28-08-2026_16-10-06.webp` | `top5_miss`    | `chateau-tamagne-signature-kaberne`                                              | `golubitskoe-rose`                               | Поместье Голубицкое (99%)            |
| `96.13_19-08-2026_19-33-10.webp` | `top5_correct` | `leto-risling-2024`                                                              | `roze-2`                                         | Табия (99%)                          |
| `96.15_22-08-2026_20-56-09.webp` | `top5_miss`    | `leto-muskat-oranzh-2024-suhoe-vino`                                             | `rozovoe-zoloto`                                 | Табия (99%)                          |
| `96.31_21-08-2026_17-24-55.webp` | `top5_miss`    | `vinodelnya-pokrovskaya-ryzhest-rkatsiteli-beloe-suhoe-11`                       | `czitronnyj-magaracha`                           | Табия (100%)                         |
| `96.45_19-08-2026_19-35-28.webp` | `top5_miss`    | `leto-muskat-oranzh-2024-suhoe-vino`                                             | `risling-1`                                      | Табия (100%)                         |
| `96.55_30-08-2026_19-34-31.webp` | `top5_miss`    | `sary-pandas-vyderzhannoe`                                                       | `novyj-svet-pino-gri-millezim`                   | Новый Свет. Дом шампанских вин (96%) |
| `96.58_05-09-2026_16-05-04.webp` | `top5_miss`    | `koktebel-casa-belyj-kupazh`                                                     | `esse-rose-sira-rozovoe-suhoe-12`                | ESSE (100%)                          |
| `96.62_07-09-2026_18-12-27.webp` | `top5_miss`    | `zmv-koktebel-shardone-beloe-suhoe-12`                                           | `esse-kaberne-fran-krasnoe-suhoe-13`             | ESSE (100%)                          |
| `96.78_25-08-2026_18-04-42.webp` | `top5_miss`    | `vinodelnya-batrak-perfekt-klassik-merlo-krasnoe-suhoe-135`                      | `czitronnyj-magaracha`                           | Табия (99%)                          |
| `96.79_20-08-2026_17-48-31.webp` | `top5_miss`    | `vinodelnya-batrak-prikumskoe-vitale-kraft-kaberne-sovinon-krasnoe-suhoe-135`    | `roze-2`                                         | Табия (100%)                         |
| `96.83_20-08-2026_17-48-13.webp` | `top5_miss`    | `vinodelnya-pokrovskaya-ryzhest-rkatsiteli-beloe-suhoe-11`                       | `roze-2`                                         | Табия (100%)                         |
| `96.8_25-08-2026_22-20-38.webp`  | `top5_miss`    | `aratti-risling-beloe-polusuhoe`                                                 | `kuban-vino-shato-tamane-aligote-beloe-suhoe-13` | Кубань-Вино (99%)                    |
| `96.98_19-08-2026_19-05-06.webp` | `top5_miss`    | `vinodelnya-batrak-perfekt-klassik-merlo-krasnoe-suhoe-135`                      | `czitronnyj-magaracha`                           | Табия (100%)                         |
