# OCR benchmark on 65 field photos

Benchmark date: 2026-09-29. The input set is `data/test/field_66_review/images`:
65 real shop photos, of which 57 have an exact wine label in the catalog and 8
are marked as absent/ambiguous.

## What is measured

- **Chars** — average number of non-space characters returned per photo. This
  measures coverage, but is not an accuracy metric by itself.
- **Field quality** — a 0–1 weighted score on the 57 exactly labelled photos:
  title 35%, manufacturer 30%, grapes 20%, catalog-token recall 15%.
- **OCR top-1/top-5** — an OCR-only catalog retrieval proxy against all 2,035
  reference wines. This intentionally excludes DINO and SIFT and shows whether
  the text remains useful to the existing catalog search.
- **Avg/P95** — measured end-to-end OCR latency per photo on this machine
  (RTX 4060 8 GB, 32 GB RAM). The product limit is 10 seconds per photo.

There is no manual character-by-character transcription of all visible label
text. Therefore “field quality” is measured against known catalog fields, while
all raw OCR output is retained for visual inspection. This prevents a noisy
engine from winning merely by emitting more characters.

## Full 65-photo results

| Configuration                                       |     Chars | Coverage vs current | Field quality | OCR top-1 | OCR top-5 |    Avg |    P95 |
| --------------------------------------------------- | --------: | ------------------: | ------------: | --------: | --------: | -----: | -----: |
| Current RapidOCR: PP-OCRv6 Tiny + PP-OCRv5 Cyrillic |      47.7 |            baseline |         0.497 |     21.1% |     49.1% | 1.04 s | 1.28 s |
| RapidOCR Small, lower threshold                     |      66.8 |              +40.1% |         0.601 |     19.3% |     45.6% | 2.32 s | 2.99 s |
| RapidOCR Small, medium + detail views               |      75.3 |              +58.0% |         0.624 |     21.1% |     42.1% | 4.48 s | 5.29 s |
| EasyOCR RU+EN, detail crop                          |      67.5 |              +41.6% |         0.568 |     12.3% |     36.8% | 0.75 s | 0.93 s |
| GLM-OCR 1B, detail crop                             |      72.4 |              +51.9% |     **0.712** |      8.8% |     43.9% | 3.65 s | 5.82 s |
| Tesseract.js RU+EN, PSM 11                          |      63.6 |              +33.5% |         0.416 |     10.5% |     33.3% | 1.12 s | 3.02 s |
| Tesseract.js RU+EN, PSM 6                           | **122.3** |         **+156.6%** |         0.396 |     10.5% |     22.8% | 1.95 s | 4.71 s |
| GLM-OCR + current RapidOCR, deduplicated            |      88.4 |              +85.6% |     **0.734** |     14.0% |     47.4% | 4.69 s | 6.97 s |

The Tesseract PSM 6 row demonstrates why coverage is not sufficient: it emits
the most characters, but inspection shows many fragments such as random Latin
letters and malformed Cyrillic. Its catalog usefulness is substantially worse
than the current OCR.

## Pilot configurations rejected before a full run

- PP-OCRv6 Medium detector: 16.4 seconds on the first photo on CPU, already
  beyond the 10-second limit.
- PaddleOCR-VL 1.5: two-photo pilot took 17.5 and 88.5 seconds. It read some
  label text, but was slower and slightly less accurate than GLM-OCR on the same
  examples.
- Several Tiny/Small resolutions, E-Slav recognizer, CLAHE/sharpening and very
  low thresholds were tested on a 12-photo pilot. Only the competitive variants
  were promoted to the full set.

## Conclusion

GLM-OCR is the best recognizer of meaningful label fields in this test. It
correctly recovers text that the current detector omits, including manufacturer,
line name and grape names, while staying below the 10-second limit.

It should not simply replace RapidOCR inside the current generic
`ocr_similarity` function. That matcher was calibrated for short RapidOCR lines
with per-line confidence; feeding all generative OCR text into it reduces
OCR-only top-1 despite improving the actual field matches.

Recommended production design:

1. Keep the current RapidOCR pass as the fast candidate-retrieval anchor.
2. Run GLM-OCR on the detail crop in parallel.
3. Use GLM text for structured fuzzy evidence: manufacturer, title, grapes and
   year. Do not give arbitrary descriptive lines equal weight.
4. Union RapidOCR candidates with manufacturer/title candidates from GLM, then
   let DINO/SIFT and the calibrated stage-5 model perform the final ranking.
5. Fall back to the existing path if GLM is empty or unavailable.

The conservative serial ensemble measured 4.69 seconds average and 6.97 seconds
P95. Parallel CPU RapidOCR + GPU GLM-OCR should be closer to the slower branch
(3.65 seconds average, 5.82 seconds P95) plus small orchestration overhead.

## Reproducibility and outputs

- Benchmark runner: `src/benchmark_ocr_65.py`
- Modern VLM adapter: `src/benchmark_ocr_vlm.py`
- EasyOCR adapter: `src/benchmark_ocr_easy.py`
- Tesseract.js adapter: `src/benchmark_ocr_tesseract.mjs`
- Machine-readable summary: `data/artifacts/stage8/ocr_benchmark_65/summary.json`
- CSV summary: `data/artifacts/stage8/ocr_benchmark_65/summary.csv`
- Per-photo metrics: `data/artifacts/stage8/ocr_benchmark_65/details.json`
- Raw output from every engine: `data/artifacts/stage8/ocr_benchmark_65/raw/*.jsonl`

The raw files contain recognized lines, confidence when available, crop source,
boxes when available, timing and errors. They allow rescoring without rerunning
the OCR models.
