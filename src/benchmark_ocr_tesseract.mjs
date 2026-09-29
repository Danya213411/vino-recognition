// Tesseract.js benchmark adapter for the 65 field photos.
// Dependencies live under the benchmark artifact directory so the web app's
// package manifests stay untouched.

import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const sourceDir = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(sourceDir, '..');
const outputDir = path.join(root, 'data', 'artifacts', 'stage8', 'ocr_benchmark_65');
const requireFromBenchmark = createRequire(path.join(outputDir, 'tesseractjs', 'package.json'));
const { createWorker, PSM } = requireFromBenchmark('tesseract.js');
const sharp = requireFromBenchmark('sharp');

const args = process.argv.slice(2);
const limitFlag = args.indexOf('--limit');
const limit = limitFlag >= 0 ? Number(args[limitFlag + 1]) : null;
const force = args.includes('--force');
const requested = args.includes('--psm11') ? ['psm11'] : args.includes('--psm6') ? ['psm6'] : ['psm6', 'psm11'];
const manifest = JSON.parse(fs.readFileSync(path.join(outputDir, 'manifest.json'), 'utf8')).slice(0, limit || undefined);

function readJsonl(file) {
  if (!fs.existsSync(file)) return [];
  return fs.readFileSync(file, 'utf8').split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line));
}

function appendJsonl(file, value) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.appendFileSync(file, `${JSON.stringify(value)}\n`, 'utf8');
}

function normalize(text) {
  return text.toLowerCase().replace(/[^0-9a-zа-яё]+/giu, ' ').trim().replace(/\s+/g, ' ');
}

async function prepareImage(file) {
  const oriented = sharp(file).rotate();
  const metadata = await oriented.metadata();
  const width = metadata.width;
  const height = metadata.height;
  return oriented
    .extract({
      left: Math.round(width * 0.20),
      top: Math.round(height * 0.22),
      width: Math.round(width * 0.60),
      height: Math.round(height * 0.72),
    })
    .resize({ width: 1600, height: 1600, fit: 'inside', withoutEnlargement: true })
    .png()
    .toBuffer();
}

const worker = await createWorker(['eng', 'rus'], 1, { logger: () => {} });
for (const psm of requested) {
  const config = `tessjs_engrus_${psm}_detail`;
  const output = path.join(outputDir, 'raw', `${config}.jsonl`);
  if (force && fs.existsSync(output)) fs.unlinkSync(output);
  const completed = new Set(readJsonl(output).map((row) => row.name));
  await worker.setParameters({
    tessedit_pageseg_mode: psm === 'psm6' ? PSM.SINGLE_BLOCK : PSM.SPARSE_TEXT,
    preserve_interword_spaces: '1',
  });
  for (let index = 0; index < manifest.length; index += 1) {
    const item = manifest[index];
    if (completed.has(item.name)) continue;
    const started = performance.now();
    let lines = [];
    let rawText = '';
    let error = null;
    try {
      const image = await prepareImage(path.join(root, ...item.path.split('/')));
      const result = await worker.recognize(image);
      rawText = result.data.text || '';
      const confidence = Number(result.data.confidence || 0) / 100;
      lines = rawText.split(/\r?\n/).map((text) => ({
        text: text.trim(),
        normalized: normalize(text),
        score: confidence,
        view: 'detail',
        box: null,
      })).filter((line) => line.normalized.length >= 2);
    } catch (exception) {
      error = `${exception?.name || 'Error'}: ${exception?.message || exception}`;
    }
    const elapsed = performance.now() - started;
    appendJsonl(output, {
      engine: 'tesseract.js',
      config,
      config_details: { languages: ['eng', 'rus'], psm, view: 'detail', max_size: 1600 },
      name: item.name,
      lines,
      raw_text: rawText,
      timing_ms: { total: elapsed },
      error,
    });
    process.stdout.write(`[${config}] ${index + 1}/${manifest.length} lines=${lines.length} ms=${elapsed.toFixed(0)} error=${error}\n`);
  }
}
await worker.terminate();
