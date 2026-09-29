"use client";

import { useCallback, useState } from "react";
import type { Analysis, Prediction } from "@/shared/api/contracts";
import { ImageLightbox } from "@/shared/ui/ImageLightbox";

function percentage(value: number) { return `${Math.round(value * 100)}%`; }
function formatValue(value: unknown): string {
  if (Array.isArray(value)) return value.join(", ") || "—";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(4);
  if (value == null || value === "") return "—";
  return String(value);
}
function readableKey(key: string) { return key.replaceAll("_", " ").replace(/^./, (value) => value.toUpperCase()); }

function Candidate({ item, rank, disabled, onCorrect, onPreview }: {
  item: Prediction;
  rank: number;
  disabled?: boolean;
  onCorrect?: () => void;
  onPreview: () => void;
}) {
  return (
    <article className="candidate-row">
      <span className="candidate-rank">{String(rank).padStart(2, "0")}</span>
      <button className="candidate-image" type="button" onClick={onPreview} aria-label={`Увеличить ${item.wine.title}`}>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={item.wine.image_url} alt="" /><span className="image-zoom-mark" aria-hidden="true" />
      </button>
      <div className="candidate-copy">
        <strong>{item.wine.title}</strong><small>{item.wine.manufacturer} · {item.wine.region}</small>
        <div className="support-track"><span style={{ width: percentage(item.relative_support) }} /></div>
        <div className="candidate-metrics">
          <span>Итог <b>{item.score.toFixed(3)}</b></span><span>DINO <b>{percentage(item.embedding_score)}</b></span>
          <span>SIFT <b>{percentage(item.sift_score)}</b></span><span>Inliers <b>{item.sift_inliers}</b></span>
          <span>OCR <b>{percentage(item.ocr_score)}</b></span>
        </div>
      </div>
      <div className="candidate-score"><b>{percentage(item.relative_support)}</b><small>доля top‑5</small></div>
      {rank > 1 && onCorrect && <button className="button button--ghost button--small" disabled={disabled} onClick={onCorrect}>Это оно</button>}
    </article>
  );
}

function DataBlock({ title, data, suffix }: { title: string; data: Record<string, unknown>; suffix?: string }) {
  return (
    <section className="diagnostic-block"><h3>{title}</h3><dl>
      {Object.entries(data).map(([key, value]) => <div key={key}><dt>{readableKey(key)}</dt><dd>{formatValue(value)}{typeof value === "number" && suffix ? suffix : ""}</dd></div>)}
    </dl></section>
  );
}

export function AnalysisDiagnostics({ analysis, disabled, onCorrect, compact = false }: {
  analysis: Analysis;
  disabled?: boolean;
  onCorrect?: (slug: string) => void;
  compact?: boolean;
}) {
  const [preview, setPreview] = useState<{ src: string; title: string; subtitle?: string }>();
  const [detailsOpen, setDetailsOpen] = useState(!compact);
  const closePreview = useCallback(() => setPreview(undefined), []);
  const confidenceData: Record<string, unknown> = {
    "Итоговая уверенность": percentage(analysis.confidence),
    "Калиброванная модель": percentage(analysis.calibrated_confidence),
    "Порог принятия": percentage(analysis.thresholds.accept),
    "Порог проверки": percentage(analysis.thresholds.review),
    "Источник решения": analysis.confidence_source,
  };
  const timings = Object.fromEntries(Object.entries(analysis.timing_ms).map(([key, value]) => [key, Math.round(value)]));
  const memory = Object.fromEntries(Object.entries(analysis.memory).map(([key, value]) => [key, `${(value / 1024 / 1024).toFixed(1)} МБ`]));
  return (
    <div className="analysis-diagnostics">
      <div className="candidate-list">
        {analysis.predictions.map((item, index) => <Candidate key={item.slug} item={item} rank={index + 1} disabled={disabled} onCorrect={onCorrect ? () => onCorrect(item.slug) : undefined} onPreview={() => setPreview({ src: item.wine.image_url, title: item.wine.title, subtitle: `${item.wine.manufacturer ?? ""} · ${item.wine.region ?? ""}` })} />)}
      </div>
      {analysis.manufacturer_match?.mode === "hard" && <div className="manufacturer-gate"><div><span className="eyebrow">OCR определил производителя</span><strong>{analysis.manufacturer_match.values?.join(" / ") || analysis.manufacturer_match.value}</strong></div><div><b>{percentage(analysis.manufacturer_match.score)}</b><small>кандидатов: {analysis.manufacturer_match.candidate_count}</small></div></div>}
      {compact && <button className="diagnostics-toggle" type="button" aria-expanded={detailsOpen} onClick={() => setDetailsOpen((value) => !value)}>{detailsOpen ? "Скрыть подробные метрики" : "Показать подробные метрики"}</button>}
      {detailsOpen && <>
        <div className="diagnostic-summary">
          <DataBlock title="Решение и пороги" data={confidenceData} />
          <DataBlock title="Evidence — признаки confidence" data={analysis.evidence} />
          <DataBlock title="Согласованность кропов" data={analysis.crop_consistency} />
          <DataBlock title="Тайминги" data={timings} suffix=" мс" />
          <DataBlock title="Память процесса" data={memory} />
        </div>
        <section className="ocr-block"><div><span className="eyebrow">Что прочитал OCR</span><h3>{analysis.ocr_lines.length ? `${analysis.ocr_lines.length} текстовых фрагментов` : "Текст не найден"}</h3></div><div className="ocr-lines">{analysis.ocr_lines.map((line, index) => <span key={`${line.text}-${index}`}><b>{line.text ?? line.normalized ?? "—"}</b><small>{line.score == null ? "" : percentage(line.score)}</small></span>)}</div></section>
      </>}
      <ImageLightbox src={preview?.src} title={preview?.title} subtitle={preview?.subtitle} onClose={closePreview} />
    </div>
  );
}
