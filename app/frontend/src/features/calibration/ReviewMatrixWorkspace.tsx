"use client";
/* eslint-disable @next/next/no-img-element */

import { useEffect, useMemo, useState } from "react";
import { loadReviewMatrix } from "@/shared/api/client";
import type { Prediction, ReviewMatrixItem } from "@/shared/api/contracts";
import { ImageLightbox } from "@/shared/ui/ImageLightbox";

type Filter = "all" | "top5" | "miss";
type Preview = { src: string; title: string; subtitle?: string };

function score(value: number | undefined) {
  return value === undefined ? "—" : value.toFixed(3);
}

function percent(value: number | undefined) {
  return value === undefined ? "—" : `${Math.round(value * 100)}%`;
}

function correctSlug(item: ReviewMatrixItem) {
  const feedback = item.recognition.feedback;
  if (feedback?.verdict === "incorrect") return feedback.correct_slug ?? null;
  if (feedback?.verdict === "correct") return item.recognition.candidate_slug;
  return null;
}

function correctRank(item: ReviewMatrixItem) {
  const slug = correctSlug(item);
  if (!slug) return null;
  const index = item.recognition.result.predictions.findIndex((candidate) => candidate.slug === slug);
  return index >= 0 ? index + 1 : null;
}

function CandidateCard({ candidate, rank, expected, onPreview }: {
  candidate?: Prediction;
  rank: number;
  expected: string | null;
  onPreview: (preview: Preview) => void;
}) {
  if (!candidate) return <div className="matrix-candidate matrix-candidate--empty"><b>#{rank}</b><span>Нет кандидата</span></div>;
  const correct = candidate.slug === expected;
  return (
    <article className={`matrix-candidate ${correct ? "is-correct" : ""}`}>
      <div className="matrix-rank"><b>#{rank}</b>{correct && <span>правильный</span>}</div>
      <button type="button" className="matrix-image" onClick={() => onPreview({
        src: candidate.wine.image_url,
        title: candidate.wine.title,
        subtitle: `${candidate.wine.manufacturer ?? ""} · кандидат #${rank}`,
      })}>
        <img src={candidate.wine.image_url} alt={candidate.wine.title} loading="lazy" decoding="async" />
      </button>
      <div className="matrix-candidate__copy">
        <strong>{candidate.wine.title}</strong>
        <span>{candidate.wine.manufacturer || "Производитель не указан"}</span>
        <small>{candidate.wine.category || candidate.wine.region || candidate.slug}</small>
      </div>
      <dl className="matrix-scores">
        <div><dt>Итог</dt><dd>{score(candidate.score)}</dd></div>
        <div><dt>DINO</dt><dd>{percent(candidate.embedding_score)}</dd></div>
        <div><dt>SIFT</dt><dd>{percent(candidate.sift_score)}</dd></div>
        <div><dt>OCR</dt><dd>{percent(candidate.ocr_score)}</dd></div>
        <div><dt>Inliers</dt><dd>{candidate.sift_inliers}</dd></div>
      </dl>
    </article>
  );
}

export function ReviewMatrixWorkspace() {
  const [items, setItems] = useState<ReviewMatrixItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>();
  const [filter, setFilter] = useState<Filter>("all");
  const [preview, setPreview] = useState<Preview>();

  useEffect(() => {
    let active = true;
    void loadReviewMatrix()
      .then((payload) => { if (active) setItems(payload.items); })
      .catch((reason) => { if (active) setError(reason instanceof Error ? reason.message : "Не удалось загрузить матрицу"); })
      .finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, []);

  const stats = useMemo(() => ({
    top5: items.filter((item) => item.recognition.feedback?.verdict === "incorrect").length,
    miss: items.filter((item) => item.recognition.feedback?.verdict === "not_in_catalog").length,
  }), [items]);
  const visible = useMemo(() => items.filter((item) => {
    if (filter === "top5") return item.recognition.feedback?.verdict === "incorrect";
    if (filter === "miss") return item.recognition.feedback?.verdict === "not_in_catalog";
    return true;
  }), [filter, items]);

  if (loading) return <div className="matrix-loading"><i className="loader" /><p>Собираем последний прогон и OCR…</p></div>;
  if (error) return <div className="notice notice--error">{error}</div>;

  return (
    <>
      <section className="matrix-toolbar">
        <div><b>{items.length}</b><span>проблемных кадров</span></div>
        <div><b>{stats.top5}</b><span>правильный ниже Top‑1</span></div>
        <div><b>{stats.miss}</b><span>правильного нет в Top‑5</span></div>
        <nav aria-label="Фильтр матрицы">
          <button className={filter === "all" ? "is-active" : ""} onClick={() => setFilter("all")}>Все</button>
          <button className={filter === "top5" ? "is-active" : ""} onClick={() => setFilter("top5")}>Есть в Top‑5</button>
          <button className={filter === "miss" ? "is-active" : ""} onClick={() => setFilter("miss")}>Полный промах</button>
        </nav>
      </section>

      <div className="matrix-scroll">
        <table className="recognition-matrix">
          <thead><tr><th>Фото для распознавания</th>{[1, 2, 3, 4, 5].map((rank) => <th key={rank}>Кандидат #{rank}</th>)}<th>OCR и извлечённые поля</th></tr></thead>
          <tbody>
            {visible.map((item) => {
              const analysis = item.recognition.result;
              const expected = correctSlug(item);
              const rank = correctRank(item);
              const manufacturer = analysis.manufacturer_match;
              const verdict = item.recognition.feedback?.verdict;
              return (
                <tr key={item.recognition.id}>
                  <td className="matrix-source-cell">
                    <button type="button" className="matrix-source-image" onClick={() => setPreview({ src: item.input_image_url, title: item.name, subtitle: "Исходный полевой кадр" })}>
                      <img src={item.input_image_url} alt={item.name} loading="lazy" decoding="async" />
                    </button>
                    <strong>{item.name}</strong>
                    <span className={`matrix-verdict matrix-verdict--${verdict}`}>{rank ? `правильный #${rank}` : "нет в Top‑5"}</span>
                    <dl className="matrix-source-meta">
                      <div><dt>Confidence</dt><dd>{percent(analysis.confidence)}</dd></div>
                      <div><dt>Время</dt><dd>{(item.recognition.total_ms / 1000).toFixed(2)} с</dd></div>
                      <div><dt>Решение</dt><dd>{analysis.decision}</dd></div>
                    </dl>
                  </td>
                  {[0, 1, 2, 3, 4].map((index) => <td key={index}><CandidateCard candidate={analysis.predictions[index]} rank={index + 1} expected={expected} onPreview={setPreview} /></td>)}
                  <td className="matrix-ocr-cell">
                    <section>
                      <b>Распознанные строки · {analysis.ocr_lines.length}</b>
                      <div className="matrix-ocr-lines">
                        {analysis.ocr_lines.length ? analysis.ocr_lines.map((line, index) => (
                          <div key={`${line.text}-${index}`}><strong>{line.text || "—"}</strong><span>{line.normalized || "—"}</span><em>{percent(line.score)}</em></div>
                        )) : <p>OCR ничего не прочитал</p>}
                      </div>
                    </section>
                    <section className="matrix-ocr-fields">
                      <b>Manufacturer gate</b>
                      <dl>
                        <div><dt>Значение</dt><dd>{manufacturer?.value || "не найдено"}</dd></div>
                        <div><dt>Режим</dt><dd>{manufacturer?.mode || "—"}</dd></div>
                        <div><dt>Текст</dt><dd>{manufacturer?.matched_text || "—"}</dd></div>
                        <div><dt>Score</dt><dd>{score(manufacturer?.score)}</dd></div>
                        <div><dt>Margin</dt><dd>{score(manufacturer?.margin)}</dd></div>
                      </dl>
                    </section>
                    <section className="matrix-ocr-fields">
                      <b>Evidence-поля</b>
                      <dl>
                        {Object.entries(analysis.evidence).filter(([key]) => key.includes("ocr") || key.includes("title") || key.includes("manufacturer") || key.includes("grape") || key.includes("year")).map(([key, value]) => <div key={key}><dt>{key}</dt><dd>{score(value)}</dd></div>)}
                      </dl>
                    </section>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {!visible.length && <div className="empty-state"><h2>Для этого фильтра строк нет</h2></div>}
      <ImageLightbox src={preview?.src} title={preview?.title} subtitle={preview?.subtitle} onClose={() => setPreview(undefined)} />
    </>
  );
}
