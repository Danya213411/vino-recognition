"use client";
/* eslint-disable @next/next/no-img-element */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AnalysisDiagnostics } from "@/entities/recognition/AnalysisDiagnostics";
import { analyzePhoto, loadReviewFile, loadReviewSet, sendFeedback } from "@/shared/api/client";
import type { Analysis, FeedbackVerdict, ReviewSetItem } from "@/shared/api/contracts";
import { getSessionId } from "@/shared/lib/session";
import { ImageLightbox } from "@/shared/ui/ImageLightbox";

const STORAGE_KEY = "vino-batch-review-field-66-blurless-v1";
const THUMB_PAGE_SIZE = 20;

type ReviewEntry = {
  analysis?: Analysis;
  verdict?: FeedbackVerdict;
  correctSlug?: string;
  error?: string;
  running?: boolean;
};

function percent(value: number) { return `${Math.round(value * 100)}%`; }

export function BatchReviewWorkspace() {
  const [items, setItems] = useState<ReviewSetItem[]>([]);
  const [results, setResults] = useState<Record<string, ReviewEntry>>({});
  const [currentIndex, setCurrentIndex] = useState(0);
  const [loading, setLoading] = useState(true);
  const [batchRunning, setBatchRunning] = useState(false);
  const [editingName, setEditingName] = useState<string>();
  const [globalError, setGlobalError] = useState<string>();
  const [preview, setPreview] = useState<string>();
  const resultsRef = useRef(results);
  const stopRef = useRef(false);
  const persistTimerRef = useRef<number | undefined>(undefined);

  useEffect(() => {
    let active = true;
    void loadReviewSet().then((payload) => {
      if (!active) return;
      setItems(payload.items);
      try {
        const saved = window.localStorage.getItem(STORAGE_KEY);
        if (saved) {
          const parsed = JSON.parse(saved) as Record<string, ReviewEntry>;
          const available = new Set(payload.items.map((item) => item.name));
          const restored = Object.fromEntries(
            Object.entries(parsed)
              .filter(([name]) => available.has(name))
              .map(([name, entry]) => [name, { ...entry, running: false }]),
          );
          resultsRef.current = restored;
          setResults(restored);
        }
      } catch {
        window.localStorage.removeItem(STORAGE_KEY);
      }
    }).catch((reason) => {
      if (active) setGlobalError(reason instanceof Error ? reason.message : "Не удалось открыть выборку");
    }).finally(() => { if (active) setLoading(false); });
    return () => {
      active = false;
      stopRef.current = true;
      if (persistTimerRef.current) window.clearTimeout(persistTimerRef.current);
    };
  }, []);

  const schedulePersistence = useCallback((next: Record<string, ReviewEntry>) => {
    if (persistTimerRef.current) window.clearTimeout(persistTimerRef.current);
    persistTimerRef.current = window.setTimeout(() => {
      try { window.localStorage.setItem(STORAGE_KEY, JSON.stringify(next)); } catch { /* Database remains the source of truth. */ }
    }, 700);
  }, []);

  const patchEntry = useCallback((name: string, patch: Partial<ReviewEntry>) => {
    const next = { ...resultsRef.current, [name]: { ...resultsRef.current[name], ...patch } };
    resultsRef.current = next;
    setResults(next);
    if ("analysis" in patch || "verdict" in patch || ("error" in patch && patch.error)) schedulePersistence(next);
  }, [schedulePersistence]);

  const recognizeItem = useCallback(async (item: ReviewSetItem) => {
    if (resultsRef.current[item.name]?.analysis) return resultsRef.current[item.name].analysis;
    patchEntry(item.name, { running: true, error: undefined });
    try {
      const file = await loadReviewFile(item);
      const analysis = await analyzePhoto(file, "calib", getSessionId());
      patchEntry(item.name, { analysis, running: false, error: undefined });
      return analysis;
    } catch (reason) {
      const message = reason instanceof Error ? reason.message : "Ошибка анализа";
      patchEntry(item.name, { running: false, error: message });
      return undefined;
    }
  }, [patchEntry]);

  async function runAll() {
    if (batchRunning || !items.length) return;
    stopRef.current = false;
    setBatchRunning(true);
    setGlobalError(undefined);
    for (const item of items) {
      if (stopRef.current) break;
      if (!resultsRef.current[item.name]?.analysis) await recognizeItem(item);
    }
    setBatchRunning(false);
  }

  const answer = useCallback(async (verdict: FeedbackVerdict, correctSlug?: string) => {
    const item = items[currentIndex];
    const entry = item ? resultsRef.current[item.name] : undefined;
    if (!item || !entry?.analysis || (entry.verdict && editingName !== item.name) || entry.running) return;
    patchEntry(item.name, { running: true, error: undefined });
    try {
      await sendFeedback(entry.analysis.id, verdict, getSessionId(), correctSlug);
      patchEntry(item.name, { verdict, correctSlug, running: false });
      setEditingName(undefined);
      const latest = resultsRef.current;
      const nextReady = items.findIndex((candidate, index) => index > currentIndex && latest[candidate.name]?.analysis && !latest[candidate.name]?.verdict);
      if (nextReady >= 0) setCurrentIndex(nextReady);
      else if (currentIndex < items.length - 1) setCurrentIndex(currentIndex + 1);
    } catch (reason) {
      patchEntry(item.name, { running: false, error: reason instanceof Error ? reason.message : "Не удалось сохранить ответ" });
    }
  }, [currentIndex, editingName, items, patchEntry]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target?.matches("input, textarea, select")) return;
      const entry = items[currentIndex] ? resultsRef.current[items[currentIndex].name] : undefined;
      if (event.key === "ArrowLeft") setCurrentIndex((value) => Math.max(0, value - 1));
      else if (event.key === "ArrowRight") setCurrentIndex((value) => Math.min(items.length - 1, value + 1));
      else if (event.key === "0") void answer("not_in_catalog");
      else if (event.key.toLocaleLowerCase("ru") === "x") void answer("not_in_store");
      else if (/^[1-5]$/.test(event.key) && entry?.analysis) {
        const rank = Number(event.key) - 1;
        const prediction = entry.analysis.predictions[rank];
        if (prediction) void answer(rank === 0 ? "correct" : "incorrect", rank === 0 ? undefined : prediction.slug);
      } else return;
      event.preventDefault();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [answer, currentIndex, items]);

  const stats = useMemo(() => {
    const values = Object.values(results);
    return {
      recognized: values.filter((entry) => entry.analysis).length,
      reviewed: values.filter((entry) => entry.verdict).length,
      correct: values.filter((entry) => entry.verdict === "correct").length,
      alternatives: values.filter((entry) => entry.verdict === "incorrect").length,
      absent: values.filter((entry) => entry.verdict === "not_in_catalog").length,
      notInStore: values.filter((entry) => entry.verdict === "not_in_store").length,
      errors: values.filter((entry) => entry.error && !entry.analysis).length,
    };
  }, [results]);

  const current = items[currentIndex];
  const entry = current ? results[current.name] : undefined;
  const analysis = entry?.analysis;
  const editingCurrent = editingName === current?.name;
  const progress = items.length ? Math.round((stats.recognized / items.length) * 100) : 0;
  const thumbPage = Math.floor(currentIndex / THUMB_PAGE_SIZE);
  const thumbPageCount = Math.ceil(items.length / THUMB_PAGE_SIZE);
  const thumbStart = thumbPage * THUMB_PAGE_SIZE;
  const visibleThumbs = items.slice(thumbStart, thumbStart + THUMB_PAGE_SIZE);

  function downloadResults() {
    const payload = {
      exported_at: new Date().toISOString(),
      total: items.length,
      stats,
      items: items.map((item) => ({ name: item.name, ...resultsRef.current[item.name] })),
    };
    const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = "field-66-blurless-review.json";
    link.click();
    URL.revokeObjectURL(url);
  }

  if (loading) return <div className="batch-loading"><div className="loader" /><h2>Готовим 66 полевых кадров</h2></div>;

  return (
    <div className="batch-review">
      {globalError && <div className="notice notice--error" role="alert">{globalError}</div>}
      <section className="batch-toolbar">
        <div className="batch-progress-copy"><span>{stats.recognized} / {items.length} распознано</span><b>{progress}%</b></div>
        <div className="batch-progress"><span style={{ width: `${progress}%` }} /></div>
        <div className="batch-stats">
          <span>Проверено <b>{stats.reviewed}</b></span><span>Top‑1 верно <b>{stats.correct}</b></span>
          <span>Другой кандидат <b>{stats.alternatives}</b></span><span>Нет в top‑5 <b>{stats.absent}</b></span>
          <span>Нет в магазине <b>{stats.notInStore}</b></span>
        </div>
        <div className="batch-toolbar__actions">
          {batchRunning ? <button className="button button--ghost" onClick={() => { stopRef.current = true; }}>Остановить после текущего</button> : <button className="button button--primary" onClick={() => void runAll()} disabled={!items.length || stats.recognized === items.length}>Распознать все {items.length}</button>}
          <button className="button button--ghost" onClick={downloadResults} disabled={!stats.reviewed}>Скачать итоги</button>
        </div>
      </section>

      <div className="batch-layout">
        <aside className="batch-queue">
          <div className="batch-queue__head"><b>Кадры</b><span>{stats.errors ? `${stats.errors} ошибок` : "1–5 кандидат · 0 нет · X нет в магазине"}</span></div>
          <div className="batch-thumbs">
            {visibleThumbs.map((item, visibleIndex) => {
              const index = thumbStart + visibleIndex;
              const value = results[item.name];
              const state = value?.error && !value.analysis ? "has-error" : value?.running ? "is-running" : value?.verdict ? `is-reviewed is-${value.verdict}` : value?.analysis ? "is-ready" : "";
              return <button key={item.name} className={`${state} ${index === currentIndex ? "is-current" : ""}`} onClick={() => setCurrentIndex(index)} title={`${index + 1}. ${item.name}`}><img src={item.image_url} alt="" loading="lazy" decoding="async" /><span>{index + 1}</span></button>;
            })}
          </div>
          {thumbPageCount > 1 && <div className="batch-pages" aria-label="Страницы кадров">
            {Array.from({ length: thumbPageCount }, (_, page) => <button key={page} className={page === thumbPage ? "is-current" : ""} onClick={() => setCurrentIndex(page * THUMB_PAGE_SIZE)}>{page + 1}</button>)}
          </div>}
        </aside>

        <main className="batch-current">
          {current ? <>
            <section className="batch-photo-panel">
              <button className="batch-photo" onClick={() => setPreview(current.image_url)} aria-label="Увеличить исходный кадр"><img src={current.image_url} alt={current.name} />{entry?.running && <span><i className="loader" />{analysis ? "Сохраняем ответ" : "Распознаём"}</span>}</button>
              <div className="batch-photo-meta"><div><span>Кадр {currentIndex + 1} из {items.length}</span><b>{current.name}</b></div><div><button onClick={() => setCurrentIndex((value) => Math.max(0, value - 1))} disabled={currentIndex === 0}>←</button><button onClick={() => setCurrentIndex((value) => Math.min(items.length - 1, value + 1))} disabled={currentIndex === items.length - 1}>→</button></div></div>
              {!analysis && !entry?.running && <button className="button button--primary batch-run-one" onClick={() => void recognizeItem(current)}>Распознать этот кадр</button>}
              {entry?.error && <div className="notice notice--error">{entry.error}</div>}
            </section>

            <section className="calibration-panel batch-analysis">
              {!analysis && <div className="empty-state"><span>{String(currentIndex + 1).padStart(2, "0")} / {items.length}</span><h2>{entry?.running ? "Сверяем признаки" : "Кадр ещё не обработан"}</h2><p>{entry?.running ? "DINOv2 → SIFT → OCR → confidence" : "Запустите всю выборку или только выбранную фотографию."}</p></div>}
              {analysis && <>
                <div className="confidence-head"><div><span className="eyebrow">Калиброванная уверенность</span><h2>{percent(analysis.confidence)}</h2></div><span className={`decision-badge decision-badge--${analysis.decision}`}>{analysis.verified_match ? "текст + геометрия" : analysis.decision === "match" ? "принято" : analysis.decision === "alternatives" ? "проверить" : "не найдено"}</span></div>
                <AnalysisDiagnostics key={analysis.id} analysis={analysis} compact disabled={Boolean((entry?.verdict && !editingCurrent) || entry?.running)} onCorrect={(slug) => void answer("incorrect", slug)} />
                <div className={`feedback-box batch-feedback ${entry?.verdict && !editingCurrent ? "is-done" : ""}`}>
                  {entry?.verdict && !editingCurrent ? <div className="saved-feedback"><p><b>Ответ сохранён:</b> {entry.verdict === "correct" ? "первый кандидат верный" : entry.verdict === "incorrect" ? `верный вариант — ${entry.correctSlug}` : entry.verdict === "not_in_store" ? "этого вина физически нет в магазине" : "правильного вина нет среди top‑5"}.</p><button className="button button--ghost button--small" onClick={() => setEditingName(current.name)}>Изменить ответ</button></div> : <><strong>{entry?.verdict ? "Выберите новый ответ" : "Какой вариант правильный?"}</strong><div className="feedback-actions"><button className="button button--primary" disabled={entry?.running} onClick={() => void answer("correct")}>1 · Первый верный</button><button className="button button--ghost" disabled={entry?.running} onClick={() => void answer("not_in_catalog")}>0 · Нет в top‑5</button><button className="button button--danger-soft" disabled={entry?.running} onClick={() => void answer("not_in_store")}>X · Этого вина нет в магазине</button>{entry?.verdict && <button className="button button--ghost" onClick={() => setEditingName(undefined)}>Отмена</button>}</div><small>Для вариантов 2–5 нажмите «Это оно» в строке или цифру. Если бутылки физически не было в магазине, нажмите X.</small></>}
                </div>
              </>}
            </section>
          </> : <div className="empty-state"><h2>В папке нет изображений</h2><p>Пересоберите набор командой `uv run python src/export_fuzzy_recheck_set.py`.</p></div>}
        </main>
      </div>
      <ImageLightbox src={preview} title={current?.name} subtitle="Полевой кадр после обновления каталога и весов" onClose={() => setPreview(undefined)} />
    </div>
  );
}
