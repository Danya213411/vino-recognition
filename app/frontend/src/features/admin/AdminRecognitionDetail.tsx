"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { AnalysisDiagnostics } from "@/entities/recognition/AnalysisDiagnostics";
import { loadAdminImage, loadAdminRecord } from "@/shared/api/client";
import type { AdminRecord } from "@/shared/api/contracts";
import { getAdminToken } from "@/shared/lib/admin-token";
import { ImageLightbox } from "@/shared/ui/ImageLightbox";

function bytes(value: number) { return value < 1024 * 1024 ? `${Math.round(value / 1024)} КБ` : `${(value / 1024 / 1024).toFixed(1)} МБ`; }

export function AdminRecognitionDetail() {
  const params = useSearchParams();
  const id = params.get("id");
  const [record, setRecord] = useState<AdminRecord>();
  const [inputImage, setInputImage] = useState<string>();
  const [preview, setPreview] = useState(false);
  const [error, setError] = useState<string>();
  const [busy, setBusy] = useState(true);

  const load = useCallback(async () => {
    if (!id) { setError("В адресе отсутствует id распознавания"); setBusy(false); return; }
    setBusy(true); setError(undefined);
    const token = getAdminToken();
    try {
      const [nextRecord, source] = await Promise.all([loadAdminRecord(id, token), loadAdminImage(id, token)]);
      setRecord(nextRecord); setInputImage(source);
    } catch (reason) { setError(reason instanceof Error ? reason.message : "Не удалось открыть распознавание"); }
    finally { setBusy(false); }
  }, [id]);

  useEffect(() => {
    const timer = window.setTimeout(() => void load(), 0);
    return () => window.clearTimeout(timer);
  }, [load]);
  useEffect(() => () => { if (inputImage) URL.revokeObjectURL(inputImage); }, [inputImage]);

  if (busy) return <div className="detail-loading"><div className="loader" /><h2>Загружаем полный разбор</h2></div>;
  if (error || !record) return <div className="notice notice--error">{error ?? "Запись не найдена"}<br /><Link href="/admin/">← Вернуться в журнал</Link></div>;

  return (
    <div className="admin-detail">
      <div className="admin-detail__top"><Link href="/admin/" className="back-link">← Все распознавания</Link><span>{record.id}</span></div>
      <section className="admin-detail__hero">
        <button className="admin-detail__photo" type="button" onClick={() => setPreview(true)}>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={inputImage} alt="Исходный кадр" /><span>Нажмите, чтобы развернуть на весь экран</span>
        </button>
        <div className="admin-detail__summary">
          <span className="eyebrow">{record.mode === "calib" ? "Калибровочный кадр" : "Пользовательский кадр"}</span>
          <h1>{record.result.wine.title}</h1>
          <p>{record.result.wine.manufacturer} · {record.result.wine.region}</p>
          <div className="admin-detail__decision"><b>{Math.round(record.confidence * 100)}%</b><span className={`decision-badge decision-badge--${record.decision}`}>{record.decision}</span></div>
          <dl>
            <div><dt>Пользователь</dt><dd>{record.session_id}</dd></div><div><dt>Создано</dt><dd>{new Date(record.created_at).toLocaleString("ru-RU")}</dd></div>
            <div><dt>Файл</dt><dd>{record.original_name} · {record.width}×{record.height} · {bytes(record.byte_size)}</dd></div><div><dt>Время</dt><dd>{Math.round(record.total_ms)} мс</dd></div>
            <div><dt>Кандидат</dt><dd>{record.candidate_slug}</dd></div><div><dt>Feedback</dt><dd>{record.feedback ? `${record.feedback.verdict}${record.feedback.correct_slug ? ` → ${record.feedback.correct_slug}` : ""}` : "Нет ответа"}</dd></div>
          </dl>
        </div>
      </section>
      <section className="admin-detail__analysis"><div className="section-heading"><div><span className="eyebrow">Почему система выбрала этот ответ</span><h2>Полная диагностика</h2></div></div><AnalysisDiagnostics analysis={record.result} /></section>
      <details className="pipeline-details"><summary>Показать конфигурацию пайплайна</summary><pre>{JSON.stringify(record.pipeline, null, 2)}</pre></details>
      <ImageLightbox src={preview ? inputImage : undefined} title={record.original_name} subtitle={`${record.width}×${record.height} · ${new Date(record.created_at).toLocaleString("ru-RU")}`} onClose={() => setPreview(false)} />
    </div>
  );
}
