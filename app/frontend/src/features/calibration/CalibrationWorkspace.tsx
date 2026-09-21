"use client";

import { useState } from "react";
import { analyzePhoto, downloadManifest, sendFeedback } from "@/shared/api/client";
import type { Analysis, FeedbackVerdict } from "@/shared/api/contracts";
import { AnalysisDiagnostics } from "@/entities/recognition/AnalysisDiagnostics";
import { getSessionId } from "@/shared/lib/session";
import { PhotoInput } from "@/shared/ui/PhotoInput";

function percentage(value: number) {
  return `${Math.round(value * 100)}%`;
}

export function CalibrationWorkspace() {
  const [analysis, setAnalysis] = useState<Analysis>();
  const [busy, setBusy] = useState(false);
  const [feedback, setFeedback] = useState<FeedbackVerdict>();
  const [error, setError] = useState<string>();

  async function run(file: File) {
    setBusy(true);
    setAnalysis(undefined);
    setFeedback(undefined);
    setError(undefined);
    try {
      setAnalysis(await analyzePhoto(file, "calib", getSessionId()));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Ошибка анализа");
    } finally {
      setBusy(false);
    }
  }

  async function answer(verdict: FeedbackVerdict, correctSlug?: string) {
    if (!analysis) return;
    try {
      await sendFeedback(analysis.id, verdict, getSessionId(), correctSlug);
      setFeedback(verdict);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Не удалось сохранить ответ");
    }
  }

  return (
    <div className="calibration-grid">
      <div>
        <PhotoInput busy={busy} compact onSelect={run} />
        <div className="calib-note">
          <span>Полевой режим</span>
          <p>Каждый кадр и ваш ответ сохраняются локально. Это данные для следующей калибровки порогов.</p>
        </div>
      </div>
      <div className="calibration-panel">
        {!analysis && !busy && (
          <div className="empty-state">
            <span>01—05</span>
            <h2>Здесь появятся кандидаты</h2>
            <p>Вы увидите калиброванную уверенность, долю каждого кандидата и вклад сигналов.</p>
          </div>
        )}
        {busy && <div className="empty-state"><div className="loader" /><h2>Сверяем признаки</h2><p>DINOv2 → SIFT → OCR → confidence</p></div>}
        {error && <div className="notice notice--error" role="alert">{error}</div>}
        {analysis && (
          <>
            <div className="confidence-head">
              <div>
                <span className="eyebrow">Калиброванная уверенность</span>
                <h2>{percentage(analysis.confidence)}</h2>
              </div>
              <span className={`decision-badge decision-badge--${analysis.decision}`}>
                {analysis.verified_match ? "текст + геометрия" : analysis.decision === "match" ? "принято" : analysis.decision === "alternatives" ? "проверить" : "не найдено"}
              </span>
            </div>
            <AnalysisDiagnostics analysis={analysis} disabled={Boolean(feedback)} onCorrect={(slug) => void answer("incorrect", slug)} />
            <div className="feedback-box">
              {feedback ? (
                <p><b>Ответ сохранён.</b> Он уже попадёт в выгрузку и следующий цикл калибровки.</p>
              ) : (
                <>
                  <strong>Первый вариант определён правильно?</strong>
                  <div className="feedback-actions">
                    <button className="button button--primary" onClick={() => answer("correct")}>Да, правильно</button>
                    <button className="button button--ghost" onClick={() => answer("not_in_catalog")}>Этого вина нет</button>
                  </div>
                  <small>Если верный вариант ниже — нажмите «Это оно» в его строке.</small>
                </>
              )}
            </div>
            <button className="button button--download" onClick={() => downloadManifest(analysis, getSessionId())}>
              Скачать полный манифест ↓
            </button>
          </>
        )}
      </div>
    </div>
  );
}
