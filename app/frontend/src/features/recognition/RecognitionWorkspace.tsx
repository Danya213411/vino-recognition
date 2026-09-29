"use client";

import { useRef, useState } from "react";
import { WineDetail } from "@/entities/wine/WineDetail";
import { analyzePhoto } from "@/shared/api/client";
import type { Analysis } from "@/shared/api/contracts";
import { getSessionId } from "@/shared/lib/session";
import { PhotoInput } from "@/shared/ui/PhotoInput";
import { TopCandidates } from "@/features/recognition/TopCandidates";

export function RecognitionWorkspace() {
  const [analysis, setAnalysis] = useState<Analysis>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string>();
  const retakeInput = useRef<HTMLInputElement>(null);

  async function recognize(file: File) {
    setBusy(true);
    setError(undefined);
    try {
      setAnalysis(await analyzePhoto(file, "user", getSessionId()));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Не удалось распознать фотографию");
    } finally {
      setBusy(false);
    }
  }

  const retake = (
    <>
      <button className="button button--retake" type="button" onClick={() => retakeInput.current?.click()} disabled={busy}>
        {busy ? "Анализируем новый кадр…" : "Выбрать другой кадр"}
      </button>
      <input ref={retakeInput} className="sr-only" type="file" accept="image/jpeg,image/png,image/webp" capture="environment" onChange={(event) => {
        const file = event.target.files?.[0];
        if (file) void recognize(file);
        event.currentTarget.value = "";
      }} />
    </>
  );

  if (analysis && analysis.decision !== "not_found") {
    return (
      <div className="recognized-page core-container">
        {busy && <div className="processing-banner"><div className="loader" />Сверяем новый кадр с каталогом…</div>}
        {error && <div className="notice notice--error" role="alert">{error}</div>}
        <WineDetail wine={analysis.wine} decision={analysis.decision} action={retake} />
        <TopCandidates predictions={analysis.predictions} />
      </div>
    );
  }

  return (
    <>
      <section className="scan-hero core-container">
        <div className="scan-hero__copy">
          <span className="eyebrow">Российское вино — ближе</span>
          <h1>Найти своё<br />вино</h1>
          <p>Сфотографируйте бутылку — мы сопоставим этикетку с независимым каталогом российских вин.</p>
          <div className="scan-hero__facts"><span>2 025 вин в каталоге</span><span>DINOv2 · SIFT · OCR</span></div>
        </div>
        <div className="recognition-workspace">
          <PhotoInput busy={busy} onSelect={recognize} />
          {error && <div className="notice notice--error" role="alert">{error}</div>}
          {analysis?.decision === "not_found" && <article className="no-match-card"><span className="eyebrow">Совпадение не подтверждено</span><h2>Не хотим угадывать</h2><p>Попробуйте снять этикетку ближе и без блика — текущий кадр не подтвердили одновременно изображение, детали и текст.</p></article>}
        </div>
      </section>
      <section className="how-it-works core-container">
        <span className="eyebrow">Как это работает</span>
        <div className="steps"><article><b>01</b><h3>Снимите этикетку</h3><p>Целиком, прямо или под небольшим углом.</p></article><article><b>02</b><h3>Сверим три сигнала</h3><p>Общий вид, локальные детали печати и текст.</p></article><article><b>03</b><h3>Изучите карточку</h3><p>Регион, сорт, категория и гастрономические сочетания.</p></article></div>
      </section>
    </>
  );
}
