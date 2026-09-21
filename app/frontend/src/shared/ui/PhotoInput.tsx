"use client";

import { useEffect, useRef, useState } from "react";

type Props = {
  busy?: boolean;
  compact?: boolean;
  onSelect: (file: File) => void;
};

export function PhotoInput({ busy = false, compact = false, onSelect }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [preview, setPreview] = useState<string>();

  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
    };
  }, [preview]);

  function choose(file?: File) {
    if (!file) return;
    if (preview) URL.revokeObjectURL(preview);
    setPreview(URL.createObjectURL(file));
    onSelect(file);
  }

  return (
    <div className={`photo-input ${compact ? "photo-input--compact" : ""}`}>
      {preview ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img className="photo-preview" src={preview} alt="Выбранная фотография этикетки" />
      ) : (
        <div className="camera-glyph" aria-hidden="true">
          <span />
        </div>
      )}
      <div className="photo-input__copy">
        <strong>{busy ? "Изучаем этикетку…" : "Сфотографируйте бутылку"}</strong>
        <p>Этикетка целиком, без сильного блика. JPEG, PNG или WebP до 15 МБ.</p>
        <button className="button button--primary" onClick={() => input.current?.click()} disabled={busy}>
          {preview ? "Выбрать другой кадр" : "Открыть камеру или галерею"}
        </button>
      </div>
      <input
        ref={input}
        className="sr-only"
        type="file"
        accept="image/jpeg,image/png,image/webp"
        capture="environment"
        onChange={(event) => choose(event.target.files?.[0])}
      />
      {busy && <div className="scan-line" aria-hidden="true" />}
    </div>
  );
}
