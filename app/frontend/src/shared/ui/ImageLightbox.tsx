"use client";

import { useEffect } from "react";

type Props = {
  src?: string;
  title?: string;
  subtitle?: string;
  onClose: () => void;
};

export function ImageLightbox({ src, title, subtitle, onClose }: Props) {
  useEffect(() => {
    if (!src) return;
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => {
      document.body.style.overflow = previous;
      window.removeEventListener("keydown", closeOnEscape);
    };
  }, [src, onClose]);

  if (!src) return null;
  return (
    <div className="lightbox" role="dialog" aria-modal="true" aria-label={title ?? "Просмотр изображения"} onMouseDown={onClose}>
      <button className="lightbox__close" type="button" onClick={onClose} aria-label="Закрыть"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="m7 7 10 10M17 7 7 17" /></svg></button>
      <div className="lightbox__content" onMouseDown={(event) => event.stopPropagation()}>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={src} alt={title ?? "Увеличенное изображение"} />
        {(title || subtitle) && (
          <div className="lightbox__caption">
            {title && <strong>{title}</strong>}
            {subtitle && <span>{subtitle}</span>}
          </div>
        )}
      </div>
    </div>
  );
}
