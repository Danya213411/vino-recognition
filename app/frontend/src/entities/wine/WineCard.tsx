import type { Decision, Wine } from "@/shared/api/contracts";

const outcomeCopy: Record<Decision, { eyebrow: string; title: string }> = {
  match: { eyebrow: "Совпадение найдено", title: "Это вино" },
  alternatives: { eyebrow: "Нужно подтверждение", title: "Больше всего похоже на" },
  not_found: { eyebrow: "Нет уверенного совпадения", title: "Возможно, этого вина нет в каталоге" },
};

export function WineCard({ wine, decision }: { wine: Wine; decision: Decision }) {
  const copy = outcomeCopy[decision];
  return (
    <article className={`wine-card wine-card--${decision}`}>
      <div className="wine-card__visual" style={{ background: wine.color_gradient ?? undefined }}>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={wine.image_url} alt={wine.title} />
      </div>
      <div className="wine-card__content">
        <span className="eyebrow">{copy.eyebrow}</span>
        <p className="result-prefix">{copy.title}</p>
        <h2>{wine.title}</h2>
        <p className="wine-maker">{wine.manufacturer}</p>
        <div className="chips">
          {wine.region && <span>{wine.region}</span>}
          {wine.category && <span>{wine.category}</span>}
          {wine.alcohol && <span>{wine.alcohol}%</span>}
        </div>
        {decision !== "not_found" && wine.description && (
          <p className="wine-description">{wine.description.split("\n")[0]}</p>
        )}
        {wine.grapes && wine.grapes.length > 0 && (
          <dl className="wine-facts">
            <div><dt>Сорта</dt><dd>{wine.grapes.join(", ")}</dd></div>
            {wine.dishes && wine.dishes.length > 0 && (
              <div><dt>К столу</dt><dd>{wine.dishes.slice(0, 3).join(", ")}</dd></div>
            )}
          </dl>
        )}
        {wine.source_url && decision === "match" && (
          <a className="text-link" href={wine.source_url} target="_blank" rel="noreferrer">
            Открыть карточку на «Своё Вино» ↗
          </a>
        )}
      </div>
    </article>
  );
}
