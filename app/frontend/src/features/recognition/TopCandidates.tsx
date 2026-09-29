import type { Prediction } from "@/shared/api/contracts";

function percentage(value: number) {
  return `${Math.round(value * 100)}%`;
}

export function TopCandidates({ predictions }: { predictions: Prediction[] }) {
  const candidates = predictions.slice(0, 5);
  if (!candidates.length) return null;

  return (
    <section className="top-candidates" aria-labelledby="top-candidates-title">
      <div className="top-candidates__intro">
        <div>
          <span className="eyebrow">Проверка результата</span>
          <h2 id="top-candidates-title">Ещё варианты распознавания</h2>
        </div>
        <p>Если главный результат не похож на вашу бутылку, посмотрите ближайшие совпадения из того же анализа.</p>
      </div>
      <div className="top-candidates__grid">
        {candidates.map((candidate, index) => (
          <article className="top-candidate" key={candidate.slug}>
            <div className="top-candidate__visual">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={candidate.wine.image_url} alt={candidate.wine.title} loading="lazy" />
              <span>#{index + 1}</span>
            </div>
            <div className="top-candidate__content">
              <h3>{candidate.wine.title}</h3>
              <p>{candidate.wine.manufacturer || "Производитель не указан"}</p>
              <small>{candidate.wine.region || candidate.wine.category || "Категория не указана"}</small>
              <div className="top-candidate__support">
                <span><i style={{ width: percentage(candidate.relative_support) }} /></span>
                <b>{percentage(candidate.relative_support)}</b>
              </div>
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}
