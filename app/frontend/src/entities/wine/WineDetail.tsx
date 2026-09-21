/* eslint-disable @next/next/no-img-element */
import type { Decision, Wine } from "@/shared/api/contracts";

function backgroundFor(wine: Wine) {
  const grapes = wine.grapes?.join(" ").toLocaleLowerCase("ru") ?? "";
  const kind = `${wine.category ?? ""} ${wine.color ?? ""}`.toLocaleLowerCase("ru");
  if (grapes.includes("пино нуар")) return "/media/wine-bg-pinot.webp";
  if (kind.includes("оранж") || kind.includes("янтар")) return "/media/wine-bg-orange.webp";
  if (kind.includes("роз")) return "/media/wine-bg-rose.webp";
  if (kind.includes("бел")) return "/media/wine-bg-white.webp";
  return "/media/wine-bg-red.webp";
}

function servingTemperature(wine: Wine) {
  if (wine.temperature) return /°|c/i.test(wine.temperature) ? wine.temperature : `${wine.temperature}°C`;
  const kind = `${wine.category ?? ""} ${wine.color ?? ""}`.toLocaleLowerCase("ru");
  if (kind.includes("игрист")) return "6–8°C";
  if (kind.includes("бел") || kind.includes("роз")) return "8–12°C";
  if (kind.includes("оранж") || kind.includes("янтар")) return "10–14°C";
  return "16–18°C";
}

function dishIcon(dish: string) {
  const value = dish.toLocaleLowerCase("ru");
  if (value.includes("выпеч") || value.includes("десерт")) return "🥐";
  if (value.includes("мяс") || value.includes("стейк")) return "🥩";
  if (value.includes("сыр")) return "🧀";
  if (value.includes("рыб") || value.includes("морепродукт")) return "🐟";
  if (value.includes("паст") || value.includes("макарон")) return "🍝";
  return "🍽";
}

function CardIcon({ kind }: { kind: "temperature" | "alcohol" | "dishes" }) {
  if (kind === "temperature") return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 5a3 3 0 0 1 6 0v8.25a5 5 0 1 1-6 0V5Z"/><path d="M12 8v7"/></svg>;
  if (kind === "alcohol") return <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="7" cy="7" r="2.5"/><circle cx="17" cy="17" r="2.5"/><path d="m18.5 5.5-13 13"/></svg>;
  return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="m4 3 7 7M8 3l-4 4 5 5 3-3M14 4l6 6M17 3l4 4-9 9M8 21l4-4M14 14l7 7"/></svg>;
}

export function WineDetail({ wine, action }: {
  wine: Wine;
  decision: Decision;
  action: React.ReactNode;
}) {
  const background = backgroundFor(wine);
  return (
    <>
      <div className="wine-breadcrumbs"><span>Главная</span><i>›</i><span>Свои вина</span><i>›</i><b>{wine.title}</b></div>
      <section className="wine-detail">
        <div className="wine-detail__info">
          <div className="wine-detail__action-mobile">{action}</div>
          <div className="wine-detail__heading">
            <h1>{wine.title}</h1>
            <p className="wine-detail__maker">{wine.manufacturer ?? "Производитель не указан"}</p>
          </div>
          <div className="wine-detail__facts">
            <div><span className="wine-detail__fact-thumb"><img src="/media/wine-bg-white.webp" alt="" /></span><p><span>Регион</span><b>{wine.region ?? "Не указан"}</b></p></div>
            <div><span className="wine-detail__fact-thumb"><img src={background} alt="" /></span><p><span>Сорт винограда</span><b>{wine.grapes?.join(", ") || "Не указан"}</b></p></div>
            <div><span className="wine-detail__fact-thumb wine-detail__fact-color" style={{ background: wine.color_gradient ?? "#941c3d" }} /><p><span>Категория и цвет</span><b>{wine.category ?? "Не указана"}</b>{wine.color && <small>{wine.color}</small>}</p></div>
          </div>
        </div>
        <div className="wine-detail__visual">
          <img className="wine-detail__backdrop" src={background} alt="" />
          <div className="wine-detail__action">{action}</div>
          <img className="wine-detail__bottle" src={wine.image_url} alt={wine.title} />
          <div className="wine-detail__cards">
            <article><i><CardIcon kind="temperature" /></i><span>Температура<br />подачи</span><b>{servingTemperature(wine)}</b></article>
            <article><i><CardIcon kind="alcohol" /></i><span>Крепость<br />вина</span><b>{wine.alcohol != null ? `${wine.alcohol}%` : "—"}</b></article>
            <article className="wine-detail__dishes"><div className="wine-detail__dishes-title"><i><CardIcon kind="dishes" /></i><span>Сочетание<br />с блюдами</span></div><div>{wine.dishes?.slice(0, 4).map((dish) => <em key={dish}><span>{dishIcon(dish)}</span><small>{dish}</small></em>) ?? <em><small>Не указано</small></em>}</div></article>
          </div>
        </div>
      </section>
      {wine.description && <p className="wine-detail__description">{wine.description}</p>}
      {wine.source_url && <div className="wine-detail__source"><a href={wine.source_url} target="_blank" rel="noreferrer">Открыть исходную карточку на vino-svoe.ru ↗</a></div>}
    </>
  );
}
