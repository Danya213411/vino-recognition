import Link from "next/link";

export function SiteFooter() {
  return (
    <footer className="site-footer">
      <div className="site-footer__top">
        <nav aria-label="Навигация в подвале">
          <Link href="/">Распознать вино</Link>
          <Link href="/calib">Калибровка</Link>
          <Link href="/review">Проверка выборки</Link>
          <Link href="/matrix">Матрица ошибок</Link>
          <Link href="/admin">Администрирование</Link>
        </nav>
        <p>Локальный сервис распознавания российских вин по фотографии этикетки.</p>
      </div>
      <div className="site-footer__bottom">
        <div>
          <p>© ХАКАТОН · распознавание российского вина</p>
          <p>Данные кадров остаются в локальном журнале сервиса</p>
        </div>
        <div className="age-mark"><span>Чрезмерное употребление алкоголя вредит вашему здоровью</span><b>18+</b></div>
      </div>
    </footer>
  );
}
