"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const links = [
  { href: "/", label: "Распознать" },
  { href: "/calib/", label: "Калибровка" },
  { href: "/review/", label: "Выборка" },
  { href: "/admin/", label: "Администрирование" },
];

export function SiteHeader() {
  const pathname = usePathname();
  return (
    <header className="site-header-wrap">
      <div className="site-header">
        <Link href="/" className="brand" aria-label="Хакатон — на главную">ХАКАТОН</Link>
        <nav aria-label="Основная навигация">
          {links.map((link) => {
            const active = link.href === "/" ? pathname === "/" : pathname.startsWith(link.href.slice(0, -1));
            return <Link key={link.href} href={link.href} className={active ? "is-active" : undefined}>{link.label}</Link>;
          })}
        </nav>
        <Link href="/" className="header-search" aria-label="Распознать вино">
          <span>Распознать вино</span>
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="m21 21-4.35-4.35m2.35-5.15a7.5 7.5 0 1 1-15 0 7.5 7.5 0 0 1 15 0Z" /></svg>
        </Link>
      </div>
    </header>
  );
}
