"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import { clearAdminRecognitions, deleteAdminRecognition, downloadAdminExport, loadAdmin, loadAdminImage } from "@/shared/api/client";
import type { AdminRecord, AdminStats } from "@/shared/api/contracts";
import { getAdminToken, setAdminToken } from "@/shared/lib/admin-token";
import { ImageLightbox } from "@/shared/ui/ImageLightbox";

const PAGE_SIZE = 30;
function formatMs(value: number | null) { return value == null ? "—" : `${Math.round(value)} мс`; }

function AdminThumb({ id, token, onOpen }: { id: string; token: string; onOpen: (source: string) => void }) {
  const [source, setSource] = useState<string>();
  useEffect(() => {
    let active = true;
    let objectUrl: string | undefined;
    loadAdminImage(id, token).then((url) => { objectUrl = url; if (active) setSource(url); }).catch(() => undefined);
    return () => { active = false; if (objectUrl) URL.revokeObjectURL(objectUrl); };
  }, [id, token]);
  return source ? <button className="admin-thumb-button" type="button" onClick={() => onOpen(source)} aria-label="Увеличить входной кадр">
    {/* eslint-disable-next-line @next/next/no-img-element */}
    <img className="admin-thumb" src={source} alt="Входной кадр" /><span className="image-zoom-mark" aria-hidden="true" />
  </button> : <span className="admin-thumb admin-thumb--empty">…</span>;
}

function ConfirmModal({ title, text, confirm, busy, onConfirm, onClose }: {
  title: string; text: string; confirm: string; busy: boolean; onConfirm: () => void; onClose: () => void;
}) {
  return <div className="confirm-backdrop" role="dialog" aria-modal="true" aria-labelledby="confirm-title" onMouseDown={onClose}><div className="confirm-modal" onMouseDown={(event) => event.stopPropagation()}><span className="confirm-modal__icon">!</span><h2 id="confirm-title">{title}</h2><p>{text}</p><div><button className="button button--ghost" onClick={onClose} disabled={busy}>Отмена</button><button className="button button--danger" onClick={onConfirm} disabled={busy}>{busy ? "Удаляем…" : confirm}</button></div></div></div>;
}

export function AdminDashboard() {
  const [token, setTokenState] = useState("");
  const [stats, setStats] = useState<AdminStats>();
  const [records, setRecords] = useState<AdminRecord[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [error, setError] = useState<string>();
  const [busy, setBusy] = useState(true);
  const [preview, setPreview] = useState<string>();
  const [confirm, setConfirm] = useState<{ kind: "one"; record: AdminRecord } | { kind: "all" }>();
  const [deleting, setDeleting] = useState(false);
  const pageCount = Math.max(1, Math.ceil(total / PAGE_SIZE));

  const refresh = useCallback(async (activePage = page, activeToken = token) => {
    setBusy(true); setError(undefined);
    try {
      const payload = await loadAdmin(activeToken, activePage, PAGE_SIZE);
      setStats(payload.stats); setRecords(payload.records.items); setTotal(payload.records.total);
    } catch (reason) { setError(reason instanceof Error ? reason.message : "Не удалось загрузить журнал"); }
    finally { setBusy(false); }
  }, [page, token]);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      const stored = getAdminToken(); setTokenState(stored); void refresh(1, stored);
    }, 0);
    return () => window.clearTimeout(timer);
    // The initial request is intentionally tied to the mounted dashboard.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function goToPage(value: number) {
    const next = Math.max(1, Math.min(pageCount, value));
    setPage(next); void refresh(next, token);
  }

  const pages = useMemo(() => {
    const start = Math.max(1, Math.min(page - 2, pageCount - 4));
    return Array.from({ length: Math.min(5, pageCount) }, (_, index) => start + index);
  }, [page, pageCount]);

  async function removeConfirmed() {
    if (!confirm) return;
    setDeleting(true); setError(undefined);
    try {
      const nextPage = confirm.kind === "all" ? 1 : records.length === 1 && page > 1 ? page - 1 : page;
      if (confirm.kind === "all") await clearAdminRecognitions(token);
      else await deleteAdminRecognition(confirm.record.id, token);
      setPage(nextPage); setConfirm(undefined); await refresh(nextPage, token);
    } catch (reason) { setError(reason instanceof Error ? reason.message : "Не удалось удалить запись"); }
    finally { setDeleting(false); }
  }

  return (
    <div className="admin-shell">
      <div className="admin-toolbar">
        <label><span>Admin token <small>(локально можно оставить пустым)</small></span><input value={token} onChange={(event) => { setTokenState(event.target.value); setAdminToken(event.target.value); }} type="password" placeholder="VINO_ADMIN_TOKEN" /></label>
        <button className="button button--ghost" onClick={() => void refresh()} disabled={busy}>Обновить</button>
        <button className="button button--primary" onClick={() => downloadAdminExport(token)}>Выгрузить журнал</button>
        <button className="button button--danger-outline" onClick={() => setConfirm({ kind: "all" })} disabled={total === 0}>Очистить таблицу</button>
      </div>
      {error && <div className="notice notice--error">{error}</div>}
      {stats && <div className="stats-grid"><article><span>Всего кадров</span><b>{stats.recognitions.total}</b></article><article><span>Уверенных</span><b>{stats.recognitions.matches}</b></article><article><span>Нужна проверка</span><b>{stats.recognitions.alternatives}</b></article><article><span>p95 ответа</span><b>{formatMs(stats.timing_ms.p95)}</b></article><article><span>Ответов людей</span><b>{stats.feedback.total}</b></article></div>}
      <section className="admin-table-wrap">
        <div className="section-heading"><div><h2>Распознавания</h2><p>Нажмите на строку, чтобы открыть полный разбор.</p></div><span>{busy ? "Обновляем…" : `${total} записей · по ${PAGE_SIZE}`}</span></div>
        <div className="admin-table">
          <div className="admin-row admin-row--head"><span>Кадр</span><span>Пользователь / время</span><span>Результат</span><span>Сигнал</span><span>Обратная связь</span><span></span></div>
          {records.map((record) => <article className="admin-row" key={record.id}>
            <AdminThumb id={record.id} token={token} onOpen={setPreview} />
            <Link className="admin-row__link" href={`/admin/detail/?id=${record.id}`} aria-label={`Открыть распознавание ${record.id}`} />
            <div><b>{record.session_id.slice(0, 18)}</b><small>{new Date(record.created_at).toLocaleString("ru-RU")} · {record.mode}</small></div>
            <div><b>{record.result.wine.title}</b><small>{record.decision} · {record.candidate_slug}</small></div>
            <div><b>{Math.round(record.confidence * 100)}%</b><small>{Math.round(record.total_ms)} мс</small></div>
            <div><b>{record.feedback?.verdict ?? "—"}</b><small>{record.feedback?.correct_slug ?? "нет ответа"}</small></div>
            <button className="icon-button icon-button--danger" type="button" onClick={() => setConfirm({ kind: "one", record })} aria-label="Удалить скан"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="m7 7 10 10M17 7 7 17" /></svg></button>
          </article>)}
          {!busy && records.length === 0 && <p className="table-empty">Пока нет загруженных фотографий.</p>}
        </div>
        {total > PAGE_SIZE && <nav className="pagination" aria-label="Страницы журнала"><button onClick={() => goToPage(page - 1)} disabled={page === 1}>←</button>{pages.map((value) => <button key={value} className={value === page ? "is-active" : undefined} onClick={() => goToPage(value)}>{value}</button>)}<button onClick={() => goToPage(page + 1)} disabled={page === pageCount}>→</button><span>Страница {page} из {pageCount}</span></nav>}
      </section>
      <ImageLightbox src={preview} title="Исходный кадр" subtitle="Фотография, отправленная пользователем" onClose={() => setPreview(undefined)} />
      {confirm && <ConfirmModal title={confirm.kind === "all" ? "Удалить весь журнал?" : "Удалить этот скан?"} text={confirm.kind === "all" ? `Будут безвозвратно удалены ${total} записей, все ответы калибровки и загруженные фотографии.` : `Запись «${confirm.record.result.wine.title}» и исходная фотография будут удалены безвозвратно.`} confirm={confirm.kind === "all" ? `Да, удалить все ${total}` : "Да, удалить скан"} busy={deleting} onConfirm={() => void removeConfirmed()} onClose={() => !deleting && setConfirm(undefined)} />}
    </div>
  );
}
