const KEY = "vino-session-id";

export function getSessionId(): string {
  const stored = window.localStorage.getItem(KEY);
  if (stored) return stored;
  const created = globalThis.crypto?.randomUUID?.() ?? `local-${Date.now()}-${Math.random()}`;
  window.localStorage.setItem(KEY, created);
  return created;
}
