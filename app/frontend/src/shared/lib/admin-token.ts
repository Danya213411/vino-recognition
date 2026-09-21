const KEY = "vino-admin-token";

export function getAdminToken(): string {
  return typeof window === "undefined" ? "" : window.sessionStorage.getItem(KEY) ?? "";
}

export function setAdminToken(token: string): void {
  if (typeof window !== "undefined") window.sessionStorage.setItem(KEY, token);
}
