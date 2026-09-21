import type {
  AdminRecord,
  AdminStats,
  Analysis,
  FeedbackVerdict,
} from "./contracts";

const API_BASE = process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ?? "";

async function checked<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new Error(payload?.detail ?? `Ошибка API: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export async function analyzePhoto(
  file: File,
  mode: "user" | "calib",
  sessionId: string,
): Promise<Analysis> {
  const form = new FormData();
  form.append("file", file);
  form.append("mode", mode);
  return checked<Analysis>(
    await fetch(`${API_BASE}/api/v1/analyze`, {
      method: "POST",
      headers: { "X-Session-ID": sessionId },
      body: form,
    }),
  );
}

export async function sendFeedback(
  recognitionId: string,
  verdict: FeedbackVerdict,
  sessionId: string,
  correctSlug?: string,
): Promise<void> {
  await checked(
    await fetch(`${API_BASE}/api/v1/recognitions/${recognitionId}/feedback`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Session-ID": sessionId,
      },
      body: JSON.stringify({ verdict, correct_slug: correctSlug }),
    }),
  );
}

export async function downloadManifest(
  analysis: Analysis,
  sessionId: string,
): Promise<void> {
  const response = await fetch(`${API_BASE}${analysis.manifest_url}`, {
    headers: { "X-Session-ID": sessionId },
  });
  if (!response.ok) throw new Error("Не удалось скачать манифест");
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `vino-manifest-${analysis.id}.json`;
  link.click();
  URL.revokeObjectURL(url);
}

function adminHeaders(token: string): HeadersInit {
  return token ? { Authorization: `Bearer ${token}` } : {};
}

export async function loadAdmin(token: string): Promise<{
  stats: AdminStats;
  records: { total: number; items: AdminRecord[] };
}>;
export async function loadAdmin(token: string, page: number, pageSize?: number): Promise<{
  stats: AdminStats;
  records: { total: number; items: AdminRecord[] };
}>;
export async function loadAdmin(token: string, page = 1, pageSize = 30): Promise<{
  stats: AdminStats;
  records: { total: number; items: AdminRecord[] };
}> {
  const headers = adminHeaders(token);
  const offset = Math.max(0, page - 1) * pageSize;
  const [stats, records] = await Promise.all([
    checked<AdminStats>(await fetch(`${API_BASE}/api/v1/admin/stats`, { headers })),
    checked<{ total: number; items: AdminRecord[] }>(
      await fetch(`${API_BASE}/api/v1/admin/recognitions?limit=${pageSize}&offset=${offset}`, { headers }),
    ),
  ]);
  return { stats, records };
}

export async function loadAdminRecord(id: string, token: string): Promise<AdminRecord> {
  return checked<AdminRecord>(
    await fetch(`${API_BASE}/api/v1/admin/recognitions/${id}`, {
      headers: adminHeaders(token),
    }),
  );
}

export async function loadAdminImage(id: string, token: string): Promise<string> {
  const response = await fetch(`${API_BASE}/api/v1/admin/recognitions/${id}/image`, {
    headers: adminHeaders(token),
  });
  if (!response.ok) throw new Error("Фото недоступно");
  return URL.createObjectURL(await response.blob());
}

export async function downloadAdminExport(token: string): Promise<void> {
  const response = await fetch(`${API_BASE}/api/v1/admin/export`, {
    headers: adminHeaders(token),
  });
  if (!response.ok) throw new Error("Экспорт недоступен");
  const url = URL.createObjectURL(await response.blob());
  const link = document.createElement("a");
  link.href = url;
  link.download = "vino-admin-export.json";
  link.click();
  URL.revokeObjectURL(url);
}

export async function deleteAdminRecognition(id: string, token: string): Promise<void> {
  await checked(
    await fetch(`${API_BASE}/api/v1/admin/recognitions/${id}`, {
      method: "DELETE",
      headers: adminHeaders(token),
    }),
  );
}

export async function clearAdminRecognitions(token: string): Promise<void> {
  await checked(
    await fetch(`${API_BASE}/api/v1/admin/recognitions`, {
      method: "DELETE",
      headers: adminHeaders(token),
    }),
  );
}
