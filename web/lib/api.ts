import type { UploadResult, TSAnalysisResponse, OLSAnalysisResponse } from "./types";

const BASE = "/api/py";

export async function uploadFile(file: File): Promise<UploadResult> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/upload`, { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Upload failed");
  }
  return res.json();
}

export async function analyzeTs(column: string): Promise<TSAnalysisResponse> {
  const res = await fetch(`${BASE}/analyze/ts?column=${encodeURIComponent(column)}`, {
    method: "POST",
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Analysis failed");
  }
  return res.json();
}

export async function analyzeOls(yCol: string, xCols: string[]): Promise<OLSAnalysisResponse> {
  const res = await fetch(`${BASE}/analyze/ols`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ y_col: yCol, x_cols: xCols }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Analysis failed");
  }
  return res.json();
}
