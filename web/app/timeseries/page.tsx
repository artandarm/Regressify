"use client";
import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import Link from "next/link";
import { uploadFile, analyzeTs } from "@/lib/api";
import type { UploadResult, TSAnalysisResponse } from "@/lib/types";
import { FileUpload, SpinnerFull } from "@/components/FileUpload";
import { AnalysisChart } from "@/components/AnalysisChart";
import { ProcessEquationGrid } from "@/components/ts/ProcessEquation";
import { SegmentCoefficientsTable } from "@/components/ts/CoefficientsTable";
import { PipelineLog } from "@/components/ts/PipelineLog";
import { DiagnosticsGrid } from "@/components/ts/DiagnosticsBlock";

export default function TimeSeriesPage() {
  const [upload, setUpload] = useState<UploadResult | null>(null);
  const [column, setColumn] = useState("");
  const [result, setResult] = useState<TSAnalysisResponse | null>(null);

  const uploadMut = useMutation({
    mutationFn: uploadFile,
    onSuccess: (data) => {
      setUpload(data);
      setColumn(data.columns[0] ?? "");
      setResult(null);
    },
  });

  const analyzeMut = useMutation({
    mutationFn: () => analyzeTs(column),
    onSuccess: setResult,
  });

  const error = uploadMut.error?.message ?? analyzeMut.error?.message ?? null;

  return (
    <div className="min-h-screen flex flex-col bg-base">
      {/* Nav */}
      <nav className="border-b border-edge px-6 py-3 flex items-center gap-3 bg-layer">
        <Link href="/" className="text-secondary hover:text-prose text-xs transition-colors">
          ← AllRegressions
        </Link>
        <span className="text-edge">·</span>
        <span className="text-xs text-prose font-medium">Time Series</span>
      </nav>

      <main className="flex-1 max-w-7xl w-full mx-auto px-6 py-8 flex flex-col gap-8">

        {/* Step 1 — Upload */}
        <section>
          <StepLabel n={1} label="Upload dataset" done={!!upload} />
          {!upload ? (
            <FileUpload onFile={uploadMut.mutate} loading={uploadMut.isPending} />
          ) : (
            <div
              className="flex items-center justify-between rounded-lg border border-edge bg-layer px-4 py-3 cursor-pointer hover:bg-raised transition-colors"
              onClick={() => {
                setUpload(null);
                setResult(null);
                uploadMut.reset();
                analyzeMut.reset();
              }}
            >
              <div>
                <p className="text-sm text-prose font-medium">{upload.filename}</p>
                <p className="text-xs text-secondary font-mono">
                  {upload.rows.toLocaleString()} rows · {upload.columns.length} columns
                </p>
              </div>
              <span className="text-xs text-muted hover:text-secondary">change ×</span>
            </div>
          )}
        </section>

        {/* Step 2 — Column + Run */}
        {upload && !analyzeMut.isPending && !result && (
          <section>
            <StepLabel n={2} label="Select column & run" done={false} />
            <div className="flex gap-3 items-end">
              <div className="flex-1">
                <label className="block text-xs text-secondary mb-1.5">
                  Time series column
                </label>
                <select
                  value={column}
                  onChange={(e) => setColumn(e.target.value)}
                  className="w-full bg-base border border-edge rounded-lg px-3 py-2.5 text-sm text-prose
                             focus:outline-none focus:border-accent transition-colors"
                >
                  {upload.columns.map((c) => (
                    <option key={c} value={c}>{c}</option>
                  ))}
                </select>
              </div>
              <button
                onClick={() => analyzeMut.mutate()}
                disabled={!column}
                className="px-5 py-2.5 rounded-lg bg-accent text-white text-sm font-medium
                           hover:opacity-90 disabled:opacity-40 transition-opacity"
              >
                Run analysis
              </button>
            </div>
          </section>
        )}

        {/* Loading */}
        {analyzeMut.isPending && (
          <SpinnerFull label="Running pipeline… this takes ~20–40 seconds" />
        )}

        {/* Error */}
        {error && (
          <div className="rounded-lg border border-bad/30 bg-bad-bg px-4 py-3 text-sm text-bad">
            {error}
          </div>
        )}

        {/* Results */}
        {result && (
          <section className="flex flex-col gap-6">
            <div className="flex items-center justify-between">
              <StepLabel n={3} label="Results" done />
              <button
                onClick={() => { setResult(null); analyzeMut.reset(); }}
                className="text-xs text-secondary hover:text-prose transition-colors"
              >
                ← re-run
              </button>
            </div>

            {/* Summary bar */}
            <div className="grid grid-cols-3 gap-3">
              <MetricCard label="Segments" value={result.segments.length} />
              <MetricCard
                label="Breakpoints"
                value={result.breakpoints.length === 0 ? "none" : result.breakpoints.length}
              />
              <MetricCard
                label="Seasonal period"
                value={result.seasonal_period ? `m = ${result.seasonal_period}` : "none"}
              />
            </div>

            {/* ── Equations — full width, stacked vertically ── */}
            <div className="flex flex-col gap-3">
              <SectionMeta label={result.segments.length > 1 ? "Process equations" : "Process equation"} />
              <ProcessEquationGrid segments={result.segments} />
            </div>

            {/* ── Chart + Diagnostics — side by side ── */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
              <div className="flex flex-col gap-3">
                <SectionMeta label="Series" />
                <AnalysisChart
                  seriesValues={result.series_values}
                  seriesOriginal={result.series_original}
                  breakpoints={result.breakpoints}
                  outliers={result.outliers}
                />
              </div>
              <div className="flex flex-col gap-3">
                <SectionMeta label="Residual diagnostics" />
                <DiagnosticsGrid segments={result.segments} />
              </div>
            </div>

            {/* ── Coefficients ── */}
            <div className="flex flex-col gap-3">
              <SectionMeta label={result.segments.length > 1 ? "Coefficients by segment" : "Coefficients"} />
              <div className={result.segments.length > 1 ? "grid grid-cols-1 lg:grid-cols-2 gap-4" : ""}>
                {result.segments.map((seg) => (
                  <div key={seg.segment_index} className="bg-layer border border-edge rounded-lg p-4">
                    {result.segments.length > 1 && (
                      <p className="text-[11px] font-mono text-muted uppercase tracking-wider mb-3">
                        Segment {seg.segment_index}
                      </p>
                    )}
                    <SegmentCoefficientsTable seg={seg} />
                  </div>
                ))}
              </div>
            </div>

            {/* ── Pipeline log — collapsed by default ── */}
            <details className="group">
              <summary className="cursor-pointer list-none flex items-center gap-2 select-none">
                <span className="text-[11px] font-mono tracking-wider text-secondary uppercase">
                  Analysis log
                </span>
                <span className="text-[10px] text-muted group-open:hidden">▼ show</span>
                <span className="text-[10px] text-muted hidden group-open:inline">▲ hide</span>
              </summary>
              <div className="mt-3">
                <PipelineLog
                  preSteps={result.pre_analysis_steps}
                  breakSteps={result.break_detection_steps}
                  segments={result.segments}
                />
              </div>
            </details>
          </section>
        )}
      </main>
    </div>
  );
}

// ── UI helpers ────────────────────────────────────────────────────────────

function StepLabel({ n, label, done }: { n: number; label: string; done: boolean }) {
  return (
    <div className="flex items-center gap-2 mb-4">
      <span
        className={[
          "flex items-center justify-center w-5 h-5 rounded-full text-xs font-mono",
          done
            ? "bg-ok-bg text-ok border border-ok/30"
            : "bg-raised text-secondary border border-edge",
        ].join(" ")}
      >
        {done ? "✓" : n}
      </span>
      <span className="text-xs font-medium text-secondary uppercase tracking-wider">
        {label}
      </span>
    </div>
  );
}

function SectionMeta({ label }: { label: string }) {
  return (
    <p className="text-[11px] font-mono tracking-wider text-secondary uppercase">{label}</p>
  );
}

function MetricCard({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="bg-layer border border-edge rounded-lg px-4 py-3">
      <p className="text-xs text-secondary mb-1">{label}</p>
      <p className="text-lg font-semibold text-prose font-mono">{value}</p>
    </div>
  );
}
