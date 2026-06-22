"use client";
import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import Link from "next/link";
import { uploadFile, analyzeTs } from "@/lib/api";
import type { UploadResult, TSAnalysisResponse } from "@/lib/types";
import { FileUpload, SpinnerFull } from "@/components/FileUpload";
import { AnalysisChart } from "@/components/AnalysisChart";
import { PipelineLog } from "@/components/PipelineLog";
import { SegmentCard } from "@/components/SegmentCard";

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
    <div className="min-h-screen flex flex-col">
      {/* Nav */}
      <nav className="border-b border-edge px-6 py-3 flex items-center gap-4">
        <Link href="/" className="text-secondary hover:text-prose text-xs transition-colors">
          ← AllRegressions
        </Link>
        <span className="text-edge">·</span>
        <span className="text-xs text-prose">Time Series</span>
      </nav>

      <main className="flex-1 max-w-4xl w-full mx-auto px-6 py-10 flex flex-col gap-10">

        {/* Step 1 — Upload */}
        <section>
          <SectionLabel n={1} label="Upload dataset" done={!!upload} />
          {!upload ? (
            <FileUpload onFile={uploadMut.mutate} loading={uploadMut.isPending} />
          ) : (
            <div
              className="flex items-center justify-between rounded-lg border border-edge bg-layer px-5 py-3 cursor-pointer hover:bg-raised transition-colors"
              onClick={() => { setUpload(null); setResult(null); uploadMut.reset(); analyzeMut.reset(); }}
            >
              <div>
                <p className="text-sm text-prose font-medium">{upload.filename}</p>
                <p className="text-xs text-secondary">{upload.rows} rows · {upload.columns.length} columns</p>
              </div>
              <span className="text-xs text-muted hover:text-secondary">change ×</span>
            </div>
          )}
        </section>

        {/* Step 2 — Column + Run */}
        {upload && !analyzeMut.isPending && !result && (
          <section>
            <SectionLabel n={2} label="Select column & run" done={false} />
            <div className="flex gap-3 items-end">
              <div className="flex-1">
                <label className="block text-xs text-secondary mb-1.5">Time series column</label>
                <select
                  value={column}
                  onChange={(e) => setColumn(e.target.value)}
                  className="w-full bg-layer border border-edge rounded-lg px-3 py-2.5 text-sm text-prose focus:outline-none focus:border-accent transition-colors"
                >
                  {upload.columns.map((c) => (
                    <option key={c} value={c}>{c}</option>
                  ))}
                </select>
              </div>
              <button
                onClick={() => analyzeMut.mutate()}
                disabled={!column}
                className="px-5 py-2.5 rounded-lg bg-accent text-base text-sm font-medium hover:opacity-90 disabled:opacity-40 transition-opacity"
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
          <div className="rounded-lg border border-bad/30 bg-bad/5 px-4 py-3 text-sm text-bad">
            {error}
          </div>
        )}

        {/* Step 3 — Results */}
        {result && (
          <section className="flex flex-col gap-8">
            <SectionLabel n={3} label="Results" done />

            {/* Summary metrics */}
            <div className="grid grid-cols-3 gap-3">
              <MetricCard label="Segments" value={result.segments.length} />
              <MetricCard label="Break points" value={result.breakpoints.length} />
              <MetricCard
                label="Seasonal period"
                value={result.seasonal_period ? `m = ${result.seasonal_period}` : "none"}
              />
            </div>

            {/* Chart */}
            <AnalysisChart
              seriesValues={result.series_values}
              breakpoints={result.breakpoints}
              outliers={result.outliers}
            />

            {/* Equations — prominently at top */}
            {result.segments.length > 0 && (
              <div>
                <h2 className="text-xs font-mono tracking-wider text-secondary uppercase mb-4">
                  Process {result.segments.length > 1 ? "equations" : "equation"}
                </h2>
                <div className={`grid gap-4 ${result.segments.length > 1 ? "grid-cols-1 lg:grid-cols-2" : "grid-cols-1"}`}>
                  {result.segments.map((seg) => (
                    <SegmentCard key={seg.segment_index} seg={seg} />
                  ))}
                </div>
              </div>
            )}

            {/* Analysis log */}
            <PipelineLog
              preSteps={result.pre_analysis_steps}
              breakSteps={result.break_detection_steps}
              segments={result.segments}
            />

            {/* Re-run */}
            <button
              onClick={() => { setResult(null); analyzeMut.reset(); }}
              className="self-start text-xs text-secondary hover:text-prose transition-colors"
            >
              ← Change column or re-run
            </button>
          </section>
        )}
      </main>
    </div>
  );
}

function SectionLabel({ n, label, done }: { n: number; label: string; done: boolean }) {
  return (
    <div className="flex items-center gap-2 mb-4">
      <span
        className={[
          "flex items-center justify-center w-5 h-5 rounded-full text-xs font-mono",
          done ? "bg-ok text-base" : "bg-raised text-secondary border border-edge",
        ].join(" ")}
      >
        {done ? "✓" : n}
      </span>
      <span className="text-xs font-medium text-secondary uppercase tracking-wider">{label}</span>
    </div>
  );
}

function MetricCard({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="bg-layer border border-edge rounded-xl px-4 py-3">
      <p className="text-xs text-secondary mb-1">{label}</p>
      <p className="text-xl font-semibold text-prose font-mono">{value}</p>
    </div>
  );
}
