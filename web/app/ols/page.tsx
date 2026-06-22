"use client";
import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import Link from "next/link";
import katex from "katex";
import "katex/dist/katex.min.css";
import { uploadFile, analyzeOls } from "@/lib/api";
import type { UploadResult, OLSAnalysisResponse, VifEntry } from "@/lib/types";
import { FileUpload, SpinnerFull } from "@/components/FileUpload";
import { OlsPipelineLog } from "@/components/OlsPipelineLog";
import { OlsScatterChart } from "@/components/OlsScatterChart";

export default function OlsPage() {
  const [upload, setUpload] = useState<UploadResult | null>(null);
  const [yCol, setYCol] = useState("");
  const [xCols, setXCols] = useState<string[]>([]);
  const [result, setResult] = useState<OLSAnalysisResponse | null>(null);

  const uploadMut = useMutation({
    mutationFn: uploadFile,
    onSuccess: (data) => {
      setUpload(data);
      setYCol(data.columns[0] ?? "");
      setXCols(data.columns.slice(1));
      setResult(null);
    },
  });

  const analyzeMut = useMutation({
    mutationFn: () => analyzeOls(yCol, xCols),
    onSuccess: setResult,
  });

  const error = uploadMut.error?.message ?? analyzeMut.error?.message ?? null;

  const xOptions = upload?.columns.filter((c) => c !== yCol) ?? [];

  const handleYChange = (newY: string) => {
    setYCol(newY);
    setXCols((prev) => prev.filter((c) => c !== newY));
  };

  const toggleX = (col: string) => {
    setXCols((prev) =>
      prev.includes(col) ? prev.filter((c) => c !== col) : [...prev, col]
    );
  };

  const canRun = !!yCol && xCols.length > 0;

  return (
    <div className="min-h-screen flex flex-col">
      {/* Nav */}
      <nav className="border-b border-edge px-6 py-3 flex items-center gap-4">
        <Link href="/" className="text-secondary hover:text-prose text-xs transition-colors">
          ← AllRegressions
        </Link>
        <span className="text-edge">·</span>
        <span className="text-xs text-prose">Cross-section OLS</span>
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
              onClick={() => {
                setUpload(null);
                setResult(null);
                uploadMut.reset();
                analyzeMut.reset();
              }}
            >
              <div>
                <p className="text-sm text-prose font-medium">{upload.filename}</p>
                <p className="text-xs text-secondary">
                  {upload.rows} rows · {upload.columns.length} columns
                </p>
              </div>
              <span className="text-xs text-muted hover:text-secondary">change ×</span>
            </div>
          )}
        </section>

        {/* Step 2 — Variables + Run */}
        {upload && !analyzeMut.isPending && !result && (
          <section>
            <SectionLabel n={2} label="Select variables & run" done={false} />
            <div className="flex flex-col gap-5">

              {/* Dataset preview */}
              <DatasetPreview columns={upload.columns} rows={upload.preview} />

              {/* Y column */}
              <div>
                <label className="block text-xs text-secondary mb-1.5">
                  Dependent variable (Y)
                </label>
                <select
                  value={yCol}
                  onChange={(e) => handleYChange(e.target.value)}
                  className="w-full bg-layer border border-edge rounded-lg px-3 py-2.5 text-sm text-prose focus:outline-none focus:border-accent transition-colors"
                >
                  {upload.columns.map((c) => (
                    <option key={c} value={c}>{c}</option>
                  ))}
                </select>
              </div>

              {/* X columns */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <label className="text-xs text-secondary">
                    Regressors (X) — select one or more
                  </label>
                  <div className="flex gap-3">
                    <button
                      onClick={() => setXCols([...xOptions])}
                      className="text-xs text-secondary hover:text-prose transition-colors"
                    >
                      all
                    </button>
                    <button
                      onClick={() => setXCols([])}
                      className="text-xs text-secondary hover:text-prose transition-colors"
                    >
                      none
                    </button>
                  </div>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                  {xOptions.map((col) => {
                    const checked = xCols.includes(col);
                    return (
                      <button
                        key={col}
                        onClick={() => toggleX(col)}
                        className={[
                          "flex items-center gap-2 px-3 py-2 rounded-lg border text-xs text-left transition-colors",
                          checked
                            ? "border-accent bg-accent/10 text-prose"
                            : "border-edge bg-layer text-secondary hover:bg-raised",
                        ].join(" ")}
                      >
                        <span
                          className={[
                            "flex items-center justify-center w-3.5 h-3.5 rounded border text-[9px] shrink-0",
                            checked ? "border-accent bg-accent text-base" : "border-edge",
                          ].join(" ")}
                        >
                          {checked && "✓"}
                        </span>
                        <span className="truncate">{col}</span>
                      </button>
                    );
                  })}
                </div>

                {xCols.length === 0 && (
                  <p className="text-xs text-muted mt-2">Select at least one regressor</p>
                )}
              </div>

              <button
                onClick={() => analyzeMut.mutate()}
                disabled={!canRun}
                className="self-start px-5 py-2.5 rounded-lg bg-accent text-base text-sm font-medium hover:opacity-90 disabled:opacity-40 transition-opacity"
              >
                Run analysis
              </button>
            </div>
          </section>
        )}

        {/* Loading */}
        {analyzeMut.isPending && (
          <SpinnerFull label="Running OLS pipeline… this may take a few seconds" />
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

            {/* Summary cards */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <MetricCard label="Observations" value={result.n_obs} />
              <MetricCard label="Adj. R²" value={result.adj_r_squared.toFixed(4)} />
              <MetricCard
                label="F-test"
                value={`${result.f_statistic.toFixed(2)}`}
                sub={`p = ${result.f_pvalue < 0.001 ? "<0.001" : result.f_pvalue.toFixed(4)}`}
              />
              <MetricCard
                label="Model"
                value={result.model_type === "OLS_robust_HC3" ? "HC3 Robust" : "OLS"}
                highlight={result.model_type === "OLS_robust_HC3" ? "warn" : "ok"}
              />
            </div>

            {/* Equation — LaTeX */}
            <div className="bg-layer border border-edge rounded-xl px-5 py-4">
              <p className="text-xs text-secondary mb-3 font-mono tracking-wider uppercase">
                Estimated equation
              </p>
              <OlsEquation latex={result.equation_latex} fallback={result.equation} />
              <div className="flex gap-4 mt-3 text-xs text-muted font-mono">
                <span>AIC = {result.aic.toFixed(2)}</span>
                <span>BIC = {result.bic.toFixed(2)}</span>
                <span>R² = {result.r_squared.toFixed(4)}</span>
              </div>
            </div>

            {/* Scatter chart */}
            <OlsScatterChart
              yActual={result.y_actual}
              yFitted={result.y_fitted}
              yCol={result.y_col}
              influentialIdx={result.influential_obs.map((o) => o.index)}
            />

            {/* Coefficients table */}
            <div>
              <h2 className="text-xs font-mono tracking-wider text-secondary uppercase mb-3">
                Coefficients
              </h2>
              <div className="overflow-x-auto rounded-xl border border-edge">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-edge bg-layer">
                      <th className="text-left px-4 py-2.5 text-secondary font-medium">Variable</th>
                      <th className="text-right px-4 py-2.5 text-secondary font-medium">Coef</th>
                      <th className="text-right px-4 py-2.5 text-secondary font-medium">Std err</th>
                      <th className="text-right px-4 py-2.5 text-secondary font-medium">t-stat</th>
                      <th className="text-right px-4 py-2.5 text-secondary font-medium">p-value</th>
                      <th className="text-center px-4 py-2.5 text-secondary font-medium">sig</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.coefficients.map((c) => (
                      <tr key={c.name} className="border-b border-edge last:border-0 hover:bg-raised">
                        <td className="px-4 py-2.5 font-mono text-prose">{c.name}</td>
                        <td className="px-4 py-2.5 font-mono text-right text-prose">{c.coef.toFixed(4)}</td>
                        <td className="px-4 py-2.5 font-mono text-right text-secondary">{c.std_err.toFixed(4)}</td>
                        <td className="px-4 py-2.5 font-mono text-right text-secondary">{c.t_stat.toFixed(3)}</td>
                        <td className={`px-4 py-2.5 font-mono text-right ${c.p_value < 0.05 ? "text-ok" : "text-warn"}`}>
                          {c.p_value < 0.001 ? "<0.001" : c.p_value.toFixed(4)}
                        </td>
                        <td className="px-4 py-2.5 text-center">
                          {c.significant
                            ? <span className="text-ok">✓</span>
                            : <span className="text-warn">—</span>}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Removed vars */}
            {result.removed_vars.length > 0 && (
              <div>
                <h2 className="text-xs font-mono tracking-wider text-secondary uppercase mb-3">
                  Removed by backward stepwise (BIC)
                </h2>
                <div className="overflow-x-auto rounded-xl border border-edge">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="border-b border-edge bg-layer">
                        <th className="text-left px-4 py-2.5 text-secondary font-medium">Variable</th>
                        <th className="text-right px-4 py-2.5 text-secondary font-medium">p-value</th>
                        <th className="text-right px-4 py-2.5 text-secondary font-medium">BIC before</th>
                        <th className="text-right px-4 py-2.5 text-secondary font-medium">BIC after</th>
                        <th className="text-right px-4 py-2.5 text-secondary font-medium">ΔBIC</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.removed_vars.map((v) => (
                        <tr key={v.variable} className="border-b border-edge last:border-0">
                          <td className="px-4 py-2.5 font-mono text-prose">{v.variable}</td>
                          <td className="px-4 py-2.5 font-mono text-right text-warn">{v.pvalue.toFixed(4)}</td>
                          <td className="px-4 py-2.5 font-mono text-right text-secondary">{v.bic_before.toFixed(2)}</td>
                          <td className="px-4 py-2.5 font-mono text-right text-ok">{v.bic_after.toFixed(2)}</td>
                          <td className="px-4 py-2.5 font-mono text-right text-ok">
                            {(v.bic_after - v.bic_before).toFixed(2)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* VIF table */}
            <div>
              <h2 className="text-xs font-mono tracking-wider text-secondary uppercase mb-3">
                Multicollinearity (VIF)
              </h2>
              <div className="overflow-x-auto rounded-xl border border-edge">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-edge bg-layer">
                      <th className="text-left px-4 py-2.5 text-secondary font-medium">Variable</th>
                      <th className="text-right px-4 py-2.5 text-secondary font-medium">VIF</th>
                      <th className="text-left px-4 py-2.5 text-secondary font-medium">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.vif_table.map((v) => (
                      <tr key={v.variable} className="border-b border-edge last:border-0 hover:bg-raised">
                        <td className="px-4 py-2.5 font-mono text-prose">{v.variable}</td>
                        <td className={`px-4 py-2.5 font-mono text-right ${vifColor(v)}`}>
                          {v.vif.toFixed(2)}
                        </td>
                        <td className={`px-4 py-2.5 ${vifColor(v)}`}>{v.note}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="text-xs text-muted mt-1.5">
                κ = {result.condition_number.toFixed(1)} (condition number)
              </p>
            </div>

            {/* Influential observations */}
            {result.influential_obs.length > 0 && (
              <div>
                <h2 className="text-xs font-mono tracking-wider text-secondary uppercase mb-1">
                  Influential observations
                </h2>
                <p className="text-xs text-muted mb-3">
                  Cook&apos;s D &gt; 4/n = {(4 / result.n_obs).toFixed(4)} — inspect manually,
                  not removed automatically
                </p>
                <div className="overflow-x-auto rounded-xl border border-edge">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="border-b border-edge bg-layer">
                        <th className="text-right px-4 py-2.5 text-secondary font-medium">Index</th>
                        <th className="text-right px-4 py-2.5 text-secondary font-medium">Cook&apos;s D</th>
                        <th className="text-right px-4 py-2.5 text-secondary font-medium">Leverage</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.influential_obs.map((o) => (
                        <tr key={o.index} className="border-b border-edge last:border-0 hover:bg-raised">
                          <td className="px-4 py-2.5 font-mono text-right text-prose">{o.index}</td>
                          <td className="px-4 py-2.5 font-mono text-right text-warn">{o.cooks_d.toFixed(4)}</td>
                          <td className="px-4 py-2.5 font-mono text-right text-secondary">{o.leverage.toFixed(4)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Pipeline log */}
            <OlsPipelineLog
              preSteps={result.pre_analysis_steps}
              multicollinearitySteps={result.multicollinearity_steps}
              modelEstimationSteps={result.model_estimation_steps}
              variableSelectionSteps={result.variable_selection_steps}
              diagnosticsSteps={result.diagnostics_steps}
            />

            {/* Re-run */}
            <button
              onClick={() => { setResult(null); analyzeMut.reset(); }}
              className="self-start text-xs text-secondary hover:text-prose transition-colors"
            >
              ← Change variables or re-run
            </button>
          </section>
        )}
      </main>
    </div>
  );
}

function vifColor(v: VifEntry): string {
  return v.verdict === "error" ? "text-bad" :
         v.verdict === "warn" ? "text-warn" :
         "text-ok";
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

function DatasetPreview({
  columns,
  rows,
}: {
  columns: string[];
  rows: Record<string, unknown>[];
}) {
  return (
    <div className="rounded-xl border border-edge overflow-hidden">
      <div className="px-4 py-2.5 bg-layer border-b border-edge flex items-center justify-between">
        <span className="text-xs font-medium text-secondary">
          Dataset preview
        </span>
        <span className="text-xs text-muted font-mono">
          {columns.length} columns · first {rows.length} rows
        </span>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-edge bg-raised">
              {columns.map((col) => (
                <th
                  key={col}
                  className="text-left px-3 py-2 text-secondary font-medium whitespace-nowrap"
                >
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={i} className="border-b border-edge last:border-0 hover:bg-raised">
                {columns.map((col) => {
                  const val = row[col];
                  const isNum = typeof val === "number";
                  return (
                    <td
                      key={col}
                      className={[
                        "px-3 py-2 font-mono whitespace-nowrap",
                        isNum ? "text-prose text-right" : "text-secondary",
                      ].join(" ")}
                    >
                      {val === null || val === undefined
                        ? <span className="text-muted italic">—</span>
                        : isNum
                        ? (Number.isInteger(val) ? String(val) : (val as number).toFixed(4))
                        : String(val)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function OlsEquation({ latex, fallback }: { latex: string; fallback: string }) {
  let html = "";
  try {
    html = katex.renderToString(latex, { throwOnError: false, displayMode: true });
  } catch {
    html = `<code class="text-sm font-mono text-prose">${fallback}</code>`;
  }
  return (
    <div
      dangerouslySetInnerHTML={{ __html: html }}
      className="overflow-x-auto text-prose py-1"
    />
  );
}

function MetricCard({
  label,
  value,
  sub,
  highlight,
}: {
  label: string;
  value: string | number;
  sub?: string;
  highlight?: "ok" | "warn";
}) {
  const valClass = highlight === "warn" ? "text-warn" : highlight === "ok" ? "text-ok" : "text-prose";
  return (
    <div className="bg-layer border border-edge rounded-xl px-4 py-3">
      <p className="text-xs text-secondary mb-1">{label}</p>
      <p className={`text-lg font-semibold font-mono ${valClass}`}>{value}</p>
      {sub && <p className="text-xs text-muted font-mono mt-0.5">{sub}</p>}
    </div>
  );
}
