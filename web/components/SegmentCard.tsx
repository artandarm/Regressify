"use client";
import katex from "katex";
import "katex/dist/katex.min.css";
import type { SegmentResult } from "@/lib/types";

function Equation({ latex }: { latex: string }) {
  let html = "";
  try {
    html = katex.renderToString(latex, { throwOnError: false, displayMode: true });
  } catch {
    html = `<code>${latex}</code>`;
  }
  return (
    <div
      dangerouslySetInnerHTML={{ __html: html }}
      className="overflow-x-auto text-prose py-1"
    />
  );
}

function Badge({ ok, label }: { ok: boolean; label: string }) {
  return (
    <span
      className={[
        "inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded font-mono",
        ok ? "text-ok bg-ok/10" : "text-bad bg-bad/10",
      ].join(" ")}
    >
      {ok ? "✓" : "✗"} {label}
    </span>
  );
}

function CoefTable({ seg }: { seg: SegmentResult }) {
  if (seg.coefficients.length === 0) return null;
  return (
    <div>
      <p className="text-xs text-secondary mb-2">Coefficients</p>
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-edge text-muted">
            <th className="text-left pb-1.5 font-normal">Name</th>
            <th className="text-right pb-1.5 font-normal">Value</th>
            <th className="text-right pb-1.5 font-normal">p-value</th>
            <th className="text-right pb-1.5 font-normal">Sig.</th>
          </tr>
        </thead>
        <tbody>
          {seg.coefficients.map((c) => (
            <tr key={c.name} className="border-b border-edge last:border-0">
              <td className="py-1.5 font-mono text-prose">{c.name}</td>
              <td className="py-1.5 text-right font-mono text-prose">{c.value.toFixed(4)}</td>
              <td className="py-1.5 text-right font-mono text-secondary">{c.p_value.toFixed(4)}</td>
              <td className="py-1.5 text-right">
                {c.significant ? (
                  <span className="text-ok">✓</span>
                ) : (
                  <span className="text-warn">✗</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ModelAveragingPanel({ seg }: { seg: SegmentResult }) {
  const ma = seg.model_averaging;
  if (!ma || !ma.ambiguous) return null;
  return (
    <div className="mt-4 pt-4 border-t border-edge">
      <p className="text-xs text-warn mb-2">
        ⚠ Model averaging — top Akaike weight {(ma.top_weight * 100).toFixed(0)}% ({"<"}70%)
      </p>
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-edge text-muted">
            <th className="text-left pb-1.5 font-normal">Model</th>
            <th className="text-right pb-1.5 font-normal">AIC</th>
            <th className="text-right pb-1.5 font-normal">Weight</th>
            <th className="text-right pb-1.5 font-normal">OOS RMSE</th>
          </tr>
        </thead>
        <tbody>
          {ma.candidates.map((c, i) => (
            <tr key={c.label} className="border-b border-edge last:border-0">
              <td className="py-1.5 font-mono text-prose">
                {i === 0 && <span className="text-accent mr-1">★</span>}
                {c.label}
              </td>
              <td className="py-1.5 text-right font-mono text-secondary">{c.aic.toFixed(1)}</td>
              <td className="py-1.5 text-right font-mono text-secondary">
                {(c.weight * 100).toFixed(1)}%
              </td>
              <td className="py-1.5 text-right font-mono text-secondary">
                {c.rmse !== null ? c.rmse.toFixed(4) : "—"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function GarchPanel({ seg }: { seg: SegmentResult }) {
  const g = seg.garch;
  if (!g.fitted) return null;
  return (
    <div className="mt-4 pt-4 border-t border-edge">
      <p className="text-xs text-secondary mb-2">GARCH(1,1)</p>
      <div className="grid grid-cols-3 gap-3">
        {[
          { label: "ω (omega)", value: g.omega },
          { label: "α (alpha)", value: g.alpha },
          { label: "β (beta)", value: g.beta },
        ].map(({ label, value }) => (
          <div key={label} className="bg-raised rounded-lg px-3 py-2">
            <p className="text-muted text-[10px] mb-0.5">{label}</p>
            <p className="font-mono text-xs text-prose">
              {value !== null ? value.toFixed(6) : "—"}
            </p>
          </div>
        ))}
      </div>
      <p className="text-muted text-xs mt-2 font-mono">
        AIC {g.aic?.toFixed(1)} · BIC {g.bic?.toFixed(1)}
      </p>
    </div>
  );
}

export function SegmentCard({ seg }: { seg: SegmentResult }) {
  return (
    <div className="bg-layer border border-edge rounded-xl p-5">
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div>
          <p className="text-xs text-muted mb-0.5">Segment {seg.segment_index}</p>
          <p className="text-sm font-medium text-prose">{seg.model_type}</p>
          <p className="text-xs text-secondary">{seg.obs} observations · t={seg.start_t}–{seg.end_t}</p>
        </div>
        <div className="flex flex-wrap gap-1.5 justify-end">
          <Badge ok={seg.ljungbox_ok} label="Ljung-Box" />
          <Badge ok={!seg.arch_effect} label="No ARCH" />
          <Badge ok={!seg.aic_bic_conflict.conflict} label="AIC/BIC" />
        </div>
      </div>

      {/* Equation */}
      {seg.equation_latex && (
        <div className="mb-4 pb-4 border-b border-edge">
          <p className="text-xs text-secondary mb-2">Process equation</p>
          <Equation latex={seg.equation_latex} />
        </div>
      )}

      {/* Insignificant warning */}
      {seg.insignificant_coefs.length > 0 && (
        <p className="text-xs text-warn mb-3">
          ⚠ Insignificant coefs: {seg.insignificant_coefs.join(", ")}
        </p>
      )}

      {/* Coefficients */}
      <CoefTable seg={seg} />

      {/* Distribution */}
      <p className="text-xs text-secondary mt-3">
        Residuals: {seg.distribution === "t" ? "Student-t" : "Normal"}
        {" · "}AIC {seg.aic.toFixed(1)} · BIC {seg.bic.toFixed(1)}
      </p>

      <GarchPanel seg={seg} />
      <ModelAveragingPanel seg={seg} />
    </div>
  );
}
