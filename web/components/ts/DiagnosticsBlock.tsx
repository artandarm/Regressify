"use client";
import type { SegmentResult, PipelineStep } from "@/lib/types";

// ── Helpers ───────────────────────────────────────────────────────────────

function findStep(steps: PipelineStep[], pattern: string): PipelineStep | undefined {
  return steps.find((s) => s.name.toLowerCase().includes(pattern.toLowerCase()));
}

function pStr(step: PipelineStep | undefined): string | null {
  if (!step || step.p_value == null) return null;
  return step.p_value < 0.001 ? "p<0.001" : `p=${step.p_value.toFixed(3)}`;
}

// ── Verdict icon for inline row ───────────────────────────────────────────

function RowIcon({ ok }: { ok: boolean }) {
  return ok ? (
    <span className="inline-flex items-center justify-center w-4 h-4 rounded-full bg-ok-bg text-ok text-[10px] font-bold border border-ok/30 shrink-0">
      ✓
    </span>
  ) : (
    <span className="inline-flex items-center justify-center w-4 h-4 rounded-full bg-warn-bg text-warn text-[10px] font-bold border border-warn/30 shrink-0">
      !
    </span>
  );
}

// ── Diagnostics row ───────────────────────────────────────────────────────

function DiagRow({
  label, ok, pValue,
}: {
  label: string;
  ok: boolean;
  pValue: string | null;
}) {
  return (
    <div className="flex items-center gap-2.5 py-2 border-b border-edge last:border-0">
      <RowIcon ok={ok} />
      <span className="flex-1 text-[13px] text-prose">{label}</span>
      {pValue && (
        <span className="text-[11px] font-mono text-muted shrink-0">{pValue}</span>
      )}
    </div>
  );
}

// ── GARCH side card ───────────────────────────────────────────────────────

function GarchCard({ seg }: { seg: SegmentResult }) {
  const g = seg.garch;
  if (!g.fitted || g.alpha == null || g.beta == null) return null;
  const persistence = g.alpha + g.beta;
  const stationary = persistence < 1;

  return (
    <div className="rounded-lg border border-edge bg-layer p-3.5 self-start">
      <p className="text-[11px] font-mono tracking-wider text-secondary uppercase mb-2.5">
        GARCH(1,1)
      </p>
      <div className="grid grid-cols-3 gap-2 mb-2.5">
        {[
          { label: "ω", value: g.omega },
          { label: "α", value: g.alpha },
          { label: "β", value: g.beta },
        ].map(({ label, value }) => (
          <div key={label} className="bg-raised rounded px-2 py-1.5">
            <p className="text-[10px] text-muted mb-0.5">{label}</p>
            <p className="text-[12px] font-mono text-prose">
              {value !== null ? value.toFixed(4) : "—"}
            </p>
          </div>
        ))}
      </div>
      <div className="flex items-center gap-1.5 text-[12px] font-mono">
        <span className="text-secondary">α+β =</span>
        <span className={stationary ? "text-prose" : "text-warn"}>
          {persistence.toFixed(4)}
        </span>
        <span className="text-muted">&lt; 1</span>
        <span className={stationary ? "text-ok" : "text-warn"}>
          {stationary ? "✓" : "✗"}
        </span>
      </div>
      {g.aic !== null && (
        <p className="text-[11px] font-mono text-muted mt-2">
          AIC {g.aic.toFixed(1)} · BIC {g.bic?.toFixed(1)}
        </p>
      )}
    </div>
  );
}

// ── Public component ──────────────────────────────────────────────────────

export function DiagnosticsBlock({ seg }: { seg: SegmentResult }) {
  const steps = seg.steps;

  const ljungStep = findStep(steps, "ljung");
  const archStep  = findStep(steps, "arch");
  const jbStep    = findStep(steps, "jarque");

  const diagnostics = [
    {
      label: "Ljung-Box (no autocorrelation)",
      ok: seg.ljungbox_ok,
      pValue: pStr(ljungStep),
    },
    {
      label: "ARCH-LM (no ARCH effect)",
      ok: !seg.arch_effect,
      pValue: pStr(archStep),
    },
    ...(jbStep
      ? [{ label: "Jarque-Bera (normality)", ok: jbStep.verdict === "ok", pValue: pStr(jbStep) }]
      : []),
  ];

  const hasGarch = seg.garch.fitted;

  return (
    <div className={`flex gap-4 ${hasGarch ? "items-start" : ""}`}>
      {/* Diagnostic tests list */}
      <div className="flex-1 rounded-lg border border-edge overflow-hidden">
        <div className="px-3 py-2 bg-layer border-b border-edge">
          <p className="text-[11px] font-mono tracking-wider text-secondary uppercase">
            Residual diagnostics
          </p>
        </div>
        <div className="px-3 bg-base">
          {diagnostics.map((d) => (
            <DiagRow key={d.label} {...d} />
          ))}
          {diagnostics.length === 0 && (
            <p className="text-[12px] text-muted py-3">No diagnostic results</p>
          )}
        </div>
      </div>

      {/* GARCH card — shown only if fitted */}
      {hasGarch && <GarchCard seg={seg} />}
    </div>
  );
}

// Grid wrapper for multi-segment results
export function DiagnosticsGrid({ segments }: { segments: SegmentResult[] }) {
  const multi = segments.length > 1;
  return (
    <div className={`grid gap-4 ${multi ? "grid-cols-1 lg:grid-cols-2" : "grid-cols-1"}`}>
      {segments.map((seg) => (
        <div key={seg.segment_index}>
          {multi && (
            <p className="text-[11px] font-mono text-muted uppercase tracking-wider mb-2">
              Segment {seg.segment_index}
            </p>
          )}
          <DiagnosticsBlock seg={seg} />
        </div>
      ))}
    </div>
  );
}
