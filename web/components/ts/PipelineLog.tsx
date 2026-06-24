"use client";
import { useState } from "react";
import type { PipelineStep, SegmentResult } from "@/lib/types";

// ── Verdict icons ─────────────────────────────────────────────────────────

type Verdict = "ok" | "warn" | "error" | "info";

const VERDICT: Record<Verdict, { sym: string; cls: string }> = {
  ok:    { sym: "✓", cls: "text-ok   bg-ok-bg   border-ok/30"   },
  warn:  { sym: "!", cls: "text-warn bg-warn-bg border-warn/30" },
  error: { sym: "✗", cls: "text-bad  bg-bad-bg  border-bad/30"  },
  info:  { sym: "i", cls: "text-info bg-info-bg border-info/30" },
};

function VerdictIcon({ verdict }: { verdict: string }) {
  const v = (VERDICT[verdict as Verdict] ?? VERDICT.info);
  return (
    <span
      className={[
        "inline-flex items-center justify-center w-4 h-4 rounded-full",
        "text-[9px] font-bold border shrink-0 mt-px",
        v.cls,
      ].join(" ")}
    >
      {v.sym}
    </span>
  );
}

// ── Single step row — collapsed / expanded ────────────────────────────────

function StepRow({ step }: { step: PipelineStep }) {
  const [open, setOpen] = useState(false);

  const pStr =
    step.p_value !== null && step.p_value !== undefined
      ? step.p_value < 0.001
        ? "p<0.001"
        : `p=${step.p_value.toFixed(3)}`
      : null;

  return (
    <div
      className="border-b border-edge last:border-0"
      onClick={() => setOpen((o) => !o)}
    >
      {/* Collapsed row */}
      <div className="flex items-start gap-2.5 px-3 py-2 cursor-pointer hover:bg-raised transition-colors select-none">
        <VerdictIcon verdict={step.verdict} />
        <div className="flex-1 min-w-0 flex items-baseline gap-1.5 flex-wrap">
          <span className="text-[13px] text-prose font-medium">{step.name}</span>
          <span className="text-[12px] text-secondary truncate max-w-xs">{step.message}</span>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          {pStr && (
            <span className="text-[11px] font-mono text-muted">{pStr}</span>
          )}
          <span className="text-[10px] text-muted">{open ? "▲" : "▼"}</span>
        </div>
      </div>

      {/* Expanded detail */}
      {open && (
        <div className="px-3 pb-3 pt-0.5 bg-layer text-[12px] leading-relaxed">
          <p className="text-secondary">{step.message}</p>
          {pStr && (
            <p className="mt-1 font-mono text-muted">{pStr}</p>
          )}
        </div>
      )}
    </div>
  );
}

// ── Collapsible section ───────────────────────────────────────────────────

function LogSection({
  title,
  steps,
  defaultOpen = false,
}: {
  title: string;
  steps: PipelineStep[];
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  if (steps.length === 0) return null;

  const warnCount = steps.filter((s) => s.verdict === "warn" || s.verdict === "error").length;

  return (
    <div className="border border-edge rounded-lg overflow-hidden">
      <button
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center justify-between px-3 py-2.5 bg-layer hover:bg-raised transition-colors text-left"
      >
        <div className="flex items-center gap-2">
          <span className="text-[13px] font-medium text-prose">{title}</span>
          {warnCount > 0 && (
            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-warn-bg text-warn border border-warn/20">
              {warnCount} warn
            </span>
          )}
        </div>
        <span className="text-[11px] font-mono text-muted">
          {steps.length} steps {open ? "▲" : "▼"}
        </span>
      </button>

      {open && (
        <div className="bg-base divide-y-0">
          {steps.map((s, i) => (
            <StepRow key={i} step={s} />
          ))}
        </div>
      )}
    </div>
  );
}

// ── Public component ──────────────────────────────────────────────────────

interface Props {
  preSteps: PipelineStep[];
  breakSteps: PipelineStep[];
  segments: SegmentResult[];
}

export function PipelineLog({ preSteps, breakSteps, segments }: Props) {
  return (
    <div className="flex flex-col gap-2">
      <LogSection title="Pre-analysis" steps={preSteps} />
      <LogSection title="Break detection" steps={breakSteps} />
      {segments.map((seg, i) => (
        <LogSection
          key={seg.segment_index}
          title={`Segment ${seg.segment_index} — ${seg.model_type}`}
          steps={seg.steps}
          defaultOpen={i === segments.length - 1}
        />
      ))}
    </div>
  );
}
