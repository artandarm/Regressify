"use client";
import { useState } from "react";
import type { PipelineStep, SegmentResult } from "@/lib/types";

function VerdictDot({ verdict }: { verdict: string }) {
  const cls =
    verdict === "ok" ? "text-ok" :
    verdict === "warn" ? "text-warn" :
    verdict === "error" ? "text-bad" :
    "text-muted";
  const sym =
    verdict === "ok" ? "✓" :
    verdict === "warn" ? "⚠" :
    verdict === "error" ? "✗" : "·";
  return <span className={`${cls} w-4 shrink-0 text-center text-xs mt-px`}>{sym}</span>;
}

function StepRow({ step }: { step: PipelineStep }) {
  return (
    <div className="flex items-start gap-2 py-2 border-b border-edge last:border-0">
      <VerdictDot verdict={step.verdict} />
      <div className="flex-1 min-w-0 text-xs leading-relaxed">
        <span className="text-secondary">{step.name}</span>
        <span className="text-muted mx-1">—</span>
        <span className="text-prose">{step.message}</span>
        {step.p_value !== null && step.p_value !== undefined && (
          <span className="ml-2 font-mono text-muted">
            p={step.p_value.toFixed(4)}
          </span>
        )}
      </div>
    </div>
  );
}

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
  const hasWarn = steps.some((s) => s.verdict === "warn" || s.verdict === "error");

  return (
    <div className="border border-edge rounded-lg overflow-hidden">
      <button
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center justify-between px-4 py-3 bg-layer hover:bg-raised transition-colors text-left"
      >
        <div className="flex items-center gap-2">
          <span className="text-xs font-medium text-prose">{title}</span>
          {hasWarn && <span className="text-warn text-xs">⚠</span>}
        </div>
        <span className="text-muted text-xs">
          {open ? "▲" : "▼"} {steps.length}
        </span>
      </button>

      {open && (
        <div className="px-4 pb-1 bg-base">
          {steps.length === 0 ? (
            <p className="text-xs text-muted py-3">No steps recorded</p>
          ) : (
            steps.map((s, i) => <StepRow key={i} step={s} />)
          )}
        </div>
      )}
    </div>
  );
}

interface Props {
  preSteps: PipelineStep[];
  breakSteps: PipelineStep[];
  segments: SegmentResult[];
}

export function PipelineLog({ preSteps, breakSteps, segments }: Props) {
  return (
    <div>
      <h2 className="text-xs font-mono tracking-wider text-secondary uppercase mb-3">
        Analysis log
      </h2>
      <div className="flex flex-col gap-2">
        <LogSection title="Pre-analysis" steps={preSteps} defaultOpen />
        <LogSection title="Break detection" steps={breakSteps} />
        {segments.map((seg) => (
          <LogSection
            key={seg.segment_index}
            title={`Segment ${seg.segment_index} — ${seg.model_type}`}
            steps={seg.steps}
          />
        ))}
      </div>
    </div>
  );
}
