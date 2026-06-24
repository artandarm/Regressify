"use client";
import { useState, useCallback } from "react";
import katex from "katex";
import "katex/dist/katex.min.css";
import type { SegmentResult } from "@/lib/types";

// Build a copy-as-code string from segment coefficients.
// Output: "dy = 0.3138 + 0.7073*y_lag1 + e"
function buildCodeString(seg: SegmentResult): string {
  const d = seg.d;
  const lhs = d === 0 ? "y" : d === 1 ? "dy" : `d${d}y`;

  const terms = seg.coefficients
    .filter((c) => c.name !== "sigma2")
    .map((c) => {
      const v = c.value.toFixed(4);
      if (c.name === "const") return v;
      if (c.name.startsWith("ar.L")) return `${v}*y_lag${c.name.slice(4)}`;
      if (c.name.startsWith("ma.L")) return `${v}*eps_lag${c.name.slice(4)}`;
      return `${v}*${c.name}`;
    });

  return `${lhs} = ${terms.length ? terms.join(" + ") + " + e" : "e"}`;
}

function KaTeXDisplay({ latex }: { latex: string }) {
  let html = "";
  try {
    html = katex.renderToString(latex, { throwOnError: false, displayMode: true });
  } catch (e) {
    console.error("[KaTeXDisplay] render error:", e, "latex:", latex);
    html = `<code class="font-mono text-sm px-1">${latex}</code>`;
  }
  if (!html) {
    console.warn("[KaTeXDisplay] empty output for latex:", latex);
    return <code className="font-mono text-sm text-secondary">{latex}</code>;
  }
  return (
    <div
      className="overflow-x-auto"
      style={{ lineHeight: "normal" }}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}

function CopyButton({ seg }: { seg: SegmentResult }) {
  const [state, setState] = useState<"idle" | "ok" | "err">("idle");

  const copy = useCallback(() => {
    const code = buildCodeString(seg);
    navigator.clipboard.writeText(code).then(
      () => { setState("ok"); setTimeout(() => setState("idle"), 2000); },
      () => { setState("err"); setTimeout(() => setState("idle"), 2000); },
    );
  }, [seg]);

  return (
    <button
      onClick={copy}
      title={`Copy: ${buildCodeString(seg)}`}
      className={[
        "shrink-0 inline-flex items-center gap-1.5 rounded px-2.5 py-1.5",
        "text-xs font-mono border transition-colors select-none whitespace-nowrap",
        state === "ok"
          ? "text-ok border-ok bg-ok-bg"
          : state === "err"
          ? "text-bad border-bad"
          : "text-secondary border-edge hover:text-accent hover:border-accent",
      ].join(" ")}
    >
      {state === "ok" ? "✓ copied" : state === "err" ? "✗ failed" : "⎘ copy code"}
    </button>
  );
}

// Separates metadata items with a centered dot.
function Meta({ items }: { items: (string | null | undefined)[] }) {
  const visible = items.filter(Boolean) as string[];
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-0.5 text-xs text-secondary font-mono mt-2.5">
      {visible.map((item, i) => (
        <span key={i}>{item}</span>
      ))}
    </div>
  );
}

export function ProcessEquation({
  seg,
  showSegmentLabel = false,
}: {
  seg: SegmentResult;
  showSegmentLabel?: boolean;
}) {
  return (
    <div
      className="rounded-r-lg border border-edge bg-layer shadow-sm pr-5 pt-4 pb-3.5 pl-5"
      style={{ borderLeft: "3px solid var(--color-accent)" }}
    >
      {/* Equation + copy button */}
      <div className="flex items-start gap-4">
        <div className="flex-1 min-w-0">
          <KaTeXDisplay latex={seg.equation_latex || String.raw`\hat{y}_t = \varepsilon_t`} />
        </div>
        <div className="pt-1">
          <CopyButton seg={seg} />
        </div>
      </div>

      {/* Footer: model · segment · obs · AIC · BIC */}
      <Meta
        items={[
          seg.model_type,
          showSegmentLabel ? `Segment ${seg.segment_index}` : null,
          `n = ${seg.obs}`,
          `AIC ${seg.aic.toFixed(1)}`,
          `BIC ${seg.bic.toFixed(1)}`,
          seg.distribution === "t" ? "Student-t" : null,
        ]}
      />
    </div>
  );
}

// Grid wrapper: always vertical (one card per row, full width).
export function ProcessEquationGrid({ segments }: { segments: SegmentResult[] }) {
  const multi = segments.length > 1;
  return (
    <div className="flex flex-col gap-4">
      {segments.map((seg) => (
        <ProcessEquation key={seg.segment_index} seg={seg} showSegmentLabel={multi} />
      ))}
    </div>
  );
}
