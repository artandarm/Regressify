"use client";
import type { Coefficient, SegmentResult } from "@/lib/types";

// Verdict badge in the Sig. column
function SigBadge({ sig }: { sig: boolean }) {
  return sig ? (
    <span className="inline-flex items-center justify-center w-4 h-4 rounded-full bg-ok-bg text-ok text-[10px] leading-none">
      ✓
    </span>
  ) : (
    <span className="inline-flex items-center justify-center w-4 h-4 rounded-full bg-warn-bg text-warn text-[10px] leading-none">
      ✗
    </span>
  );
}

// Row background based on significance
function rowBg(c: Coefficient): string {
  if (!c.significant) return "bg-warn-bg";
  return "";
}

function fmt(v: number | null | undefined, decimals = 4): string {
  if (v == null) return "—";
  return v.toFixed(decimals);
}

function pFmt(p: number): string {
  if (p < 0.001) return "<0.001";
  return p.toFixed(3);
}

export function CoefficientsTable({
  coefficients,
  modelType,
  seType,
}: {
  coefficients: Coefficient[];
  modelType: string;
  seType?: string;
}) {
  if (coefficients.length === 0) return null;

  const hasStdErr = coefficients.some((c) => c.std_err != null);
  const hasTStat  = coefficients.some((c) => c.t_stat  != null);

  return (
    <div>
      <div className="overflow-x-auto rounded-lg border border-edge">
        <table className="w-full text-[13px] font-mono">
          <thead>
            <tr className="border-b border-edge bg-layer">
              <th className="text-left px-3 py-2 font-medium text-secondary text-[11px] tracking-wide font-sans">
                Term
              </th>
              <th className="text-right px-3 py-2 font-medium text-secondary text-[11px] tracking-wide font-sans">
                Coef
              </th>
              {hasStdErr && (
                <th className="text-right px-3 py-2 font-medium text-secondary text-[11px] tracking-wide font-sans">
                  Std Err
                </th>
              )}
              {hasTStat && (
                <th className="text-right px-3 py-2 font-medium text-secondary text-[11px] tracking-wide font-sans">
                  t
                </th>
              )}
              <th className="text-right px-3 py-2 font-medium text-secondary text-[11px] tracking-wide font-sans">
                p-value
              </th>
              <th className="text-center px-3 py-2 font-medium text-secondary text-[11px] tracking-wide font-sans">
                Sig.
              </th>
            </tr>
          </thead>
          <tbody>
            {coefficients.map((c) => (
              <tr
                key={c.name}
                className={[
                  "border-b border-edge last:border-0 transition-colors",
                  rowBg(c),
                ].join(" ")}
              >
                {/* Term name — sans-serif, not monospace */}
                <td className="px-3 py-2 text-prose font-sans text-[13px] font-medium whitespace-nowrap">
                  {c.name}
                </td>
                <td className="px-3 py-2 text-right text-prose tabular-nums">
                  {fmt(c.value)}
                </td>
                {hasStdErr && (
                  <td className="px-3 py-2 text-right text-secondary tabular-nums">
                    {fmt(c.std_err)}
                  </td>
                )}
                {hasTStat && (
                  <td className="px-3 py-2 text-right text-secondary tabular-nums">
                    {fmt(c.t_stat, 2)}
                  </td>
                )}
                <td className="px-3 py-2 text-right text-secondary tabular-nums">
                  {pFmt(c.p_value)}
                </td>
                <td className="px-3 py-2 text-center">
                  <SigBadge sig={c.significant} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Footer: SE type */}
      <p className="mt-2 text-[11px] text-muted font-mono">
        {modelType}
        {seType ? ` · SE: ${seType}` : ""}
        {coefficients.some((c) => !c.significant) && (
          <span className="ml-2 text-warn">
            · ✗ = p {">"} 0.05
          </span>
        )}
      </p>
    </div>
  );
}

// Wrapper that extracts data from a SegmentResult
export function SegmentCoefficientsTable({ seg }: { seg: SegmentResult }) {
  const seType = seg.distribution === "t" ? "ML / Student-t" : "ML";
  return (
    <CoefficientsTable
      coefficients={seg.coefficients}
      modelType={seg.model_type}
      seType={seType}
    />
  );
}
