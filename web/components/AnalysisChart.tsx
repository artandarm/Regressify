"use client";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine, ReferenceDot,
  ReferenceArea,
} from "recharts";
import type { OutlierPoint } from "@/lib/types";

// Segment line colors — readable on a white/light background
const SEG_COLORS   = ["#1a4480", "#15803d", "#b45309", "#7c3aed", "#0e7490"];
// Very light tints for segment background fills
const SEG_BG       = ["#eff4fc", "#f0fdf4", "#fffbeb", "#f5f3ff", "#ecfeff"];

interface Props {
  seriesValues: number[];
  seriesOriginal: number[];
  breakpoints: number[];
  outliers: OutlierPoint[];
}

function downsample(values: number[], max = 3000) {
  if (values.length <= max) return values;
  const step = Math.ceil(values.length / max);
  return values.filter((_, i) => i % step === 0 || i === values.length - 1);
}

// Custom tooltip shown on hover
function ChartTooltip({
  active, payload, label, outlierMap,
}: {
  active?: boolean;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  payload?: any[];
  label?: number;
  outlierMap: Map<number, OutlierPoint>;
}) {
  if (!active || !payload?.length) return null;
  const t = label as number;
  const outlier = outlierMap.get(t);
  // Find first numeric value across all series keys
  const numericEntry = payload.find((p) => typeof p.value === "number");
  if (!numericEntry) return null;
  return (
    <div className="rounded-lg border border-edge bg-base shadow-md px-3 py-2 text-[12px]">
      <p className="text-secondary font-mono mb-1">t = {t}</p>
      <p className="font-mono text-prose">{(numericEntry.value as number).toFixed(4)}</p>
      {outlier && (
        <p className="mt-1 text-warn font-mono">
          orig: {outlier.original_value.toFixed(4)} → cleaned: {outlier.cleaned_value.toFixed(4)}
        </p>
      )}
    </div>
  );
}

export function AnalysisChart({ seriesValues, seriesOriginal, breakpoints, outliers }: Props) {
  const hasOriginal = outliers.length > 0;

  const sampled = downsample(seriesValues);
  const ratio   = seriesValues.length / sampled.length;

  const segBounds = [0, ...breakpoints, seriesValues.length];
  const segCount  = segBounds.length - 1;

  // Build chartData: one point per sampled index with per-segment series keys
  const data = sampled.map((v, idx) => {
    const t = Math.round(idx * ratio);
    const row: Record<string, number | null> = { t };
    for (let s = 0; s < segCount; s++) {
      row[`s${s}`] = t >= segBounds[s] && t < segBounds[s + 1] ? v : null;
    }
    // Original series only at outlier positions (dashed overlay)
    if (hasOriginal && seriesOriginal[t] !== undefined) {
      row["orig"] = seriesOriginal[t];
    }
    return row;
  });

  // Fast lookup for outlier tooltips
  const outlierMap = new Map(outliers.map((o) => [o.index, o]));

  return (
    <div className="bg-layer rounded-xl border border-edge p-4">
      {/* Legend */}
      <div className="flex items-center justify-between mb-3">
        <span className="text-[11px] font-mono tracking-wider text-secondary uppercase">
          Time series
        </span>
        <div className="flex items-center gap-3 flex-wrap justify-end">
          {segCount > 1 && Array.from({ length: segCount }, (_, i) => (
            <span key={i} className="flex items-center gap-1.5 text-[11px] text-secondary">
              <span
                className="inline-block w-8 h-0.5 rounded"
                style={{ background: SEG_COLORS[i % SEG_COLORS.length] }}
              />
              Seg {i + 1}
            </span>
          ))}
          {hasOriginal && (
            <span className="flex items-center gap-1.5 text-[11px] text-secondary">
              <span className="inline-block w-5 border-t border-dashed border-muted" />
              original
            </span>
          )}
          {outliers.length > 0 && (
            <span className="flex items-center gap-1.5 text-[11px] text-secondary">
              <span className="inline-block w-2 h-2 rounded-full bg-warn" />
              {outliers.length} outlier{outliers.length > 1 ? "s" : ""}
            </span>
          )}
        </div>
      </div>

      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
          {/* Segment background fills */}
          {segCount > 1 && segBounds.slice(0, -1).map((start, i) => (
            <ReferenceArea
              key={`bg${i}`}
              x1={start}
              x2={segBounds[i + 1] - 1}
              fill={SEG_BG[i % SEG_BG.length]}
              fillOpacity={1}
              strokeOpacity={0}
            />
          ))}

          <CartesianGrid stroke="#e2e6ef" strokeDasharray="3 4" vertical={false} />
          <XAxis
            dataKey="t"
            tick={{ fill: "#9ba3af", fontSize: 10 }}
            tickLine={false}
            axisLine={{ stroke: "#e2e6ef" }}
          />
          <YAxis
            tick={{ fill: "#9ba3af", fontSize: 10 }}
            tickLine={false}
            axisLine={false}
            width={48}
            tickFormatter={(v: number) => v.toFixed(2)}
          />
          <Tooltip
            content={
              <ChartTooltip outlierMap={outlierMap} />
            }
          />

          {/* Original series — dashed overlay (shown only when outliers cleaned) */}
          {hasOriginal && (
            <Line
              type="linear"
              dataKey="orig"
              stroke="#9ba3af"
              strokeWidth={1}
              strokeDasharray="4 3"
              dot={false}
              connectNulls
              isAnimationActive={false}
              name="original"
            />
          )}

          {/* Per-segment cleaned series */}
          {Array.from({ length: segCount }, (_, i) => (
            <Line
              key={`s${i}`}
              type="linear"
              dataKey={`s${i}`}
              stroke={SEG_COLORS[i % SEG_COLORS.length]}
              strokeWidth={1.5}
              dot={false}
              connectNulls={false}
              isAnimationActive={false}
              name={`Segment ${i + 1}`}
            />
          ))}

          {/* Break reference lines */}
          {breakpoints.map((bp) => (
            <ReferenceLine
              key={`bp${bp}`}
              x={bp}
              stroke="#b91c1c"
              strokeDasharray="4 3"
              strokeOpacity={0.6}
              label={{
                value: `Break t=${bp}`,
                fill: "#b91c1c",
                fontSize: 9,
                position: "insideTopRight",
              }}
            />
          ))}

          {/* Outlier dots */}
          {outliers.map((o) => (
            <ReferenceDot
              key={`o${o.index}`}
              x={o.index}
              y={o.original_value}
              r={4}
              fill="#b45309"
              stroke="#ffffff"
              strokeWidth={1.5}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
