"use client";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine, ReferenceDot,
} from "recharts";
import type { OutlierPoint } from "@/lib/types";

const SEG_COLORS = ["#818cf8", "#34d399", "#fbbf24", "#f87171", "#a78bfa", "#67e8f9"];

interface Props {
  seriesValues: number[];
  breakpoints: number[];
  outliers: OutlierPoint[];
}

function downsample(values: number[], max = 3000) {
  if (values.length <= max) return values;
  const step = Math.ceil(values.length / max);
  return values.filter((_, i) => i % step === 0 || i === values.length - 1);
}

export function AnalysisChart({ seriesValues, breakpoints, outliers }: Props) {
  const sampled = downsample(seriesValues);
  const ratio = seriesValues.length / sampled.length;

  const segBounds = [0, ...breakpoints, seriesValues.length];
  const segCount = segBounds.length - 1;

  const data = sampled.map((v, idx) => {
    const t = Math.round(idx * ratio);
    const row: Record<string, number | null> = { t };
    for (let s = 0; s < segCount; s++) {
      row[`s${s}`] = t >= segBounds[s] && t < segBounds[s + 1] ? v : null;
    }
    return row;
  });

  return (
    <div className="bg-layer rounded-xl border border-edge p-5">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-xs font-mono tracking-wider text-secondary uppercase">
          Time series
        </h3>
        {breakpoints.length > 0 && (
          <div className="flex items-center gap-3">
            {Array.from({ length: segCount }, (_, i) => (
              <span key={i} className="flex items-center gap-1.5 text-xs text-secondary">
                <span className="inline-block w-3 h-0.5 rounded" style={{ background: SEG_COLORS[i % SEG_COLORS.length] }} />
                Seg {i + 1}
              </span>
            ))}
          </div>
        )}
      </div>

      <ResponsiveContainer width="100%" height={240}>
        <LineChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid stroke="#2a2a3a" strokeDasharray="3 4" vertical={false} />
          <XAxis
            dataKey="t"
            tick={{ fill: "#44445a", fontSize: 10 }}
            tickLine={false}
            axisLine={{ stroke: "#2a2a3a" }}
          />
          <YAxis
            tick={{ fill: "#44445a", fontSize: 10 }}
            tickLine={false}
            axisLine={false}
            width={46}
            tickFormatter={(v: number) => v.toFixed(2)}
          />
          <Tooltip
            contentStyle={{
              background: "#1c1c28",
              border: "1px solid #2a2a3a",
              borderRadius: 8,
              fontSize: 12,
            }}
            labelStyle={{ color: "#7878a0" }}
            itemStyle={{ color: "#e0e0f0" }}
            formatter={(v) => [typeof v === "number" ? v.toFixed(4) : v, ""]}
            labelFormatter={(t) => `t = ${t}`}
          />
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
          {breakpoints.map((bp) => (
            <ReferenceLine
              key={`bp-${bp}`}
              x={bp}
              stroke="#f87171"
              strokeDasharray="4 3"
              strokeOpacity={0.7}
              label={{
                value: `t=${bp}`,
                fill: "#f87171",
                fontSize: 10,
                position: "insideTopRight",
              }}
            />
          ))}
          {outliers.map((o) => (
            <ReferenceDot
              key={`ao-${o.index}`}
              x={o.index}
              y={o.original_value}
              r={4}
              fill="#fbbf24"
              stroke="#0d0d12"
              strokeWidth={1.5}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>

      {outliers.length > 0 && (
        <p className="mt-3 text-xs text-secondary">
          <span className="inline-block w-2.5 h-2.5 rounded-full bg-warn mr-1.5 align-middle" />
          {outliers.length} additive outlier{outliers.length > 1 ? "s" : ""} detected and interpolated
        </p>
      )}
    </div>
  );
}
