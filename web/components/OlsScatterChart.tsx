"use client";
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine, Cell,
} from "recharts";

interface Props {
  yActual: number[];
  yFitted: number[];
  yCol: string;
  influentialIdx: number[];
}

export function OlsScatterChart({ yActual, yFitted, yCol, influentialIdx }: Props) {
  const infSet = new Set(influentialIdx);

  const points = yActual.map((y, i) => ({
    fitted: yFitted[i],
    actual: y,
    inf: infSet.has(i),
  }));

  // Independent axis domains — scale to each variable's own range
  const minFitted = Math.min(...yFitted);
  const maxFitted = Math.max(...yFitted);
  const minActual = Math.min(...yActual);
  const maxActual = Math.max(...yActual);
  const padX = (maxFitted - minFitted) * 0.08 || 0.5;
  const padY = (maxActual - minActual) * 0.08 || 0.5;
  const xLo = minFitted - padX;
  const xHi = maxFitted + padX;
  const yLo = minActual - padY;
  const yHi = maxActual + padY;

  // OLS trend line: regress yActual on yFitted
  const n = yActual.length;
  const meanX = yFitted.reduce((s, v) => s + v, 0) / n;
  const meanY = yActual.reduce((s, v) => s + v, 0) / n;
  const ssXY = yFitted.reduce((s, v, i) => s + (v - meanX) * (yActual[i] - meanY), 0);
  const ssXX = yFitted.reduce((s, v) => s + (v - meanX) ** 2, 0);
  const slope = ssXX > 0 ? ssXY / ssXX : 0;
  const intercept = meanY - slope * meanX;

  return (
    <div className="bg-layer rounded-xl border border-edge p-5">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-xs font-mono tracking-wider text-secondary uppercase">
          Actual vs Fitted
        </h3>
        <div className="flex items-center gap-4 text-xs text-muted">
          <span className="flex items-center gap-1.5">
            <span className="inline-block w-5 border-t-2 border-accent opacity-80" />
            fit line
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-accent inline-block opacity-70" />
            obs
          </span>
          {influentialIdx.length > 0 && (
            <span className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-bad inline-block" />
              influential
            </span>
          )}
        </div>
      </div>

      <ResponsiveContainer width="100%" height={280}>
        <ScatterChart margin={{ top: 8, right: 16, bottom: 32, left: 8 }}>
          <CartesianGrid stroke="#2a2a3a" strokeDasharray="3 4" />
          <XAxis
            dataKey="fitted"
            type="number"
            domain={[xLo, xHi]}
            tick={{ fill: "#44445a", fontSize: 10 }}
            tickLine={false}
            axisLine={{ stroke: "#2a2a3a" }}
            tickFormatter={(v: number) => v.toFixed(1)}
            label={{
              value: "Fitted Ŷ",
              position: "insideBottom",
              offset: -18,
              style: { fill: "#7878a0", fontSize: 10 },
            }}
          />
          <YAxis
            dataKey="actual"
            type="number"
            domain={[yLo, yHi]}
            tick={{ fill: "#44445a", fontSize: 10 }}
            tickLine={false}
            axisLine={false}
            width={50}
            tickFormatter={(v: number) => v.toFixed(1)}
            label={{
              value: yCol,
              angle: -90,
              position: "insideLeft",
              offset: 12,
              style: { fill: "#7878a0", fontSize: 10 },
            }}
          />
          <Tooltip
            cursor={{ strokeDasharray: "3 3", stroke: "#2a2a3a" }}
            contentStyle={{
              background: "#1c1c28",
              border: "1px solid #2a2a3a",
              borderRadius: 8,
              fontSize: 12,
            }}
            formatter={(v: unknown) =>
              [typeof v === "number" ? v.toFixed(4) : String(v), ""]
            }
            labelFormatter={() => ""}
          />
          {/* Estimated regression line through the scatter */}
          <ReferenceLine
            segment={[
              { x: xLo, y: intercept + slope * xLo },
              { x: xHi, y: intercept + slope * xHi },
            ]}
            stroke="#818cf8"
            strokeWidth={2}
            ifOverflow="extendDomain"
          />
          <Scatter data={points} isAnimationActive={false}>
            {points.map((p, i) => (
              <Cell
                key={i}
                fill={p.inf ? "#f87171" : "#818cf8"}
                fillOpacity={p.inf ? 0.9 : 0.65}
                stroke={p.inf ? "#f87171" : "#818cf8"}
                strokeOpacity={0.25}
                r={p.inf ? 5 : 4}
              />
            ))}
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}
