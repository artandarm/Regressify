import Link from "next/link";

const MODES = [
  {
    href: "/timeseries",
    label: "Time Series",
    description: "ARIMA/SARIMA, structural breaks, GARCH, walk-forward validation",
    available: true,
  },
  {
    href: "#",
    label: "Panel Data",
    description: "Fixed / random effects, Hausman test, heteroskedasticity",
    available: false,
  },
  {
    href: "#",
    label: "Cross-section",
    description: "OLS, robust SE, variable selection, outlier diagnostics",
    available: false,
  },
];

export default function HomePage() {
  return (
    <main className="flex flex-col items-center justify-center min-h-screen px-6 py-20">
      <div className="max-w-2xl w-full">
        <p className="text-xs font-mono tracking-widest text-secondary uppercase mb-6">
          AllRegressions
        </p>
        <h1 className="text-3xl font-semibold text-prose mb-2 tracking-tight">
          Automated econometric analysis
        </h1>
        <p className="text-secondary mb-12 leading-relaxed">
          Upload a dataset, select a variable. The pipeline runs the tests,
          selects models, and shows you why.
        </p>

        <div className="grid gap-3">
          {MODES.map((m) => (
            <ModeCard key={m.label} {...m} />
          ))}
        </div>
      </div>
    </main>
  );
}

function ModeCard({
  href,
  label,
  description,
  available,
}: {
  href: string;
  label: string;
  description: string;
  available: boolean;
}) {
  const inner = (
    <div
      className={[
        "group flex items-start gap-4 rounded-lg border px-5 py-4 transition-colors",
        available
          ? "border-edge bg-layer hover:bg-raised hover:border-accent cursor-pointer"
          : "border-edge bg-layer opacity-40 cursor-default",
      ].join(" ")}
    >
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-1">
          <span className="text-sm font-medium text-prose">{label}</span>
          {!available && (
            <span className="text-[10px] font-mono tracking-wider text-muted border border-edge rounded px-1.5 py-0.5">
              soon
            </span>
          )}
        </div>
        <p className="text-xs text-secondary leading-relaxed">{description}</p>
      </div>
      {available && (
        <span className="text-secondary text-xs group-hover:text-accent transition-colors mt-0.5">
          →
        </span>
      )}
    </div>
  );

  return available ? <Link href={href}>{inner}</Link> : <div>{inner}</div>;
}
