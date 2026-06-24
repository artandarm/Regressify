"""
Iteration 3 test: residual diagnostics (Wooldridge AR(1), Pesaran CD, DK SE).
Both datasets are FE cases (correlated effects, N=100, T=10).

Dataset 1 — Clean: iid errors, no AR(1), no cross-sectional dependence.
  Expected: both tests green, se_type_final=clustered.

Dataset 2 — Common shock: strong common factor per period.
  Expected: CD significant, se_type_final=driscoll_kraay, SE change visible.
"""
import requests, json, io
import numpy as np
import pandas as pd

BASE = "http://localhost:8080"


def post_panel(df, entity_col, time_col, dep_var, regressors):
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    r = requests.post(f"{BASE}/upload", files={"file": ("panel.csv", buf, "text/csv")})
    assert r.status_code == 200, f"upload failed: {r.text}"
    r = requests.post(f"{BASE}/analyze/panel", json={
        "entity_col": entity_col, "time_col": time_col,
        "dep_var": dep_var, "regressors": regressors,
    })
    return r.status_code, r.json()


rng = np.random.default_rng(7)
N, T = 100, 10

# ── Dataset 1: Clean FE (iid errors) ─────────────────────────────────────
rows1 = []
for i in range(N):
    x_mean_i = rng.normal(0, 1)
    alpha_i  = 0.9 * x_mean_i + rng.normal(0, 0.3)
    for t in range(T):
        x_it = x_mean_i + rng.normal(0, 0.5)
        eps  = rng.normal(0, 0.5)            # pure iid noise
        y_it = alpha_i + 1.5 * x_it + eps
        rows1.append({"entity": f"E{i:03d}", "time": t, "y": y_it, "x1": x_it})
df1 = pd.DataFrame(rows1)

# ── Dataset 2: FE + strong common shock per period ────────────────────────
# common_t is the same for all i in period t — induces CD
common_shocks = rng.normal(0, 2.0, size=T)    # sigma=2 is deliberately large
rows2 = []
for i in range(N):
    x_mean_i = rng.normal(0, 1)
    alpha_i  = 0.9 * x_mean_i + rng.normal(0, 0.3)
    for t in range(T):
        x_it = x_mean_i + rng.normal(0, 0.5)
        eps  = common_shocks[t] + rng.normal(0, 0.3)   # dominated by common shock
        y_it = alpha_i + 1.5 * x_it + eps
        rows2.append({"entity": f"E{i:03d}", "time": t, "y": y_it, "x1": x_it})
df2 = pd.DataFrame(rows2)


def print_result(label, status, body):
    print(f"\n{'='*70}")
    print(f"  {label}   HTTP {status}")
    print(f"{'='*70}")
    if status not in (200, 422):
        print(f"  ERROR: {body}")
        return
    data = body if status == 200 else body.get("detail", body)
    if not isinstance(data, dict):
        print(f"  RAW: {data}")
        return

    # recommendation
    rec = data.get("recommendation", {})
    print(f"  Recommendation: {rec.get('recommended_model')}  confidence={rec.get('confidence')}")

    # diagnostics
    diag = data.get("diagnostics", {})
    print(f"\n  diagnostics.se_type_final = {diag.get('se_type_final')}")
    for step in diag.get("steps", []):
        name    = step.get("name", "?")
        stat    = step.get("statistic")
        pval    = step.get("pvalue")
        verdict = step.get("verdict", "?")
        msg     = step.get("message", "")[:90]
        stat_s  = f"stat={stat}" if stat is not None else ""
        pval_s  = f"p={pval}"   if pval is not None else ""
        nums    = "  ".join(x for x in [stat_s, pval_s] if x)
        print(f"  [{verdict.upper():4s}] {name}")
        if nums:
            print(f"         {nums}")
        print(f"         {msg}")

    # final_model coefficients (shows SE change for DK case)
    fm = data.get("final_model", {})
    print(f"\n  final_model: {fm.get('model_type')}  n_obs={fm.get('n_obs')}")
    for c in fm.get("coefficients", []):
        print(f"    {c['name']:8s}  coef={c['coef']:8.4f}  se={c['std_err']:7.4f}  "
              f"p={c['pvalue']:6.4f}  [{c['verdict']}]")


for label, df, regs in [
    ("Dataset 1 — Clean FE (expect no issues, clustered SE)", df1, ["x1"]),
    ("Dataset 2 — FE + common shock (expect CD warn, DK SE)", df2, ["x1"]),
]:
    status, body = post_panel(df, "entity", "time", "y", regs)
    print_result(label, status, body)
