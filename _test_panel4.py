"""
Iteration 4 test: beta_std, observations, model_comparison.

Dataset 1 — FE + cross-sectional dependence:
  N=100, T=12, correlated effects, common shock sigma=0.3 (small -> FE recommended).
  X1 within-std=2.0, X2 within-std=0.2 (ratio=10), same true beta=1.5.
  Expected: beta_std(X1)/beta_std(X2) ~ 10.
  Expected: CD significant -> DK SE, se_type_final=driscoll_kraay.
  One outlier added -> at least 1 influential observation.

Dataset 2 — RE clean:
  N=100, T=10, true random effects, no common shock.
  X1 raw-std~1.0.
  Expected: RE high confidence, CD ok -> clustered SE.
"""
import requests, json, io
import numpy as np
import pandas as pd

BASE = "http://localhost:8080"


def post_panel(df, entity_col, time_col, dep_var, regressors):
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    r = requests.post(f"{BASE}/upload", files={"file": ("p.csv", buf, "text/csv")})
    assert r.status_code == 200, f"upload: {r.text}"
    r = requests.post(f"{BASE}/analyze/panel", json={
        "entity_col": entity_col, "time_col": time_col,
        "dep_var": dep_var, "regressors": regressors,
    })
    return r.status_code, r.json()


rng = np.random.default_rng(42)
N, T = 100, 12

# ── Dataset 1: FE + CD ─────────────────────────────────────────────────────
common_shocks = rng.normal(0, 0.3, size=T)   # small sigma -> FE still recommended
rows1 = []
for i in range(N):
    xm1 = rng.normal(0, 1)
    xm2 = rng.normal(0, 0.1)
    alpha_i = 0.9 * xm1 + rng.normal(0, 0.3)   # correlated with X1
    for t in range(T):
        x1 = xm1 + rng.normal(0, 2.0)   # within-std ~ 2.0
        x2 = xm2 + rng.normal(0, 0.2)   # within-std ~ 0.2
        eps = common_shocks[t] + rng.normal(0, 0.5)
        y = alpha_i + 1.5 * x1 + 1.5 * x2 + eps
        rows1.append({"entity": f"E{i:03d}", "time": t, "y": y, "x1": x1, "x2": x2})

df1 = pd.DataFrame(rows1)
# Add outlier: 3 obs with y shifted by +20
for k in range(3):
    df1.loc[df1.index[k], "y"] += 20

# ── Dataset 2: RE clean ────────────────────────────────────────────────────
rows2 = []
for i in range(N):
    alpha_i = rng.normal(0, 1)   # independent of X
    for t in range(T):
        x1 = rng.normal(0, 1)
        eps = rng.normal(0, 0.5)
        y = alpha_i + 1.5 * x1 + eps
        rows2.append({"entity": f"E{i:03d}", "time": t, "y": y, "x1": x1})

df2 = pd.DataFrame(rows2)


def print_result(label, status, body):
    print(f"\n{'='*72}")
    print(f"  {label}   HTTP {status}")
    print(f"{'='*72}")
    if status not in (200, 422):
        print(f"  ERROR body: {str(body)[:300]}")
        return
    data = body if status == 200 else body.get("detail", body)

    # recommendation
    rec = data.get("recommendation", {})
    print(f"  Recommendation: {rec.get('recommended_model')}  confidence={rec.get('confidence')}")

    # diagnostics
    diag = data.get("diagnostics", {})
    print(f"  se_type_final: {diag.get('se_type_final')}")
    for s in diag.get("steps", []):
        verdict = s.get("verdict", "?")
        pval = s.get("pvalue")
        pval_s = f" p={pval}" if pval is not None else ""
        print(f"    [{verdict.upper():4}] {s.get('name')}  stat={s.get('statistic')}{pval_s}")

    # final_model coefficients with beta_std
    fm = data.get("final_model", {})
    print(f"\n  final_model: {fm.get('model_type')}")
    for c in fm.get("coefficients", []):
        print(f"    {c['name']:6s}  coef={c.get('coef',0):8.4f}  se={c.get('std_err',0):7.4f}"
              f"  beta_std={c.get('beta_std')}  [{c.get('verdict')}]")

    # beta_std ratio for X1 / X2
    coefs = {c["name"]: c for c in fm.get("coefficients", [])}
    if "x1" in coefs and "x2" in coefs:
        b1 = coefs["x1"].get("beta_std")
        b2 = coefs["x2"].get("beta_std")
        ratio = round(b1 / b2, 2) if (b1 and b2 and b2 != 0) else "N/A"
        print(f"\n  beta_std(x1)/beta_std(x2) = {ratio}  (expected ~ 10)")

    # model_comparison
    mc = data.get("model_comparison", {})
    models_in_table = mc.get("models", [])
    se_row = mc.get("se_type", [])
    print(f"\n  model_comparison.models = {models_in_table}")
    print(f"  model_comparison.se_type = {se_row}")
    print(f"  model_comparison.recommended = {mc.get('recommended')}")
    coef_table = mc.get("coefficients", {})
    for varname, per_model in coef_table.items():
        row = f"  [{varname}]"
        for m in models_in_table:
            entry = per_model.get(m, {})
            coef = entry.get("coef")
            pval = entry.get("pvalue")
            row += f"  {m}: coef={coef} p={pval}"
        print(row)
    fit = mc.get("fit", {})
    for m, stats in fit.items():
        print(f"  fit[{m}]: {stats}")

    # observations summary
    obs = data.get("observations", [])
    influential = [o for o in obs if o.get("influential")]
    print(f"\n  observations total={len(obs)}  influential={len(influential)}")
    print(f"  First 3 observations:")
    for o in obs[:3]:
        print(f"    entity={o['entity']} t={o['time']} y={o['y_actual']} "
              f"fitted={o['y_fitted']} resid={o['residual']} influential={o['influential']}")
    if influential:
        print(f"  Sample influential obs:")
        for o in influential[:3]:
            print(f"    entity={o['entity']} t={o['time']} resid={o['residual']}")


for label, df, regs in [
    ("Dataset 1 — FE+CD (expect FE, DK SE, beta_std ratio~10)", df1, ["x1", "x2"]),
    ("Dataset 2 — RE clean (expect RE, clustered SE)", df2, ["x1"]),
]:
    status, body = post_panel(df, "entity", "time", "y", regs)
    print_result(label, status, body)
