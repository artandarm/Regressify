"""
Iteration 2 test: POLS/FE/RE estimation + model selection.
Dataset 1 — correlated random effects (FE case): should recommend FE, high confidence.
Dataset 2 — true random effects (RE case): should recommend RE, high confidence.
"""
import requests, json, io
import numpy as np
import pandas as pd

BASE = "http://localhost:8080"


def post_panel(df, entity_col, time_col, dep_var, regressors):
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    r_up = requests.post(f"{BASE}/upload", files={"file": ("panel.csv", buf, "text/csv")})
    assert r_up.status_code == 200, f"upload failed: {r_up.text}"
    r = requests.post(f"{BASE}/analyze/panel", json={
        "entity_col": entity_col,
        "time_col": time_col,
        "dep_var": dep_var,
        "regressors": regressors,
    })
    return r.status_code, r.json()


rng = np.random.default_rng(42)
N, T = 100, 10

# ── Dataset 1: Correlated Random Effects (FE case) ─────────────────────────
# alpha_i is a function of X mean -> correlated effects -> Hausman should reject RE
rows1 = []
for i in range(N):
    x_mean_i = rng.normal(0, 1)
    alpha_i = 0.9 * x_mean_i + rng.normal(0, 0.3)   # strong correlation
    for t in range(T):
        x_it = x_mean_i + rng.normal(0, 0.5)
        y_it = alpha_i + 1.5 * x_it + rng.normal(0, 0.5)
        rows1.append({"entity": f"E{i:03d}", "time": t, "y": y_it, "x1": x_it})

df1 = pd.DataFrame(rows1)

# ── Dataset 2: True Random Effects (RE case) ───────────────────────────────
# alpha_i is independent of X -> Hausman should not reject RE
rows2 = []
for i in range(N):
    alpha_i = rng.normal(0, 1)                        # independent of X
    x_mean_i = rng.normal(0, 1)
    for t in range(T):
        x_it = x_mean_i + rng.normal(0, 0.5)
        y_it = alpha_i + 1.5 * x_it + rng.normal(0, 0.5)
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

    # panel_structure
    ps = data.get("panel_structure", {})
    print(f"  panel_structure: N={ps.get('N')} T_min={ps.get('T_min')} "
          f"T_max={ps.get('T_max')} balanced={ps.get('balanced')}")

    # models summary
    models = data.get("models", {})
    for mkey in ("pols", "fe", "re", "twfe"):
        m = models.get(mkey)
        if m is None:
            print(f"  [{mkey.upper()}] not estimated")
            continue
        coef_names = [c["name"] for c in m.get("coefficients", [])]
        r2 = m.get("r_squared")
        r2w = m.get("r_squared_within", "N/A")
        theta = m.get("theta", "N/A")
        extra = ""
        if mkey == "fe":
            extra = f"  r2_within={r2w}"
        if mkey == "re":
            extra = f"  theta={theta}"
        print(f"  [{mkey.upper()}] R2={r2}  coefs={coef_names}{extra}")

    # selection tests
    tests = data.get("selection_tests", {})
    print()
    for tkey in ("f_test_pols_fe", "hausman", "mundlak", "twfe_f_test"):
        t = tests.get(tkey)
        if t is None:
            continue
        name = t.get("name", tkey)
        stat = t.get("statistic")
        pval = t.get("pvalue")
        verdict = t.get("verdict", "?")
        msg = t.get("message", "")[:80]
        print(f"  [{verdict.upper()}] {name}: stat={stat} p={pval}")
        print(f"         {msg}")

    # recommendation
    rec = data.get("recommendation", {})
    print(f"\n  RECOMMENDATION: {rec.get('recommended_model')}  "
          f"confidence={rec.get('confidence')}  "
          f"show_alternative={rec.get('show_alternative')}")
    for r in rec.get("reasoning", []):
        print(f"    - {r}")

    # final model
    fm = data.get("final_model")
    if fm:
        print(f"\n  FINAL MODEL: {fm.get('model_type')}  n_obs={fm.get('n_obs')}  "
              f"n_entities={fm.get('n_entities')}  n_periods={fm.get('n_periods')}")
        for c in fm.get("coefficients", []):
            print(f"    {c['name']:12s}  coef={c['coef']:8.4f}  "
                  f"se={c['std_err']:7.4f}  p={c['pvalue']:6.4f}  [{c['verdict']}]")


for label, df, regs in [
    ("Dataset 1 — CRE (expect FE, high confidence)", df1, ["x1"]),
    ("Dataset 2 — TRE (expect RE, high confidence)", df2, ["x1"]),
]:
    status, body = post_panel(df, "entity", "time", "y", regs)
    print_result(label, status, body)
