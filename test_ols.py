import numpy as np
import pandas as pd
from app.core.engine_ols import OLSPipeline


def print_log(log):
    icons = {"ok": "v", "warn": "!", "error": "X", "info": "."}
    for e in log:
        icon = icons.get(e["verdict"], "?")
        pv = f"  p={e['pvalue']:.4f}" if "pvalue" in e else ""
        phase = e["phase"][:10].ljust(10)
        print(f"  [{phase}] {icon}  {e['step']}: {e['decision'][:85]}{pv}")


# ── ДАТАСЕТ 1: чистый ─────────────────────────────────────────────────────
np.random.seed(42)
n = 200
X1 = np.random.normal(0, 1, n)
X2 = np.random.normal(0, 1, n)
Y  = 2 + 3*X1 - 1.5*X2 + np.random.normal(0, 1, n)

df_clean = pd.DataFrame({"Y": Y, "X1": X1, "X2": X2})
pipe1 = OLSPipeline(df_clean, "Y", ["X1", "X2"])
res1  = pipe1.run()

print("=" * 70)
print("DATASET 1: CLEAN  (Y = 2 + 3*X1 - 1.5*X2 + N(0,1))")
print("=" * 70)
print(f"  Model     : {res1['model_type']}")
print(f"  Equation  : {res1['equation']}")
print(f"  Adj. R2   : {res1['adj_r_squared']:.4f}")
print(f"  AIC/BIC   : {res1['aic']:.1f} / {res1['bic']:.1f}")
print(f"  Removed   : {[v['variable'] for v in res1['removed_vars']]}")
print(f"  Influential obs: {[o['index'] for o in res1['influential_obs']]}")
print()
print_log(pipe1.log)

# ── ДАТАСЕТ 2: проблемный ─────────────────────────────────────────────────
np.random.seed(7)
n = 200
X2b = np.random.normal(0, 1, n)

# Влиятельный выброс в X1 — умеренный, чтобы BP ещё видел гетероскедастичность
X1b = np.random.normal(0, 1, n)
X1b[42] = 5.0   # высокий leverage, но не экстремальный

# Сильно коррелирующий регрессор (VIF >> 10) — вычислен ИЗ финального X1
X3b = X1b * 0.98 + np.random.normal(0, 0.1, n)

# Гетероскедастичность: дисперсия ошибки растёт с X1
sigma = 0.3 + 1.5 * np.abs(X1b)
eps = np.array([np.random.normal(0, s) for s in sigma])
Yb = 2 + 3*X1b - 1.5*X2b + eps

df_bad = pd.DataFrame({"Y": Yb, "X1": X1b, "X2": X2b, "X3": X3b})

# Делаем obs 42 влиятельным по Y тоже (большой остаток = большой Cook's D)
df_bad.loc[42, "Y"] = df_bad.loc[42, "Y"] + 25.0

pipe2 = OLSPipeline(df_bad, "Y", ["X1", "X2", "X3"])
res2  = pipe2.run()

print()
print("=" * 70)
print("DATASET 2: PROBLEMATIC (heteroskedasticity + VIF + influential obs)")
print("=" * 70)
print(f"  Model     : {res2['model_type']}")
print(f"  Equation  : {res2['equation']}")
print(f"  Adj. R2   : {res2['adj_r_squared']:.4f}")
print(f"  AIC/BIC   : {res2['aic']:.1f} / {res2['bic']:.1f}")
print(f"  Removed   : {[v['variable'] for v in res2['removed_vars']]}")
print(f"  Influential obs ({len(res2['influential_obs'])}): "
      f"{[o['index'] for o in res2['influential_obs']]}")
print(f"  VIF table : {[(v['variable'], v['vif'], v['verdict']) for v in res2['vif_table']]}")
print()
print_log(pipe2.log)
