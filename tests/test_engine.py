import numpy as np
import pandas as pd
import sys
sys.path.append(".")
from app.core.engine import TSAnalysisPipeline

# Генерируем синтетический AR(1) ряд с разрывом
np.random.seed(42)
n = 200
series1 = np.cumsum(np.random.normal(0, 1, n))        # первая половина
series2 = np.cumsum(np.random.normal(2, 1, n))        # вторая половина со сдвигом
full_series = pd.Series(np.concatenate([series1, series2]))

print("=== Запускаем TSAnalysisPipeline ===\n")
pipeline = TSAnalysisPipeline(full_series)
results = pipeline.run()

print("=== ЛОГ ШАГОВ ===")
for entry in results["log"]:
    pv = f"  p-value={entry['pvalue']}" if "pvalue" in entry else ""
    print(f"[{entry['step']}] → {entry['decision']}{pv}")

print("\n=== ТОЧКИ РАЗРЫВА ===")
print(results["breakpoints"])

print("\n=== МОДЕЛИ ПО СЕГМЕНТАМ ===")
for seg in results["segments"]:
    print(f"\nСегмент {seg['segment']} ({seg['obs']} наблюдений)")
    print(f"  Модель:       ARMA{seg['arma_order']}")
    print(f"  AIC/BIC:      {seg['aic']} / {seg['bic']}")
    print(f"  Ljung-Box OK: {seg['ljungbox_ok']}")
    print(f"  ARCH-эффект:  {seg['arch_effect']}")
    print(f"  Распределение:{seg['distribution']}")
    print(f"  Коэффициенты: {seg['coefficients']}")