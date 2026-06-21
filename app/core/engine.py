import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.stats import jarque_bera, zscore
from scipy import stats
from pmdarima import auto_arima
from arch import arch_model
import ruptures as rpt
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# BASE
# ─────────────────────────────────────────────

class BasePipeline:
    def __init__(self, data):
        self.data = data
        self.log = []
        self.results = {}

    def run(self) -> dict:
        raise NotImplementedError

    def _log(self, step: str, decision: str, pvalue=None):
        entry = {"step": step, "decision": decision}
        if pvalue is not None:
            entry["pvalue"] = round(float(pvalue), 4)
        self.log.append(entry)


# ─────────────────────────────────────────────
# TIME SERIES PIPELINE
# ─────────────────────────────────────────────

class TSAnalysisPipeline(BasePipeline):

    def __init__(self, series: pd.Series):
        super().__init__(series.dropna().reset_index(drop=True))

    # ── 1. Outlier detection ──────────────────

    def remove_outliers(self, series: pd.Series) -> pd.Series:
        z = np.abs(zscore(series))
        n_outliers = int((z > 3.5).sum())
        cleaned = series.copy()
        cleaned[z > 3.5] = np.nan
        cleaned = cleaned.interpolate(method="linear").dropna()
        self._log(
            step="Outlier detection (Z-score > 3.5)",
            decision=f"Removed {n_outliers} outlier(s), interpolated"
        )
        return cleaned

    # ── 2. Seasonality detection ──────────────

    def detect_seasonality(self, series: pd.Series):
        from statsmodels.tsa.stattools import acf
        n = len(series)
        candidates = [m for m in [4, 12, 7, 52] if n > 2 * m]
        best_m = None
        for m in candidates:
            acf_vals = acf(series, nlags=m, fft=True)
            if abs(acf_vals[m]) > 1.96 / np.sqrt(n):
                best_m = m
                break
        self._log(
            step="Seasonality detection (ACF)",
            decision=f"Seasonal period m={best_m}" if best_m else "No seasonality detected"
        )
        return best_m

    # ── 3. Stationarity ───────────────────────

    def test_stationarity(self, series: pd.Series) -> bool:
        adf_p = adfuller(series)[1]
        pp_p = adfuller(series, regression="ct")[1]
        kpss_p = kpss(series, regression="c", nlags="auto")[1]
        stationary_votes = sum([adf_p < 0.05, pp_p < 0.05, kpss_p >= 0.05])
        is_stationary = stationary_votes >= 2
        self._log(
            step="Stationarity (ADF/PP/KPSS)",
            decision="Stationary" if is_stationary else "Non-stationary",
            pvalue=adf_p
        )
        return is_stationary

    def make_stationary(self, series: pd.Series):
        d = 0
        while not self.test_stationarity(series) and d < 2:
            series = series.diff().dropna()
            d += 1
            self._log(step=f"Differencing d={d}", decision="Applied diff()")
        return series, d

    # ── 4. Model selection ────────────────────

    def select_arma_order(self, series: pd.Series, seasonal_m=None):
        if seasonal_m:
            model = auto_arima(
                series,
                start_p=0, max_p=5,
                start_q=0, max_q=5,
                d=0,
                seasonal=True, m=seasonal_m,
                start_P=0, max_P=2,
                start_Q=0, max_Q=2,
                information_criterion="aic",
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore"
            )
            p, q = model.order[0], model.order[2]
            P, Q = model.seasonal_order[0], model.seasonal_order[2]

            if P == 0 and Q == 0:
                self._log(
                    step="Seasonality check",
                    decision="Seasonal orders P=Q=0 → treated as non-seasonal"
                )
                self._log(
                    step="Model order selection (auto_arima AIC)",
                    decision=f"ARMA({p},{q})"
                )
                return p, q, 0, 0, None

            self._log(
                step="Model order selection (auto_arima AIC)",
                decision=f"SARIMA({p},0,{q})({P},0,{Q})[{seasonal_m}]"
            )
            return p, q, P, Q, seasonal_m

        else:
            model = auto_arima(
                series,
                start_p=0, max_p=5,
                start_q=0, max_q=5,
                d=0,
                seasonal=False,
                information_criterion="aic",
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore"
            )
            p, q = model.order[0], model.order[2]
            self._log(
                step="Model order selection (auto_arima AIC)",
                decision=f"ARMA({p},{q})"
            )
            return p, q, 0, 0, None

    # ── 5. Coefficient significance ───────────

    def check_coef_significance(self, model) -> tuple:
        pvalues = model.pvalues
        coef_report = {}
        insignificant = []

        for name, pval in pvalues.items():
            if name == "sigma2":
                continue
            sig = pval < 0.05
            coef_report[name] = {
                "coef": round(float(model.params[name]), 4),
                "pvalue": round(float(pval), 4),
                "significant": sig
            }
            if not sig:
                insignificant.append(name)

        if insignificant:
            self._log(
                step="Coefficient significance (t-test)",
                decision=f"⚠️ Insignificant: {insignificant} — consider simpler model"
            )
        else:
            self._log(
                step="Coefficient significance (t-test)",
                decision="All coefficients significant ✓"
            )

        return coef_report, insignificant

    # ── 6. AIC/BIC conflict check ─────────────

    def check_aic_bic_conflict(self, series: pd.Series, p: int, q: int,
                                d: int, m=None, P=0, Q=0) -> dict:
        try:
            if m:
                aic_model = SARIMAX(
                    series, order=(p, d, q),
                    seasonal_order=(P, 0, Q, m)
                ).fit(disp=False)
            else:
                aic_model = ARIMA(series, order=(p, d, q)).fit()

            bic_winner = auto_arima(
                series, d=0, seasonal=False,
                information_criterion="bic",
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore"
            )
            p_bic, q_bic = bic_winner.order[0], bic_winner.order[2]
            conflict = not (p == p_bic and q == q_bic)

            if conflict:
                if m:
                    bic_model = SARIMAX(
                        series, order=(p_bic, d, q_bic),
                        seasonal_order=(P, 0, Q, m)
                    ).fit(disp=False)
                else:
                    bic_model = ARIMA(series, order=(p_bic, d, q_bic)).fit()

                delta = round(float(aic_model.aic) - float(bic_model.aic), 2)
                self._log(
                    step="AIC/BIC conflict check",
                    decision=(
                        f"⚠️ AIC→ARIMA({p},{d},{q}) vs BIC→ARIMA({p_bic},{d},{q_bic}) "
                        f"| ΔAIC={delta} — check coefficient significance"
                    )
                )
                return {
                    "conflict": True,
                    "aic_model": f"ARIMA({p},{d},{q})",
                    "bic_model": f"ARIMA({p_bic},{d},{q_bic})",
                    "delta_aic": delta
                }
            else:
                self._log(
                    step="AIC/BIC conflict check",
                    decision=f"AIC and BIC agree → ARIMA({p},{d},{q}) ✓"
                )
                return {"conflict": False}

        except Exception as e:
            self._log(step="AIC/BIC conflict check", decision=f"Skipped: {str(e)}")
            return {"conflict": False}

    # ── 7. Walk-forward validation ────────────

    def walk_forward(self, series: pd.Series, p: int, q: int, d: int,
                     p2: int, q2: int, test_size: float = 0.2) -> dict:
        n = len(series)
        split = int(n * (1 - test_size))

        if split < 20:
            self._log(step="Walk-forward validation", decision="Skipped — series too short")
            return {}

        label1 = f"ARIMA({p},{d},{q})"
        label2 = f"ARIMA({p2},{d},{q2})"
        errors = {label1: [], label2: []}

        for t in range(split, n):
            train = series.iloc[:t]
            actual = float(series.iloc[t])
            for (mp, mq), label in [((p, q), label1), ((p2, q2), label2)]:
                try:
                    m_fit = ARIMA(train, order=(mp, d, mq)).fit()
                    forecast = float(m_fit.forecast(steps=1).iloc[0])
                    errors[label].append((actual - forecast) ** 2)
                except Exception:
                    pass

        rmse = {}
        for label, errs in errors.items():
            if errs:
                rmse[label] = round(float(np.sqrt(np.mean(errs))), 6)

        if len(rmse) == 2:
            winner = min(rmse, key=rmse.get)
            loser = max(rmse, key=rmse.get)
            self._log(
                step="Walk-forward validation (RMSE)",
                decision=f"✓ {winner} RMSE={rmse[winner]} beats {loser} RMSE={rmse[loser]}"
            )
        else:
            winner = label1
            self._log(step="Walk-forward validation (RMSE)", decision="Only one model evaluated")

        return {"rmse": rmse, "winner": winner}

    # ── 8. Diagnostics ────────────────────────

    def test_ljungbox(self, residuals: np.ndarray) -> bool:
        result = acorr_ljungbox(residuals, lags=[10], return_df=True)
        pvalue = result["lb_pvalue"].values[0]
        ok = pvalue >= 0.05
        self._log(
            step="Ljung-Box (residuals)",
            decision="No autocorrelation ✓" if ok else "Autocorrelation detected ✗",
            pvalue=pvalue
        )
        return ok

    def test_arch_effect(self, residuals: np.ndarray) -> bool:
        _, pvalue, _, _ = het_arch(residuals, nlags=10)
        has_arch = pvalue < 0.05
        self._log(
            step="ARCH-LM test",
            decision="ARCH effect detected → fit GARCH" if has_arch else "No ARCH effect ✓",
            pvalue=pvalue
        )
        return has_arch

    def fit_garch(self, residuals: np.ndarray) -> dict:
        try:
            garch = arch_model(residuals, vol="Garch", p=1, q=1, dist="normal")
            res = garch.fit(disp="off")
            params = {k: round(float(v), 6) for k, v in res.params.items()}
            self._log(
                step="GARCH(1,1) estimation",
                decision=f"omega={params.get('omega','?')}, alpha={params.get('alpha[1]','?')}, beta={params.get('beta[1]','?')}"
            )
            return {
                "fitted": True,
                "aic": round(float(res.aic), 2),
                "bic": round(float(res.bic), 2),
                "params": params
            }
        except Exception as e:
            self._log(step="GARCH(1,1) estimation", decision=f"Failed: {str(e)}")
            return {"fitted": False}

    def select_distribution(self, residuals: np.ndarray) -> str:
        _, pvalue = jarque_bera(residuals)
        dist = "t" if pvalue < 0.05 else "normal"
        self._log(
            step="Jarque-Bera (distribution)",
            decision="Student-t" if dist == "t" else "Normal",
            pvalue=pvalue
        )
        return dist

    # ── 9. Structural breaks ──────────────────

    def find_breakpoints(self, series: pd.Series) -> list:
        arr = series.values
        algo = rpt.Pelt(model="rbf").fit(arr)
        breakpoints = algo.predict(pen=10)
        breakpoints = [b for b in breakpoints if b < len(arr)]
        self._log(
            step="Structural breaks (PELT)",
            decision=f"Found {len(breakpoints)} breakpoint(s): {breakpoints}"
        )
        return breakpoints

    # ── 10. Run ───────────────────────────────

    def run(self) -> dict:
        series = self.data.copy()

        # Шаг 1: очистка выбросов
        series = self.remove_outliers(series)

        # Шаг 2: сезонность
        seasonal_m = self.detect_seasonality(series)

        # Шаг 3: структурные разрывы
        breakpoints = self.find_breakpoints(series)

        # Шаг 4: разбивка на сегменты
        segments_raw = []
        prev = 0
        for bp in breakpoints + [len(series)]:
            segments_raw.append(series.iloc[prev:bp])
            prev = bp

        # Шаг 5: анализ каждого сегмента
        segment_models = []
        for i, seg in enumerate(segments_raw):
            if len(seg) < 20:
                self._log(
                    step=f"Segment {i+1} ({len(seg)} obs)",
                    decision="Skipped — too short (<20 obs)"
                )
                continue

            self._log(step=f"--- Segment {i+1} ({len(seg)} obs) ---", decision="")

            # стационарность
            stationary_seg, d = self.make_stationary(seg)

            # подбор порядка
            p, q, P, Q, m = self.select_arma_order(stationary_seg, seasonal_m)

            # оценка модели
            if m:
                model = SARIMAX(
                    stationary_seg,
                    order=(p, 0, q),
                    seasonal_order=(P, 0, Q, m)
                ).fit(disp=False)
            else:
                model = ARIMA(stationary_seg, order=(p, 0, q)).fit()

            resid = model.resid.values

            # значимость коэффициентов
            coef_report, insignificant = self.check_coef_significance(model)

            # AIC/BIC конфликт
            aic_bic_info = self.check_aic_bic_conflict(stationary_seg, p, q, d, m, P, Q)

            # walk-forward если конфликт и |ΔAIC| < 2
            walk_result = {}
            if aic_bic_info.get("conflict"):
                delta = aic_bic_info.get("delta_aic", 99)
                if abs(delta) < 2:
                    bic_label = aic_bic_info.get("bic_model", "")
                    try:
                        parts = bic_label.replace("ARIMA(", "").replace(")", "").split(",")
                        p2, q2 = int(parts[0]), int(parts[2])
                        walk_result = self.walk_forward(stationary_seg, p, q, d, p2, q2)
                        if walk_result.get("winner") == bic_label:
                            p, q = p2, q2
                            model = ARIMA(stationary_seg, order=(p, d, q)).fit()
                            resid = model.resid.values
                            coef_report, insignificant = self.check_coef_significance(model)
                            self._log(
                                step="Model update after walk-forward",
                                decision=f"Switched to {bic_label} based on RMSE"
                            )
                    except Exception as e:
                        self._log(step="Walk-forward", decision=f"Skipped: {str(e)}")

            # диагностика остатков
            lb_ok = self.test_ljungbox(resid)
            has_arch = self.test_arch_effect(resid)
            dist = self.select_distribution(resid)

            # GARCH если нужно
            garch_result = self.fit_garch(resid) if has_arch else {"fitted": False}

            model_label = (
                f"SARIMA({p},{d},{q})({P},0,{Q})[{m}]" if m
                else f"ARIMA({p},{d},{q})"
            )

            segment_models.append({
                "segment": i + 1,
                "obs": len(seg),
                "model_type": model_label,
                "arma_order": [p, q],
                "seasonal_order": [P, Q, m] if m else None,
                "d": d,
                "aic": round(float(model.aic), 2),
                "bic": round(float(model.bic), 2),
                "aic_bic_conflict": aic_bic_info,
                "walk_forward": walk_result,
                "ljungbox_ok": lb_ok,
                "arch_effect": has_arch,
                "garch": garch_result,
                "distribution": dist,
                "coefficients": coef_report,
                "insignificant_coefs": insignificant
            })

        self.results["pipeline_type"] = "timeseries"
        self.results["segments"] = segment_models
        self.results["breakpoints"] = breakpoints
        self.results["seasonal_period"] = seasonal_m
        self.results["log"] = self.log

        return self.results