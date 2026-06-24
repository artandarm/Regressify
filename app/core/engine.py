import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox, breaks_cusumolsresid
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

    def _log(self, step: str, decision: str, pvalue=None,
             verdict: str = "info"):
        entry = {
            "step": step,
            "decision": decision,
            "verdict": verdict,
            "phase": getattr(self, "_current_phase", "pre_analysis"),
        }
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
        n = len(series)
        # diff_vals[j] = series[j+1] - series[j], so for outlier at position t:
        #   diff INTO  t = diff_vals[t-1]
        #   diff OUT OF t = diff_vals[t]
        diff_vals = series.diff().dropna().values
        z_diff = np.abs(zscore(diff_vals))

        threshold = 3.5
        outlier_mask = pd.Series(False, index=series.index)

        # Additive outlier: spike at t creates a large diff in AND an equally large
        # diff out with the opposite sign (the return). Level shift has a large diff in
        # but a normal diff out — so it is NOT flagged here.
        for t in range(1, n - 1):
            if (z_diff[t - 1] > threshold and
                    z_diff[t] > threshold and
                    np.sign(diff_vals[t - 1]) != np.sign(diff_vals[t])):
                outlier_mask.iloc[t] = True

        # Edge fallback: raw z-score for t=0 and t=n-1 (no neighbour on one side)
        z_raw = np.abs(zscore(series.values))
        if z_raw[0] > threshold:
            outlier_mask.iloc[0] = True
        if z_raw[-1] > threshold:
            outlier_mask.iloc[-1] = True

        outlier_idx = list(series.index[outlier_mask])
        n_out = len(outlier_idx)
        cleaned = series.copy()
        cleaned[outlier_mask] = np.nan
        cleaned = cleaned.interpolate(method="linear").dropna()

        self._outlier_info = [
            {
                "index": int(idx),
                "original_value": round(float(series.iloc[idx]), 6),
                "cleaned_value": round(float(cleaned.loc[idx]), 6),
            }
            for idx in outlier_idx
        ]

        self._log(
            step="Outlier detection (AO test on diffs, Z>3.5)",
            decision=(
                f"Removed {n_out} additive outlier(s) at t={outlier_idx}, interpolated"
                if n_out else "No outliers detected"
            ),
            verdict="warn" if n_out else "ok",
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
            decision=f"Seasonal period m={best_m}" if best_m else "No seasonality detected",
            verdict="info",
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
            pvalue=adf_p,
            verdict="ok" if is_stationary else "warn",
        )
        return is_stationary

    def make_stationary(self, series: pd.Series):
        d = 0
        while not self.test_stationarity(series) and d < 2:
            series = series.diff().dropna()
            d += 1
            self._log(step=f"Differencing d={d}", decision="Applied diff()", verdict="info")
        return series, d

    # ── 4. Structural breaks ──────────────────

    def find_breakpoints(self, series: pd.Series, stationary_series: pd.Series) -> list:
        arr = stationary_series.values
        n = len(arr)

        # ── Калибровка штрафа PELT на синтетическом RW ──
        np.random.seed(42)
        rw = np.cumsum(np.random.normal(0, np.std(arr), n))
        rw_diff = np.diff(rw)
        algo_rw = rpt.Pelt(model="rbf").fit(rw_diff)
        pen = 10
        while len(algo_rw.predict(pen=pen)) > 3 and pen < 200:
            pen += 10
        self._log(
            step="PELT penalty calibration (sanity-check RW)",
            decision=f"Calibrated pen={pen} (RW false positives suppressed)",
            verdict="info",
        )

        # ── Детектор 1: PELT (rbf) ──
        algo_pelt = rpt.Pelt(model="rbf").fit(arr)
        pelt_bkps = algo_pelt.predict(pen=pen)
        pelt_bkps = [b for b in pelt_bkps if b < n]
        pelt_vote = len(pelt_bkps) > 0
        self._log(
            step="Break detection #1: PELT (rbf)",
            decision=(
                f"Found {len(pelt_bkps)} break(s) at {pelt_bkps}"
                if pelt_vote else "No breaks detected"
            ),
            verdict="warn" if pelt_vote else "ok",
        )

        # ── Детектор 2: CUSUM (OLS residuals) ──
        cusum_vote = False
        try:
            cusum_model = ARIMA(stationary_series, order=(1, 0, 0)).fit()
            cusum_result = breaks_cusumolsresid(cusum_model.resid)
            cusum_pvalue = float(cusum_result[1])
            cusum_vote = cusum_pvalue < 0.05
            self._log(
                step="Break detection #2: CUSUM (OLS residuals)",
                decision="Breaks detected" if cusum_vote else "No breaks",
                pvalue=cusum_pvalue,
                verdict="warn" if cusum_vote else "ok",
            )
        except Exception as e:
            self._log(step="Break detection #2: CUSUM", decision=f"Skipped: {str(e)}",
                      verdict="info")

        # ── Детектор 3: Binseg (l2, BIC-optimal k) ──
        binseg_vote = False
        try:
            algo_binseg = rpt.Binseg(model="l2").fit(arr)
            best_k, best_bic = 0, float("inf")
            k_range = range(0, min(6, n // 10 + 1))
            for k in k_range:
                if k == 0:
                    cost_k = float(np.sum((arr - arr.mean()) ** 2))
                else:
                    bkps_k = algo_binseg.predict(n_bkps=k)
                    cost_k = float(algo_binseg.cost.sum_of_costs(bkps_k))
                bic_k = n * np.log(max(cost_k / n, 1e-10)) + k * np.log(n)
                if bic_k < best_bic:
                    best_bic = bic_k
                    best_k = k
            binseg_vote = best_k > 0
            self._log(
                step="Break detection #3: Binseg (l2, BIC)",
                decision=(
                    f"BIC-optimal k={best_k} break(s)"
                    if binseg_vote else "BIC-optimal k=0 — no breaks"
                ),
                verdict="warn" if binseg_vote else "ok",
            )
        except Exception as e:
            self._log(step="Break detection #3: Binseg (BIC)", decision=f"Skipped: {str(e)}",
                      verdict="info")

        self._pelt_candidates = pelt_bkps

        # ── Голосование 2/3 ──
        votes = sum([pelt_vote, cusum_vote, binseg_vote])
        self._break_votes = votes
        if votes >= 2 and pelt_bkps:
            self._log(
                step="Break detection vote (2/3)",
                decision=f"{votes}/3 — segmenting at {pelt_bkps}",
                verdict="ok",
            )
            return pelt_bkps
        else:
            self._log(
                step="Break detection vote (2/3)",
                decision=f"{votes}/3 — treating as no structural breaks",
                verdict="info",
            )
            return []

    # ── 4б. Variance break detection ─────────

    def detect_variance_breaks(self, stationary_series: pd.Series) -> list:
        try:
            ar1 = ARIMA(stationary_series, order=(1, 0, 0)).fit()
            sq_resid = ar1.resid.values ** 2
            n = len(sq_resid)
            algo = rpt.Pelt(model="l2").fit(sq_resid)
            pen = float(np.var(sq_resid) * np.log(n))
            var_bkps = [b for b in algo.predict(pen=pen) if b < n]
            if var_bkps:
                self._log(
                    step="Variance break detection (squared residuals)",
                    decision=f"{len(var_bkps)} variance break(s) at {var_bkps} — possible volatility regime shift",
                    verdict="warn",
                )
            else:
                self._log(
                    step="Variance break detection (squared residuals)",
                    decision="No variance breaks detected",
                    verdict="ok",
                )
            return var_bkps
        except Exception as e:
            self._log(step="Variance break detection", decision=f"Skipped: {str(e)}",
                      verdict="info")
            return []

    # ── 4в. Co-location check ─────────────────

    def _check_colocation(self, mean_candidates: list,
                          variance_bkps: list, window: int = 20) -> list:
        """Return (mean_bp, var_bp) pairs where a PELT candidate is within `window` of a variance break."""
        pairs = []
        for m_bp in mean_candidates:
            for v_bp in variance_bkps:
                if abs(m_bp - v_bp) <= window:
                    pairs.append((m_bp, v_bp))
                    break
        return pairs

    # ── 4г. OOS: segmented vs unified ────────

    def compare_segmented_vs_unified(self, series: pd.Series,
                                     candidates: list, d: int) -> dict:
        n = len(series)
        split = int(n * 0.8)

        if split < 20:
            self._log(step="OOS: segmented vs unified", decision="Skipped — series too short",
                      verdict="info")
            return {}

        valid_bps = [bp for bp in candidates if 10 <= bp <= split - 10]
        if not valid_bps:
            self._log(step="OOS: segmented vs unified",
                      decision="Skipped — no candidate breakpoint inside training window",
                      verdict="info")
            return {}

        last_bp = valid_bps[-1]
        errors_u, errors_s = [], []

        for t in range(split, n):
            actual = float(series.iloc[t])
            train_u = series.iloc[:t]
            train_s = series.iloc[last_bp:t]
            if len(train_s) <= 10 + d:
                continue
            for train, err_list in [(train_u, errors_u), (train_s, errors_s)]:
                try:
                    m = ARIMA(train, order=(1, d, 0)).fit()
                    fc = float(m.forecast(steps=1).iloc[0])
                    err_list.append((actual - fc) ** 2)
                except Exception:
                    pass

        if len(errors_u) < 5 or len(errors_s) < 5:
            self._log(step="OOS: segmented vs unified",
                      decision="Skipped — insufficient test observations",
                      verdict="info")
            return {}

        rmse_u = round(float(np.sqrt(np.mean(errors_u))), 6)
        rmse_s = round(float(np.sqrt(np.mean(errors_s))), 6)

        if rmse_s < rmse_u * 0.98:
            winner = "segmented"
            decision = f"Segmented wins: RMSE={rmse_s} vs unified RMSE={rmse_u}"
            oos_verdict = "ok"
        elif rmse_u < rmse_s * 0.98:
            winner = "unified"
            decision = f"Unified wins: RMSE={rmse_u} vs segmented RMSE={rmse_s}"
            oos_verdict = "warn"
        else:
            winner = "tie"
            decision = f"Tie: unified RMSE={rmse_u}, segmented RMSE={rmse_s} — deferring to break-test votes"
            oos_verdict = "info"

        self._log(step="OOS: segmented vs unified (RMSE)", decision=decision,
                  verdict=oos_verdict)
        return {"winner": winner, "rmse_unified": rmse_u, "rmse_segmented": rmse_s}

    # ── 5. Model selection ────────────────────

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
                    decision="Seasonal orders P=Q=0 — treated as non-seasonal",
                    verdict="info",
                )
                self._log(
                    step="Model order selection (auto_arima AIC)",
                    decision=f"ARMA({p},{q})",
                    verdict="ok",
                )
                return p, q, 0, 0, None

            self._log(
                step="Model order selection (auto_arima AIC)",
                decision=f"SARIMA({p},0,{q})({P},0,{Q})[{seasonal_m}]",
                verdict="ok",
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
                decision=f"ARMA({p},{q})",
                verdict="ok",
            )
            return p, q, 0, 0, None

    # ── 6. Coefficient significance ───────────

    def check_coef_significance(self, model) -> tuple:
        pvalues = model.pvalues
        coef_report = {}
        insignificant = []

        for name, pval in pvalues.items():
            if name == "sigma2":
                continue
            sig = pval < 0.05
            coef_report[name] = {
                "coef":     round(float(model.params[name]), 4),
                "std_err":  round(float(model.bse[name]), 4),
                "t_stat":   round(float(model.tvalues[name]), 4),
                "pvalue":   round(float(pval), 4),
                "significant": sig,
            }
            if not sig:
                insignificant.append(name)

        if insignificant:
            self._log(
                step="Coefficient significance (t-test)",
                decision=f"Insignificant: {insignificant} — consider simpler model",
                verdict="warn",
            )
        else:
            self._log(
                step="Coefficient significance (t-test)",
                decision="All coefficients significant",
                verdict="ok",
            )

        return coef_report, insignificant

    # ── 7. LaTeX equation builder ─────────────

    def build_equation(self, p: int, q: int, d: int,
                       coefficients: dict, m=None, P=0, Q=0) -> str:
        coefs = {k: v["coef"] for k, v in coefficients.items()}

        const = coefs.get("const", None)
        ar_terms = [coefs[f"ar.L{i}"] for i in range(1, p + 1) if f"ar.L{i}" in coefs]
        ma_terms = [coefs[f"ma.L{i}"] for i in range(1, q + 1) if f"ma.L{i}" in coefs]

        # левая часть
        if d == 1:
            lhs = r"\Delta y_t"
            lag_lhs = r"\Delta y_{t-%d}"
        elif d == 2:
            lhs = r"\Delta^2 y_t"
            lag_lhs = r"\Delta^2 y_{t-%d}"
        else:
            lhs = r"y_t"
            lag_lhs = r"y_{t-%d}"

        rhs_parts = []

        # константа
        if const is not None:
            rhs_parts.append(f"{const:.4f}")

        # AR члены
        for i, coef in enumerate(ar_terms, 1):
            sign = "+" if coef >= 0 else "-"
            rhs_parts.append(rf"{sign} {abs(coef):.4f}\," + (lag_lhs % i))

        # MA члены
        if ma_terms:
            rhs_parts.append(r"+ \varepsilon_t")
            for i, coef in enumerate(ma_terms, 1):
                sign = "+" if coef >= 0 else "-"
                rhs_parts.append(rf"{sign} {abs(coef):.4f}\,\varepsilon_{{t-{i}}}")
        else:
            rhs_parts.append(r"+ \varepsilon_t")

        return lhs + " = " + " ".join(rhs_parts)

    # ── 8. AIC/BIC conflict check ─────────────

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
                        f"AIC→ARIMA({p},{d},{q}) vs BIC→ARIMA({p_bic},{d},{q_bic}) "
                        f"| delta_aic={delta} — check coefficient significance"
                    ),
                    verdict="warn",
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
                    decision=f"AIC and BIC agree — ARIMA({p},{d},{q})",
                    verdict="ok",
                )
                return {"conflict": False}

        except Exception as e:
            self._log(step="AIC/BIC conflict check", decision=f"Skipped: {str(e)}",
                      verdict="info")
            return {"conflict": False}

    # ── 9. Walk-forward validation ────────────

    def walk_forward(self, series: pd.Series, p: int, q: int, d: int,
                     p2: int, q2: int, test_size: float = 0.2) -> dict:
        n = len(series)
        split = int(n * (1 - test_size))

        if split < 20:
            self._log(step="Walk-forward validation", decision="Skipped — series too short",
                      verdict="info")
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
                decision=f"{winner} RMSE={rmse[winner]} beats {loser} RMSE={rmse[loser]}",
                verdict="ok",
            )
        else:
            winner = label1
            self._log(step="Walk-forward validation (RMSE)", decision="Only one model evaluated",
                      verdict="info")

        return {"rmse": rmse, "winner": winner}

    # ── 9б. Model averaging ───────────────────

    def build_model_candidates(self, series: pd.Series, d: int,
                               p_winner: int, q_winner: int, top_n: int = 3) -> dict:
        # Candidate grid: ±1 neighborhood of winner + minimal baselines
        grid = set()
        for dp in (-1, 0, 1):
            for dq in (-1, 0, 1):
                cp, cq = p_winner + dp, q_winner + dq
                if 0 <= cp <= 5 and 0 <= cq <= 5:
                    grid.add((cp, cq))
        grid.update({(0, 0), (1, 0), (0, 1)})

        fitted = []
        for cp, cq in sorted(grid):
            try:
                m = ARIMA(series, order=(cp, 0, cq)).fit()
                fitted.append({
                    "label": f"ARIMA({cp},{d},{cq})",
                    "p": cp, "q": cq,
                    "aic": round(float(m.aic), 2),
                    "bic": round(float(m.bic), 2),
                    "weight": None,
                    "rmse": None,
                })
            except Exception:
                pass

        if not fitted:
            return {"ambiguous": False, "top_weight": 1.0, "candidates": []}

        fitted.sort(key=lambda x: x["aic"])
        top = fitted[:top_n]

        # Akaike weights: w_i = exp(-0.5·Δ_i) / Σ exp(-0.5·Δ_j)
        aic_min = top[0]["aic"]
        raw_w = [np.exp(-0.5 * (c["aic"] - aic_min)) for c in top]
        w_total = sum(raw_w)
        for c, w in zip(top, raw_w):
            c["weight"] = round(w / w_total, 4)

        best_weight = top[0]["weight"]
        ambiguous = best_weight < 0.70

        if ambiguous:
            summary = ", ".join(
                f"{c['label']} w={c['weight']:.2f}" for c in top
            )
            self._log(
                step="Model averaging (Akaike weights)",
                decision=f"Top weight={best_weight:.2f} (<0.70) — evidence spread: {summary}",
                verdict="warn",
            )
            # Walk-forward RMSE for each candidate when ambiguous
            n = len(series)
            split = int(n * 0.8)
            if split >= 20:
                for c in top:
                    errors = []
                    for t in range(split, n):
                        train = series.iloc[:t]
                        actual = float(series.iloc[t])
                        try:
                            m = ARIMA(train, order=(c["p"], 0, c["q"])).fit()
                            fc = float(m.forecast(steps=1).iloc[0])
                            errors.append((actual - fc) ** 2)
                        except Exception:
                            pass
                    c["rmse"] = round(float(np.sqrt(np.mean(errors))), 6) if errors else None
        else:
            self._log(
                step="Model averaging (Akaike weights)",
                decision=f"{top[0]['label']} dominates: weight={best_weight:.2f} (>=0.70)",
                verdict="ok",
            )

        return {"ambiguous": ambiguous, "top_weight": best_weight, "candidates": top}

    # ── 10. Diagnostics ───────────────────────

    def test_ljungbox(self, residuals: np.ndarray) -> bool:
        result = acorr_ljungbox(residuals, lags=[10], return_df=True)
        pvalue = result["lb_pvalue"].values[0]
        ok = pvalue >= 0.05
        self._log(
            step="Ljung-Box (residuals)",
            decision="No autocorrelation" if ok else "Autocorrelation detected",
            pvalue=pvalue,
            verdict="ok" if ok else "error",
        )
        return ok

    def test_arch_effect(self, residuals: np.ndarray) -> bool:
        _, pvalue, _, _ = het_arch(residuals, nlags=10)
        has_arch = pvalue < 0.05
        self._log(
            step="ARCH-LM test",
            decision="ARCH effect detected — fitting GARCH" if has_arch else "No ARCH effect",
            pvalue=pvalue,
            verdict="warn" if has_arch else "ok",
        )
        return has_arch

    def fit_garch(self, residuals: np.ndarray) -> dict:
        try:
            garch = arch_model(residuals, vol="Garch", p=1, q=1, dist="normal")
            res = garch.fit(disp="off")
            params = {k: round(float(v), 6) for k, v in res.params.items()}
            self._log(
                step="GARCH(1,1) estimation",
                decision=f"omega={params.get('omega','?')}, alpha={params.get('alpha[1]','?')}, beta={params.get('beta[1]','?')}",
                verdict="ok",
            )
            return {
                "fitted": True,
                "aic": round(float(res.aic), 2),
                "bic": round(float(res.bic), 2),
                "params": params
            }
        except Exception as e:
            self._log(step="GARCH(1,1) estimation", decision=f"Failed: {str(e)}",
                      verdict="warn")
            return {"fitted": False}

    def select_distribution(self, residuals: np.ndarray) -> str:
        _, pvalue = jarque_bera(residuals)
        dist = "t" if pvalue < 0.05 else "normal"
        self._log(
            step="Jarque-Bera (distribution)",
            decision="Student-t residuals" if dist == "t" else "Normal residuals",
            pvalue=pvalue,
            verdict="ok",
        )
        return dist

    # ── 11. Run ───────────────────────────────

    def run(self) -> dict:
        series_original = self.data.copy()

        # Шаг 1: очистка выбросов
        self._current_phase = "pre_analysis"
        series = self.remove_outliers(series_original.copy())

        # Шаг 2: сезонность
        seasonal_m = self.detect_seasonality(series)

        # Шаг 3: стационарность на ПОЛНОМ ряду
        self._log(step="--- Full series pre-analysis ---", decision="", verdict="info")
        stationary_full, d_full = self.make_stationary(series.copy())

        # Шаг 4а: break detection в уровне/тренде (голосование 2/3)
        self._current_phase = "break_detection"
        breakpoints = self.find_breakpoints(series, stationary_full)
        pelt_candidates = getattr(self, "_pelt_candidates", [])

        # Шаг 4б: break detection в дисперсии (независимый)
        variance_bkps = self.detect_variance_breaks(stationary_full)

        # Шаг 4в: co-location — variance breaks corroborate PELT mean-break candidates
        colocation_pairs = self._check_colocation(pelt_candidates, variance_bkps)
        if colocation_pairs:
            for m_bp, v_bp in colocation_pairs:
                self._log(
                    step="Break co-location check",
                    decision=(
                        f"Mean-break candidate t={m_bp} coincides with variance break "
                        f"t={v_bp} (delta={abs(m_bp - v_bp)} obs) — likely volatility regime shift. "
                        f"GARCH on full series is a parsimonious alternative to mean segmentation."
                    ),
                    verdict="warn",
                )
        elif pelt_candidates:
            self._log(
                step="Break co-location check",
                decision="No variance break near mean-break candidate(s) — shift likely in mean/trend",
                verdict="ok",
            )

        # Шаг 4г: OOS сравнение + финальный арбитраж по политике:
        #   votes >= 2  → сегментация (vote решает, OOS информационно)
        #   votes == 1  → OOS tie-breaker только при наличии co-location
        #   votes == 0  → нет сегментации (pelt_candidates будет пуст)
        oos_result = {}
        votes = getattr(self, "_break_votes", 0)
        if pelt_candidates:
            oos_result = self.compare_segmented_vs_unified(series, pelt_candidates, d_full)
            winner = oos_result.get("winner")

            if votes >= 2:
                if winner == "unified":
                    self._log(
                        step="Final segmentation decision",
                        decision=(
                            f"Segmenting by vote ({votes}/3). "
                            f"OOS RMSE prefers unified — informational signal only."
                        ),
                        verdict="ok",
                    )
                else:
                    self._log(
                        step="Final segmentation decision",
                        decision=f"Segmentation confirmed by vote ({votes}/3) and OOS RMSE",
                        verdict="ok",
                    )
                # breakpoints уже выставлены find_breakpoints

            elif votes == 1:
                # PELT в меньшинстве — override только если variance co-location + OOS согласны
                if colocation_pairs and winner == "segmented":
                    self._log(
                        step="Final segmentation decision",
                        decision=(
                            f"Break-vote 1/3 (PELT only), but variance co-location + OOS both confirm "
                            f"— segmenting at {pelt_candidates}. "
                            f"Consider GARCH on full series as parsimonious alternative."
                        ),
                        verdict="warn",
                    )
                    breakpoints = pelt_candidates
                else:
                    reason = (
                        "no variance break co-location"
                        if not colocation_pairs
                        else f"OOS does not confirm ({winner})"
                    )
                    self._log(
                        step="Final segmentation decision",
                        decision=f"Break-vote 1/3 (PELT only), {reason} — no segmentation",
                        verdict="ok",
                    )

        # Шаг 5: разбивка ИСХОДНОГО ряда
        segments_raw = []
        seg_bounds = []
        prev = 0
        for bp in breakpoints + [len(series)]:
            segments_raw.append(series.iloc[prev:bp])
            seg_bounds.append((prev, bp))
            prev = bp

        # Шаг 6: анализ каждого сегмента
        segment_models = []
        for i, (seg, (seg_start, seg_end)) in enumerate(zip(segments_raw, seg_bounds)):
            self._current_phase = f"segment_{i+1}"
            if len(seg) < 20:
                self._log(
                    step=f"Segment {i+1} ({len(seg)} obs)",
                    decision="Skipped — too short (<20 obs)",
                    verdict="warn",
                )
                continue

            self._log(step=f"--- Segment {i+1} ({len(seg)} obs) ---", decision="",
                      verdict="info")

            # стационарность сегмента
            stationary_seg, d = self.make_stationary(seg)

            # подбор порядка
            p, q, P, Q, m = self.select_arma_order(stationary_seg, seasonal_m)

            # model averaging — Akaike weights по кандидатам вокруг AIC-победителя
            candidate_result = self.build_model_candidates(stationary_seg, d, p, q)

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

            # fallback SARIMA → ARIMA если сезонные коэфы незначимы
            if m and insignificant:
                seasonal_insig = [c for c in insignificant if "S." in c or "seasonal" in c.lower()]
                if seasonal_insig:
                    self._log(
                        step="SARIMA seasonal fallback",
                        decision=f"Insignificant seasonal coefs {seasonal_insig} — trying ARIMA",
                        verdict="info",
                    )
                    try:
                        model_noseas = ARIMA(stationary_seg, order=(p, 0, q)).fit()
                        if model_noseas.bic < model.bic:
                            model = model_noseas
                            m, P, Q = None, 0, 0
                            resid = model.resid.values
                            coef_report, insignificant = self.check_coef_significance(model)
                            self._log(
                                step="SARIMA seasonal fallback",
                                decision=f"ARIMA({p},{d},{q}) preferred by BIC",
                                verdict="ok",
                            )
                        else:
                            self._log(
                                step="SARIMA seasonal fallback",
                                decision="SARIMA kept — BIC still lower",
                                verdict="info",
                            )
                    except Exception as e:
                        self._log(step="SARIMA seasonal fallback", decision=f"Failed: {str(e)}",
                                  verdict="warn")

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
                                decision=f"Switched to {bic_label} based on RMSE",
                                verdict="ok",
                            )
                    except Exception as e:
                        self._log(step="Walk-forward", decision=f"Skipped: {str(e)}",
                                  verdict="info")

            # диагностика остатков
            lb_ok = self.test_ljungbox(resid)
            has_arch = self.test_arch_effect(resid)
            dist = self.select_distribution(resid)

            # GARCH если нужно
            garch_result = self.fit_garch(resid) if has_arch else {"fitted": False}

            # уравнение процесса
            equation = self.build_equation(p, q, d, coef_report, m, P, Q)

            model_label = (
                f"SARIMA({p},{d},{q})({P},0,{Q})[{m}]" if m
                else f"ARIMA({p},{d},{q})"
            )

            segment_models.append({
                "segment": i + 1,
                "obs": len(seg),
                "start_t": seg_start,
                "end_t": seg_end,
                "model_type": model_label,
                "equation": equation,
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
                "insignificant_coefs": insignificant,
                "model_candidates": candidate_result,
            })

        self.results["pipeline_type"] = "timeseries"
        self.results["series_values"] = [round(float(v), 6) for v in series.values]
        self.results["series_original"] = [round(float(v), 6) for v in series_original.values]
        self.results["outliers"] = getattr(self, "_outlier_info", [])
        self.results["segments"] = segment_models
        self.results["breakpoints"] = breakpoints
        self.results["variance_breakpoints"] = variance_bkps
        self.results["oos_comparison"] = oos_result
        self.results["seasonal_period"] = seasonal_m
        self.results["log"] = self.log

        return self.results