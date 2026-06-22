import numpy as np
import pandas as pd
from scipy.stats import jarque_bera
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import (
    het_breuschpagan, het_white, acorr_breusch_godfrey,
)
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.api import linear_reset
import warnings
warnings.filterwarnings("ignore")

from app.core.engine import BasePipeline


# ─────────────────────────────────────────────
# OLS PIPELINE
# ─────────────────────────────────────────────

class OLSPipeline(BasePipeline):

    def __init__(self, df: pd.DataFrame, y_col: str, x_cols: list):
        super().__init__(df)          # self.data = df (DataFrame)
        self.y_col = y_col
        self.x_cols = list(x_cols)

    # ══════════════════════════════════════════
    # БЛОК А: предобработка
    # ══════════════════════════════════════════

    def detect_y_type(self, y: pd.Series) -> str:
        """
        Определяет тип зависимой переменной.
        Возвращает "continuous", "binary" или "count".
        Логирует предупреждение если тип нестандартный для OLS.
        """
        unique_vals = y.dropna().unique()
        is_binary = set(unique_vals).issubset({0, 1, 0.0, 1.0})
        is_count = (
            not is_binary
            and np.issubdtype(y.dtype, np.integer) or
            (y.dropna() % 1 == 0).all() and (y.dropna() >= 0).all()
        )

        if is_binary:
            self._log(
                step="Y variable type detection",
                decision=(
                    "Binary outcome detected (only 0/1 values). "
                    "OLS will produce unbiased estimates but predicted values may fall outside [0,1]. "
                    "Consider Logit or Probit for a proper probability model."
                ),
                verdict="warn",
            )
            return "binary"

        if is_count and not is_binary:
            self._log(
                step="Y variable type detection",
                decision=(
                    "Count outcome detected (non-negative integers). "
                    "OLS is applied as requested, but Poisson or Negative Binomial "
                    "may be more appropriate if variance ≈ mean."
                ),
                verdict="warn",
            )
            return "count"

        self._log(
            step="Y variable type detection",
            decision="Continuous outcome — OLS is appropriate",
            verdict="ok",
        )
        return "continuous"

    def handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Логирует долю пропусков по каждой переменной.
        Предупреждает если > 50% по любой.
        Применяет listwise deletion.
        """
        cols = [self.y_col] + self.x_cols
        n_total = len(df)

        for col in cols:
            miss_frac = df[col].isna().mean()
            if miss_frac > 0.50:
                self._log(
                    step=f"Missing data check: {col}",
                    decision=(
                        f"{miss_frac*100:.1f}% missing in '{col}' — "
                        "results may be unreliable. Consider imputation or excluding this variable."
                    ),
                    verdict="warn",
                )
            elif miss_frac > 0:
                self._log(
                    step=f"Missing data check: {col}",
                    decision=f"{miss_frac*100:.1f}% missing in '{col}'",
                    verdict="info",
                )

        df_clean = df[cols].dropna()
        n_dropped = n_total - len(df_clean)

        if n_dropped > 0:
            self._log(
                step="Listwise deletion",
                decision=(
                    f"Dropped {n_dropped} row(s) with any missing value. "
                    f"{len(df_clean)} observations remain."
                ),
                verdict="info",
            )
        else:
            self._log(
                step="Listwise deletion",
                decision=f"No missing values — all {n_total} observations retained",
                verdict="ok",
            )

        return df_clean

    def detect_influential_obs(self, model, X: pd.DataFrame) -> list:
        """
        Cook's distance и leverage для каждого наблюдения.
        Порог Cook's D: 4/n.
        Логирует влиятельные наблюдения — НЕ удаляет.
        """
        n = model.nobs
        threshold_cooks = 4.0 / n

        influence = model.get_influence()
        cooks_d = influence.cooks_distance[0]
        leverage = influence.hat_matrix_diag

        influential = [
            {
                "index": int(i),
                "cooks_d": round(float(cooks_d[i]), 6),
                "leverage": round(float(leverage[i]), 6),
            }
            for i in range(int(n))
            if cooks_d[i] > threshold_cooks
        ]

        if influential:
            idx_list = [obs["index"] for obs in influential]
            self._log(
                step=f"Influential observations (Cook's D > 4/n = {threshold_cooks:.4f})",
                decision=(
                    f"{len(influential)} influential observation(s) at index(es) {idx_list}. "
                    "These observations have outsized leverage on the coefficients. "
                    "Inspect manually — not removed automatically."
                ),
                verdict="warn",
            )
        else:
            self._log(
                step=f"Influential observations (Cook's D > 4/n = {threshold_cooks:.4f})",
                decision="No influential observations detected",
                verdict="ok",
            )

        return influential

    # ══════════════════════════════════════════
    # БЛОК Б: мультиколлинеарность
    # ══════════════════════════════════════════

    def check_vif(self, X: pd.DataFrame) -> list:
        """
        VIF для каждого регрессора.
        < 5 → ok, 5–10 → warn, > 10 → error.
        """
        X_arr = add_constant(X.values, has_constant="add")
        vif_table = []

        for i, col in enumerate(X.columns):
            col_idx = i + 1  # +1 because add_constant prepends
            try:
                vif = float(variance_inflation_factor(X_arr, col_idx))
            except Exception:
                vif = float("nan")

            if vif > 10:
                verdict = "error"
                note = "high — consider removing this variable"
            elif vif > 5:
                verdict = "warn"
                note = "moderate multicollinearity"
            else:
                verdict = "ok"
                note = "acceptable"

            vif_table.append({
                "variable": col,
                "vif": round(vif, 3),
                "verdict": verdict,
                "note": note,
            })

            self._log(
                step=f"VIF: {col}",
                decision=f"VIF = {vif:.2f} — {note}",
                verdict=verdict,
            )

        return vif_table

    def check_condition_number(self, X: pd.DataFrame) -> float:
        """
        Condition number матрицы X (с константой).
        > 30 → warn, > 100 → error.
        """
        X_arr = add_constant(X.values, has_constant="add")
        _, sv, _ = np.linalg.svd(X_arr)
        cond = float(sv[0] / sv[-1]) if sv[-1] > 1e-12 else float("inf")

        if cond > 100:
            verdict, note = "error", "severe multicollinearity or near-singular matrix"
        elif cond > 30:
            verdict, note = "warn", "moderate collinearity"
        else:
            verdict, note = "ok", "acceptable"

        self._log(
            step="Condition number (X matrix)",
            decision=f"kappa = {cond:.1f} — {note}",
            verdict=verdict,
        )
        return cond

    # ══════════════════════════════════════════
    # БЛОК В: оценка OLS
    # ══════════════════════════════════════════

    def fit_ols_both(self, y: pd.Series, X: pd.DataFrame):
        """
        Оценивает plain OLS и HC3-робастную версию.
        Сравнивает SE: если отличие > 20% по любому коэфу — переключается на HC3.
        Возвращает (финальная модель, use_robust: bool, plain_model).
        """
        Xc = add_constant(X, has_constant="add")
        m_plain = OLS(y, Xc).fit()
        m_robust = OLS(y, Xc).fit(cov_type="HC3")

        se_plain = m_plain.bse
        se_robust = m_robust.bse

        # Относительное отличие SE по каждому коэфу (без константы — она менее интересна)
        param_names = [p for p in se_plain.index if p != "const"]
        max_rel_diff = 0.0
        worst_param = ""
        for p in param_names:
            rel = abs(se_robust[p] - se_plain[p]) / (abs(se_plain[p]) + 1e-12)
            if rel > max_rel_diff:
                max_rel_diff = rel
                worst_param = p

        use_robust = max_rel_diff > 0.20

        if use_robust:
            self._log(
                step="HC3 robust SE comparison",
                decision=(
                    f"SE differ by {max_rel_diff*100:.1f}% on '{worst_param}' (threshold 20%) — "
                    "switching to HC3 robust standard errors as final model. "
                    "Plain OLS SE would understate uncertainty under heteroskedasticity."
                ),
                verdict="warn",
            )
        else:
            self._log(
                step="HC3 robust SE comparison",
                decision=(
                    f"Max SE difference = {max_rel_diff*100:.1f}% (threshold 20%) — "
                    "plain OLS standard errors are stable. Using OLS."
                ),
                verdict="ok",
            )

        final_model = m_robust if use_robust else m_plain
        return final_model, use_robust, m_plain

    def check_coef_significance_ols(self, model) -> tuple:
        """
        Значимость коэффициентов OLS: p-value, t-stat, std_err, вердикт.
        Возвращает (coef_report dict, insignificant list).
        """
        coef_report = {}
        insignificant = []

        for name in model.params.index:
            pval = float(model.pvalues[name])
            sig = pval < 0.05
            verdict = "ok" if sig else "warn"
            coef_report[name] = {
                "coef": round(float(model.params[name]), 6),
                "std_err": round(float(model.bse[name]), 6),
                "t_stat": round(float(model.tvalues[name]), 4),
                "pvalue": round(pval, 4),
                "significant": sig,
                "verdict": verdict,
            }
            if not sig:
                insignificant.append(name)

        if insignificant:
            self._log(
                step="Coefficient significance (t-test)",
                decision=f"Insignificant at α=0.05: {insignificant}",
                verdict="warn",
            )
        else:
            self._log(
                step="Coefficient significance (t-test)",
                decision="All coefficients significant at α=0.05",
                verdict="ok",
            )

        return coef_report, insignificant

    def build_ols_equation(self, model, y_col: str, x_cols: list) -> str:
        """Строит читаемое уравнение Y = a + b1*X1 + b2*X2 ..."""
        const = round(float(model.params.get("const", 0)), 4)
        parts = [str(const)]
        for col in x_cols:
            if col in model.params.index:
                coef = round(float(model.params[col]), 4)
                sign = "+" if coef >= 0 else "-"
                parts.append(f"{sign} {abs(coef)}·{col}")
        return f"{y_col} = " + " ".join(parts)

    def get_model_stats(self, model, use_robust: bool) -> dict:
        """Собирает сводную статистику модели."""
        return {
            "model_type": "OLS_robust_HC3" if use_robust else "OLS",
            "r_squared": round(float(model.rsquared), 6),
            "adj_r_squared": round(float(model.rsquared_adj), 6),
            "f_statistic": round(float(model.fvalue), 4),
            "f_pvalue": round(float(model.f_pvalue), 6),
            "aic": round(float(model.aic), 4),
            "bic": round(float(model.bic), 4),
            "n_obs": int(model.nobs),
        }

    # ══════════════════════════════════════════
    # БЛОК Г: отбор переменных
    # ══════════════════════════════════════════

    def backward_stepwise_bic(self, y: pd.Series, X: pd.DataFrame):
        """
        Backward stepwise по BIC.
        На каждой итерации удаляем переменную с наибольшим p-value
        если это улучшает BIC. Останавливаемся когда BIC не улучшается.
        Возвращает (X_final, removed_list).
        """
        remaining = list(X.columns)
        removed = []

        Xc = add_constant(X[remaining], has_constant="add")
        current_bic = float(OLS(y, Xc).fit().bic)

        self._log(
            step="Backward stepwise BIC — start",
            decision=f"Full model BIC = {current_bic:.2f} with vars: {remaining}",
            verdict="info",
        )

        while len(remaining) > 1:
            m_full = OLS(y, add_constant(X[remaining], has_constant="add")).fit()

            # Находим переменную с наибольшим p-value (исключая const)
            pvals = {
                k: v for k, v in m_full.pvalues.items()
                if k != "const" and k in remaining
            }
            if not pvals:
                break

            worst_var = max(pvals, key=pvals.get)
            worst_pval = pvals[worst_var]

            # Пробуем убрать и считаем BIC
            candidate = [v for v in remaining if v != worst_var]
            Xc_try = add_constant(X[candidate], has_constant="add")
            bic_try = float(OLS(y, Xc_try).fit().bic)

            if bic_try < current_bic:
                self._log(
                    step=f"Backward stepwise — remove '{worst_var}'",
                    decision=(
                        f"Removed '{worst_var}' (p={worst_pval:.4f}): "
                        f"BIC {current_bic:.2f} → {bic_try:.2f} (Δ={bic_try - current_bic:.2f})"
                    ),
                    verdict="info",
                )
                removed.append({"variable": worst_var, "pvalue": round(worst_pval, 4),
                                 "bic_before": round(current_bic, 4),
                                 "bic_after": round(bic_try, 4)})
                remaining = candidate
                current_bic = bic_try
            else:
                self._log(
                    step="Backward stepwise — converged",
                    decision=(
                        f"Removing '{worst_var}' would worsen BIC "
                        f"({bic_try:.2f} > {current_bic:.2f}) — stopping. "
                        f"Final vars: {remaining}"
                    ),
                    verdict="ok",
                )
                break

        return X[remaining], removed

    # ══════════════════════════════════════════
    # БЛОК Д: диагностика остатков
    # ══════════════════════════════════════════

    def test_normality_jb(self, resid: np.ndarray, n: int) -> bool:
        """
        Jarque-Bera. При N > 100 ненормальность некритична.
        """
        _, pval = jarque_bera(resid)
        normal = pval >= 0.05

        if normal:
            self._log(
                step="Jarque-Bera (normality of residuals)",
                decision="Residuals consistent with normality",
                pvalue=pval,
                verdict="ok",
            )
        elif n > 100:
            self._log(
                step="Jarque-Bera (normality of residuals)",
                decision=(
                    f"Non-normal residuals (p={pval:.4f}), but N={n} > 100 — "
                    "CLT ensures asymptotic validity of inference. Informational only."
                ),
                pvalue=pval,
                verdict="info",
            )
        else:
            self._log(
                step="Jarque-Bera (normality of residuals)",
                decision=(
                    f"Non-normal residuals (p={pval:.4f}) with N={n} ≤ 100 — "
                    "inference may be unreliable. Consider bootstrap SE."
                ),
                pvalue=pval,
                verdict="warn",
            )
        return normal

    def test_heteroskedasticity(self, resid: np.ndarray, X: pd.DataFrame) -> bool:
        """
        Breusch-Pagan + White test.
        Если BP p < 0.05 → гетероскедастичность подтверждена → нужны робастные SE.
        White используется как кросс-проверка.
        Возвращает True если нужно переключиться на HC3.
        """
        Xc = add_constant(X, has_constant="add")

        # Breusch-Pagan
        bp_stat, bp_pval, _, _ = het_breuschpagan(resid, Xc)
        bp_reject = bp_pval < 0.05

        self._log(
            step="Breusch-Pagan test (heteroskedasticity)",
            decision=(
                "Heteroskedasticity detected — switching to HC3 robust SE"
                if bp_reject
                else "Homoskedasticity — OLS SE are efficient"
            ),
            pvalue=float(bp_pval),
            verdict="warn" if bp_reject else "ok",
        )

        # White test (cross-check)
        try:
            white_stat, white_pval, _, _ = het_white(resid, Xc)
            white_reject = white_pval < 0.05
            self._log(
                step="White test (heteroskedasticity cross-check)",
                decision=(
                    f"Confirms heteroskedasticity (p={white_pval:.4f})"
                    if white_reject
                    else f"Consistent with homoskedasticity (p={white_pval:.4f})"
                ),
                pvalue=float(white_pval),
                verdict="warn" if white_reject else "ok",
            )
        except Exception as e:
            self._log(
                step="White test (heteroskedasticity cross-check)",
                decision=f"Skipped: {e}",
                verdict="info",
            )

        return bp_reject

    def test_linearity_reset(self, model, y: pd.Series, X: pd.DataFrame,
                             suggest_log_y: bool) -> bool:
        """
        Ramsey RESET. Если p < 0.05 → предупреждение с конкретными предложениями.
        suggest_log_y: True если Y > 0 везде.
        """
        try:
            reset_result = linear_reset(model, power=3, use_f=True)
            pval = float(reset_result.pvalue)
            reject = pval < 0.05

            if reject:
                suggestions = []
                if suggest_log_y:
                    suggestions.append("log(Y) as dependent variable (Y is strictly positive)")
                suggestions.append("squared terms X² for nonlinear predictors")
                suggestions.append("interaction terms between regressors")
                sug_str = "; ".join(suggestions)
                self._log(
                    step="Ramsey RESET test (functional form)",
                    decision=(
                        f"Nonlinearity detected (p={pval:.4f}) — linear form may be misspecified. "
                        f"Consider: {sug_str}"
                    ),
                    pvalue=pval,
                    verdict="warn",
                )
            else:
                self._log(
                    step="Ramsey RESET test (functional form)",
                    decision=f"Linear functional form not rejected (p={pval:.4f})",
                    pvalue=pval,
                    verdict="ok",
                )
            return reject
        except Exception as e:
            self._log(
                step="Ramsey RESET test (functional form)",
                decision=f"Skipped: {e}",
                verdict="info",
            )
            return False

    def test_autocorrelation(self, resid: np.ndarray, X: pd.DataFrame):
        """
        Durbin-Watson (быстрый) + Breusch-Godfrey (строгий).
        Предупреждает при DW < 1.5 или > 2.5, но объясняет контекст.
        """
        dw = float(durbin_watson(resid))

        if dw < 1.5:
            dw_verdict, dw_note = "warn", "positive autocorrelation suspected"
        elif dw > 2.5:
            dw_verdict, dw_note = "warn", "negative autocorrelation suspected"
        else:
            dw_verdict, dw_note = "ok", "no autocorrelation"

        self._log(
            step="Durbin-Watson test (autocorrelation)",
            decision=(
                f"DW = {dw:.3f} — {dw_note}. "
                "Note: for pure cross-sections without natural ordering, "
                "autocorrelation is usually not meaningful."
            ),
            verdict=dw_verdict,
        )

        # Breusch-Godfrey
        try:
            Xc = add_constant(X, has_constant="add")
            bg_stat, bg_pval, _, _ = acorr_breusch_godfrey(
                OLS(pd.Series(resid), Xc).fit(), nlags=1
            )
            bg_reject = bg_pval < 0.05
            self._log(
                step="Breusch-Godfrey test (autocorrelation, lag=1)",
                decision=(
                    f"Autocorrelation detected (p={bg_pval:.4f})"
                    if bg_reject
                    else f"No autocorrelation (p={bg_pval:.4f})"
                ),
                pvalue=float(bg_pval),
                verdict="warn" if bg_reject else "ok",
            )
        except Exception as e:
            self._log(
                step="Breusch-Godfrey test (autocorrelation)",
                decision=f"Skipped: {e}",
                verdict="info",
            )

    # ══════════════════════════════════════════
    # RUN
    # ══════════════════════════════════════════

    def run(self) -> dict:
        df = self.data

        # ── Блок А: предобработка ─────────────────────────────────
        self._current_phase = "pre_analysis"

        y_type = self.detect_y_type(df[self.y_col])
        df_clean = self.handle_missing(df)

        y = df_clean[self.y_col].astype(float)
        X = df_clean[self.x_cols].astype(float)
        n = len(y)

        # ── Блок Б: мультиколлинеарность ─────────────────────────
        self._current_phase = "multicollinearity"

        vif_table = self.check_vif(X)
        cond_number = self.check_condition_number(X)

        # ── Блок В: оценка OLS ────────────────────────────────────
        self._current_phase = "model_estimation"

        final_model, use_robust, plain_model = self.fit_ols_both(y, X)
        coef_report, insignificant = self.check_coef_significance_ols(final_model)
        equation = self.build_ols_equation(final_model, self.y_col, self.x_cols)
        model_stats = self.get_model_stats(final_model, use_robust)

        f_pval = model_stats["f_pvalue"]
        self._log(
            step="F-test (joint significance)",
            decision=(
                f"F = {model_stats['f_statistic']:.2f}, p = {f_pval:.4f} — "
                + ("model jointly significant" if f_pval < 0.05 else "model NOT jointly significant")
            ),
            pvalue=f_pval,
            verdict="ok" if f_pval < 0.05 else "error",
        )
        self._log(
            step="Goodness of fit",
            decision=(
                f"Adj. R² = {model_stats['adj_r_squared']:.4f}, "
                f"AIC = {model_stats['aic']:.2f}, BIC = {model_stats['bic']:.2f}"
            ),
            verdict="info",
        )

        # Cook's D нужен на plain OLS (final_model может быть HC3, но influence одинаков)
        influential_obs = self.detect_influential_obs(plain_model, X)

        # ── Блок Г: отбор переменных ──────────────────────────────
        self._current_phase = "variable_selection"

        X_final, removed_vars = self.backward_stepwise_bic(y, X)

        # Переоцениваем финальную модель если что-то убрали
        if removed_vars:
            final_model, use_robust, plain_model = self.fit_ols_both(y, X_final)
            coef_report, insignificant = self.check_coef_significance_ols(final_model)
            equation = self.build_ols_equation(final_model, self.y_col, list(X_final.columns))
            model_stats = self.get_model_stats(final_model, use_robust)
            influential_obs = self.detect_influential_obs(plain_model, X_final)
            self._log(
                step="Model re-estimated after variable selection",
                decision=(
                    f"Final model: {self.y_col} ~ {list(X_final.columns)}. "
                    f"Adj. R² = {model_stats['adj_r_squared']:.4f}, "
                    f"BIC = {model_stats['bic']:.2f}"
                ),
                verdict="ok",
            )

        # ── Блок Д: диагностика ───────────────────────────────────
        self._current_phase = "diagnostics"

        resid = final_model.resid.values
        self.test_normality_jb(resid, n)

        suggest_log_y = bool((y > 0).all())
        needs_robust = self.test_heteroskedasticity(resid, X_final)

        # Если BP нашёл гетероскедастичность, а модель ещё не робастная — переключаем
        if needs_robust and not use_robust:
            self._log(
                step="HC3 upgrade (post-diagnostic)",
                decision=(
                    "Breusch-Pagan confirms heteroskedasticity — switching to HC3 robust SE. "
                    "SE comparison in Block C did not trigger the switch (< 20% diff), "
                    "but formal test confirms the problem."
                ),
                verdict="warn",
            )
            Xc_final = add_constant(X_final, has_constant="add")
            final_model = OLS(y, Xc_final).fit(cov_type="HC3")
            use_robust = True
            coef_report, insignificant = self.check_coef_significance_ols(final_model)
            model_stats = self.get_model_stats(final_model, use_robust)

        self.test_linearity_reset(final_model, y, X_final, suggest_log_y)
        self.test_autocorrelation(resid, X_final)

        # ── Результат ─────────────────────────────────────────────
        self.results = {
            "pipeline_type": "ols",
            "y_col": self.y_col,
            "x_cols": list(X_final.columns),
            "x_cols_original": self.x_cols,
            "n_obs": n,
            "y_type": y_type,
            "influential_obs": influential_obs,
            "vif_table": vif_table,
            "condition_number": round(cond_number, 2),
            "model_type": model_stats["model_type"],
            "equation": equation,
            "coefficients": coef_report,
            "insignificant_coefs": insignificant,
            "r_squared": model_stats["r_squared"],
            "adj_r_squared": model_stats["adj_r_squared"],
            "f_statistic": model_stats["f_statistic"],
            "f_pvalue": model_stats["f_pvalue"],
            "aic": model_stats["aic"],
            "bic": model_stats["bic"],
            "removed_vars": removed_vars,
            "log": self.log,
        }

        return self.results
