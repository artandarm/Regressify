import numpy as np
import pandas as pd
from scipy.stats import f as f_dist
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import warnings
warnings.filterwarnings("ignore")

from app.core.engine import BasePipeline

try:
    from linearmodels.panel import PooledOLS, PanelOLS, RandomEffects
    _HAS_LINEARMODELS = True
except ImportError:
    _HAS_LINEARMODELS = False


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

_DISCLAIMERS = [
    "Endogeneity of regressors is not tested. IV/2SLS planned for next release.",
    "Spatial dependence across entities is not tested.",
    "Staggered treatment effects in TWFE are not checked.",
]


def _sf(v, decimals=6):
    """Safe float: returns None for NaN/Inf, else rounded float."""
    if v is None:
        return None
    try:
        f = float(v)
        if np.isnan(f) or np.isinf(f):
            return None
        return round(f, decimals)
    except Exception:
        return None


# ─────────────────────────────────────────────
# PANEL PIPELINE
# ─────────────────────────────────────────────

class PanelPipeline(BasePipeline):

    def __init__(self, df: pd.DataFrame, entity_col: str, time_col: str,
                 dep_var: str, regressors: list):
        super().__init__(df)
        self.entity_col = entity_col
        self.time_col = time_col
        self.dep_var = dep_var
        self.regressors = list(regressors)

    # ══════════════════════════════════════════
    # DIAGNOSE PANEL STRUCTURE
    # ══════════════════════════════════════════

    def _is_dynamic(self) -> bool:
        y = self.dep_var
        lag_patterns = {
            f"{y}_lag", f"L.{y}", f"L1.{y}", f"l.{y}", f"l1.{y}", f"{y}_1",
        }
        return any(r in lag_patterns for r in self.regressors)

    def diagnose_panel_structure(self) -> dict:
        df = self.data
        cols = [self.entity_col, self.time_col, self.dep_var] + self.regressors

        available = [c for c in cols if c in df.columns]
        df_clean = df[available].dropna()

        entity_col = self.entity_col
        time_col = self.time_col

        entities = df_clean[entity_col].unique()
        N = int(len(entities))
        t_counts = df_clean.groupby(entity_col)[time_col].nunique()
        T_min = int(t_counts.min())
        T_max = int(t_counts.max())
        T_avg = round(float(t_counts.mean()), 2)
        balanced = bool(T_min == T_max)
        n_obs = int(len(df_clean))
        n_obs_raw = int(len(df[[entity_col, time_col]].dropna()))
        is_dynamic = self._is_dynamic()

        panel_structure = {
            "N": N,
            "T_min": T_min,
            "T_max": T_max,
            "T_avg": T_avg,
            "balanced": balanced,
            "n_obs": n_obs,
            "is_dynamic": is_dynamic,
        }

        hard_stops = []
        warnings_list = []

        if T_min <= 1:
            hard_stops.append({
                "name": "T = 1",
                "verdict": "error",
                "message": (
                    f"Each entity has only T={T_min} time period(s). "
                    "Panel analysis requires T >= 2. Use cross-section (OLS) instead."
                ),
            })

        if N < 10:
            hard_stops.append({
                "name": "N too small",
                "verdict": "error",
                "message": (
                    f"N = {N} entities is too small for reliable clustered SE and panel tests. "
                    "Minimum recommended: N >= 10."
                ),
            })

        if is_dynamic and T_min < 5:
            hard_stops.append({
                "name": "Dynamic panel -- T too small",
                "verdict": "error",
                "message": (
                    f"Dynamic panel detected (lagged Y among regressors) but T_min = {T_min} < 5. "
                    "GMM/Arellano-Bond instruments require T >= 5. "
                    "Drop the lagged dependent variable or collect more periods."
                ),
            })

        if not hard_stops:
            if T_min < 5:
                warnings_list.append({
                    "name": "Small T",
                    "verdict": "warn",
                    "message": (
                        f"T_min = {T_min} < 5. Fixed effects estimates are consistent but "
                        "interpret with caution -- incidental parameters bias is non-negligible."
                    ),
                })

            if N < 30:
                warnings_list.append({
                    "name": "Small N",
                    "verdict": "warn",
                    "message": (
                        f"N = {N} < 30. Hausman test may be unreliable. "
                        "Clustered SE require N >= 30 for asymptotic validity."
                    ),
                })

            if not balanced:
                theoretical = N * T_max
                missing_frac = 1.0 - n_obs / theoretical if theoretical > 0 else 0.0
                msg = (
                    f"Panel is unbalanced (T varies from {T_min} to {T_max}). "
                    "All estimators use unbalanced-compatible implementations."
                )
                v = "warn" if missing_frac > 0.30 else "info"
                if missing_frac > 0.30:
                    msg += (
                        f" Missing observations: {missing_frac*100:.1f}% of a balanced panel. "
                        "Check the missingness mechanism (MCAR/MAR/MNAR)."
                    )
                warnings_list.append({"name": "Unbalanced panel", "verdict": v, "message": msg})

            if is_dynamic:
                warnings_list.append({
                    "name": "Dynamic panel",
                    "verdict": "info",
                    "message": (
                        "Dynamic panel detected (lagged Y among regressors). "
                        "GMM/Arellano-Bond not yet implemented. "
                        "FE estimator is biased (Nickell Bias, O(1/T)) -- "
                        "interpret results with caution, especially for small T."
                    ),
                })

        disclaimers = [{"verdict": "info", "message": d} for d in _DISCLAIMERS]

        return {
            "panel_structure": panel_structure,
            "hard_stops": hard_stops,
            "warnings": warnings_list,
            "disclaimers": disclaimers,
        }

    # ══════════════════════════════════════════
    # ESTIMATE MODELS
    # ══════════════════════════════════════════

    def _result_to_dict(self, res, model_type: str, N: int, T_unique: int) -> dict | None:
        if res is None:
            return None
        coefs = []
        for name in res.params.index:
            pval = _sf(float(res.pvalues[name]))
            coefs.append({
                "name": name,
                "coef": _sf(float(res.params[name])),
                "std_err": _sf(float(res.std_errors[name])),
                "t_stat": _sf(float(res.tstats[name])),
                "pvalue": pval,
                "verdict": "ok" if (pval is not None and pval < 0.05) else "warn",
            })

        try:
            f_stat = _sf(float(res.f_statistic.stat))
            f_pval = _sf(float(res.f_statistic.pval))
        except Exception:
            f_stat, f_pval = None, None

        d = {
            "model_type": model_type,
            "coefficients": coefs,
            "r_squared": _sf(float(res.rsquared)),
            "f_statistic": f_stat,
            "f_pvalue": f_pval,
            "n_obs": int(res.nobs),
            "n_entities": N,
            "n_periods": T_unique,
        }

        if model_type in ("FE", "TWFE"):
            try:
                d["r_squared_within"] = _sf(float(res.rsquared_within))
            except Exception:
                d["r_squared_within"] = None

        if model_type == "RE":
            try:
                theta_vals = res.theta.values.flatten()
                d["theta"] = _sf(float(np.mean(theta_vals)))
            except Exception:
                d["theta"] = None

        return d

    def estimate_models(self, diag: dict) -> dict:
        if not _HAS_LINEARMODELS:
            raise RuntimeError(
                "linearmodels is not installed. Run: pip install linearmodels"
            )

        # ── Data prep ─────────────────────────────
        df = self.data
        entity_col = self.entity_col
        time_col = self.time_col
        dep_var = self.dep_var
        regressors = self.regressors

        cols = [entity_col, time_col, dep_var] + regressors
        df_clean = df[[c for c in cols if c in df.columns]].copy()
        for col in [dep_var] + regressors:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
        df_clean = df_clean.dropna().reset_index(drop=True)

        df_panel = df_clean.set_index([entity_col, time_col])
        Y = df_panel[dep_var]
        X = df_panel[regressors].copy().astype(float)
        X_c = X.copy()
        X_c["const"] = 1.0

        N = int(df_panel.index.get_level_values(0).nunique())
        T_unique = int(df_panel.index.get_level_values(1).nunique())

        # ── Block 1: Estimate POLS / FE / RE ──────
        self._current_phase = "model_estimation"

        pols_res = fe_res = re_res = None

        try:
            pols_res = PooledOLS(Y, X_c).fit(cov_type="clustered", cluster_entity=True)
            self._log(step="POLS", decision="Pooled OLS fitted with clustered SE", verdict="ok")
        except Exception as e:
            self._log(step="POLS", decision=f"Failed: {e}", verdict="error")

        try:
            fe_res = PanelOLS(Y, X, entity_effects=True).fit(
                cov_type="clustered", cluster_entity=True
            )
            self._log(step="FE", decision="Fixed Effects fitted with clustered SE", verdict="ok")
        except Exception as e:
            self._log(step="FE", decision=f"Failed: {e}", verdict="error")

        try:
            re_res = RandomEffects(Y, X_c).fit(cov_type="clustered", cluster_entity=True)
            self._log(step="RE", decision="Random Effects fitted with clustered SE", verdict="ok")
        except Exception as e:
            self._log(step="RE", decision=f"Failed: {e}", verdict="error")

        models_dict = {
            "pols": self._result_to_dict(pols_res, "POLS", N, T_unique),
            "fe":   self._result_to_dict(fe_res,   "FE",   N, T_unique),
            "re":   self._result_to_dict(re_res,   "RE",   N, T_unique),
            "twfe": None,
        }

        # ── Block 2: Selection tests ───────────────
        self._current_phase = "selection_tests"

        # Test 1: F-test (POLS vs FE)
        f_test_result = None
        f_pval_raw = None

        if fe_res is not None:
            try:
                fp = fe_res.f_pooled
                f_stat_raw = float(fp.stat)
                f_pval_raw = float(fp.pval)
                f_reject = f_pval_raw < 0.05
                f_test_result = {
                    "name": "F-test (POLS vs FE)",
                    "statistic": _sf(f_stat_raw),
                    "pvalue": _sf(f_pval_raw),
                    "verdict": "warn" if f_reject else "ok",
                    "message": (
                        "Individual effects are significant -> FE preferred over POLS"
                        if f_reject
                        else "Individual effects not significant -> POLS is sufficient"
                    ),
                }
                self._log(
                    step="F-test (POLS vs FE)",
                    decision=f"F={f_stat_raw:.3f} p={f_pval_raw:.4f}",
                    verdict="warn" if f_reject else "ok",
                )
            except Exception as e:
                f_test_result = {"name": "F-test (POLS vs FE)", "verdict": "error", "message": str(e)}

        # Test 2: Robust Hausman (Wooldridge auxiliary regression)
        hausman_result = None
        hausman_pval_raw = None
        mundlak_coefs = []

        if fe_res is not None and re_res is not None:
            try:
                means = df_clean.groupby(entity_col)[regressors].transform("mean")
                mean_cols = [f"mean_{r}" for r in regressors]
                means.columns = mean_cols

                X_aug = pd.concat([df_clean[regressors], means], axis=1).astype(float)
                X_aug_c = add_constant(X_aug, has_constant="add")
                y_flat = df_clean[dep_var].values.astype(float)
                entity_arr = df_clean[entity_col].values

                aug_model = OLS(y_flat, X_aug_c).fit(
                    cov_type="cluster", cov_kwds={"groups": entity_arr}
                )

                param_names = list(aug_model.params.index)
                mean_idx = [param_names.index(mc) for mc in mean_cols if mc in param_names]

                if mean_idx:
                    n_params = len(param_names)
                    r_mat = np.zeros((len(mean_idx), n_params))
                    for i, col_i in enumerate(mean_idx):
                        r_mat[i, col_i] = 1.0

                    f_res_h = aug_model.f_test(r_mat)
                    raw_stat = f_res_h.statistic
                    raw_pval = f_res_h.pvalue
                    hausman_stat_val = float(raw_stat.item() if hasattr(raw_stat, "item") else raw_stat)
                    hausman_pval_raw = float(raw_pval.item() if hasattr(raw_pval, "item") else raw_pval)

                    h_reject_strict = hausman_pval_raw < 0.04
                    h_borderline    = 0.04 <= hausman_pval_raw < 0.10

                    hausman_result = {
                        "name": "Hausman test (Wooldridge robust)",
                        "statistic": _sf(hausman_stat_val),
                        "pvalue": _sf(hausman_pval_raw),
                        "verdict": "warn" if h_reject_strict else ("info" if h_borderline else "ok"),
                        "message": (
                            "RE is inconsistent -> FE recommended"
                            if h_reject_strict
                            else (
                                "Borderline result -> Mundlak tiebreaker applied"
                                if h_borderline
                                else "RE is consistent -> RE preferred"
                            )
                        ),
                    }

                    for mc in mean_cols:
                        if mc in param_names:
                            pv = float(aug_model.pvalues[mc])
                            mundlak_coefs.append({
                                "name": mc,
                                "pvalue": _sf(pv),
                                "significant": pv < 0.05,
                            })

                    self._log(
                        step="Hausman test (Wooldridge robust)",
                        decision=f"F={hausman_stat_val:.3f} p={hausman_pval_raw:.4f}",
                        verdict=hausman_result["verdict"],
                    )

            except Exception as e:
                hausman_result = {
                    "name": "Hausman test (Wooldridge robust)",
                    "verdict": "error",
                    "message": str(e),
                }
                self._log(step="Hausman test", decision=f"Failed: {e}", verdict="error")

        # Test 3: Mundlak tiebreaker (0.04 <= p < 0.10)
        mundlak_result = None
        if hausman_pval_raw is not None and 0.04 <= hausman_pval_raw < 0.10:
            sig_means = [c["name"] for c in mundlak_coefs if c["significant"]]
            mundlak_result = {
                "name": "Mundlak test (tiebreaker)",
                "verdict": "warn" if sig_means else "ok",
                "message": (
                    f"Group means significant for {sig_means} -> "
                    "correlation between effects and regressors confirmed -> FE preferred"
                    if sig_means
                    else "No group means significant -> RE preferred (effects uncorrelated with regressors)"
                ),
                "mundlak_coefs": mundlak_coefs,
            }
            self._log(
                step="Mundlak tiebreaker",
                decision=mundlak_result["message"],
                verdict=mundlak_result["verdict"],
            )

        # ── Block 3: Decision matrix ───────────────
        self._current_phase = "model_selection"

        f_reject = f_pval_raw is not None and f_pval_raw < 0.05
        reasoning = []

        if not f_reject or f_test_result is None:
            recommended = "POLS"
            confidence = "high"
            show_alternative = False
            if f_test_result:
                reasoning.append(
                    f"F-test not significant (p={_sf(f_pval_raw)}) -> individual effects absent -> POLS sufficient"
                )
        else:
            reasoning.append(
                f"F-test significant (p={_sf(f_pval_raw)}) -> individual effects exist"
            )
            if hausman_result is None or hausman_pval_raw is None:
                recommended = "FE"
                confidence = "moderate"
                show_alternative = True
                reasoning.append("Hausman test unavailable -> defaulting to FE (conservative)")
            elif hausman_pval_raw < 0.04:
                recommended = "FE"
                confidence = "high"
                show_alternative = False
                reasoning.append(
                    f"Hausman test significant (p={_sf(hausman_pval_raw)}) -> RE inconsistent"
                )
                reasoning.append("Recommendation: Fixed Effects with clustered SE")
            elif hausman_pval_raw >= 0.10:
                recommended = "RE"
                confidence = "high"
                show_alternative = False
                reasoning.append(
                    f"Hausman test not significant (p={_sf(hausman_pval_raw)}) -> RE consistent"
                )
                reasoning.append("Recommendation: Random Effects with clustered SE")
            else:
                reasoning.append(
                    f"Hausman test borderline (p={_sf(hausman_pval_raw)}) -> Mundlak tiebreaker applied"
                )
                if mundlak_result and mundlak_result["verdict"] == "warn":
                    recommended = "FE"
                    confidence = "moderate"
                    show_alternative = True
                    reasoning.append("Mundlak: significant group means -> FE preferred (moderate confidence)")
                else:
                    recommended = "RE"
                    confidence = "moderate"
                    show_alternative = True
                    reasoning.append("Mundlak: no significant group means -> RE preferred (moderate confidence)")

        recommendation = {
            "recommended_model": recommended,
            "confidence": confidence,
            "show_alternative": show_alternative,
            "reasoning": reasoning,
        }
        self._log(
            step="Model recommendation",
            decision=f"Recommended: {recommended}, confidence: {confidence}",
            verdict="ok",
        )

        # ── Block 4: TWFE if FE recommended & T > 1 ─
        self._current_phase = "twfe_estimation"
        twfe_f_test = None

        if recommended == "FE" and T_unique > 1 and fe_res is not None:
            try:
                twfe_res = PanelOLS(Y, X, entity_effects=True, time_effects=True).fit(
                    cov_type="clustered", cluster_entity=True
                )
                models_dict["twfe"] = self._result_to_dict(twfe_res, "TWFE", N, T_unique)
                self._log(step="TWFE", decision="Two-way FE fitted with clustered SE", verdict="ok")

                rss_fe   = float(np.sum(fe_res.resids.values ** 2))
                rss_twfe = float(np.sum(twfe_res.resids.values ** 2))
                df_num = T_unique - 1
                df_den = float(twfe_res.df_resid)

                if df_den > 0 and rss_twfe > 1e-12 and rss_fe > rss_twfe:
                    f_time = ((rss_fe - rss_twfe) / df_num) / (rss_twfe / df_den)
                    pval_time = float(f_dist.sf(f_time, df_num, df_den))
                    t_reject = pval_time < 0.05

                    twfe_f_test = {
                        "name": "TWFE F-test (time effects)",
                        "statistic": _sf(f_time),
                        "pvalue": _sf(pval_time),
                        "verdict": "warn" if t_reject else "ok",
                        "message": (
                            "Time effects are significant -> TWFE preferred over FE. "
                            "WARNING: If treatment timing varies across units (staggered), "
                            "TWFE estimates may be biased -- algorithm does not check this."
                            if t_reject
                            else "Time effects not significant -> plain FE is sufficient"
                        ),
                    }
                    self._log(
                        step="TWFE F-test",
                        decision=f"F={f_time:.3f} p={pval_time:.4f}",
                        verdict="warn" if t_reject else "ok",
                    )

                    if t_reject:
                        recommendation["recommended_model"] = "TWFE"
                        recommendation["reasoning"].append(
                            f"TWFE F-test significant (p={_sf(pval_time)}) -> time effects present -> TWFE preferred"
                        )
                else:
                    twfe_f_test = {
                        "name": "TWFE F-test (time effects)",
                        "statistic": None,
                        "pvalue": None,
                        "verdict": "info",
                        "message": "TWFE RSS >= FE RSS -- time effects are negligible or collinear",
                    }

            except Exception as e:
                twfe_f_test = {
                    "name": "TWFE F-test (time effects)",
                    "verdict": "error",
                    "message": str(e),
                }
                self._log(step="TWFE", decision=f"Failed: {e}", verdict="error")

        # ── Final model ────────────────────────────
        final_key = recommendation["recommended_model"].lower()
        final_model_dict = models_dict.get(final_key) or models_dict.get("fe")

        return {
            "panel_structure": diag["panel_structure"],
            "hard_stops": [],
            "warnings": diag.get("warnings", []),
            "disclaimers": diag.get("disclaimers", []),
            "models": models_dict,
            "selection_tests": {
                "f_test_pols_fe": f_test_result,
                "hausman": hausman_result,
                "mundlak": mundlak_result,
                "twfe_f_test": twfe_f_test,
            },
            "recommendation": recommendation,
            "final_model": final_model_dict,
        }

    # ══════════════════════════════════════════
    # RUN
    # ══════════════════════════════════════════

    def run(self) -> dict:
        self._current_phase = "pre_analysis"
        diag = self.diagnose_panel_structure()
        if diag["hard_stops"]:
            return diag
        return self.estimate_models(diag)
