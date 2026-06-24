import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from app.core.engine import BasePipeline


# ─────────────────────────────────────────────
# PANEL PIPELINE
# ─────────────────────────────────────────────

_DISCLAIMERS = [
    "Endogeneity of regressors is not tested. IV/2SLS planned for next release.",
    "Spatial dependence across entities is not tested.",
    "Staggered treatment effects in TWFE are not checked.",
]


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
        """True if dep_var lagged version appears in regressors."""
        y = self.dep_var
        lag_patterns = {
            f"{y}_lag",
            f"L.{y}",
            f"L1.{y}",
            f"l.{y}",
            f"l1.{y}",
        }
        for r in self.regressors:
            if r in lag_patterns:
                return True
        # Also check numeric lag: if dep_var column shifted by 1 matches a regressor name
        # (heuristic: regressor name equals dep_var with a numeric suffix _1)
        if f"{y}_1" in self.regressors:
            return True
        return False

    def diagnose_panel_structure(self) -> dict:
        df = self.data
        cols = [self.entity_col, self.time_col, self.dep_var] + self.regressors

        # ── Listwise deletion on Y and X ──────────
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

        # ── Hard stops ────────────────────────────

        if T_min <= 1:
            hard_stops.append({
                "name": "T = 1",
                "verdict": "error",
                "message": (
                    f"Each entity has only T={T_min} time period(s). "
                    "Panel analysis requires T ≥ 2. Use cross-section (OLS) instead."
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
                "name": "Dynamic panel — T too small",
                "verdict": "error",
                "message": (
                    f"Dynamic panel detected (lagged Y among regressors) but T_min = {T_min} < 5. "
                    "GMM/Arellano-Bond instruments require T >= 5. "
                    "Drop the lagged dependent variable or collect more periods."
                ),
            })

        # ── Warnings (only if no hard stops) ──────

        if not hard_stops:
            if T_min < 5:
                warnings_list.append({
                    "name": "Small T",
                    "verdict": "warn",
                    "message": (
                        f"T_min = {T_min} < 5. Fixed effects estimates are consistent but "
                        "interpret with caution — incidental parameters bias is non-negligible."
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
                missing_frac = 1.0 - n_obs / max(n_obs_raw, 1)
                # Theoretical balanced obs = N * T_max
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
                warnings_list.append({
                    "name": "Unbalanced panel",
                    "verdict": v,
                    "message": msg,
                })

            if is_dynamic:
                warnings_list.append({
                    "name": "Dynamic panel",
                    "verdict": "info",
                    "message": (
                        "Dynamic panel detected (lagged Y among regressors). "
                        "GMM/Arellano-Bond not yet implemented. "
                        "FE estimator is biased (Nickell Bias, O(1/T)) — "
                        "interpret results with caution, especially for small T."
                    ),
                })

        # ── Fixed disclaimers ──────────────────────
        disclaimers = [{"verdict": "info", "message": d} for d in _DISCLAIMERS]

        return {
            "panel_structure": panel_structure,
            "hard_stops": hard_stops,
            "warnings": warnings_list,
            "disclaimers": disclaimers,
        }

    def run(self) -> dict:
        self._current_phase = "pre_analysis"
        return self.diagnose_panel_structure()
