import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import jarque_bera
from pmdarima import auto_arima
import ruptures as rpt
import warnings
warnings.filterwarnings("ignore")


class TSAnalysisPipeline:

    def __init__(self, series: pd.Series):
        self.series = series.dropna()
        self.log = []
        self.results = {}

    def _log(self, step: str, decision: str, pvalue: float = None):
        entry = {"step": step, "decision": decision}
        if pvalue is not None:
            entry["pvalue"] = round(pvalue, 4)
        self.log.append(entry)

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
        self.results["d"] = d
        return series

    def select_arma_order(self, series: pd.Series):
        model = auto_arima(
            series,
            start_p=0, max_p=5,
            start_q=0, max_q=5,
            d=0,
            information_criterion="bic",
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore"
        )
        p, q = model.order[0], model.order[2]
        self._log(
            step="ARMA order selection (auto_arima BIC)",
            decision=f"ARMA({p},{q})"
        )
        return p, q

    def test_ljungbox(self, residuals: np.ndarray, p: int, q: int) -> bool:
        result = acorr_ljungbox(residuals, lags=[10], return_df=True)
        pvalue = result["lb_pvalue"].values[0]
        ok = pvalue >= 0.05
        self._log(
            step="Ljung-Box (residuals)",
            decision="No autocorrelation ✓" if ok else "Autocorrelation detected",
            pvalue=pvalue
        )
        return ok

    def test_arch_effect(self, residuals: np.ndarray) -> bool:
        _, pvalue, _, _ = het_arch(residuals, nlags=10)
        has_arch = pvalue < 0.05
        self._log(
            step="ARCH-LM test",
            decision="ARCH effect → fit GARCH" if has_arch else "No ARCH effect ✓",
            pvalue=pvalue
        )
        return has_arch

    def select_distribution(self, residuals: np.ndarray) -> str:
        _, pvalue = jarque_bera(residuals)
        dist = "t" if pvalue < 0.05 else "normal"
        self._log(
            step="Jarque-Bera (distribution)",
            decision="Student-t" if dist == "t" else "Normal",
            pvalue=pvalue
        )
        return dist

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

    def run(self) -> dict:
        series = self.series.copy()

        breakpoints = self.find_breakpoints(series)

        segments_raw = []
        prev = 0
        for bp in breakpoints + [len(series)]:
            segments_raw.append(series.iloc[prev:bp])
            prev = bp

        segment_models = []
        for i, seg in enumerate(segments_raw):
            if len(seg) < 20:
                self._log(step=f"Segment {i+1}", decision="Too short, skipped (<20 obs)")
                continue

            self._log(step=f"--- Segment {i+1} ({len(seg)} obs) ---", decision="")

            stationary_seg = self.make_stationary(seg)
            p, q = self.select_arma_order(stationary_seg)

            model = ARIMA(stationary_seg, order=(p, 0, q)).fit()
            resid = model.resid.values

            lb_ok = self.test_ljungbox(resid, p, q)
            has_arch = self.test_arch_effect(resid)
            dist = self.select_distribution(resid)

            segment_models.append({
                "segment": i + 1,
                "obs": len(seg),
                "arma_order": (p, q),
                "d": self.results.get("d", 0),
                "aic": round(model.aic, 2),
                "bic": round(model.bic, 2),
                "ljungbox_ok": lb_ok,
                "arch_effect": has_arch,
                "distribution": dist,
                "coefficients": {k: round(v, 4) for k, v in model.params.items()}
            })
            self.results["d"] = 0

        self.results["segments"] = segment_models
        self.results["breakpoints"] = breakpoints
        self.results["log"] = self.log

        return self.results