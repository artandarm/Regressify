# Regressify

Automated econometric analysis for researchers and analysts who need rigorous statistical models without writing code. Upload a dataset, select variables — the pipeline runs the tests, selects the model, and explains every decision.

---

## What it does

Most econometric tools either require you to know which model to run upfront (Stata, R) or hide all the methodology behind a black box (AutoML). Regressify sits in between: it runs the full diagnostic workflow that a trained econometrician would run, exposes every test result and p-value, and gives you a model you can actually cite.

**Three analysis modes:**

| Mode | Use case |
|---|---|
| **Time Series** | Single variable observed over time — ARIMA/SARIMA, structural breaks, volatility |
| **Cross-section OLS** | Observations at a single point in time — regression with automatic SE correction |
| **Panel Data** | Multiple entities observed over multiple time periods — FE/RE/TWFE selection |

---

## Tech stack

- **Backend:** Python · FastAPI · statsmodels · linearmodels · arch
- **Frontend:** Next.js 16 · Tailwind CSS v4 · Recharts · KaTeX
- **Fonts:** Inter + IBM Plex Mono

---

## Running locally

```bash
# Backend (Python 3.11+)
pip install -r requirements.txt
uvicorn app.main:app --port 8080

# Frontend (Node 18+)
cd web
npm install
npm run dev
```

Open `http://localhost:3000`.

---

## Time Series

### What it does

Fits ARIMA/SARIMA models to a univariate time series. Detects structural breaks and fits separate models per segment. Detects volatility clustering and fits GARCH(1,1) if needed. Validates the final model out-of-sample.

### Logic tree

```
1. OUTLIER DETECTION
   └─ IQR + residual-based detection → additive outliers interpolated

2. SEASONALITY DETECTION
   └─ ACF/PACF periodogram → seasonal period m (if found)

3. STATIONARITY
   ├─ ADF + PP + KPSS (majority vote)
   ├─ Stationary → d = 0
   └─ Non-stationary → difference → retest (max d = 2)

4. STRUCTURAL BREAK DETECTION
   ├─ CUSUM test → candidate breakpoints
   ├─ Zivot-Andrews test → single break location
   ├─ Chow test at each candidate → confirm/reject
   └─ Colocation filter → merge breaks closer than min_segment_len

5. PER-SEGMENT MODEL FITTING
   ├─ Grid search: ARIMA(p, d, q), p ≤ 4, q ≤ 4
   ├─ Seasonal: SARIMA(p, d, q)(P, D, Q)[m] if m detected
   ├─ Order selection: AIC (primary) + BIC (tiebreaker)
   ├─ AIC/BIC conflict → model averaging (Akaike weights)
   ├─ Coefficient significance → t-tests, drop if all insig.
   ├─ Residual diagnostics:
   │   ├─ Ljung-Box → serial correlation
   │   └─ ARCH-LM → volatility clustering
   ├─ ARCH detected → fit GARCH(1,1)
   └─ Distribution: Normal vs Student-t (AIC comparison)

6. MULTI-SEGMENT VALIDATION
   └─ Walk-forward OOS: segmented vs unified model → RMSE comparison
```

### Data requirements & constraints

**Works best with:**
- 100 – 2 000 observations
- Regular frequency (daily, monthly, quarterly, annual)
- One clear trend or one structural break

**Hard limits:**
- Maximum **10 000 rows** (dev cap, first 10k used)
- Single numeric column only — no date parsing, no multivariate
- At most **2 structural breaks** detected
- Maximum differencing **d = 2**

**Known limitations:**
- No exogenous regressors (ARIMAX not supported)
- No long-memory models (ARFIMA)
- GARCH order fixed at (1,1) — higher orders not tested
- Seasonal period auto-detected from ACF; wrong detection → wrong model
- Very short series (< 30 obs): AIC grid search unreliable, results indicative only
- Highly irregular or intermittent series: outlier detection may over-clean

---

## Cross-section OLS

### What it does

Fits an OLS regression with automatic heteroskedasticity correction, VIF-based multicollinearity handling, backward BIC variable selection, Cook's D influential observation detection, and Chow structural stability tests.

### Logic tree

```
1. PRE-ANALYSIS
   ├─ Y type detection: continuous / binary / count
   ├─ Non-numeric columns → rejected with error
   └─ Missing values → listwise deletion

2. MULTICOLLINEARITY CHECK
   ├─ VIF per variable
   │   ├─ VIF < 5 → OK
   │   ├─ 5 ≤ VIF < 10 → warn
   │   └─ VIF ≥ 10 → drop variable, re-check
   └─ Condition number → global collinearity flag

3. MODEL ESTIMATION
   ├─ OLS (statsmodels)
   ├─ Breusch-Pagan test → heteroskedasticity?
   │   ├─ Yes → refit with HC3 robust SE
   │   └─ No → keep classical SE
   └─ F-test overall significance

4. VARIABLE SELECTION
   └─ Backward BIC stepwise:
       ├─ Drop variable with highest p-value if BIC improves
       └─ Repeat until no BIC improvement

5. DIAGNOSTICS
   ├─ Cook's D → influential observations (threshold: 4/n)
   ├─ Leverage (hat matrix diagonal)
   ├─ Jarque-Bera → residual normality
   ├─ RESET test → functional form (linearity)
   ├─ Durbin-Watson → serial correlation (informational)
   └─ Chow test per X variable → structural stability

6. OUTPUT
   └─ Equation · Coefficients + β* · VIF table · R² · AIC/BIC
```

### Data requirements & constraints

**Works best with:**
- 50 + observations
- Continuous numeric Y
- 2 – 15 regressors (before VIF pruning)
- No strong multicollinearity between Xs

**Hard limits:**
- Maximum **10 000 rows**
- All columns must be **numeric** — categorical variables must be pre-encoded as dummies before upload
- No automatic dummy creation for string columns

**Known limitations:**
- Binary Y (0/1) and count Y detected and flagged, but **OLS is still fitted** — no logistic or Poisson regression
- No interaction terms or polynomial features
- No instrumental variables (IV/2SLS)
- No time series correction — if data has a time dimension, use the Panel module instead
- Backward BIC stepwise can miss suppressor variable structures
- Influential observation threshold (Cook's D > 4/n) is heuristic — domain knowledge required

---

## Panel Data

### What it does

Estimates Pooled OLS, Fixed Effects, Random Effects, and Two-Way Fixed Effects models with clustered standard errors. Runs a full selection procedure (F-test → Hausman → Mundlak → TWFE F-test) to recommend the most appropriate specification. Diagnoses serial correlation and cross-sectional dependence in residuals, upgrading to Driscoll-Kraay SE when needed.

### Logic tree

```
1. PANEL STRUCTURE DIAGNOSTICS
   ├─ Balance check: unbalanced → warn
   ├─ T per entity:
   │   ├─ T < 2 → HARD STOP (use OLS)
   │   └─ T < 5 → warn (FE within-variation unreliable)
   ├─ Dynamic panel detection (lag variable names):
   │   └─ T < 5 → HARD STOP (need GMM/Arellano-Bond)
   ├─ N < 30 → warn (clustered SE asymptotic validity)
   └─ Within-variation check per regressor

2. MODEL ESTIMATION (all four, clustered SE)
   ├─ POLS  — Pooled OLS
   ├─ FE    — Fixed Effects (entity demeaning)
   ├─ RE    — Random Effects (GLS, Swamy-Arora)
   └─ TWFE  — Two-Way FE (entity + time effects)

3. MODEL SELECTION
   ├─ F-test (POLS vs FE): entity effects significant?
   │   └─ Not significant → POLS (high confidence)
   │
   ├─ Hausman test (FE vs RE): Wooldridge auxiliary regression
   │   ├─ p < 0.04  → FE (high confidence)
   │   ├─ p ≥ 0.10  → RE (high confidence)
   │   └─ 0.04 ≤ p < 0.10 → Mundlak tiebreaker
   │       ├─ Any X̄_i significant → FE (moderate)
   │       └─ None significant    → RE (moderate)
   │
   └─ TWFE F-test: time effects significant?
       └─ Yes → upgrade to TWFE

4. RESIDUAL DIAGNOSTICS
   ├─ Wooldridge AR(1) test (first-differenced residuals)
   │   └─ Serial correlation → warn
   ├─ Pesaran CD test (requires T ≥ 10)
   │   └─ Cross-sectional dependence → upgrade SE
   └─ SE upgrade: clustered → Driscoll-Kraay (kernel)

5. OUTPUT
   └─ Recommended model · All 4 models · β* (within-std) ·
      Observation-level fitted values · Model comparison table
```

### Data requirements & constraints

**Works best with:**
- **N ≥ 30** entities, **T ≥ 10** time periods
- Strongly balanced panel (same T for all entities)
- Entity and time columns clearly separated
- Time column as integer (1, 2, 3… or year: 2010, 2011…)

**Hard limits:**
- Maximum **10 000 rows**
- **T < 2** per entity → hard stop
- **T < 5** with lag variables → hard stop (dynamic panel)
- **Pesaran CD test requires T ≥ 10** — skipped for shorter panels

**Known limitations:**
- No endogeneity correction — IV/2SLS not implemented (biggest gap)
- No dynamic panel estimators (Arellano-Bond, Blundell-Bond)
- No spatial dependence testing
- No staggered DiD / heterogeneous treatment effects in TWFE
- AIC not reported for FE/RE models (profile log-likelihood unreliable)
- Unbalanced panels supported but some tests assume balance
- String time columns (e.g. "2020-Q1") not parsed — convert to integers before upload

---

## Data format

All three modes accept **CSV** (comma or semicolon separated, `.` or `,` decimal). Excel `.xlsx` also supported.

```
# Time Series: single numeric column
value
1.23
1.31
1.28
...

# OLS: Y + X columns, all numeric
gdp,investment,trade_openness,inflation
2.1,0.34,0.61,0.02
...

# Panel: entity + time + Y + X columns
country,year,gdp_growth,investment,trade
DEU,2010,4.1,0.22,0.41
DEU,2011,3.8,0.24,0.43
FRA,2010,1.9,0.19,0.39
...
```

---

## Current limitations (all modes)

- **Row cap:** 10 000 rows (first 10k used if exceeded)
- **No missing value imputation** — rows with NaN in selected columns are dropped
- **No categorical encoding** — all variables must be numeric at upload
- **English-only UI**
- **Single dataset in memory** — uploading a new file replaces the previous one

---

## Roadmap

- [ ] IV / 2SLS for endogenous regressors (Panel)
- [ ] Logistic regression and Poisson for binary/count Y (OLS)
- [ ] ARIMAX — exogenous regressors in time series
- [ ] Arellano-Bond GMM for dynamic panels
- [ ] OLS panel page in the frontend
- [ ] Panel data page in the frontend
- [ ] Export results to PDF / LaTeX table
