export interface PipelineStep {
  name: string;
  message: string;
  verdict: "ok" | "warn" | "error" | "info";
  p_value: number | null;
}

export interface OutlierPoint {
  index: number;
  original_value: number;
  cleaned_value: number;
}

export interface Coefficient {
  name: string;
  value: number;
  p_value: number;
  significant: boolean;
}

export interface GarchResult {
  fitted: boolean;
  omega: number | null;
  alpha: number | null;
  beta: number | null;
  aic: number | null;
  bic: number | null;
}

export interface ModelCandidate {
  label: string;
  aic: number;
  bic: number;
  weight: number;
  rmse: number | null;
}

export interface ModelAveraging {
  ambiguous: boolean;
  top_weight: number;
  candidates: ModelCandidate[];
}

export interface AicBicConflict {
  conflict: boolean;
  aic_model: string | null;
  bic_model: string | null;
  delta_aic: number | null;
}

export interface SegmentResult {
  segment_index: number;
  obs: number;
  start_t: number;
  end_t: number;
  model_type: string;
  equation_latex: string;
  d: number;
  aic: number;
  bic: number;
  aic_bic_conflict: AicBicConflict;
  ljungbox_ok: boolean;
  arch_effect: boolean;
  distribution: "normal" | "t";
  coefficients: Coefficient[];
  insignificant_coefs: string[];
  garch: GarchResult;
  model_averaging: ModelAveraging | null;
  steps: PipelineStep[];
}

export interface OOSComparison {
  winner: "segmented" | "unified" | "tie";
  rmse_unified: number;
  rmse_segmented: number;
}

export interface TSAnalysisResponse {
  pipeline_type: "timeseries";
  series_values: number[];
  series_original: number[];
  series_length: number;
  outliers: OutlierPoint[];
  seasonal_period: number | null;
  breakpoints: number[];
  variance_breakpoints: number[];
  oos_comparison: OOSComparison | null;
  pre_analysis_steps: PipelineStep[];
  break_detection_steps: PipelineStep[];
  segments: SegmentResult[];
}

export interface UploadResult {
  filename: string;
  rows: number;
  columns: string[];
  preview: Record<string, unknown>[];
}

// ── OLS types ──────────────────────────────────────────────

export interface OLSCoefficient {
  name: string;
  coef: number;
  std_err: number;
  t_stat: number;
  p_value: number;
  significant: boolean;
  verdict: "ok" | "warn";
}

export interface VifEntry {
  variable: string;
  vif: number;
  verdict: "ok" | "warn" | "error";
  note: string;
}

export interface InfluentialObs {
  index: number;
  cooks_d: number;
  leverage: number;
}

export interface RemovedVar {
  variable: string;
  pvalue: number;
  bic_before: number;
  bic_after: number;
}

export interface OLSAnalysisResponse {
  pipeline_type: "ols";
  y_col: string;
  x_cols: string[];
  x_cols_original: string[];
  n_obs: number;
  y_type: "continuous" | "binary" | "count";
  model_type: "OLS" | "OLS_robust_HC3";
  equation: string;
  equation_latex: string;
  y_actual: number[];
  y_fitted: number[];
  coefficients: OLSCoefficient[];
  insignificant_coefs: string[];
  r_squared: number;
  adj_r_squared: number;
  f_statistic: number;
  f_pvalue: number;
  aic: number;
  bic: number;
  condition_number: number;
  vif_table: VifEntry[];
  influential_obs: InfluentialObs[];
  removed_vars: RemovedVar[];
  pre_analysis_steps: PipelineStep[];
  multicollinearity_steps: PipelineStep[];
  model_estimation_steps: PipelineStep[];
  variable_selection_steps: PipelineStep[];
  diagnostics_steps: PipelineStep[];
}
