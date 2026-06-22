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
