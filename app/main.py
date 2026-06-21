from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal, Optional
import pandas as pd
import numpy as np
import io
from app.core.engine import TSAnalysisPipeline

app = FastAPI(title="AllRegressions API", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_uploaded_df: dict = {}


# ── Helpers ────────────────────────────────────────────────────────────────

def _np_clean(obj):
    """Recursively convert numpy scalars to Python primitives."""
    if isinstance(obj, dict):
        return {k: _np_clean(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_np_clean(i) for i in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ── Pydantic response models ───────────────────────────────────────────────

class PipelineStep(BaseModel):
    name: str
    message: str
    verdict: Literal["ok", "warn", "error", "info"]
    p_value: Optional[float] = None


class OutlierPoint(BaseModel):
    index: int
    original_value: float
    cleaned_value: float


class Coefficient(BaseModel):
    name: str
    value: float
    p_value: float
    significant: bool


class GarchResult(BaseModel):
    fitted: bool
    omega: Optional[float] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None
    aic: Optional[float] = None
    bic: Optional[float] = None


class ModelCandidate(BaseModel):
    label: str
    aic: float
    bic: float
    weight: float
    rmse: Optional[float] = None


class ModelAveraging(BaseModel):
    ambiguous: bool
    top_weight: float
    candidates: list[ModelCandidate]


class AicBicConflict(BaseModel):
    conflict: bool
    aic_model: Optional[str] = None
    bic_model: Optional[str] = None
    delta_aic: Optional[float] = None


class SegmentResult(BaseModel):
    segment_index: int
    obs: int
    start_t: int
    end_t: int
    model_type: str
    equation_latex: str
    d: int
    aic: float
    bic: float
    aic_bic_conflict: AicBicConflict
    ljungbox_ok: bool
    arch_effect: bool
    distribution: Literal["normal", "t"]
    coefficients: list[Coefficient]
    insignificant_coefs: list[str]
    garch: GarchResult
    model_averaging: Optional[ModelAveraging] = None
    steps: list[PipelineStep]


class OOSComparison(BaseModel):
    winner: Literal["segmented", "unified", "tie"]
    rmse_unified: float
    rmse_segmented: float


class TSAnalysisResponse(BaseModel):
    pipeline_type: Literal["timeseries"]
    series_values: list[float]
    series_original: list[float]
    series_length: int
    outliers: list[OutlierPoint]
    seasonal_period: Optional[int]
    breakpoints: list[int]
    variance_breakpoints: list[int]
    oos_comparison: Optional[OOSComparison]
    pre_analysis_steps: list[PipelineStep]
    break_detection_steps: list[PipelineStep]
    segments: list[SegmentResult]


# ── Log restructuring ──────────────────────────────────────────────────────

def _to_step(entry: dict) -> PipelineStep:
    return PipelineStep(
        name=entry["step"],
        message=entry["decision"],
        verdict=entry.get("verdict", "info"),
        p_value=entry.get("pvalue"),
    )


def _restructure_log(
    raw_log: list,
    segment_count: int,
) -> tuple:
    """Split flat log by phase into pre_analysis, break_detection, segment_N buckets."""
    pre: list[PipelineStep] = []
    breaks: list[PipelineStep] = []
    segs: dict = {f"segment_{i+1}": [] for i in range(segment_count)}

    for entry in raw_log:
        if entry["step"].startswith("---"):
            continue
        phase = entry.get("phase", "pre_analysis")
        step = _to_step(entry)
        if phase == "pre_analysis":
            pre.append(step)
        elif phase == "break_detection":
            breaks.append(step)
        elif phase in segs:
            segs[phase].append(step)
        else:
            pre.append(step)

    return pre, breaks, segs


def _build_garch(g: dict) -> GarchResult:
    if not g.get("fitted"):
        return GarchResult(fitted=False)
    params = g.get("params", {})
    return GarchResult(
        fitted=True,
        omega=params.get("omega"),
        alpha=params.get("alpha[1]"),
        beta=params.get("beta[1]"),
        aic=g.get("aic"),
        bic=g.get("bic"),
    )


def _build_segment(seg: dict, steps: list) -> SegmentResult:
    coef_list = [
        Coefficient(
            name=name,
            value=info["coef"],
            p_value=info["pvalue"],
            significant=info["significant"],
        )
        for name, info in seg.get("coefficients", {}).items()
    ]

    conflict_raw = seg.get("aic_bic_conflict", {})
    conflict = AicBicConflict(
        conflict=conflict_raw.get("conflict", False),
        aic_model=conflict_raw.get("aic_model"),
        bic_model=conflict_raw.get("bic_model"),
        delta_aic=conflict_raw.get("delta_aic"),
    )

    mc_raw = seg.get("model_candidates")
    model_avg = None
    if mc_raw and mc_raw.get("candidates"):
        model_avg = ModelAveraging(
            ambiguous=mc_raw["ambiguous"],
            top_weight=mc_raw["top_weight"],
            candidates=[
                ModelCandidate(
                    label=c["label"],
                    aic=c["aic"],
                    bic=c["bic"],
                    weight=c["weight"],
                    rmse=c.get("rmse"),
                )
                for c in mc_raw["candidates"]
            ],
        )

    return SegmentResult(
        segment_index=seg["segment"],
        obs=seg["obs"],
        start_t=seg.get("start_t", 0),
        end_t=seg.get("end_t", seg["obs"]),
        model_type=seg.get("model_type", ""),
        equation_latex=seg.get("equation", ""),
        d=seg.get("d", 0),
        aic=seg["aic"],
        bic=seg["bic"],
        aic_bic_conflict=conflict,
        ljungbox_ok=seg["ljungbox_ok"],
        arch_effect=seg["arch_effect"],
        distribution=seg.get("distribution", "normal"),
        coefficients=coef_list,
        insignificant_coefs=seg.get("insignificant_coefs", []),
        garch=_build_garch(seg.get("garch", {})),
        model_averaging=model_avg,
        steps=steps,
    )


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "version": "0.3.0"}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()

    try:
        text = contents.decode("utf-8", errors="replace")
        lines = [l for l in text.split("\n") if l.strip()]
        first_data_line = lines[1] if len(lines) > 1 else lines[0]

        if ";" in first_data_line:
            sep, decimal = ";", ","
        else:
            comma_count = first_data_line.count(",")
            dot_count = first_data_line.count(".")
            fields = first_data_line.split(",")
            all_short = all(len(f.strip()) <= 4 for f in fields[1:])
            if comma_count == 1 and dot_count == 0 and all_short:
                sep, decimal = "\t", ","
                df_test = pd.read_csv(io.BytesIO(contents), sep=sep, decimal=decimal, nrows=3)
                if df_test.select_dtypes(include="number").shape[1] == 0:
                    sep, decimal = ",", ","
            else:
                sep, decimal = ",", "."

        df = pd.read_csv(io.BytesIO(contents), sep=sep, decimal=decimal)
        df = df.dropna(how="all")

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File read error: {str(e)}")

    _uploaded_df["data"] = df
    return {
        "filename": file.filename,
        "rows": len(df),
        "columns": list(df.columns),
        "preview": df.head(5).to_dict(orient="records"),
    }


@app.post("/analyze/ts", response_model=TSAnalysisResponse)
async def analyze_ts(column: str):
    if "data" not in _uploaded_df:
        raise HTTPException(status_code=400, detail="Upload a file first via /upload")

    df = _uploaded_df["data"]

    if column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{column}' not found")

    series = pd.Series(df[column].values, dtype=float)

    try:
        pipeline = TSAnalysisPipeline(series)
        raw = _np_clean(pipeline.run())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

    seg_count = len(raw.get("segments", []))
    pre_steps, break_steps, seg_steps = _restructure_log(raw.get("log", []), seg_count)

    oos_raw = raw.get("oos_comparison", {})
    oos = None
    if oos_raw and "winner" in oos_raw:
        oos = OOSComparison(
            winner=oos_raw["winner"],
            rmse_unified=oos_raw["rmse_unified"],
            rmse_segmented=oos_raw["rmse_segmented"],
        )

    segments = [
        _build_segment(seg, seg_steps.get(f"segment_{seg['segment']}", []))
        for seg in raw.get("segments", [])
    ]

    return TSAnalysisResponse(
        pipeline_type="timeseries",
        series_values=raw.get("series_values", []),
        series_original=raw.get("series_original", []),
        series_length=len(raw.get("series_values", [])),
        outliers=[OutlierPoint(**o) for o in raw.get("outliers", [])],
        seasonal_period=raw.get("seasonal_period"),
        breakpoints=raw.get("breakpoints", []),
        variance_breakpoints=raw.get("variance_breakpoints", []),
        oos_comparison=oos,
        pre_analysis_steps=pre_steps,
        break_detection_steps=break_steps,
        segments=segments,
    )
