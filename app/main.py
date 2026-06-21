from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import io
from app.core.engine import TSAnalysisPipeline

app = FastAPI(title="AllRegressions API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_uploaded_df: dict = {}


def convert(obj):
    if isinstance(obj, dict):
        return {k: convert(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert(i) for i in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


@app.get("/")
def root():
    return {"status": "ok", "version": "0.2.0"}


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
        raise HTTPException(status_code=400, detail=f"Ошибка чтения файла: {str(e)}")

    _uploaded_df["data"] = df
    return {
        "filename": file.filename,
        "rows": len(df),
        "columns": list(df.columns),
        "preview": df.head(5).to_dict(orient="records"),
    }


@app.post("/analyze/ts")
async def analyze_ts(column: str):
    if "data" not in _uploaded_df:
        raise HTTPException(status_code=400, detail="Сначала загрузите файл через /upload")

    df = _uploaded_df["data"]

    if column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Колонка '{column}' не найдена")

    series = pd.Series(df[column].values, dtype=float)

    try:
        pipeline = TSAnalysisPipeline(series)
        results = pipeline.run()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка анализа: {str(e)}")

    return convert(results)