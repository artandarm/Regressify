from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import io
from app.core.engine import TSAnalysisPipeline

app = FastAPI(title="AllRegressions API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_uploaded_df: dict = {}


def convert(obj):
    """Рекурсивно конвертирует numpy-типы в стандартные Python-типы"""
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
    return {"status": "ok", "message": "AllRegressions API is running"}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        if file.filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Поддерживаются только CSV и Excel")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка чтения файла: {str(e)}")

    _uploaded_df["data"] = df

    return {
        "filename": file.filename,
        "rows": len(df),
        "columns": list(df.columns),
        "preview": df.head(5).to_dict(orient="records")
    }


@app.post("/analyze")
def analyze(column: str):
    if "data" not in _uploaded_df:
        raise HTTPException(status_code=400, detail="Сначала загрузи файл через /upload")

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