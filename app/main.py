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


def detect_csv_format(contents: bytes):
    text = contents.decode("utf-8", errors="replace")
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    data_line = lines[1] if len(lines) > 1 else lines[0]

    if ";" in data_line:
        return ";", ","
    elif "\t" in data_line:
        return "\t", "."
    else:
        # Нет точки и есть запятая — скорее всего European decimal: 0,1181
        has_dot = "." in data_line
        has_comma = "," in data_line
        if has_comma and not has_dot:
            # Пробуем sep=';' decimal=',' (одна колонка без разделителя)
            try:
                df_test = pd.read_csv(io.BytesIO(contents), sep=";", decimal=",", nrows=3)
                if df_test.shape[1] == 1 and df_test.select_dtypes(include="number").shape[1] == 1:
                    return ";", ","
            except Exception:
                pass
            # Fallback: читаем через engine с заменой запятой на точку
            return "EUROPEAN", ","
        else:
            return ",", "."


@app.get("/")
def root():
    return {"status": "ok", "message": "AllRegressions API is running"}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        if file.filename.endswith(".csv"):
            sep, decimal = detect_csv_format(contents)

            if sep == "EUROPEAN":
                text = contents.decode("utf-8", errors="replace")
                lines = text.split("\n")
                header = lines[0].strip()
                data_lines = [l.strip().replace(",", ".") for l in lines[1:] if l.strip()]
                fixed_text = header + "\n" + "\n".join(data_lines)
                df = pd.read_csv(io.StringIO(fixed_text))
            else:
                df = pd.read_csv(io.BytesIO(contents), sep=sep, decimal=decimal)

        elif file.filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Поддерживаются только CSV и Excel")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка чтения файла: {str(e)}")

    _uploaded_df["data"] = df

    return {
        "filename": file.filename,
        "rows": len(df),
        "columns": list(df.columns),
        "preview": df.head(5).to_dict(orient="records"),
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