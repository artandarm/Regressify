import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import io

API = "http://127.0.0.1:8000"

st.set_page_config(page_title="AllRegressions", layout="wide")
st.title("📈 AllRegressions — Анализ временных рядов")

uploaded = st.file_uploader("Загрузи CSV или Excel", type=["csv", "xlsx"])

if uploaded:
    file_bytes = uploaded.read()

    res = requests.post(
        f"{API}/upload",
        files={"file": (uploaded.name, file_bytes, uploaded.type)}
    )
    meta = res.json()
    st.success(f"Загружено: {meta['rows']} строк, колонки: {meta['columns']}")

    column = st.selectbox("Выбери колонку с временным рядом", meta["columns"])

    # ── Умный парсинг для графика ─────────────
    try:
        if uploaded.name.endswith(".csv"):
            text = file_bytes.decode("utf-8", errors="replace")
            lines = [l for l in text.split("\n") if l.strip()]
            first_data = lines[1] if len(lines) > 1 else lines[0]
            if ";" in first_data:
                df = pd.read_csv(io.BytesIO(file_bytes), sep=";", decimal=",")
            else:
                comma_count = first_data.count(",")
                fields = first_data.split(",")
                all_short = all(len(f.strip()) <= 4 for f in fields[1:])
                if comma_count == 1 and all_short:
                    df = pd.read_csv(io.BytesIO(file_bytes), sep="\t", decimal=",")
                    if df.select_dtypes(include="number").shape[1] == 0:
                        df = pd.read_csv(io.BytesIO(file_bytes), sep=",", decimal=",")
                else:
                    df = pd.read_csv(io.BytesIO(file_bytes), sep=",", decimal=".")
        else:
            df = pd.read_excel(io.BytesIO(file_bytes))
    except Exception:
        df = pd.read_csv(io.BytesIO(file_bytes))

    if column in df.columns:
        series_vals = df[column].dropna().values
    else:
        num_cols = df.select_dtypes(include="number").columns.tolist()
        series_vals = df[num_cols[0]].dropna().values if num_cols else []

    if st.button("🚀 Запустить анализ", type="primary"):
        with st.spinner("Прогоняем тесты... (~30 сек)"):
            res = requests.post(f"{API}/analyze/ts", params={"column": column})

        if res.status_code != 200:
            st.error(f"Ошибка: {res.text}")
        else:
            data = res.json()
            breakpoints = data["breakpoints"]

            # ── Метаинфо ──────────────────────────────────────────
            col1, col2, col3 = st.columns(3)
            col1.metric("Сегментов", len(data["segments"]))
            col2.metric("Точек разрыва", len(breakpoints))
            sp = data.get("seasonal_period")
            col3.metric("Сезонный период", f"m={sp}" if sp else "Нет")

            # ── График ряда с разрывами ────────────────────────────
            st.subheader("📊 Временной ряд и точки разрыва")
            colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3"]
            fig = go.Figure()

            segments_bounds = [0] + breakpoints + [len(series_vals)]
            for i in range(len(segments_bounds) - 1):
                start = segments_bounds[i]
                end = segments_bounds[i + 1]
                fig.add_trace(go.Scatter(
                    x=list(range(start, end)),
                    y=series_vals[start:end].tolist(),
                    mode="lines",
                    name=f"Сегмент {i+1}",
                    line=dict(color=colors[i % len(colors)], width=1.5)
                ))

            for bp in breakpoints:
                fig.add_vline(
                    x=bp, line_dash="dash",
                    line_color="red", opacity=0.6,
                    annotation_text=f"t={bp}",
                    annotation_position="top"
                )

            fig.update_layout(
                xaxis_title="t", yaxis_title="value",
                legend_title="Сегменты",
                height=400, margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

            # ── Лог шагов ─────────────────────────────────────────
            st.subheader("🔍 Пошаговый лог тестов")
            for entry in data["log"]:
                if entry["step"].startswith("---"):
                    st.markdown(f"**{entry['step'].replace('---', '').strip()}**")
                    st.divider()
                else:
                    pv = f" &nbsp;`p={entry['pvalue']}`" if "pvalue" in entry else ""
                    good = any(x in entry["decision"] for x in [
                        "✓", "Stationary", "Normal", "Student", "Applied",
                        "No seasonality", "Seasonal", "agree", "Switched", "confirms"
                    ])
                    warn = any(x in entry["decision"] for x in [
                        "ARCH effect detected", "Autocorrelation detected",
                        "Skipped", "⚠️", "Insignificant"
                    ])
                    icon = "✅" if good else ("⚠️" if warn else "🔹")
                    st.markdown(
                        f"{icon} **{entry['step']}** → {entry['decision']}{pv}",
                        unsafe_allow_html=True
                    )

            # ── Таблица моделей ───────────────────────────────────
            st.subheader("📋 Итоговые модели по сегментам")
            rows = []
            for seg in data["segments"]:
                garch = seg.get("garch", {})
                conflict = seg.get("aic_bic_conflict", {})
                wf = seg.get("walk_forward", {})
                rows.append({
                    "Сегмент": seg["segment"],
                    "Наблюдений": seg["obs"],
                    "Модель": seg.get("model_type", "—"),
                    "AIC": seg["aic"],
                    "BIC": seg["bic"],
                    "AIC/BIC": "⚠️ Конфликт" if conflict.get("conflict") else "✅ Согласны",
                    "Walk-forward": wf.get("winner", "—").replace("ARIMA", "") if wf else "—",
                    "Ljung-Box": "✅ OK" if seg["ljungbox_ok"] else "❌ Автокорр.",
                    "ARCH": "⚠️ Есть" if seg["arch_effect"] else "✅ Нет",
                    "GARCH(1,1)": "✅" if garch.get("fitted") else "—",
                    "Распределение": "Student-t" if seg["distribution"] == "t" else "Normal",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # ── Model averaging (только при неоднозначном выборе) ─
            ambiguous_segs = [
                s for s in data["segments"]
                if s.get("model_candidates", {}).get("ambiguous")
            ]
            if ambiguous_segs:
                st.subheader("📊 Model averaging — неоднозначный выбор")
                st.caption(
                    "Akaike weight топ-кандидата < 0.70: доказательства размазаны между моделями. "
                    "Таблица показывает in-sample (AIC/BIC/вес) и out-of-sample (RMSE) сравнение."
                )
                for seg in ambiguous_segs:
                    mc = seg["model_candidates"]
                    st.markdown(
                        f"**Сегмент {seg['segment']}** — "
                        f"топ-вес {mc['top_weight']:.0%}, "
                        f"{len(mc['candidates'])} кандидата"
                    )
                    cand_rows = []
                    for i, c in enumerate(mc["candidates"]):
                        cand_rows.append({
                            "Модель": ("★ " if i == 0 else "  ") + c["label"],
                            "AIC": c["aic"],
                            "BIC": c["bic"],
                            "Вес Akaike": f"{c['weight'] * 100:.1f}%",
                            "OOS RMSE": c["rmse"] if c["rmse"] is not None else "—",
                        })
                    st.dataframe(
                        pd.DataFrame(cand_rows),
                        use_container_width=True,
                        hide_index=True
                    )

            # ── Коэффициенты + уравнение + GARCH ─────────────────
            st.subheader("🔢 Коэффициенты моделей")
            cols = st.columns(min(len(data["segments"]), 3))
            for i, seg in enumerate(data["segments"]):
                with cols[i % 3]:
                    st.markdown(f"**Сегмент {seg['segment']}** — {seg.get('model_type', '—')}")

                    # уравнение LaTeX
                    eq = seg.get("equation", "")
                    if eq:
                        st.markdown("**Уравнение процесса:**")
                        st.latex(eq)

                    # предупреждение о незначимых коэфах
                    insig = seg.get("insignificant_coefs", [])
                    if insig:
                        st.warning(f"Незначимые коэффициенты: {', '.join(insig)}")

                    # таблица коэффициентов с p-values
                    coef_rows = []
                    for name, info in seg["coefficients"].items():
                        coef_rows.append({
                            "Коэффициент": name,
                            "Значение": info["coef"],
                            "p-value": info["pvalue"],
                            "Значим": "✅" if info["significant"] else "❌"
                        })
                    if coef_rows:
                        st.dataframe(
                            pd.DataFrame(coef_rows),
                            use_container_width=True,
                            hide_index=True
                        )

                    # GARCH
                    garch = seg.get("garch", {})
                    if garch.get("fitted"):
                        st.markdown("**GARCH(1,1) параметры:**")
                        garch_rows = [
                            {"Параметр": k, "Значение": v}
                            for k, v in garch["params"].items()
                        ]
                        st.dataframe(
                            pd.DataFrame(garch_rows),
                            use_container_width=True,
                            hide_index=True
                        )
                        st.caption(f"AIC: {garch['aic']} | BIC: {garch['bic']}")