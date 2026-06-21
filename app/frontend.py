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

    # Читаем ряд для графика
    if uploaded.name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(file_bytes))
    else:
        df = pd.read_excel(io.BytesIO(file_bytes))

    series_vals = df[column].dropna().values

    if st.button("🚀 Запустить анализ", type="primary"):
        with st.spinner("Прогоняем тесты... (~20 сек)"):
            res = requests.post(f"{API}/analyze/ts", params={"column": column})

        if res.status_code != 200:
            st.error(f"Ошибка: {res.text}")
        else:
            data = res.json()
            breakpoints = data["breakpoints"]

            # ── Метаинфо ─────────────────────────────────────────
            col1, col2, col3 = st.columns(3)
            col1.metric("Сегментов", len(data["segments"]))
            col2.metric("Точек разрыва", len(breakpoints))
            sp = data.get("seasonal_period")
            col3.metric("Сезонный период", f"m={sp}" if sp else "Нет")

            # ── График ряда с разрывами ──────────────────────────
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

            # ── Лог шагов ────────────────────────────────────────
            st.subheader("🔍 Пошаговый лог тестов")
            for entry in data["log"]:
                if entry["step"].startswith("---"):
                    st.markdown(f"**{entry['step'].replace('---', '').strip()}**")
                    st.divider()
                else:
                    pv = f" &nbsp;`p={entry['pvalue']}`" if "pvalue" in entry else ""
                    good = any(x in entry["decision"] for x in [
                        "✓", "Stationary", "Normal", "Student", "Applied", "No seasonality", "Seasonal"
                    ])
                    warn = any(x in entry["decision"] for x in ["ARCH effect detected", "Autocorrelation detected", "Skipped"])
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
                rows.append({
                    "Сегмент": seg["segment"],
                    "Наблюдений": seg["obs"],
                    "Модель": seg.get("model_type", "—"),
                    "AIC": seg["aic"],
                    "BIC": seg["bic"],
                    "Ljung-Box": "✅ OK" if seg["ljungbox_ok"] else "❌ Автокорр.",
                    "ARCH-эффект": "⚠️ Есть" if seg["arch_effect"] else "✅ Нет",
                    "GARCH(1,1)": "✅ Оценён" if garch.get("fitted") else "—",
                    "Распределение": "Student-t" if seg["distribution"] == "t" else "Normal",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # ── Коэффициенты + GARCH ──────────────────────────────
            st.subheader("🔢 Коэффициенты моделей")
            cols = st.columns(min(len(data["segments"]), 3))
            for i, seg in enumerate(data["segments"]):
                with cols[i % 3]:
                    st.markdown(f"**Сегмент {seg['segment']}** — {seg.get('model_type', '—')}")
                    st.json(seg["coefficients"])

                    garch = seg.get("garch", {})
                    if garch.get("fitted"):
                        st.markdown("**GARCH(1,1) параметры:**")
                        st.json(garch["params"])
                        st.caption(f"AIC: {garch['aic']} | BIC: {garch['bic']}")