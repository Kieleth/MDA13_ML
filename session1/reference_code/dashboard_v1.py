# ============================================================
# dashboard_v1.py — S1 dashboard (versión completa)
# ============================================================
#
# Esta es la versión completa del dashboard al final de la S1.
# Es lo que paso_3.py debe quedar cuando rellenes todos los huecos.
#
# Úsalo si te has perdido entre paso_1 y paso_3 y quieres ver
# cómo funciona todo junto sin huecos.
#
# Cuatro paneles, todos sobre datos limpios de Cañadata:
#   - Clasificación (probabilidad de conversión)
#   - Regresión (ACV predicho)
#   - Clustering (cluster id, mapeable a arquetipo)
#   - Series temporales (forecast mensual de conversiones)
#
# Cómo ejecutar:
#
#   streamlit run session1/reference_code/dashboard_v1.py
#
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

st.set_page_config(page_title="Cañadata — S1", page_icon="🎯", layout="wide")


# ── Carga ──────────────────────────────────────────────────

@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "canadata_leads_clean.csv")


@st.cache_resource
def load_classifier() -> dict:
    return joblib.load(ROOT / "session1" / "models" / "classifier.pkl")


@st.cache_resource
def load_regressor() -> dict:
    return joblib.load(ROOT / "session1" / "models" / "regressor.pkl")


@st.cache_resource
def load_clusterer() -> dict:
    return joblib.load(ROOT / "session1" / "models" / "clusterer.pkl")


@st.cache_resource
def load_timeseries() -> dict:
    return joblib.load(ROOT / "session1" / "models" / "timeseries.pkl")


# ── Helpers ────────────────────────────────────────────────

CLF_INPUT_COLS = [
    "industry", "company_size", "country", "source", "demo_requested",
    "emails_opened", "response_time_hours", "n_meetings",
    "decision_maker_contacted", "quoted_acv_eur",
]
REG_INPUT_COLS = [c for c in CLF_INPUT_COLS if c != "quoted_acv_eur"]


def build_X(lead: dict, columns: list[str], training_features: list[str]) -> pd.DataFrame:
    df = pd.DataFrame([{k: lead[k] for k in columns}])
    cat_cols = [c for c in ["industry", "country", "source"] if c in columns]
    X = pd.get_dummies(df, columns=cat_cols)
    for col in ["demo_requested", "decision_maker_contacted"]:
        if col in X.columns:
            X[col] = X[col].astype(int)
    return X.reindex(columns=training_features, fill_value=0)


@st.cache_data
def cluster_archetype_map(_df: pd.DataFrame, _clusterer: dict) -> dict[int, str]:
    """Mapea cada cluster id al arquetipo más frecuente que cae en él."""
    rows = []
    for _, row in _df.iterrows():
        X = build_X(row.to_dict(), REG_INPUT_COLS, _clusterer["feature_names"])
        Xs = _clusterer["scaler"].transform(X)
        rows.append({
            "cluster": _clusterer["model"].predict(Xs)[0],
            "archetype": row["lead_segment_truth"],
        })
    df_clu = pd.DataFrame(rows)
    return df_clu.groupby("cluster")["archetype"].agg(lambda s: s.mode().iloc[0]).to_dict()


# ── App ────────────────────────────────────────────────────

df = load_data()
classifier = load_classifier()
regressor = load_regressor()
clusterer = load_clusterer()
ts = load_timeseries()
cluster_to_archetype = cluster_archetype_map(df, clusterer)

st.title("🎯 Cañadata — Dashboard S1")
st.caption(f"{len(df):,} leads limpios · 4 modelos clásicos servidos")

with st.sidebar:
    st.subheader("Lead")
    lead_id = st.selectbox("lead_id", df["lead_id"].tolist())
    st.divider()
    st.subheader("Forecast")
    forecast_months = st.slider("Meses a predecir", min_value=1, max_value=12, value=6)

lead = df[df["lead_id"] == lead_id].iloc[0].to_dict()

# ── Datos del lead ─────────────────────────────────────────
st.subheader("Datos del lead")
visible = ["company_name", "industry", "company_size", "country", "source",
           "demo_requested", "emails_opened", "response_time_hours",
           "n_meetings", "decision_maker_contacted", "quoted_acv_eur"]
st.json({k: lead[k] for k in visible})

if isinstance(lead.get("company_description"), str) and lead["company_description"]:
    st.caption("**Descripción libre** (lo que verá el LLM en S2):")
    st.write(f"_{lead['company_description']}_")

# ── Cuatro paneles ─────────────────────────────────────────
col_clf, col_reg, col_clu = st.columns(3)

with col_clf:
    st.subheader("🎯 Clasificación")
    X_clf = build_X(lead, CLF_INPUT_COLS, classifier["feature_names"])
    proba = classifier["model"].predict_proba(X_clf)[0, 1]
    st.metric("P(convertir)", f"{proba:.1%}")
    if proba >= 0.5:
        st.success("✓ Lead probable")
    else:
        st.warning("✗ Lead poco probable")
    st.caption(f"realidad: `{lead['converted']}`")

with col_reg:
    st.subheader("💰 Regresión (ACV)")
    X_reg = build_X(lead, REG_INPUT_COLS, regressor["feature_names"])
    acv_pred = float(np.exp(regressor["model"].predict(X_reg))[0])
    st.metric("ACV predicho", f"{acv_pred:,.0f} €")
    st.caption(f"cotizado real: `{lead['quoted_acv_eur']:,.0f} €`")

with col_clu:
    st.subheader("🔮 Clustering")
    X_clu = build_X(lead, REG_INPUT_COLS, clusterer["feature_names"])
    X_scaled = clusterer["scaler"].transform(X_clu)
    cluster_id = int(clusterer["model"].predict(X_scaled)[0])
    archetype_pred = cluster_to_archetype.get(cluster_id, "?")
    st.metric("Cluster", f"#{cluster_id} · {archetype_pred}")
    st.caption(f"arquetipo plantado: `{lead['lead_segment_truth']}`")

# ── Series temporales ──────────────────────────────────────
st.divider()
st.subheader("📈 Histórico de conversiones + forecast")

history: pd.Series = ts["history"]
ts_model = ts["model"]   # Prophet

future = ts_model.make_future_dataframe(periods=forecast_months, freq="MS")
fc = ts_model.predict(future)
forecast: pd.Series = fc.set_index("ds")["yhat"].iloc[-forecast_months:]
ci_df = fc.set_index("ds")[["yhat_lower", "yhat_upper"]].iloc[-forecast_months:]

chart_df = pd.DataFrame({
    "histórico": history,
    "forecast": forecast,
})
# Bridge: copia el último histórico al primer punto de forecast
# para que las dos líneas conecten visualmente.
if len(history) > 0 and len(forecast) > 0:
    chart_df.loc[history.index[-1], "forecast"] = history.iloc[-1]

st.line_chart(chart_df, height=320)
col_a, col_b, col_c = st.columns(3)
col_a.metric("Historia (meses)", f"{len(history)}")
col_b.metric(f"Forecast ({forecast_months}m)", f"{forecast.sum():.0f} conversiones")
col_c.metric("MAPE hold-out", f"{ts.get('validation_mape', 0):.1f}%")

with st.expander("Ver intervalo de confianza (Prophet yhat_lower / yhat_upper)"):
    st.dataframe(ci_df.round(1))
