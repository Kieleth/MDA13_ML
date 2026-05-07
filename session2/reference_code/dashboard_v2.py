# ============================================================
# dashboard_v2.py — S2 dashboard (versión completa)
# ============================================================
#
# Lo que paso_6 debería ser cuando rellenas todos los huecos, más
# integración con paso_4 (panel zero-shot) y paso_5 (analista) en
# la misma app.
#
# Modos LLM disponibles:
#   - 🤖 Naive (zero-shot, sólo company_description)
#   - 🧮 Analista (text-to-code sobre `df`)
#   - 🧠 Operador (text-to-code con acceso a los 4 modelos entrenados)
#
# Más:
#   - los 4 paneles del dashboard_v1 (clásicos por lead)
#   - vista de comparación: misma pregunta en analista vs operador
#
# Cómo ejecutar:
#
#   streamlit run session2/reference_code/dashboard_v2.py
#
# ============================================================

import os
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / ".env")

st.set_page_config(page_title="Cañadata — S2", page_icon="🧠", layout="wide")

MODEL = "gpt-4.1-mini"


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


@st.cache_resource
def get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY no está. Crea `.env` en la raíz.")
        st.stop()
    return OpenAI(api_key=api_key)


# ── Helpers ────────────────────────────────────────────────

CLF_INPUT_COLS = [
    "industry", "company_size", "country", "source", "demo_requested",
    "emails_opened", "response_time_hours", "n_meetings",
    "decision_maker_contacted", "quoted_acv_eur",
]
REG_INPUT_COLS = [c for c in CLF_INPUT_COLS if c != "quoted_acv_eur"]


def build_X(lead: dict, columns: list[str], training_features: list[str]) -> pd.DataFrame:
    d = pd.DataFrame([{k: lead[k] for k in columns}])
    cat_cols = [c for c in ["industry", "country", "source"] if c in columns]
    X = pd.get_dummies(d, columns=cat_cols)
    for col in ["demo_requested", "decision_maker_contacted"]:
        if col in X.columns:
            X[col] = X[col].astype(int)
    return X.reindex(columns=training_features, fill_value=0)


@st.cache_data
def cluster_archetype_map(_df, _clusterer):
    rows = []
    for _, row in _df.iterrows():
        X = build_X(row.to_dict(), REG_INPUT_COLS, _clusterer["feature_names"])
        Xs = _clusterer["scaler"].transform(X)
        rows.append({"cluster": _clusterer["model"].predict(Xs)[0],
                     "archetype": row["lead_segment_truth"]})
    df_clu = pd.DataFrame(rows)
    return df_clu.groupby("cluster")["archetype"].agg(lambda s: s.mode().iloc[0]).to_dict()


# ── LLM helpers ────────────────────────────────────────────

def build_scoring_prompt(description: str) -> str:
    return (
        "Eres un analista comercial B2B en Cañadata, una SaaS de gestión de pipeline "
        "para equipos comerciales. Lee la descripción y estima la probabilidad "
        "(0-100) de que esta empresa se convierta en cliente de Cañadata.\n"
        "Devuelve SÓLO el número entero entre 0 y 100, sin explicación.\n\n"
        f"Descripción: {description}"
    )


@st.cache_data(show_spinner=False)
def llm_score_lead(_client, lead_id: str, description: str) -> int:
    resp = _client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": build_scoring_prompt(description)}],
        max_tokens=10,
        temperature=0.0,
    )
    text = resp.choices[0].message.content.strip()
    m = re.search(r"\d+", text)
    return int(m.group(0)) if m else -1


SYSTEM_PROMPT_ANALISTA = """Eres un analista de datos. Tienes un DataFrame de pandas llamado `df` con datos de leads B2B (Cañadata).

Columnas: lead_id (str), company_name (str), industry (str ∈ {SaaS, fintech, retail, logistics, healthcare, unknown}), company_size (int), country (str ∈ {ES, FR, DE, UK, IT, PT, unknown}), signup_date (str YYYY-MM-DD), source (str ∈ {organic, paid, referral, conference, outbound, unknown}), demo_requested (bool), emails_opened (int), response_time_hours (float), n_meetings (int), decision_maker_contacted (bool), quoted_acv_eur (float), company_description (str), converted (bool), converted_within_days (float, NaN si no convirtió), lead_segment_truth (str, USA SÓLO PARA EVALUAR, no como feature).

Genera código Python pandas para responder la pregunta del usuario.
Termina asignando el resultado a una variable llamada `resultado`.
Devuelve SÓLO un bloque ```python ... ```, sin explicación."""


SYSTEM_PROMPT_OPERADOR = """Eres un asistente con acceso a:
  - `df`: DataFrame de leads de Cañadata (columnas: industry, company_size, country, source, demo_requested, emails_opened, response_time_hours, n_meetings, decision_maker_contacted, quoted_acv_eur, converted, ...).
  - `classifier` (dict con 'model', 'feature_names'). Predice conversión sobre 1 lead así:
        X = build_X(lead_dict, CLF_INPUT_COLS, classifier['feature_names'])
        proba = classifier['model'].predict_proba(X)[0, 1]
  - `regressor` (igual estructura, target en log space; convierte con np.exp).
  - `clusterer` (dict con 'model', 'scaler', 'feature_names'). Para 1 lead:
        Xs = clusterer['scaler'].transform(build_X(lead_dict, REG_INPUT_COLS, clusterer['feature_names']))
        cluster_id = clusterer['model'].predict(Xs)[0]
  - `timeseries` (dict con 'model' (SARIMAX fit), 'history' (Series mensual)).
        forecast = timeseries['model'].get_forecast(steps=N).predicted_mean
  - `build_X(lead_dict, columns, training_features)` helper para construir features de 1 lead.
  - `CLF_INPUT_COLS`, `REG_INPUT_COLS` constantes.

Reglas:
  - Para "¿qué probabilidad tiene este lead de convertir?" usa el clasificador entrenado.
  - Para "¿cuál es la tasa de conversión por industria en los datos?" usa `df` directamente.
  - Para predicción de ACV de un lead específico: regressor con np.exp.
  - Para forecast temporal: timeseries.
  - Para descubrir cluster de un lead: clusterer.
  - Termina con `resultado = ...`.
  - Devuelve SÓLO ```python ... ```, sin explicación."""


def ask_llm_for_code(_client, pregunta: str, system_prompt: str) -> str:
    resp = _client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": pregunta},
        ],
        temperature=0.0,
    )
    return resp.choices[0].message.content


def extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    return m.group(1) if m else text.strip()


def run_code(code: str, df: pd.DataFrame, mode: str):
    ns = {"df": df, "pd": pd, "np": np}
    if mode == "operador":
        ns.update({
            "classifier": classifier,
            "regressor": regressor,
            "clusterer": clusterer,
            "timeseries": timeseries,
            "build_X": build_X,
            "CLF_INPUT_COLS": CLF_INPUT_COLS,
            "REG_INPUT_COLS": REG_INPUT_COLS,
        })
    try:
        exec(code, ns)
        return ns.get("resultado", "(no se asignó `resultado`)"), None
    except Exception as e:
        return None, str(e)


# ── Cargas ─────────────────────────────────────────────────

df = load_data()
classifier = load_classifier()
regressor = load_regressor()
clusterer = load_clusterer()
timeseries = load_timeseries()
client = get_openai_client()
cluster_to_archetype = cluster_archetype_map(df, clusterer)


# ── App ────────────────────────────────────────────────────

st.title("🧠 Cañadata — Dashboard S2")
st.caption(f"{len(df):,} leads · 4 modelos clásicos · 3 modos LLM (`{MODEL}`)")

with st.sidebar:
    st.subheader("Lead")
    lead_id = st.selectbox("lead_id", df["lead_id"].tolist())
    st.divider()
    st.subheader("Forecast")
    forecast_months = st.slider("Meses", 1, 12, 6)

lead = df[df["lead_id"] == lead_id].iloc[0].to_dict()

# Lead summary
st.subheader("Datos del lead")
visible = ["company_name", "industry", "company_size", "country", "source",
           "demo_requested", "emails_opened", "response_time_hours",
           "n_meetings", "decision_maker_contacted", "quoted_acv_eur"]
st.json({k: lead[k] for k in visible})
desc = lead.get("company_description")
if isinstance(desc, str) and desc:
    st.caption("**Descripción libre** (lo que ve el LLM zero-shot):")
    st.write(f"_{desc}_")

# Cinco paneles: clásicos + LLM zero-shot
col_clf, col_reg, col_clu, col_llm = st.columns(4)

with col_clf:
    st.subheader("🎯 Clasificador")
    X_clf = build_X(lead, CLF_INPUT_COLS, classifier["feature_names"])
    proba_clf = classifier["model"].predict_proba(X_clf)[0, 1]
    st.metric("P(convertir)", f"{proba_clf:.1%}")
    st.caption(f"realidad: `{lead['converted']}`")

with col_reg:
    st.subheader("💰 Regresor")
    X_reg = build_X(lead, REG_INPUT_COLS, regressor["feature_names"])
    acv_pred = float(np.exp(regressor["model"].predict(X_reg))[0])
    st.metric("ACV", f"{acv_pred:,.0f} €")
    st.caption(f"cotizado: `{lead['quoted_acv_eur']:,.0f} €`")

with col_clu:
    st.subheader("🔮 Cluster")
    X_clu = build_X(lead, REG_INPUT_COLS, clusterer["feature_names"])
    cluster_id = int(clusterer["model"].predict(clusterer["scaler"].transform(X_clu))[0])
    arche = cluster_to_archetype.get(cluster_id, "?")
    st.metric("Cluster", f"#{cluster_id} · {arche}")
    st.caption(f"plantado: `{lead['lead_segment_truth']}`")

with col_llm:
    st.subheader("🤖 LLM zero-shot")
    if isinstance(desc, str) and desc.strip():
        with st.spinner("LLM…"):
            score = llm_score_lead(client, lead_id, desc)
        proba_llm = score / 100.0 if score >= 0 else None
        st.metric("P(convertir)", f"{score}%" if score >= 0 else "n/d")
        st.caption("Sólo lee `company_description`.")
    else:
        st.warning("Sin descripción")
        proba_llm = None

# Resumen comparativo
st.divider()
st.markdown("**Lado a lado, este lead:**")
ca, cb, cc, cd = st.columns(4)
ca.metric("Clasificador", f"{proba_clf:.0%}")
cb.metric("LLM zero-shot", f"{proba_llm:.0%}" if proba_llm is not None else "n/d")
cc.metric("Realidad", "✓ sí" if lead["converted"] else "✗ no")
cd.metric("Arquetipo", lead["lead_segment_truth"])

# Time series (de S1)
st.divider()
st.subheader("📈 Histórico + forecast (S1)")
history = timeseries["history"]
fc = timeseries["model"].get_forecast(steps=forecast_months)
chart_df = pd.DataFrame({"histórico": history, "forecast": fc.predicted_mean})
st.line_chart(chart_df, height=280)

# Chat con modo seleccionable
st.divider()
st.subheader("💬 Pregunta en lenguaje natural")
st.caption("Elige el modo: `analista` ve `df`, `operador` ve `df` + los 4 modelos.")

col_mode, col_q = st.columns([1, 4])
with col_mode:
    modo = st.radio("Modo", ["analista", "operador"], index=1)
with col_q:
    pregunta = st.text_area("Pregunta:", height=80, placeholder="P.ej. 'predice el ACV de un lead fintech de 200 empleados con 4 reuniones'")

cta_a, cta_b = st.columns(2)
do_run = cta_a.button("Preguntar", type="primary", disabled=not pregunta.strip())
do_compare = cta_b.button("Comparar modos", disabled=not pregunta.strip())

if do_run:
    sys_prompt = SYSTEM_PROMPT_OPERADOR if modo == "operador" else SYSTEM_PROMPT_ANALISTA
    with st.spinner(f"LLM (modo {modo})…"):
        raw = ask_llm_for_code(client, pregunta, sys_prompt)
        code = extract_code(raw)
    with st.expander(f"Código generado ({modo})"):
        st.code(code, language="python")
    res, err = run_code(code, df, modo)
    if err:
        st.error(err)
    else:
        st.subheader("Resultado")
        if isinstance(res, (pd.DataFrame, pd.Series)):
            st.dataframe(res, use_container_width=True)
        else:
            st.write(res)

if do_compare:
    col_an, col_op = st.columns(2)
    for col, name, sys_p in [
        (col_an, "🧮 Analista (sólo df)", SYSTEM_PROMPT_ANALISTA),
        (col_op, "🧠 Operador (df + modelos)", SYSTEM_PROMPT_OPERADOR),
    ]:
        with col:
            st.markdown(f"### {name}")
            with st.spinner("…"):
                raw = ask_llm_for_code(client, pregunta, sys_p)
                code = extract_code(raw)
            with st.expander("Código"):
                st.code(code, language="python")
            mode_key = "analista" if "analista" in name.lower() else "operador"
            res, err = run_code(code, df, mode_key)
            if err:
                st.error(err)
            else:
                if isinstance(res, (pd.DataFrame, pd.Series)):
                    st.dataframe(res, use_container_width=True)
                else:
                    st.write(res)
