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


def _preflight_pkls() -> None:
    """Falla loud si los .pkl de S1 faltan o tienen shape rota."""
    expected = {
        "classifier.pkl": {"model", "feature_names"},
        "regressor.pkl": {"model", "feature_names"},
        "clusterer.pkl": {"model", "scaler", "feature_names"},
        "timeseries.pkl": {"model", "history"},
    }
    models_dir = ROOT / "session1" / "models"
    for fname, expected_keys in expected.items():
        path = models_dir / fname
        if not path.exists():
            st.error(
                f"Falta `{path.relative_to(ROOT)}`. Corre el notebook de pre-clase "
                f"(`pre_class/1_classical_models.ipynb`) para regenerarlo."
            )
            st.stop()
        try:
            obj = joblib.load(path)
        except Exception as e:
            st.error(f"No se pudo cargar `{fname}`: {e}. Re-genera con el notebook de pre-clase.")
            st.stop()
        if not isinstance(obj, dict) or not expected_keys.issubset(obj.keys()):
            keys_found = set(obj.keys()) if isinstance(obj, dict) else type(obj).__name__
            st.error(
                f"`{fname}` shape inesperada. Esperaba keys ⊇ {expected_keys}, "
                f"encontradas: {keys_found}."
            )
            st.stop()


_preflight_pkls()


MODEL = "gpt-4.1-mini"
# Precio gpt-4.1-mini (ene-2026): $0.40/1M input, $1.60/1M output
COST_IN = 0.40 / 1_000_000
COST_OUT = 1.60 / 1_000_000
USD_TO_EUR = 0.93


def track_cost(key: str, cost_eur: float) -> None:
    bag = st.session_state.setdefault("llm_call_costs", {})
    bag[key] = cost_eur


def render_cost_sidebar() -> None:
    bag = st.session_state.get("llm_call_costs", {})
    if not bag:
        return
    total = sum(bag.values())
    with st.sidebar:
        st.divider()
        st.metric("💰 Coste LLM", f"{total*1000:.2f} m€", f"{len(bag)} llamadas únicas")


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
def llm_score_lead(_client, lead_id: str, description: str) -> tuple[int, float]:
    resp = _client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": build_scoring_prompt(description)}],
        max_tokens=10,
        temperature=0.0,
    )
    text = resp.choices[0].message.content.strip()
    m = re.search(r"\d+", text)
    score = int(m.group(0)) if m else -1
    cost = (resp.usage.prompt_tokens * COST_IN
            + resp.usage.completion_tokens * COST_OUT) * USD_TO_EUR
    return score, cost


SYSTEM_PROMPT_ANALISTA = """Eres un analista de datos. Tienes un DataFrame de pandas llamado `df` con datos de leads B2B (Cañadata).

Columnas: lead_id (str), company_name (str), industry (str ∈ {SaaS, fintech, retail, logistics, healthcare, unknown}), company_size (int), country (str ∈ {ES, FR, DE, UK, IT, PT, unknown}), source (str ∈ {organic, paid, referral, conference, outbound}), demo_requested (bool), emails_opened (float), response_time_hours (float), n_meetings (int), decision_maker_contacted (bool), quoted_acv_eur (float), company_description (str), converted (bool), converted_within_days (float, NaN si no convirtió), lead_segment_truth (str, USA SÓLO PARA EVALUAR, no como feature).

**`lead_id` es una COLUMNA, no el índice.** Para buscar un lead por id: `df[df['lead_id'] == 'L0050'].iloc[0].to_dict()`.

Genera código Python pandas para responder la pregunta del usuario.
Termina asignando el resultado a una variable llamada `resultado`.
Devuelve SÓLO un bloque ```python ... ```, sin explicación."""


SYSTEM_PROMPT_OPERADOR = """Eres un asistente con acceso a:
  - `df`: DataFrame de leads de Cañadata. **`lead_id` es una COLUMNA, no el índice.** Para buscar un lead por id, usa: `df[df['lead_id'] == 'L0050'].iloc[0].to_dict()`.
  - Columnas de df: lead_id, company_name, industry, company_size, country, source, signup_date, demo_requested, emails_opened, response_time_hours, n_meetings, decision_maker_contacted, quoted_acv_eur, company_description, converted, converted_within_days, lead_segment_truth.

  - `classifier` (dict con 'model', 'feature_names'). Para predecir conversión sobre 1 lead:
        X = build_X(lead_dict, CLF_INPUT_COLS, classifier['feature_names'])
        proba = classifier['model'].predict_proba(X)[0, 1]
  - `regressor` (target en log space; convierte con np.exp):
        X = build_X(lead_dict, REG_INPUT_COLS, regressor['feature_names'])
        acv = float(np.exp(regressor['model'].predict(X))[0])
  - `clusterer` (dict con 'model', 'scaler', 'feature_names'). Para 1 lead:
        Xs = clusterer['scaler'].transform(build_X(lead_dict, REG_INPUT_COLS, clusterer['feature_names']))
        cluster_id = clusterer['model'].predict(Xs)[0]
  - `timeseries` (dict con 'model' (Prophet entrenado), 'history' (Series mensual)). Para forecast:
        future = timeseries['model'].make_future_dataframe(periods=N, freq='MS')
        fc = timeseries['model'].predict(future)
        forecast = fc.set_index('ds')['yhat'].iloc[-N:]
  - `build_X(lead_dict, columns, training_features)` helper.
  - `CLF_INPUT_COLS`, `REG_INPUT_COLS` constantes.

**IMPORTANTE — construcción de leads hipotéticos**:
Si el usuario describe un lead hipotético sin todas las features, RELLENA LOS HUECOS con estos defaults razonables:
    DEFAULTS = {
        'industry': 'SaaS', 'company_size': 100, 'country': 'ES', 'source': 'organic',
        'demo_requested': False, 'emails_opened': 5, 'response_time_hours': 24,
        'n_meetings': 1, 'decision_maker_contacted': False, 'quoted_acv_eur': 5000.0
    }
    lead = {**DEFAULTS, **lo_que_el_usuario_dijo}

Reglas:
  - Para "¿qué probabilidad tiene este lead?" usa el clasificador.
  - Para "¿tasa de conversión por X en df?" usa df.
  - Para predicción de ACV individual: regressor con np.exp.
  - Para forecast temporal: timeseries con Prophet.
  - Para cluster: clusterer con scaler.
  - Termina con `resultado = ...`. Devuelve SÓLO ```python ... ```."""


@st.cache_data(show_spinner=False)
def ask_llm_for_code(_client, pregunta: str, system_prompt: str) -> tuple[str, float]:
    resp = _client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": pregunta},
        ],
        temperature=0.0,
    )
    cost = (resp.usage.prompt_tokens * COST_IN
            + resp.usage.completion_tokens * COST_OUT) * USD_TO_EUR
    return resp.choices[0].message.content, cost


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

render_cost_sidebar()

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
            score, cost_zs = llm_score_lead(client, lead_id, desc)
        track_cost(f"v2-score:{lead_id}", cost_zs)
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
ts_m = timeseries["model"]   # Prophet
future = ts_m.make_future_dataframe(periods=forecast_months, freq="MS")
fc = ts_m.predict(future)
forecast = fc.set_index("ds")["yhat"].iloc[-forecast_months:]
yhat_lower = fc.set_index("ds")["yhat_lower"].iloc[-forecast_months:]
yhat_upper = fc.set_index("ds")["yhat_upper"].iloc[-forecast_months:]

import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=history.index, y=history.values, name="histórico", line=dict(color="steelblue")))
fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, name="forecast", line=dict(color="darkorange")))
fig.add_trace(go.Scatter(
    x=list(forecast.index) + list(forecast.index[::-1]),
    y=list(yhat_upper) + list(yhat_lower[::-1]),
    fill="toself", fillcolor="rgba(255,165,0,0.15)",
    line=dict(color="rgba(0,0,0,0)"),
    name="banda 80%",
))
fig.update_layout(height=280, margin=dict(l=0, r=0, t=20, b=0))
st.plotly_chart(fig, use_container_width=True)

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
        raw, cost = ask_llm_for_code(client, pregunta, sys_prompt)
        track_cost(f"v2-chat-{modo}:{pregunta}", cost)
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
                raw, cost = ask_llm_for_code(client, pregunta, sys_p)
                code = extract_code(raw)
            mode_key = "analista" if "analista" in name.lower() else "operador"
            track_cost(f"v2-cmp-{mode_key}:{pregunta}", cost)
            with st.expander("Código"):
                st.code(code, language="python")
            res, err = run_code(code, df, mode_key)
            if err:
                st.error(err)
            else:
                if isinstance(res, (pd.DataFrame, pd.Series)):
                    st.dataframe(res, use_container_width=True)
                else:
                    st.write(res)
