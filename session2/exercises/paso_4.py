# ============================================================
# paso_4.py — Naive LLM: zero-shot scoring de un lead
# ============================================================
#
# ── Reto ────────────────────────────────────────────────────
#
# Ayer construimos cuatro herramientas. Hoy se las damos al LLM,
# en tres niveles. Empezamos por el primer nivel: **sin herramientas**.
#
# El LLM ve sólo la descripción libre del lead (`company_description`)
# y le pedimos un número 0-100. Sin features estructuradas, sin
# DataFrame, sin modelos entrenados. **Es el techo del LLM solo.**
#
# Por qué lo hacemos: para sentir el techo del LLM solo. Hoy en
# paso_5 y paso_6 le damos herramientas y observamos cómo cambia
# su comportamiento. El jueves (S3, paso_7) MEDIMOS formalmente
# sobre el holdout cuánto sube cada nivel. Hoy es cualitativo.
#
# ── Huecos ──────────────────────────────────────────────────
#
# TRES huecos marcados con `___`:
#   - HUECO 1: el prompt de scoring (sistema + user con la descripción)
#   - HUECO 2: la llamada a la API de OpenAI (chat.completions.create)
#   - HUECO 3: parsear el número del texto de respuesta
#
# ── Cómo ejecutar ──────────────────────────────────────────
#
#   streamlit run session2/exercises/paso_4.py
#
# Necesitas .env con OPENAI_API_KEY válida.
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

st.set_page_config(page_title="Cañadata — paso 4", page_icon="🤖", layout="wide")


def _preflight() -> None:
    """Falla loud si datos o .pkl de S1 faltan o tienen shape rota."""
    csv_path = ROOT / "data" / "canadata_leads_clean.csv"
    if not csv_path.exists():
        st.error(
            f"Falta `{csv_path.relative_to(ROOT)}`. Corre el notebook de pre-clase "
            f"(`pre_class/1_classical_models.ipynb`) — lo genera al limpiar."
        )
        st.stop()
    expected = {
        "classifier.pkl": {"model", "feature_names"},
        "regressor.pkl": {"model", "feature_names"},
        "clusterer.pkl": {"model", "scaler", "feature_names"},
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


_preflight()


# ── Sentinel para huecos sin rellenar ────────────────────
# Muestra warning friendly y para la app limpio. Los componentes ya
# renderizados arriba quedan visibles; los de abajo no renderizan
# hasta que rellenes el hueco.
def _hueco(n: int, desc: str = ""):
    st.warning(f"👉 **HUECO {n} pendiente**: {desc}\n\nRellénalo en este archivo y refresca.")
    st.stop()


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


# ── Carga (igual que paso_3, todo en cache) ───────────────

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
def get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY no está. Crea `.env` en la raíz con la clave de Luis.")
        st.stop()
    client = OpenAI(api_key=api_key, timeout=10.0)
    # Ping ~free para detectar 401/red rota antes del primer click del alumno.
    try:
        client.models.list()
    except Exception as e:
        st.error(
            f"No se pudo contactar OpenAI: `{type(e).__name__}`. "
            f"Verifica red + que la API key es válida.\n\nDetalle: {e}"
        )
        st.stop()
    return client


# ── Helpers (de paso_3) ────────────────────────────────────

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


# ── LLM zero-shot: lo NUEVO ────────────────────────────────

MODEL = "gpt-4.1-mini"


# ── HUECO 1 ────────────────────────────────────────────────
# Construye el prompt de scoring. Le decimos al LLM:
#   - que es un analista comercial B2B,
#   - qué empresa estamos puntuando (la descripción),
#   - que devuelva SÓLO un número entero 0-100, sin explicación.
#
# Ejemplo (pista, pero úsalo a tu manera):
#   f"""Eres un analista comercial B2B en Cañadata, una SaaS de
#   gestión de pipeline. Lee la descripción y estima la probabilidad
#   (0-100) de que esta empresa se convierta en cliente.
#   Devuelve SÓLO el número entero, sin explicación.
#
#   Descripción: {description}"""
# ──────────────────────────────────────────────────────────
def build_scoring_prompt(description: str) -> str:
    return """Eres un analista comercial B2B en Cañadata, una SaaS de
   gestión de pipeline. Lee la descripción y estima la probabilidad
   (0-100) de que esta empresa se convierta en cliente.
   Devuelve SÓLO el número entero, sin explicación.

   Descripción: {description}

   More data:
   columns are:
   lead_id,company_name,industry,company_size,country,signup_date,source,demo_requested,emails_opened,response_time_hours,n_meetings,decision_maker_contacted,quoted_acv_eur,company_description,converted,converted_within_days,lead_segment_truth

   and data for this lead is:

   L0140,Verbena Works,fintech,24,DE,2025-11-29,Conf.,True,11,1.5,0,False,4114.95,"pequeña jugador de fintech en DE, ciclo de compra corto. 🚀💰",False,,fast_mover

   """.format(description=description)


# ── HUECO 2 ────────────────────────────────────────────────
# Llama a la API y captura el `response` completo (lo usaremos para
# extraer texto Y coste). Patrón:
#   response = _client.chat.completions.create(
#       model=MODEL,
#       messages=[{"role": "user", "content": prompt}],
#       max_tokens=10,
#       temperature=0.0,   # determinismo: mismo lead → mismo score
#   )
# Luego: text = response.choices[0].message.content.strip()
# ──────────────────────────────────────────────────────────
# Precio gpt-4.1-mini (ene-2026): $0.40/1M tokens input, $1.60/1M output
COST_PER_INPUT_TOKEN_USD = 0.40 / 1_000_000
COST_PER_OUTPUT_TOKEN_USD = 1.60 / 1_000_000
USD_TO_EUR = 0.93


@st.cache_data(show_spinner=False)
def llm_score_lead(_client, lead_id: str, description: str) -> tuple[int, float]:
    """Devuelve (score 0-100 o -1, coste en € de esta llamada).

    Cached por (lead_id, description). _client opta fuera del hash.
    """
    prompt = build_scoring_prompt(description)
    print(f"DEBUG: prompt para lead {lead_id}:\n{prompt}\n---")
    response = response = _client.chat.completions.create(
       model=MODEL,
       messages=[{"role": "user", "content": prompt}],
       max_tokens=10,
       temperature=0.0,   # determinismo: mismo lead → mismo score
   )

    text = response.choices[0].message.content.strip()
    cost_eur = (
        response.usage.prompt_tokens * COST_PER_INPUT_TOKEN_USD
        + response.usage.completion_tokens * COST_PER_OUTPUT_TOKEN_USD
    ) * USD_TO_EUR

    # ── HUECO 3 ────────────────────────────────────────────
    # `text` viene como "73" o "73%" o "Probabilidad: 73". Saca
    # el número entero. Si no encuentras nada, devuelve -1.
    # Pista: usa re.search(r"\d+", text) y .group(0).
    # ──────────────────────────────────────────────────────
    print(f"DEBUG: respuesta LLM para lead {lead_id}: `{text}` (coste {cost_eur:.6f} €)")
    m = re.search(r"\d+", text)
    score = int(m.group(0)) if m else -1
    return score, cost_eur


# ── Cargas (igual patrón que paso_5 y paso_6) ──────────────

df = load_data()
classifier = load_classifier()
regressor = load_regressor()
clusterer = load_clusterer()
client = get_openai_client()


# ── App ────────────────────────────────────────────────────

st.title("🤖 Cañadata — paso 4: clasificador clásico vs LLM zero-shot")
st.caption(f"{len(df):,} leads · clasificador `{classifier['kind']}` · LLM `{MODEL}`")

with st.sidebar:
    st.subheader("Selecciona un lead")
    lead_id = st.selectbox("lead_id", df["lead_id"].tolist())

render_cost_sidebar()

lead = df[df["lead_id"] == lead_id].iloc[0].to_dict()

# Descripción visible primero — es lo que ve el LLM
st.subheader("Descripción de la empresa (lo único que verá el LLM)")
desc = lead.get("company_description") or "(sin descripción)"
st.info(f"_{desc}_")

# Cuatro paneles en columnas: clásicos + LLM zero-shot
col_clf, col_reg, col_clu, col_llm = st.columns(4)

with col_clf:
    st.subheader("🎯 Clasificador (clásico)")
    X_clf = build_X(lead, CLF_INPUT_COLS, classifier["feature_names"])
    proba_clf = classifier["model"].predict_proba(X_clf)[0, 1]
    st.metric("P(convertir)", f"{proba_clf:.1%}")
    st.caption("Entrenado sobre features estructuradas de 700 leads.")

with col_reg:
    st.subheader("💰 Regresión (clásica)")
    X_reg = build_X(lead, REG_INPUT_COLS, regressor["feature_names"])
    acv_pred = float(np.exp(regressor["model"].predict(X_reg))[0])
    st.metric("ACV predicho", f"{acv_pred:,.0f} €")

with col_clu:
    st.subheader("🔮 Cluster")
    X_clu = build_X(lead, REG_INPUT_COLS, clusterer["feature_names"])
    cluster_id = int(clusterer["model"].predict(clusterer["scaler"].transform(X_clu))[0])
    st.metric("Cluster", f"#{cluster_id}")

with col_llm:
    st.subheader("🤖 LLM zero-shot")
    if not isinstance(desc, str) or not desc.strip() or desc == "(sin descripción)":
        st.warning("Sin descripción → no podemos preguntarle al LLM")
        proba_llm = None
        cost_llm = 0.0
    else:
        with st.spinner("Pregunto al LLM…"):
            score, cost_llm = llm_score_lead(client, lead_id, desc)
        track_cost(f"paso4:{lead_id}", cost_llm)
        if score < 0:
            st.error("No conseguí parsear un número. Revisa el HUECO 3.")
            proba_llm = None
        else:
            proba_llm = score / 100.0
            st.metric("P(convertir)", f"{score}%")
            st.caption(
                f"Sin entrenar. Lee `company_description`.\n"
                f"Coste: **{cost_llm*1000:.3f} m€** "
                f"(~{cost_llm*100:.3f} €/100 leads)"
            )

# Línea de comparación final
st.divider()
st.subheader("🪞 Lado a lado")
col_a, col_b, col_c = st.columns(3)
col_a.metric("Clasificador entrenado", f"{proba_clf:.0%}")
col_b.metric("LLM zero-shot", f"{proba_llm:.0%}" if proba_llm is not None else "n/d")
col_c.metric("Realidad", "✓ convirtió" if lead["converted"] else "✗ no convirtió")

st.caption(
    f"Arquetipo plantado: `{lead['lead_segment_truth']}`. "
    "El clasificador acierta porque ha visto features. El LLM acierta (o falla) "
    "leyendo la descripción. Mira si están de acuerdo o si discrepan."
)

st.divider()
with st.expander("✅ Valores esperados (sanity check)"):
    st.markdown("""
- **paso_4**: L0001 → score LLM ~10 (rango 5-20). Coste ~0.04 m€/call.
- **paso_5**: `tasa de conversión por industria` → 5 filas, valores entre 0.30 y 0.55.
- **paso_6**: L0050 con operador → cluster #2 (strategic en este dataset). Coste ~0.14 m€/call.
""")
st.divider()
st.subheader("🚀 Si te quedas con ganas")
st.markdown(
    """
- **Cambia el lead** y observa cuándo el LLM y el clasificador discrepan más. ¿Hay un patrón? Pista: prueba con leads cuya descripción es vaga o está en inglés.
- **Mejora el prompt**: añade el contexto del negocio (precios, segmentos, qué hace que un lead sea bueno) y vuelve a comparar. ¿Mejora la correlación con el clasificador?
- **Modo confianza**: pídele al LLM que devuelva además su nivel de confianza ("alta/media/baja") y muéstralo. Útil para discutir cuándo confiar en el zero-shot.
- **Batch eval**: evalúa el LLM sobre 50 leads del holdout (`canadata_holdout.csv`) y compara su ROC-AUC con el del clasificador. Coste aproximado: ~0.004 € por 100 leads.
- **Coste real**: `response.usage` te da los tokens. Calcula el coste por lead a precios actuales de gpt-4.1-mini.
"""
)
