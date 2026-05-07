# ============================================================
# paso_1.py — Tu primera Streamlit + clasificador
# ============================================================
#
# ── ¿Qué construyes? ────────────────────────────────────────
#
# Una página web con Streamlit que te deja:
#   1. Elegir un lead de Cañadata por su lead_id
#   2. Ver sus features
#   3. Ver la probabilidad de conversión que predice el
#      clasificador que entrenaste en pre_class/1_classical_models.ipynb
#
# ── ¿Qué tienes que arreglar? ──────────────────────────────
#
# Hay DOS huecos marcados con `___`. Tu trabajo es rellenarlos.
# Las pistas están en los comentarios justo encima.
#
# ── Cómo ejecutar ──────────────────────────────────────────
#
# Desde la raíz del proyecto MDA13_ML:
#
#   streamlit run session1/exercises/paso_1.py
#
# Si Streamlit abre el navegador y ves un selector de leads
# en la barra lateral, vas bien. Si la app se rompe, lee el
# error en rojo y mira el `___` correspondiente.
#
# ============================================================

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

# ── Configuración de la página ─────────────────────────────
st.set_page_config(page_title="Cañadata — paso 1", page_icon="🎯", layout="wide")


# ── Carga de datos y modelo (cacheada) ─────────────────────
#
# Streamlit re-ejecuta el archivo en cada interacción.
# `@st.cache_data` y `@st.cache_resource` guardan el resultado
# de estas funciones para que no se recargue en cada click.

@st.cache_data
def load_data() -> pd.DataFrame:
    """Carga el dataset Cañadata limpio (el que produjo el warm-up)."""
    return pd.read_csv(ROOT / "data" / "canadata_leads_clean.csv")


@st.cache_resource
def load_classifier() -> dict:
    """Carga el clasificador que entrenaste en el warm-up.

    Devuelve un dict con: 'model', 'feature_names', 'kind'.
    """
    # ── HUECO 1 ────────────────────────────────────────────
    # Usa joblib.load(...) para cargar el archivo:
    #     ROOT / "session1" / "models" / "classifier.pkl"
    #
    # Ejemplo de uso:
    #     joblib.load("ruta/al/archivo.pkl")
    # ──────────────────────────────────────────────────────
    return ___


# ── Helper: construye la matriz de features para un lead ───
#
# El modelo se entrenó con un X one-hot encoded. Para predecir
# sobre UN lead, tenemos que reconstruir ese mismo formato.
# Esto NO es un hueco — léelo y entiende qué hace.

CLF_INPUT_COLS = [
    "industry", "company_size", "country", "source", "demo_requested",
    "emails_opened", "response_time_hours", "n_meetings",
    "decision_maker_contacted", "quoted_acv_eur",
]


def build_X_for_lead(lead: dict, training_features: list[str]) -> pd.DataFrame:
    """Devuelve un DataFrame de 1 fila con las mismas columnas que vio el modelo."""
    df = pd.DataFrame([{k: lead[k] for k in CLF_INPUT_COLS}])
    X = pd.get_dummies(df, columns=["industry", "country", "source"])
    for col in ["demo_requested", "decision_maker_contacted"]:
        X[col] = X[col].astype(int)
    return X.reindex(columns=training_features, fill_value=0)


# ── App ────────────────────────────────────────────────────

df = load_data()
classifier = load_classifier()
clf_model = classifier["model"]
clf_features = classifier["feature_names"]

st.title("🎯 Cañadata — paso 1: clasifica un lead")
st.caption(
    f"{len(df):,} leads cargados · clasificador: `{classifier['kind']}` "
    f"con {len(clf_features)} features"
)

# ── Sidebar: selección de lead ─────────────────────────────
with st.sidebar:
    st.subheader("Selecciona un lead")
    lead_id = st.selectbox("lead_id", df["lead_id"].tolist())

lead = df[df["lead_id"] == lead_id].iloc[0].to_dict()

# ── Layout principal: datos a la izquierda, predicción a la derecha ──
col_data, col_pred = st.columns([2, 1])

with col_data:
    st.subheader("Datos del lead")
    visible_fields = [
        "company_name", "industry", "company_size", "country", "source",
        "signup_date", "demo_requested", "emails_opened", "response_time_hours",
        "n_meetings", "decision_maker_contacted", "quoted_acv_eur",
    ]
    st.json({k: lead[k] for k in visible_fields})

    if isinstance(lead.get("company_description"), str) and lead["company_description"]:
        st.caption("**Descripción libre** (lo que verá el LLM en S2):")
        st.write(f"_{lead['company_description']}_")

with col_pred:
    st.subheader("Predicción del clasificador")

    X_lead = build_X_for_lead(lead, clf_features)

    # ── HUECO 2 ────────────────────────────────────────────
    # `clf_model.predict_proba(X_lead)` devuelve una matriz
    # (n_samples, n_classes). Para 1 lead y clasificación
    # binaria es de forma (1, 2). La columna 0 es la
    # probabilidad de NO convertir; la columna 1 es la
    # probabilidad de SÍ convertir.
    #
    # Quédate con la probabilidad de convertir: [0, 1].
    # ──────────────────────────────────────────────────────
    proba = ___

    st.metric("P(convertir)", f"{proba:.1%}")

    if proba >= 0.5:
        st.success("✓ Lead probable de convertir")
    else:
        st.warning("✗ Lead poco probable de convertir")

    st.divider()
    st.caption(
        f"**Realidad** · convirtió: `{lead['converted']}` · "
        f"arquetipo plantado: `{lead['lead_segment_truth']}`"
    )

st.divider()
st.subheader("🚀 Si te quedas con ganas")
st.markdown(
    """
- Cambia el lead seleccionado y observa cómo cambia la probabilidad. ¿Es coherente con los features?
- Filtra el dropdown a solo los `tire_kicker` (verás muchos no-convertibles). ¿Cuántos predice el modelo como convertibles? Esos son tus falsos positivos.
- Añade un `st.slider` para que puedas overrider `n_meetings` del lead seleccionado. Construye un nuevo `lead` con ese override y vuelve a predecir. ¿Cómo cambia la probabilidad?
- Pinta el top 10 de `feature_importances_` del modelo (tienes acceso a `clf_model.feature_importances_` y `clf_features`).
"""
)
