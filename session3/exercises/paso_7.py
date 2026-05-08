# ============================================================
# paso_7.py — Comparison harness: clásico vs LLM en el holdout
# ============================================================
#
# ── Reto ────────────────────────────────────────────────────
#
# Hasta ahora hemos comparado clásico vs LLM en leads SUELTOS.
# Aquí lo hacemos en serio: 100 leads del holdout (que ningún
# modelo ha visto), tres métricas, y leemos juntos la tabla.
#
# El harness corre por debajo:
#   - Para cada lead del holdout, predice con el clasificador.
#   - Para cada lead, pide al LLM zero-shot un score 0-100.
#   - Calcula ROC-AUC, coste medio por predicción, latencia.
#
# ── Huecos ──────────────────────────────────────────────────
#
# TRES huecos pequeños (es Together — Luis los tipea contigo):
#   - HUECO 1: el prompt zero-shot (réplica del paso_4).
#   - HUECO 2: roc_auc_score del clasificador y del LLM.
#   - HUECO 3: la barra horizontal con st.bar_chart.
#
# ── Cómo ejecutar ──────────────────────────────────────────
#
#   streamlit run session3/exercises/paso_7.py
#
# Ojo: cada lead procesado = 1 llamada a la API. El default es
# 20 leads (~0.002 €). El slider sube hasta 100.
#
# ============================================================

import os
import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / ".env")

st.set_page_config(page_title="Cañadata — paso 7", page_icon="📊", layout="wide")
MODEL = "gpt-4.1-mini"


# ── Carga ──────────────────────────────────────────────────

@st.cache_data
def load_holdout() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "canadata_holdout.csv")
    # Limpieza mínima para que el clasificador no se queje en columns mismatch.
    # En producción usarías la misma clean_canadata del warm-up; aquí basta con
    # normalizar industry/country/source y forzar tipos.
    df["industry"] = df["industry"].astype("string").str.lower().map({
        "saas": "SaaS", "fintech": "fintech", "retail": "retail",
        "logistics": "logistics", "healthcare": "healthcare",
    }).fillna("unknown")
    df["country"] = df["country"].astype("string").str.upper().str.strip().where(
        df["country"].astype("string").str.upper().str.strip().isin(["ES","FR","DE","UK","IT","PT"]),
        "unknown",
    )
    df["source"] = df["source"].astype("string").str.lower().str.strip().map({
        "organic": "organic", "paid": "paid", "referral": "referral",
        "conference": "conference", "outbound": "outbound",
    }).fillna("unknown")
    for col in ["company_size", "emails_opened", "response_time_hours", "quoted_acv_eur"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["emails_opened"] = df["emails_opened"].fillna(df["emails_opened"].median())
    df["response_time_hours"] = df["response_time_hours"].fillna(df["response_time_hours"].median())
    df["quoted_acv_eur"] = df["quoted_acv_eur"].fillna(df["quoted_acv_eur"].median())
    df["demo_requested"] = df["demo_requested"].astype(str).str.lower().isin(["true", "1", "yes", "y", "sí"])
    df["decision_maker_contacted"] = df["decision_maker_contacted"].astype(str).str.lower().isin(["true", "1", "yes", "y", "sí"])
    df["company_description"] = df["company_description"].fillna("")
    return df


@st.cache_resource
def load_classifier():
    return joblib.load(ROOT / "session1" / "models" / "classifier.pkl")


@st.cache_resource
def get_openai_client():
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        st.error("OPENAI_API_KEY no está. Crea `.env` en la raíz.")
        st.stop()
    return OpenAI(api_key=key)


CLF_INPUT_COLS = [
    "industry", "company_size", "country", "source", "demo_requested",
    "emails_opened", "response_time_hours", "n_meetings",
    "decision_maker_contacted", "quoted_acv_eur",
]


def build_X(df_subset: pd.DataFrame, training_features: list[str]) -> pd.DataFrame:
    X = pd.get_dummies(df_subset[CLF_INPUT_COLS], columns=["industry", "country", "source"])
    for col in ["demo_requested", "decision_maker_contacted"]:
        if col in X.columns:
            X[col] = X[col].astype(int)
    return X.reindex(columns=training_features, fill_value=0)


# ── HUECO 1 ────────────────────────────────────────────────
# Réplica del prompt zero-shot que escribiste en paso_4.
# Lo necesitamos aquí porque vamos a llamarlo 1 vez por lead.
# ──────────────────────────────────────────────────────────
def build_scoring_prompt(description: str) -> str:
    return ___


@st.cache_data(show_spinner=False)
def llm_score(_client, lead_id: str, description: str) -> tuple[int, int, int]:
    """Devuelve (score, prompt_tokens, completion_tokens)."""
    desc = description if isinstance(description, str) and description.strip() else "(sin descripción)"
    resp = _client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": build_scoring_prompt(desc)}],
        max_tokens=10,
        temperature=0.0,
    )
    text = resp.choices[0].message.content.strip()
    m = re.search(r"\d+", text)
    score = int(m.group(0)) if m else 50  # fallback al medio si no parsea
    return score, resp.usage.prompt_tokens, resp.usage.completion_tokens


# ── App ────────────────────────────────────────────────────

df = load_holdout()
classifier = load_classifier()
client = get_openai_client()

st.title("📊 Cañadata — paso 7: comparación honesta sobre el holdout")
st.caption(f"{len(df)} leads en el holdout · clasificador `{classifier['kind']}` · LLM `{MODEL}`")

n = st.slider("Número de leads a evaluar", min_value=5, max_value=len(df), value=20, step=5)
sample = df.head(n)

if st.button("Correr el harness", type="primary"):
    # 1. Predicciones del clasificador (vectorizado, instantáneo)
    t0 = time.time()
    X = build_X(sample, classifier["feature_names"])
    proba_clf = classifier["model"].predict_proba(X)[:, 1]
    t_clf = time.time() - t0

    # 2. Predicciones del LLM zero-shot (1 por lead, lento)
    progress = st.progress(0.0, text="LLM zero-shot…")
    proba_llm = []
    total_in_tok, total_out_tok = 0, 0
    t0 = time.time()
    for i, row in enumerate(sample.itertuples()):
        score, in_tok, out_tok = llm_score(client, row.lead_id, row.company_description)
        proba_llm.append(score / 100.0)
        total_in_tok += in_tok
        total_out_tok += out_tok
        progress.progress((i + 1) / n)
    progress.empty()
    t_llm = time.time() - t0
    proba_llm = np.array(proba_llm)

    y_true = sample["converted"].astype(int).values

    # ── HUECO 2 ────────────────────────────────────────────
    # Calcula ROC-AUC para cada modelo. Si el holdout sale
    # con sólo una clase (todos True o todos False), AUC no
    # se puede calcular y devuelve NaN — en ese caso usa np.nan.
    # Pista:
    #   from sklearn.metrics import roc_auc_score
    #   auc_clf = roc_auc_score(y_true, proba_clf) if len(set(y_true)) > 1 else float("nan")
    # ──────────────────────────────────────────────────────
    auc_clf = ___
    auc_llm = ___

    # ── Bootstrap CI (humildad estadística) ─────────────────
    #
    # Con sólo 20-100 leads, la diferencia entre AUC=0.85 y AUC=0.78
    # podría ser ruido. Bootstrap te da el intervalo de confianza al 95%.
    # Si los CI se solapan, NO puedes decir que un modelo es mejor que el otro.
    #
    # 1000 remuestreos con reemplazo. ~0.5s extra. Te ahorra discusiones.
    def bootstrap_auc_ci(y, p, n_boot=1000, seed=0):
        rng = np.random.default_rng(seed)
        m = len(y)
        aucs = []
        for _ in range(n_boot):
            idx = rng.choice(m, size=m, replace=True)
            if len(set(y[idx])) > 1:
                aucs.append(roc_auc_score(y[idx], p[idx]))
        a = np.array(aucs)
        return np.percentile(a, 2.5), np.percentile(a, 97.5)

    if not np.isnan(auc_clf):
        clf_lo, clf_hi = bootstrap_auc_ci(y_true, proba_clf)
        llm_lo, llm_hi = bootstrap_auc_ci(y_true, proba_llm)
    else:
        clf_lo = clf_hi = llm_lo = llm_hi = float("nan")

    # Costes (tarifas a fecha de hoy: gpt-4.1-mini ≈ 0.40€ / 1M input, 1.60€ / 1M output)
    cost = (total_in_tok * 0.40 + total_out_tok * 1.60) / 1_000_000
    cost_per_lead = cost / n

    st.subheader("Resultados agregados")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "ROC-AUC clasificador",
        f"{auc_clf:.3f}" if not np.isnan(auc_clf) else "n/d",
        help=f"95% CI: [{clf_lo:.3f}, {clf_hi:.3f}]" if not np.isnan(clf_lo) else None,
    )
    c2.metric(
        "ROC-AUC LLM zero-shot",
        f"{auc_llm:.3f}" if not np.isnan(auc_llm) else "n/d",
        help=f"95% CI: [{llm_lo:.3f}, {llm_hi:.3f}]" if not np.isnan(llm_lo) else None,
    )
    c3.metric("Coste LLM total", f"{cost*100:.3f} cent")
    c4.metric("Latencia LLM media", f"{t_llm/n:.2f} s/lead")

    # Humildad estadística explícita
    if not np.isnan(clf_lo):
        ci_overlap = max(clf_lo, llm_lo) <= min(clf_hi, llm_hi)
        st.markdown(
            f"**Intervalos de confianza al 95%** (bootstrap, 1000 remuestreos):\n"
            f"- Clasificador: [{clf_lo:.3f}, {clf_hi:.3f}]\n"
            f"- LLM zero-shot: [{llm_lo:.3f}, {llm_hi:.3f}]"
        )
        if ci_overlap:
            st.warning(
                "⚠ Los CIs se SOLAPAN. Con este tamaño de muestra NO puedes "
                "afirmar que un modelo sea mejor que el otro. Necesitas más "
                "datos (sube el slider) o aceptar que la diferencia podría ser ruido."
            )
        else:
            st.success(
                "Los CIs NO se solapan: la diferencia es estadísticamente significativa "
                "a este tamaño de muestra."
            )

    st.caption(
        f"Clasificador: {t_clf*1000:.1f} ms para los {n} leads (vectorizado). "
        f"LLM: {t_llm:.1f} s totales = {t_llm/n:.2f} s/lead."
    )

    # ── HUECO 3 ────────────────────────────────────────────
    # Pinta una barra comparativa con st.bar_chart.
    # Pista:
    #   chart_df = pd.DataFrame({
    #       "clasificador": [auc_clf],
    #       "LLM zero-shot": [auc_llm],
    #   }, index=["ROC-AUC"])
    #   st.bar_chart(chart_df, height=200)
    # ──────────────────────────────────────────────────────
    ___

    # Tabla detallada (siempre útil)
    st.subheader("Detalle lead a lead")
    detail = pd.DataFrame({
        "lead_id": sample["lead_id"].values,
        "industry": sample["industry"].values,
        "P_clasificador": proba_clf.round(3),
        "P_LLM": proba_llm.round(3),
        "convirtió": y_true.astype(bool),
        "arquetipo": sample["lead_segment_truth"].values,
    })
    st.dataframe(detail, use_container_width=True, hide_index=True)


st.divider()
st.subheader("🚀 Si te quedas con ganas")
st.markdown(
    """
- **Más métricas**: añade precision, recall, F1 al umbral 0.5. ¿Qué se ve distinto?
- **Curvas ROC superpuestas**: usa `sklearn.metrics.roc_curve` y `plotly` para dibujar las dos curvas en el mismo gráfico.
- **Threshold tuning**: encuentra el threshold óptimo por F1 para cada modelo. ¿Coinciden?
- **Coste por punto de AUC**: `cost_per_lead / (auc_llm - auc_random_baseline)`. Métrica fea pero práctica para decidir.
- **Rebaja la calidad de la descripción**: trunca `company_description` a 30 caracteres y vuelve a evaluar el LLM. ¿Cuánta información perdió?
"""
)
