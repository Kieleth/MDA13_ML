# ============================================================
# paso_7.py — Comparison harness: clásico vs LLM en el holdout
# ============================================================
#
# ── Reto ────────────────────────────────────────────────────
#
# Ayer le dimos al LLM las herramientas. Hoy MEDIMOS cómo las
# usa. Sobre 100 leads del holdout (que ningún modelo ha visto):
#
#   - Cuánto cuesta cada predicción del LLM (€/pred, latencia)
#   - Si el LLM con sus herramientas acierta MÁS que las
#     herramientas usadas directamente
#   - Si las diferencias que vemos son señal o ruido (bootstrap CI)
#
# El harness corre por debajo:
#   - Para cada lead del holdout, predice con la herramienta
#     directamente (clasificador clásico, sin LLM).
#   - Para cada lead, pide al LLM un score 0-100 leyendo la
#     descripción libre (paso_4 baseline: LLM sin herramientas).
#   - Calcula ROC-AUC, coste medio por predicción, latencia, e
#     intervalos de confianza por bootstrap.
#
# La pregunta que estamos respondiendo: ¿el LLM con herramientas
# justifica su coste vs ejecutar la herramienta directa?
#
# ── Huecos ──────────────────────────────────────────────────
#
# CUATRO huecos pequeños (es Together — Luis los tipea contigo):
#   - HUECO 1: el prompt zero-shot (réplica del paso_4).
#   - HUECO 2: roc_auc_score del clasificador y del LLM.
#   - HUECO 3: la barra horizontal con st.bar_chart.
#   - HUECO 4: la fórmula de Expected Value (TP*ganancia − llamadas*coste)
#             para decidir el threshold óptimo según economía del negocio.
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
import warnings
from pathlib import Path

# Los .pkl se pickleron con sklearn 1.8; si tienes una versión menor, sklearn
# avisa pero los modelos funcionan bien. Silenciamos el ruido.
warnings.filterwarnings("ignore", message="Trying to unpickle estimator.*")

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


def _preflight() -> None:
    """Falla loud si datos o .pkl de S1 faltan."""
    holdout_path = ROOT / "data" / "canadata_holdout.csv"
    if not holdout_path.exists():
        st.error(
            f"Falta `{holdout_path.relative_to(ROOT)}`. Lo necesitamos para el harness."
        )
        st.stop()
    clf_path = ROOT / "session1" / "models" / "classifier.pkl"
    if not clf_path.exists():
        st.error(
            f"Falta `{clf_path.relative_to(ROOT)}`. Corre `pre_class/1_classical_models.ipynb`."
        )
        st.stop()


_preflight()


# ── Sentinel para huecos sin rellenar ────────────────────
# Muestra warning friendly y para la app limpio. Los componentes
# de arriba ya han renderizado; los de abajo no, hasta que rellenes.
def _hueco(n, desc: str = ""):
    st.warning(f"👉 **HUECO {n} pendiente**: {desc}\n\nRellénalo en este archivo y refresca.")
    st.stop()


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
    client = OpenAI(api_key=key, timeout=10.0)
    try:
        client.models.list()
    except Exception as e:
        st.error(
            f"No se pudo contactar OpenAI: `{type(e).__name__}`. "
            f"Verifica red + que la API key es válida.\n\nDetalle: {e}"
        )
        st.stop()
    return client


CLF_INPUT_COLS = [
    "industry", "company_size", "country", "source", "demo_requested",
    "emails_opened", "response_time_hours", "n_meetings",
    "decision_maker_contacted", "quoted_acv_eur",
]


def build_X(df_subset: pd.DataFrame, training_features: list[str]) -> pd.DataFrame:
    # get_dummies = one-hot encoding: convierte "industry=fintech" en 5 columnas
    # binarias (una por valor posible). Si en el holdout NO aparece "healthcare",
    # la columna no se crea. reindex con fill_value=0 fuerza la matriz a tener
    # las MISMAS columnas que vio el clasificador en training (en el mismo orden),
    # rellenando con ceros las que falten. Sin esto, predict_proba peta por
    # mismatch de features.
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
    return _hueco(1, "réplica del prompt zero-shot de paso_4 (rol + tarea + descripción)")


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

with st.expander("🧯 Warm-up: 5 conceptos en 10 minutos (ábrelo si vienes de S2 con dudas)", expanded=False):
    st.markdown("""
**1. AUC vs accuracy.** Accuracy = "¿acerté esta predicción?" (pasa/falla por lead). AUC = "¿el modelo rankea mejor que aleatorio?" Mecánica: coges un par random (1 lead que convirtió, 1 que no). Si tu score para el positivo es más alto que el del negativo, ese par cuenta 1. AUC = promedio sobre todos los pares posibles. **0.5 = moneda; 1.0 = ranking perfecto; entre 0.5 y 1.0 = qué de bien rankeas, no qué de acertado.**

**2. Train / test / holdout.** En S1 partiste el dataset: train (entrenas), test (evalúas durante desarrollo, ya lo viste). Hoy estrenamos **holdout** = un tercer split que NADIE ha tocado. Es el único test honesto de generalización: si tu modelo lo aprueba aquí, no es porque haya memorizado.

**3. Bootstrap CI.** Con 20-100 leads, tu AUC podría ser suerte. Bootstrap: remuestreas 1000 veces los `n` leads con reemplazo (algunos repiten, otros no aparecen), calculas AUC en cada uno, y te quedas con el rango central 95% (percentiles 2.5 a 97.5). Eso es tu CI. **Si los CIs de dos modelos se solapan, no puedes decir que uno gana — la diferencia podría ser ruido.**

**4. Vectorización.** El clasificador procesa los 50 leads en UNA operación de matrices (~24 ms total para todos). El LLM procesa un lead por llamada HTTP (~650 ms cada uno). Por lead: clf 0.5 ms vs LLM 650 ms (~1300×). Por batch: clf 24 ms vs LLM 33 s (~1300× también, porque el batch del LLM es secuencial). Cuando ves "30× más lento", es una versión amable; la realidad por lead es mucho peor.

**5. Unidades de coste.** En paso_7 lo verás en céntimos y en € por 1000 leads. En S2 viste `m€` = milésimas de euro (NO millones, lo decía la "m" minúscula). 1 m€ = 0.001 €. Para no confundir, hoy ponemos céntimos y euros directos.

**Si algo de esto no te cuadra, dilo en voz alta. No avances con el modelo mental flojo.**
""")

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

    # `converted` es la columna del holdout que dice si el lead firmó o no.
    # En S1/S2 era una columna más del df; aquí cambia de papel: es la GROUND TRUTH
    # contra la que evaluamos clf y LLM. La sacamos del holdout NO la calculamos,
    # porque para 100 leads del histórico ya sabemos qué pasó.
    y_true = sample["converted"].astype(int).values

    # ── HUECO 2a + HUECO 2b ────────────────────────────────
    # Calcula ROC-AUC para cada modelo. Son DOS asignaciones (clf y LLM),
    # pero el patrón es idéntico. Si el holdout sale con sólo una clase
    # (todos True o todos False), AUC no se puede calcular y devuelve NaN.
    #
    # Anclaje conceptual: AUC = probabilidad de que un par random
    # (positivo, negativo) tenga el positivo con score más alto. 0.5 =
    # aleatorio (moneda), 1.0 = perfecto (separación total).
    #
    # Pista:
    #   from sklearn.metrics import roc_auc_score
    #   auc_clf = roc_auc_score(y_true, proba_clf) if len(set(y_true)) > 1 else float("nan")
    # ──────────────────────────────────────────────────────
    auc_clf = _hueco("2a", "roc_auc_score(y_true, proba_clf) con fallback a NaN si len(set(y_true)) < 2")
    auc_llm = _hueco("2b", "mismo patrón con proba_llm")

    # ── Bootstrap CI (humildad estadística) ─────────────────
    #
    # Con sólo 20-100 leads, la diferencia entre AUC=0.85 y AUC=0.78
    # podría ser ruido. Bootstrap te da el intervalo de confianza al 95%.
    # Si los CI se solapan, NO puedes decir que un modelo es mejor que el otro.
    #
    # Mecánica: 1000 veces remuestreamos los `n` leads con reemplazo
    # (algunos repiten, otros no aparecen), calculamos el AUC, y al final
    # tomamos los percentiles 2.5 y 97.5 de esa distribución → CI 95%.
    #
    # ⚠ Nota técnica: usamos bootstrap INDEPENDIENTE por modelo. Lo más
    # correcto sería pareado sobre la diferencia de AUCs (ambos modelos
    # sobre el mismo remuestreo). El conservador "si se solapan, no afirmes"
    # nos sirve para clase. Si te lo cuestionan en producción, hablamos de
    # bootstrap pareado o DeLong test.
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

    # Costes (tarifas gpt-4.1-mini ene-2026: $0.40 / 1M input, $1.60 / 1M output, 0.93 USD→EUR)
    cost_usd = (total_in_tok * 0.40 + total_out_tok * 1.60) / 1_000_000
    cost_eur = cost_usd * 0.93
    cost_per_lead_meur = cost_eur * 1000 / n  # m€ = milésimas de euro

    st.subheader("Resultados agregados")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "ROC-AUC clasificador",
        f"{auc_clf:.3f}" if not np.isnan(auc_clf) else "n/d",
        help=f"95% CI: [{clf_lo:.3f}, {clf_hi:.3f}]. AUC = probabilidad de ranking correcto entre un par random (positivo, negativo). 0.5 aleatorio, 1.0 perfecto." if not np.isnan(clf_lo) else None,
    )
    c2.metric(
        "ROC-AUC LLM zero-shot",
        f"{auc_llm:.3f}" if not np.isnan(auc_llm) else "n/d",
        help=f"95% CI: [{llm_lo:.3f}, {llm_hi:.3f}]" if not np.isnan(llm_lo) else None,
    )
    cost_per_1000 = cost_eur * 1000 / n
    c3.metric(
        "Coste LLM total",
        f"{cost_eur*100:.3f} céntimos",
        help=f"= {cost_eur:.5f} € totales sobre {n} leads. Proyectado a 1000 leads: ~{cost_per_1000:.2f} €.",
    )
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
    _hueco(3, "st.bar_chart con un DataFrame de 1 fila comparando AUCs")

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
    st.dataframe(detail, width='stretch', hide_index=True)

    # ── Threshold + matriz de confusión + valor esperado ──────
    #
    # El modelo no decide, te da un número entre 0 y 1. TÚ eliges el corte.
    # A threshold 0.3 llamas a casi todos los leads pero pocos firman.
    # A threshold 0.8 sólo llamas a los más seguros, pero pierdes a los del
    # medio. El AUC mide cuánto SABE el modelo. El EV mide cuánto te PAGA.
    # No siempre coinciden.
    #
    # Puente conceptual: AUC promedia sobre TODOS los thresholds posibles
    # (toda la curva ROC). Aquí fijamos UN threshold concreto para decidir.
    # Es el paso de "saber rankear" a "tomar la decisión de a quién llamar".
    st.divider()
    st.subheader("Mueve el threshold y mira qué pasa")
    st.markdown(
        "**El AUC mide cuánto SABE el modelo. El EV (expected value) mide cuánto te PAGA.** "
        "AUC = saber rankear. EV = sumar el dinero que gano (las firmas reales) menos lo que gasto "
        "(las llamadas que hago a TODOS los predichos como positivos, firmen o no). El threshold "
        "convierte un score continuo en una decisión binaria: llamo si score ≥ corte."
    )

    threshold = st.slider("Threshold de decisión", 0.0, 1.0, 0.5, 0.05)

    pred_clf = proba_clf >= threshold
    tp_clf = int(((pred_clf == 1) & (y_true == 1)).sum())
    fp_clf = int(((pred_clf == 1) & (y_true == 0)).sum())
    tn_clf = int(((pred_clf == 0) & (y_true == 0)).sum())
    fn_clf = int(((pred_clf == 0) & (y_true == 1)).sum())

    st.markdown("**Matriz de confusión del clasificador**")
    cm_col1, cm_col2 = st.columns(2)
    cm_col1.metric("✓ True Positives", tp_clf, help="Predicho convertir y convirtió")
    cm_col1.metric("⚠ False Positives", fp_clf, help="Predicho convertir, no convirtió")
    cm_col2.metric("✓ True Negatives", tn_clf, help="Predicho no, no convirtió")
    cm_col2.metric("⚠ False Negatives", fn_clf, help="Predicho no, sí convirtió")

    # ── HUECO 4 ────────────────────────────────────────────
    # Calcula el Expected Value (€) del clasificador a este threshold:
    #   - llamar a un lead cuesta `cost_per_call` €
    #   - cada lead que firma trae `gain_per_signing` €
    #   - llamamos a TODOS los predichos como positivos: tp + fp
    #   - sólo nos pagan los TRUE positives (los que firman): tp
    #
    # Derivación:
    #   ingreso  = tp * gain_per_signing     (los que sí firman, sí pagan)
    #   gasto    = (tp + fp) * cost_per_call (a TODOS los predichos llamamos)
    #   ev       = ingreso - gasto
    #
    # Lo que NO modelamos aquí: FN (los positivos que dejamos pasar). Son
    # COSTE DE OPORTUNIDAD, no coste contable. En tu negocio real puedes
    # añadirlos restando `fn * gain_per_signing * prob_recuperacion` si
    # mides la probabilidad de re-engage.
    # ──────────────────────────────────────────────────────
    # Economía del negocio (sliders: cambia los números y observa cómo se mueve el EV)
    eco_col1, eco_col2 = st.columns(2)
    gain_per_signing = eco_col1.slider(
        "Ganancia por firma (€)", 500, 20_000, 5_000, 500,
        help="Para Cañadata, ACV medio ~9000 €. Ajusta a tu negocio.",
    )
    cost_per_call = eco_col2.slider(
        "Coste por llamada (€)", 1, 50, 5, 1,
        help="Coste cargado del SDR + tiempo. Valor ilustrativo, ajusta a tu economía.",
    )
    ev_clf = _hueco(4, "tp_clf * gain_per_signing - (tp_clf + fp_clf) * cost_per_call")

    # Mismo cálculo para el LLM (pre-rellenado para que veas el patrón)
    pred_llm = proba_llm >= threshold
    tp_llm = int(((pred_llm == 1) & (y_true == 1)).sum())
    fp_llm = int(((pred_llm == 1) & (y_true == 0)).sum())
    ev_llm = tp_llm * gain_per_signing - (tp_llm + fp_llm) * cost_per_call

    st.markdown("**Beneficio esperado a este threshold**")
    e1, e2 = st.columns(2)
    e1.metric("EV clasificador", f"{ev_clf:,} €")
    e2.metric("EV LLM zero-shot", f"{ev_llm:,} €")
    st.caption(
        f"Sobre los {n} leads del holdout con la economía de los sliders. Mueve el threshold "
        "hasta que el EV deje de subir — ese corte es el óptimo para ESTOS números. Si los "
        "costes cambian, el threshold óptimo cambia. Por eso el threshold no se decide a 0.5 "
        "por defecto, se decide con la economía del caso."
    )


st.divider()
with st.expander("✅ Valores esperados (sanity check sobre 50 leads del holdout)"):
    st.markdown("""
**Tolerancia: ±0.05 en AUCs, ±20% en costes y latencias.** Si te sales de ese rango, mira la consola por si pasó algo (rate limit, descripción vacía, etc.) y vuelve a correr. Si insiste, no es bug, es ruido del muestreo — recoge más leads.

- **AUC clasificador**: ~0.91 (CI 95%: ~0.82-0.98).
- **AUC LLM zero-shot**: ~0.69 (CI 95%: ~0.55-0.82).
- **Solapamiento de CIs**: a n=50 los intervalos casi se tocan — no afirmes señal limpia. A n=100 se separan.
- **Coste total**: ~0.15 céntimos para 50 leads (≈ 3 céntimos por 1000 leads).
- **Latencia LLM**: ~650 ms/lead (1 llamada secuencial). Clf vectorizado: 24 ms para los 50.
- **EV @ threshold 0.5** con 5000 €/firma y 5 €/llamada: clf gana al LLM por factor 5-10×.
""")
st.divider()
st.subheader("🚀 Opcional: caminos para profundizar")
st.markdown(
    """
- **Más métricas a este threshold**: añade precision, recall, F1 manualmente con TP/FP/FN. ¿Cuál se mueve más al variar el corte?
- **Curvas ROC superpuestas**: usa `sklearn.metrics.roc_curve` y `plotly` para dibujar las dos curvas en el mismo gráfico.
- **Threshold óptimo automático**: barre threshold de 0.0 a 1.0 y dibuja EV vs threshold. ¿Dónde está el máximo? ¿Coincide para clf y LLM?
- **Coste por punto de AUC**: `cost_per_lead / (auc_llm - auc_random_baseline)`. Métrica fea pero práctica para decidir.
- **Rebaja la calidad de la descripción**: trunca `company_description` a 30 caracteres y vuelve a evaluar el LLM. ¿Cuánta información perdió?
"""
)
