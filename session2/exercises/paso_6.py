# ============================================================
# paso_6.py — LLM como operador del modelo entrenado
# ============================================================
#
# ── Reto ────────────────────────────────────────────────────
#
# En paso_5 el LLM escribe pandas. Bien para análisis sobre los datos
# históricos. Pero si la pregunta es "¿qué probabilidad tiene este
# lead de convertir?", lo correcto NO es que el LLM escriba un
# clasificador inventado — es que cargue el clasificador entrenado
# en S1 y llame `predict_proba`.
#
# Aquí el LLM tiene acceso a TRES cosas:
#   1. el DataFrame `df` (como antes)
#   2. los modelos clásicos cargados (`classifier`, `regressor`,
#      `clusterer`, `timeseries`, `build_X`)
#   3. una guía sobre cómo usarlos
#
# Y termina con la VISTA DE COMPARACIÓN: misma pregunta, modo analista
# vs modo operador. Cuándo da igual, cuándo cambia la respuesta.
# (Zero-shot ya estuvo en paso_4; los tres modos medidos llegan en S3
# paso_7. Aquí contrastamos solo las dos formas de DAR HERRAMIENTAS.)
#
# ── Huecos ──────────────────────────────────────────────────
#
# El SYSTEM_PROMPT_OPERADOR viene PRE-ESCRITO (es donde está la
# documentación de cada herramienta — léelo, no es la lección).
# Tu trabajo son TRES huecos donde se ve la lección del paso:
#
#   - HUECO 1: el namespace de exec() incluye los modelos cargados
#              y los helpers (esto es lo que diferencia analista
#              de operador en runtime).
#   - HUECO 2: la llamada al LLM con el system_prompt seleccionable.
#   - HUECO 3: la vista de comparación: misma pregunta en MODO_ANALISTA
#              y MODO_OPERADOR, lado a lado.
#
# ── Cómo ejecutar ──────────────────────────────────────────
#
#   streamlit run session2/exercises/paso_6.py
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

st.set_page_config(page_title="Cañadata — paso 6", page_icon="🧠", layout="wide")


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


_preflight()


# ── Sentinel para huecos sin rellenar ────────────────────
# Warning friendly + para limpio. Los componentes ya renderizados
# arriba quedan visibles; los de abajo no, hasta que rellenes.
def _hueco(n: int, desc: str = ""):
    st.warning(f"👉 **HUECO {n} pendiente**: {desc}\n\nRellénalo en este archivo y refresca.")
    st.stop()


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


CLF_INPUT_COLS = [
    "industry", "company_size", "country", "source", "demo_requested",
    "emails_opened", "response_time_hours", "n_meetings",
    "decision_maker_contacted", "quoted_acv_eur",
]
REG_INPUT_COLS = [c for c in CLF_INPUT_COLS if c != "quoted_acv_eur"]


def build_X(lead: dict, columns: list[str], training_features: list[str]) -> pd.DataFrame:
    """Construye una matriz de features de 1 fila para un lead. El LLM puede llamarla."""
    d = pd.DataFrame([{k: lead[k] for k in columns}])
    cat_cols = [c for c in ["industry", "country", "source"] if c in columns]
    X = pd.get_dummies(d, columns=cat_cols)
    for col in ["demo_requested", "decision_maker_contacted"]:
        if col in X.columns:
            X[col] = X[col].astype(int)
    return X.reindex(columns=training_features, fill_value=0)


@st.cache_data
def cluster_archetype_map(_df, _clusterer):
    """Mapea cluster_id → arquetipo dominante (quick_mover / strategic / tire_kicker).

    Útil para que el operador devuelva 'cluster 2 (strategic)' en vez de solo '2'.
    """
    rows = []
    for _, row in _df.iterrows():
        X = build_X(row.to_dict(), REG_INPUT_COLS, _clusterer["feature_names"])
        Xs = _clusterer["scaler"].transform(X)
        rows.append({"cluster": _clusterer["model"].predict(Xs)[0],
                     "archetype": row["lead_segment_truth"]})
    df_clu = pd.DataFrame(rows)
    return df_clu.groupby("cluster")["archetype"].agg(lambda s: s.mode().iloc[0]).to_dict()


df = load_data()
classifier = load_classifier()
regressor = load_regressor()
clusterer = load_clusterer()
timeseries = load_timeseries()
client = get_openai_client()
with st.spinner("Mapeando clusters → arquetipos (primera carga, ~2-5 s)…"):
    cluster_to_archetype = cluster_archetype_map(df, clusterer)


# ── SYSTEM_PROMPT_OPERADOR (PRE-ESCRITO — léelo, no es la lección) ────
# El prompt vive aquí, completo, porque escribir un buen system prompt
# es un oficio largo: documentación de cada herramienta + reglas de
# enrutado + defaults para inputs incompletos. La lección de paso_6 está
# en el NAMESPACE (HUECO 1) y en la COMPARACIÓN (HUECO 3): lo que cambia
# entre analista y operador es lo que está disponible en runtime.
# ──────────────────────────────────────────────────────────
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
  - `cluster_to_archetype`: dict {cluster_id: archetype}. Tras predecir cluster, mapea con:
        archetype = cluster_to_archetype.get(cluster_id, "?")
    Para que la respuesta sea 'cluster 2 (strategic)' en vez de solo '2'.

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


# Mismo prompt simplificado para el modo ANALISTA (sólo `df`, sin modelos)
SYSTEM_PROMPT_ANALISTA = (
    "Eres un analista de datos. Tienes un DataFrame de pandas llamado `df` "
    "con datos de leads B2B de Cañadata.\n\n"
    "Columnas: lead_id (str), company_name (str), industry (str ∈ {SaaS, fintech, "
    "retail, logistics, healthcare, unknown}), company_size (int), country (str ∈ "
    "{ES, FR, DE, UK, IT, PT, unknown}), source (str ∈ {organic, paid, referral, "
    "conference, outbound}), demo_requested (bool), emails_opened (float), "
    "response_time_hours (float), n_meetings (int), decision_maker_contacted "
    "(bool), quoted_acv_eur (float), company_description (str), converted (bool), "
    "lead_segment_truth (str — USA SÓLO PARA EVALUAR, no como feature).\n\n"
    "**`lead_id` es una COLUMNA, no el índice.** Para buscar un lead por id: "
    "`df[df['lead_id'] == 'L0050'].iloc[0].to_dict()`.\n\n"
    "Genera código Python pandas para responder la pregunta del usuario.\n"
    "Termina asignando el resultado a una variable llamada `resultado`.\n"
    "Devuelve SÓLO un bloque ```python ... ```, sin explicación."
)


@st.cache_data(show_spinner=False)
def ask_llm_for_code(_client, pregunta: str, system_prompt: str) -> tuple[str, float]:
    """Cached por (pregunta, system_prompt). `_client` opta fuera del hash."""
    # ── HUECO 2 ────────────────────────────────────────────
    # Llama al LLM con system + user. Patrón (igual que paso_5
    # pero con system_prompt parametrizable):
    #   response = _client.chat.completions.create(
    #       model=MODEL,
    #       messages=[
    #           {"role": "system", "content": system_prompt},
    #           {"role": "user", "content": pregunta},
    #       ],
    #       temperature=0.0,
    #   )
    # ──────────────────────────────────────────────────────
    response = _client.chat.completions.create(
            model=MODEL,
            messages=[
               {"role": "system", "content": system_prompt},
               {"role": "user", "content": pregunta},
           ],
           temperature=0.0,
        )
    cost = (response.usage.prompt_tokens * COST_IN
            + response.usage.completion_tokens * COST_OUT) * USD_TO_EUR
    return response.choices[0].message.content, cost


def extract_code(text: str) -> str:
    match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    return match.group(1) if match else text.strip()


def run_code(code: str, df: pd.DataFrame, mode: str):
    if mode == "operador":
        ns = {
            "df": df, "pd": pd, "np": np,
            "classifier": classifier, "regressor": regressor,
            "clusterer": clusterer, "timeseries": timeseries,
            "build_X": build_X,
            "CLF_INPUT_COLS": CLF_INPUT_COLS,
            "REG_INPUT_COLS": REG_INPUT_COLS,
            "cluster_to_archetype": cluster_to_archetype,
        }
    else:  # analista
        ns = {"df": df, "pd": pd, "np": np}

    try:
        exec(code, ns)
        return ns.get("resultado", "(no se asignó `resultado`)"), None
    except Exception as e:
        return None, str(e)


# ── App ────────────────────────────────────────────────────

st.title("🧠 Cañadata — paso 6: LLM operador del modelo entrenado")
st.caption(
    f"{len(df):,} leads · 4 modelos clásicos disponibles · modelo LLM `{MODEL}`"
)

render_cost_sidebar()

st.subheader("Hazle una pregunta")
st.caption(
    "El LLM puede usar `df` para análisis Y los 4 modelos entrenados para "
    "predicciones individuales. Comparamos modo analista (sólo df) vs modo operador "
    "(df + modelos)."
)

ejemplos = [
    "¿Cuál es la probabilidad de conversión del lead L0001 según el clasificador?",
    "¿Cuál es la tasa de conversión por industria en df?",
    "Predice el ACV de un lead 'fintech' de 200 empleados con 4 reuniones.",
    "Forecast de conversiones para los próximos 4 meses.",
    "¿En qué cluster cae el lead L0050?",
]

cols = st.columns(len(ejemplos))
clicked = None
for i, ej in enumerate(ejemplos):
    if cols[i].button(ej, key=f"ej_{i}", width='stretch'):
        clicked = ej

pregunta = st.text_area("O escribe la tuya:", value=clicked or "", height=80)


if st.button("Preguntar (modo OPERADOR)", type="primary", disabled=not pregunta.strip()):
    with st.spinner("LLM redactando código (modo operador)…"):
        raw, cost = ask_llm_for_code(client, pregunta, SYSTEM_PROMPT_OPERADOR)
        track_cost(f"paso6-op:{pregunta}", cost)
        code = extract_code(raw)
    with st.expander("👁 Código (operador)"):
        st.code(code, language="python")
    resultado, error = run_code(code, df, "operador")
    if error:
        st.error(f"Error: {error}")
    else:
        st.subheader("Resultado (operador)")
        if isinstance(resultado, (pd.DataFrame, pd.Series)):
            st.dataframe(resultado, width='stretch')
        else:
            st.write(resultado)


if st.button("Comparar modos", disabled=not pregunta.strip()):
    col_an, col_op = st.columns(2)

    with col_an:
        st.markdown("### 🧮 Modo analista (sólo df)")
        with st.spinner("LLM (analista)…"):
            raw_an, cost_an = ask_llm_for_code(client, pregunta, SYSTEM_PROMPT_ANALISTA)
            track_cost(f"p6-an:{pregunta}", cost_an)
            code_an = extract_code(raw_an)
        with st.expander("Código generado"):
            st.code(code_an, language="python")
        res_an, err_an = run_code(code_an, df, "analista")
        if err_an:
            st.error(f"Error: {err_an}")
        else:
            if isinstance(res_an, (pd.DataFrame, pd.Series)):
                st.dataframe(res_an, use_container_width=True)
            else:
                st.write(res_an)

    with col_op:
        st.markdown("### 🧠 Modo operador (df + modelos)")
        with st.spinner("LLM (operador)…"):
            raw_op, cost_op = ask_llm_for_code(client, pregunta, SYSTEM_PROMPT_OPERADOR)
            track_cost(f"p6-op:{pregunta}", cost_op)
            code_op = extract_code(raw_op)
        with st.expander("Código generado"):
            st.code(code_op, language="python")
        res_op, err_op = run_code(code_op, df, "operador")
        if err_op:
            st.error(f"Error: {err_op}")
        else:
            if isinstance(res_op, (pd.DataFrame, pd.Series)):
                st.dataframe(res_op, use_container_width=True)
            else:
                st.write(res_op)


# ── Cierre: puente a S3 (el jueves) ────────────────────────
st.divider()
st.info(
    "**El jueves en S3 (paso_7)**: medimos clf-vs-LLM-zero-shot-vs-LLM-operador "
    "sobre el holdout (100 leads que nadie ha visto), con ROC-AUC, intervalos de "
    "confianza por bootstrap, coste y latencia. Hoy es cualitativo; mañana es "
    "cuantitativo."
)


st.divider()
with st.expander("✅ Valores esperados (sanity check)"):
    st.markdown("""
- **paso_4**: L0001 → score LLM ~10 (rango 5-20). Coste ~0.04 m€/call.
- **paso_5**: `tasa de conversión por industria` → 5 filas, valores entre 0.30 y 0.55.
- **paso_6**: L0050 con operador → cluster #2 (strategic en este dataset). Coste ~0.14 m€/call.
- **paso_6**: ACV fintech, 200 emp, 4 reuniones → ~9.000-10.000 €.
""")
st.divider()
st.subheader("🚀 Si te quedas con ganas")
st.markdown(
    """
- **Tres modos en paralelo**: añade el modo NAIVE (zero-shot del paso_4) a la comparación. Para cada pregunta verás 3 respuestas + (cuando aplica) la respuesta clásica directa.
- **Sándbox real**: en vez de exec(), usa `subprocess.run([sys.executable, "-c", code])` con timeout. Más seguro frente a código malicioso/erróneo.
- **Function calling**: re-implementa esto con OpenAI function calling tipado (`predict_conversion(lead_dict)`, `forecast(months: int)`). Compara robustez vs text-to-code.
- **Latency + coste**: mide tiempo y tokens en cada modo. ¿Cuál es el más caro? ¿Vale la pena?
- **Memoria conversacional**: el modo operador con memoria es más útil para iterar ("y para fintech?").
- **Validación cruzada**: para una pregunta numérica, ejecuta el modo operador 3 veces. ¿Da el mismo número? Si no, hay no determinismo en el código generado (aunque temperature=0).
"""
)
