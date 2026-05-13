# ============================================================
# paso_5.py — LLM como analista (text-to-code sobre los datos)
# ============================================================
#
# ── Reto ────────────────────────────────────────────────────
#
# El LLM zero-shot del paso_4 sólo veía 1-2 frases. Bien para una
# corazonada, mal para "¿cuál es la conversión por industria?".
#
# Aquí el LLM va a ESCRIBIR código pandas que ejecutamos contra el
# DataFrame real de Cañadata. La pregunta del usuario en español →
# código Python → resultado real.
#
# Es el patrón text-to-code: el LLM traduce lenguaje natural a código
# Python que tú ejecutas contra tus datos.
#
# ⚠ AVISO DE SEGURIDAD — léelo antes de empezar
#
# Este patrón usa `exec()` sobre código que escribió un LLM.
# En LOCAL, sobre un `df` que tú conoces, el peor caso es un error
# de pandas (columna inexistente, filtro mal puesto). Lo verás en
# pantalla, no rompe la app.
#
# En PRODUCCIÓN esto va con sandbox (subprocess + timeout),
# function calling tipado, o SQL gen contra una DB read-only. NUNCA
# con exec() directo. Lo usamos aquí porque es la forma más simple
# de enseñar el patrón.
#
# ── Huecos ──────────────────────────────────────────────────
#
# CUATRO huecos marcados con `___`:
#   - HUECO 1: SYSTEM_PROMPT con el schema del DataFrame
#   - HUECO 2: la llamada a la API para generar código
#   - HUECO 3: extraer el código del bloque markdown que devuelve el LLM
#   - HUECO 4: ejecutar el código contra `df` con exec() y capturar
#              el resultado en una variable `resultado`
#
# ── Cómo ejecutar ──────────────────────────────────────────
#
#   streamlit run session2/exercises/paso_5.py
#
# ============================================================

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / ".env")

st.set_page_config(page_title="Cañadata — paso 5", page_icon="🧮", layout="wide")


def _preflight() -> None:
    """Falla loud si los datos de S1 faltan."""
    csv_path = ROOT / "data" / "canadata_leads_clean.csv"
    if not csv_path.exists():
        st.error(
            f"Falta `{csv_path.relative_to(ROOT)}`. Corre el notebook de pre-clase "
            f"(`pre_class/1_classical_models.ipynb`) — lo genera al limpiar el CSV."
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
    """Acumula coste por llave única en st.session_state."""
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


df = load_data()
client = get_openai_client()


# ── HUECO 1 ────────────────────────────────────────────────
# El prompt de sistema. Tiene que decirle al LLM:
#   - que es un asistente de análisis sobre un DataFrame `df`
#   - el schema (columnas + tipos)
#   - que el código debe terminar con `resultado = ...`
#   - que devuelva SÓLO un bloque ```python ... ```
#
# Pista: Cañadata tiene estas columnas:
#   lead_id (str), company_name (str), industry (str ∈ {SaaS, fintech,
#   retail, logistics, healthcare, unknown}), company_size (int),
#   country (str ∈ {ES, FR, DE, UK, IT, PT, unknown}), signup_date
#   (str YYYY-MM-DD), source (str ∈ {organic, paid, referral, conference,
#   outbound}), demo_requested (bool), emails_opened (float),
#   response_time_hours (float), n_meetings (int),
#   decision_maker_contacted (bool), quoted_acv_eur (float),
#   company_description (str), converted (bool), converted_within_days
#   (float, NaN si no convirtió), lead_segment_truth (str ∈ {quick_mover,
#   strategic, tire_kicker} — USA SÓLO PARA EVALUAR, no como feature de
#   modelos supervisados).
#
# Plantilla mínima:
#   """Eres un analista de datos. Tienes un DataFrame de pandas
#   llamado `df` con estas columnas: ...
#   Genera código Python pandas para responder la pregunta del usuario.
#   Termina asignando el resultado a una variable llamada `resultado`.
#   Devuelve SÓLO un bloque ```python ... ```, sin explicación."""
#
# (Hueco envuelto en función para que la app arranque sin error de import.
#  La NameError sólo dispara cuando clicas "Preguntar".)
# ──────────────────────────────────────────────────────────
def _build_system_prompt() -> str:
    return _hueco(1, "el system prompt como string con el schema completo de df")


@st.cache_data(show_spinner=False)
def ask_llm_for_code(_client, pregunta: str, system_prompt: str) -> tuple[str, float]:
    """Cached por (pregunta, system_prompt). `_client` con underscore opta fuera del hash.

    Pasamos system_prompt como argumento explícito (en vez de leerlo del global)
    para que la cache lo incluya. Si editas el prompt en vivo y vuelves a preguntar
    lo mismo, queremos cache miss → llamada fresca → ver el efecto del cambio.
    """
    # ── HUECO 2 ────────────────────────────────────────────
    # Llama al LLM con system + user. Patrón:
    #   response = _client.chat.completions.create(
    #       model=MODEL,
    #       messages=[
    #           {"role": "system", "content": system_prompt},
    #           {"role": "user", "content": pregunta},
    #       ],
    #       temperature=0.0,   # código determinista
    #   )
    # ──────────────────────────────────────────────────────
    response = _hueco(2, "llamada `_client.chat.completions.create(...)` con system + user, temperature=0")

    cost = (response.usage.prompt_tokens * COST_IN
            + response.usage.completion_tokens * COST_OUT) * USD_TO_EUR
    return response.choices[0].message.content, cost


# ── HUECO 3 ────────────────────────────────────────────────
# El LLM devuelve algo así:
#   ```python
#   resultado = df.groupby('industry')['converted'].mean()
#   ```
# Saca SÓLO el código de dentro de los backticks.
# Pista: re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
#        y luego .group(1) si encuentra match.
# Si no encuentra, devuelve el texto pelado (puede que el LLM no haya
# usado backticks).
# ──────────────────────────────────────────────────────────
def extract_code(text: str) -> str:
    code = _hueco(3, "extrae el código del bloque ```python ... ``` con re.search")
    return code


# ── HUECO 4 ────────────────────────────────────────────────
# Ejecuta el código en un namespace controlado y devuelve el valor
# de `resultado`. Tres líneas:
#   ns = {"df": df, "pd": pd, "np": np}
#   exec(code, ns)
#   return ns.get("resultado", "(no se asignó `resultado`)"), None
#
# El try/except ya está montado para que un fallo no rompa la app.
# ──────────────────────────────────────────────────────────
def run_code(code: str, df: pd.DataFrame):
    try:
        # ← Borra el _hueco(...) y pon tres líneas: ns dict, exec(code, ns), return ns.get("resultado", "(no se asignó)"), None
        _hueco(4, "tres líneas: ns dict, exec(code, ns), return ns.get('resultado'), None")
    except Exception as e:
        return None, str(e)


# ── App ────────────────────────────────────────────────────

st.title("🧮 Cañadata — paso 5: el LLM como analista (text-to-code)")
st.caption(f"{len(df):,} leads · {len(df.columns)} columnas · modelo `{MODEL}`")

render_cost_sidebar()

with st.expander("📋 Schema del DataFrame `df` (lo que ve el LLM)"):
    st.dataframe(df.dtypes.rename("dtype").to_frame(), use_container_width=True)

# Ejemplos clicables
st.subheader("Hazle una pregunta a los datos")

ejemplos = [
    "¿Cuál es la tasa de conversión por industria?",
    "¿Cuál es el ACV medio por arquetipo (lead_segment_truth)?",
    "Top 10 leads de mayor tamaño que NO convirtieron.",
    "¿Cómo varía la conversión entre los que pidieron demo vs los que no?",
    "Distribución de conversiones por mes de signup.",
]

cols = st.columns(len(ejemplos))
clicked = None
for i, ej in enumerate(ejemplos):
    if cols[i].button(ej, key=f"ej_{i}", use_container_width=True):
        clicked = ej

pregunta = st.text_area("O escribe la tuya:", value=clicked or "", height=80)

if st.button("Preguntar", type="primary", disabled=not pregunta.strip()):
    with st.spinner("LLM redactando código…"):
        raw, cost = ask_llm_for_code(client, pregunta, _build_system_prompt())
        track_cost(f"paso5:{pregunta}", cost)
        code = extract_code(raw)

    with st.expander("👁 Código que generó el LLM"):
        st.code(code, language="python")

    resultado, error = run_code(code, df)

    if error:
        st.error(f"Error al ejecutar el código:\n\n{error}")
        st.caption(
            "**El LLM no sabe los nombres de columna a menos que se los digas.** "
            "Mira el código arriba. ¿La columna existe? Si no, añádela al "
            "SYSTEM_PROMPT y vuelve a preguntar."
        )
    else:
        st.subheader("Resultado")
        if isinstance(resultado, pd.DataFrame):
            st.dataframe(resultado, use_container_width=True)
        elif isinstance(resultado, pd.Series):
            st.dataframe(resultado.to_frame("valor"), use_container_width=True)
        else:
            st.write(resultado)

st.divider()
with st.expander("✅ Valores esperados (sanity check)"):
    st.markdown("""
- **paso_4**: L0001 → score LLM ~10 (rango 5-20). Coste ~0.04 m€/call.
- **paso_5**: `tasa de conversión por industria` → 5 filas, valores entre 0.30 y 0.55.
- **paso_5**: `ACV medio por arquetipo` → 3 filas (quick_mover, strategic, tire_kicker).
- **paso_6**: L0050 con operador → cluster #2 (strategic en este dataset).
""")
st.divider()
st.subheader("🚀 Si te quedas con ganas")
st.markdown(
    """
- **Memoria conversacional**: en vez de una pregunta única, mantén el historial en `st.session_state.messages` y deja que el alumno haga preguntas de seguimiento ("y filtrado por España"). Pista: pasa el historial al LLM como `messages`.
- **Auto-reparación**: si el código falla, vuelve a llamar al LLM con el error y pídele que corrija. Hasta 2 intentos. Patrón estándar de text-to-code en producción.
- **Validación del resultado**: añade reglas de sanidad (¿el porcentaje está entre 0 y 100? ¿la cuenta de filas tiene sentido?). Si no, avisa al usuario.
- **Restricciones de seguridad**: bloquea `import`, `open`, `os.system` en el código antes de exec(). En este taller no hace falta porque corremos local, en prod sí.
- **Few-shot**: añade 2-3 ejemplos de pregunta → código al system prompt. Mejora mucho la calidad sobre preguntas que requieren cruzar columnas.
"""
)
