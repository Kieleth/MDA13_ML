# ============================================================
# dashboard_v3.py · S3 reveal · asistente comercial completo
# ============================================================
#
# Versión COMPLETA del paso_7 con las 4 huecos rellenos y UNA
# tool extra (`draft_outreach_email`) como demo de cómo añadir
# una capacidad nueva al bot.
#
# Lo lanza el profesor al final del bloque paso_7 como reveal.
#
#   streamlit run session3/reference_code/dashboard_v3.py
#
# ============================================================

import json
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="Trying to unpickle estimator.*")

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / ".env")

st.set_page_config(page_title="Cañadata · S3 reveal", page_icon="🚀", layout="wide")

MODEL = "gpt-4.1-mini"
COST_IN = 0.40 / 1_000_000
COST_OUT = 1.60 / 1_000_000
USD_TO_EUR = 0.93


def _preflight() -> None:
    """Falla loud si datos o modelos de S1 faltan."""
    csv_path = ROOT / "data" / "canadata_leads_clean.csv"
    if not csv_path.exists():
        st.error(
            f"Falta `{csv_path.relative_to(ROOT)}`. Corre el notebook de pre-clase "
            f"(`pre_class/1_classical_models.ipynb`)."
        )
        st.stop()
    for fname in ["classifier.pkl", "regressor.pkl", "clusterer.pkl"]:
        path = ROOT / "session1" / "models" / fname
        if not path.exists():
            st.error(f"Falta `{path.relative_to(ROOT)}`. Regenera con el notebook de pre-clase.")
            st.stop()


_preflight()


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
        st.metric("💰 Coste LLM", f"{total*100:.3f} céntimos", f"{len(bag)} interacciones")


# ── Carga de datos y modelos ────────────────────────────────

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
    client = OpenAI(api_key=api_key, timeout=15.0)
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
    """Reconstruye la matriz de features para un lead (one-hot + reindex a las columnas de training)."""
    d = pd.DataFrame([{k: lead.get(k) for k in columns}])
    cat_cols = [c for c in ["industry", "country", "source"] if c in columns]
    X = pd.get_dummies(d, columns=cat_cols)
    for col in ["demo_requested", "decision_maker_contacted"]:
        if col in X.columns:
            X[col] = X[col].astype(int)
    return X.reindex(columns=training_features, fill_value=0)


@st.cache_data
def cluster_archetype_map(_df, _clusterer) -> dict:
    """Mapea cluster_id → arquetipo dominante."""
    rows = []
    for _, row in _df.iterrows():
        X = build_X(row.to_dict(), REG_INPUT_COLS, _clusterer["feature_names"])
        Xs = _clusterer["scaler"].transform(X)
        rows.append({"cluster": int(_clusterer["model"].predict(Xs)[0]),
                     "archetype": row["lead_segment_truth"]})
    df_clu = pd.DataFrame(rows)
    return df_clu.groupby("cluster")["archetype"].agg(lambda s: s.mode().iloc[0]).to_dict()


df = load_data()
classifier = load_classifier()
regressor = load_regressor()
clusterer = load_clusterer()
client = get_openai_client()
with st.spinner("Mapeando clusters → arquetipos (primera carga, ~2-5 s)…"):
    cluster_to_archetype = cluster_archetype_map(df, clusterer)


# ── DEFAULTS para rellenar features que el LLM no extraiga ──

DEFAULTS = {
    "industry": "SaaS", "company_size": 100, "country": "ES", "source": "organic",
    "demo_requested": False, "emails_opened": 5, "response_time_hours": 24.0,
    "n_meetings": 1, "decision_maker_contacted": False, "quoted_acv_eur": 5000.0,
}


# ── Las 5 tools que el LLM puede llamar ─────────────────────

def extract_lead_from_text(text: str) -> dict:
    """Extrae features estructuradas de una descripción libre del lead.

    Hace una llamada interna al LLM con JSON mode para forzar formato.
    Es una tool que internamente usa LLM (patrón muy común).
    """
    schema_prompt = (
        "Extrae las features de este lead B2B en JSON con esta forma:\n"
        '{"industry": "<SaaS|fintech|retail|logistics|healthcare|unknown>",\n'
        ' "country": "<ES|FR|DE|UK|IT|PT|unknown>",\n'
        ' "company_size": <int>,\n'
        ' "source": "<organic|paid|referral|conference|outbound>",\n'
        ' "demo_requested": <bool>,\n'
        ' "decision_maker_contacted": <bool>,\n'
        ' "n_meetings": <int>,\n'
        ' "emails_opened": <int>,\n'
        ' "response_time_hours": <float>,\n'
        ' "quoted_acv_eur": <float>}.\n'
        "Si un campo no está en el texto, NO lo pongas. Sólo extrae lo explícito o muy implícito."
    )
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": schema_prompt},
            {"role": "user", "content": text},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    cost = (resp.usage.prompt_tokens * COST_IN + resp.usage.completion_tokens * COST_OUT) * USD_TO_EUR
    track_cost(f"extract:{hash(text) % 10000}", cost)
    extracted = json.loads(resp.choices[0].message.content)
    lead = {**DEFAULTS, **extracted}
    lead["_extracted_from_text"] = sorted(extracted.keys())
    lead["_filled_with_defaults"] = sorted(k for k in DEFAULTS if k not in extracted)
    lead["_confidence_hint"] = (
        "alta" if len(extracted) >= 7 else
        "media" if len(extracted) >= 4 else
        "baja: la mayoría son defaults; comunícalo al usuario"
    )
    return lead


def predict_conversion(lead: dict) -> dict:
    """P(convertir) del clasificador entrenado."""
    X = build_X(lead, CLF_INPUT_COLS, classifier["feature_names"])
    proba = float(classifier["model"].predict_proba(X)[0, 1])
    return {"probability": round(proba, 3), "interpretation": f"{int(proba*100)}% de probabilidad de firmar"}


def predict_acv(lead: dict) -> dict:
    """ACV estimado en euros del regresor."""
    X = build_X(lead, REG_INPUT_COLS, regressor["feature_names"])
    acv = float(np.exp(regressor["model"].predict(X))[0])
    return {"acv_eur": round(acv, 0), "interpretation": f"{int(acv):,} € estimados de contrato anual"}


def get_archetype(lead: dict) -> dict:
    """Arquetipo del clusterer entrenado."""
    X = build_X(lead, REG_INPUT_COLS, clusterer["feature_names"])
    Xs = clusterer["scaler"].transform(X)
    cluster_id = int(clusterer["model"].predict(Xs)[0])
    archetype = cluster_to_archetype.get(cluster_id, "unknown")
    return {"cluster_id": cluster_id, "archetype": archetype}


def draft_outreach_email(lead: dict, tone: str = "profesional") -> dict:
    """Redacta un email corto de outreach para el lead. Tool bonus de dashboard_v3."""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": (
                f"Redacta un email corto de outreach en castellano, tono {tone}. "
                "5-7 líneas máximo. Saludo, contexto (cita una señal concreta del lead), "
                "valor de Cañadata (1 frase), CTA (sugerir 30 min de llamada). Sin firma."
            )},
            {"role": "user", "content": f"Lead: {json.dumps(lead, default=str, ensure_ascii=False)}"},
        ],
        temperature=0.4,
    )
    cost = (resp.usage.prompt_tokens * COST_IN + resp.usage.completion_tokens * COST_OUT) * USD_TO_EUR
    track_cost(f"email:{hash(json.dumps(lead, default=str)) % 10000}", cost)
    return {"email": resp.choices[0].message.content}


def find_similar_leads(lead: dict, k: int = 3) -> dict:
    """Top-k leads históricos más parecidos por distancia euclídea sobre features escaladas."""
    X_new = build_X(lead, REG_INPUT_COLS, clusterer["feature_names"])
    Xs_new = clusterer["scaler"].transform(X_new)[0]
    # Build matrix for all historical leads
    rows = []
    for _, row in df.iterrows():
        X_hist = build_X(row.to_dict(), REG_INPUT_COLS, clusterer["feature_names"])
        Xs_hist = clusterer["scaler"].transform(X_hist)[0]
        dist = float(np.linalg.norm(Xs_new - Xs_hist))
        rows.append({
            "lead_id": row["lead_id"],
            "company_name": row.get("company_name", ""),
            "industry": row["industry"],
            "country": row["country"],
            "company_size": int(row["company_size"]),
            "converted": bool(row["converted"]),
            "distance": dist,
        })
    rows.sort(key=lambda r: r["distance"])
    top = rows[:k]
    summary = {
        "k": k,
        "matches": [{k_: v for k_, v in r.items() if k_ != "distance"} for r in top],
        "conversion_rate_among_similar": round(sum(r["converted"] for r in top) / max(len(top), 1), 2),
    }
    return summary


# ── Schemas de las tools (lo que el LLM "ve") ───────────────

TOOL_SCHEMAS = {
    "extract_lead_from_text": {
        "type": "function",
        "function": {
            "name": "extract_lead_from_text",
            "description": "Extrae features estructuradas (industry, country, company_size, demo_requested, etc.) de la descripción libre de un lead B2B. ÚSALA SIEMPRE COMO PRIMER PASO cuando te pasen un lead nuevo en texto libre.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string", "description": "Descripción libre del lead"}},
                "required": ["text"],
            },
        },
    },
    "predict_conversion": {
        "type": "function",
        "function": {
            "name": "predict_conversion",
            "description": "Devuelve P(convertir) usando el clasificador entrenado sobre 700 leads históricos de Cañadata. Más fiable que tu corazonada.",
            "parameters": {
                "type": "object",
                "properties": {"lead": {"type": "object", "description": "dict de features estructuradas del lead"}},
                "required": ["lead"],
            },
        },
    },
    "predict_acv": {
        "type": "function",
        "function": {
            "name": "predict_acv",
            "description": "Devuelve el ACV (Annual Contract Value) estimado en euros usando el regresor entrenado.",
            "parameters": {
                "type": "object",
                "properties": {"lead": {"type": "object"}},
                "required": ["lead"],
            },
        },
    },
    "get_archetype": {
        "type": "function",
        "function": {
            "name": "get_archetype",
            "description": "Devuelve el arquetipo (quick_mover, strategic, tire_kicker) del clusterer entrenado.",
            "parameters": {
                "type": "object",
                "properties": {"lead": {"type": "object"}},
                "required": ["lead"],
            },
        },
    },
    "find_similar_leads": {
        "type": "function",
        "function": {
            "name": "find_similar_leads",
            "description": "Devuelve los k leads históricos más similares con sus desenlaces reales (firmaron o no). Útil para anclar la recomendación en casos pasados concretos.",
            "parameters": {
                "type": "object",
                "properties": {
                    "lead": {"type": "object"},
                    "k": {"type": "integer", "default": 3},
                },
                "required": ["lead"],
            },
        },
    },
    "draft_outreach_email": {
        "type": "function",
        "function": {
            "name": "draft_outreach_email",
            "description": "Redacta un email corto de outreach personalizado para el lead. Úsala SÓLO si el usuario lo pide explícitamente ('redacta un email', 'escríbeme el primer contacto', etc.).",
            "parameters": {
                "type": "object",
                "properties": {
                    "lead": {"type": "object"},
                    "tone": {"type": "string", "description": "profesional / cercano / breve"},
                },
                "required": ["lead"],
            },
        },
    },
}


# ── TOOL_FUNCS: registro de qué tools están ACTIVAS ─────────
#
# El bot SOLO ve las tools que están aquí registradas. Empieza
# con extract_lead_from_text activa. Tu trabajo: encender las
# otras cuatro, una por hueco. Cada hueco es UNA LÍNEA.
#
# Cuando rellenes HUECO 1, refresca el navegador. Pega un lead.
# Verás que el bot ahora tiene una capacidad nueva.

TOOL_FUNCS = {
    "extract_lead_from_text": extract_lead_from_text,
    "predict_conversion": predict_conversion,
    "predict_acv": predict_acv,
    "get_archetype": get_archetype,
    "find_similar_leads": find_similar_leads,
    "draft_outreach_email": draft_outreach_email,  # tool bonus de dashboard_v3
}


# Sólo exponemos al LLM las tools que están activas en TOOL_FUNCS.
TOOLS = [TOOL_SCHEMAS[name] for name in TOOL_FUNCS if name in TOOL_SCHEMAS]


# ── SYSTEM_PROMPT del asistente ─────────────────────────────

SYSTEM_PROMPT = (
    "Eres un asistente comercial de Cañadata, una SaaS B2B. Tu trabajo es ayudar al comercial "
    "a analizar leads nuevos.\n\n"
    f"Tools activas: {', '.join(TOOL_FUNCS.keys())}.\n\n"
    "Cuando el usuario te pase un lead (descripción libre), sigue este flujo SECUENCIAL:\n"
    "1. Llama `extract_lead_from_text` con el texto completo del usuario. Espera el resultado.\n"
    "2. Llama `predict_conversion` con el dict de features que devolvió extract.\n"
    "3. Llama `predict_acv` con el mismo dict.\n"
    "4. Llama `get_archetype` con el mismo dict.\n"
    "5. Llama `find_similar_leads` con el dict + k=3.\n"
    "6. Resume en bullets con: P(convertir), ACV, arquetipo, 3 leads parecidos y sus desenlaces, "
    "   y UNA recomendación accionable.\n\n"
    "Reglas:\n"
    "- SIEMPRE pasa el dict completo de features (lead=...) a las tools de predict/archetype/similar.\n"
    "- Cuando extract devuelva un dict con `_confidence_hint='baja'` o `_filled_with_defaults` largo, "
    "  menciónaselo al usuario.\n"
    "- Si la descripción es pobre, USA igualmente los defaults que devuelve extract. NO pidas confirmación.\n"
    "- Sólo llama `draft_outreach_email` si el usuario lo pide explícitamente.\n\n"
    "Sé conciso. Bullets, no párrafos largos. Peninsular profesional, sin marketing."
)


# ── Loop de tool calling ────────────────────────────────────

def run_assistant(user_message: str, history: list[dict]) -> tuple[str, list[dict]]:
    """Ejecuta el bucle de function calling hasta que el LLM responda sin tool_calls.

    Devuelve (respuesta_final_texto, lista_de_tool_calls_hechas_para_debug).
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history, {"role": "user", "content": user_message}]
    tool_calls_log = []
    total_cost = 0.0

    for _ in range(10):
        kwargs = {"model": MODEL, "messages": messages, "temperature": 0.2}
        if TOOLS:
            kwargs["tools"] = TOOLS
            kwargs["tool_choice"] = "auto"
            kwargs["parallel_tool_calls"] = False  # secuencial: las tools dependen entre sí
        resp = client.chat.completions.create(**kwargs)
        total_cost += (resp.usage.prompt_tokens * COST_IN + resp.usage.completion_tokens * COST_OUT) * USD_TO_EUR
        msg = resp.choices[0].message

        if not msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content})
            track_cost(f"chat:{hash(user_message) % 10000}", total_cost)
            return msg.content, tool_calls_log

        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [{"id": tc.id, "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                           for tc in msg.tool_calls],
        })

        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)
            if name in TOOL_FUNCS:
                try:
                    result = TOOL_FUNCS[name](**args)
                except Exception as e:
                    result = {"error": f"{type(e).__name__}: {e}"}
            else:
                result = {"error": f"Tool '{name}' no está registrada en TOOL_FUNCS. Mira los huecos de paso_7.py."}
            tool_calls_log.append({"tool": name, "args": args, "result": result})
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, default=str, ensure_ascii=False),
            })

    track_cost(f"chat:{hash(user_message) % 10000}", total_cost)
    return "(El bot dio demasiadas vueltas sin acabar. Reformula la pregunta.)", tool_calls_log


# ── UI ──────────────────────────────────────────────────────

st.title("🚀 Cañadata · S3 reveal · 6 tools activas")
st.caption("Versión completa de paso_7. Las 4 huecos rellenos + `draft_outreach_email` añadida como demo.")

with st.sidebar:
    st.markdown("### Estado del bot")
    st.markdown(f"**Tools activas**: {len(TOOL_FUNCS)} de {len(TOOL_SCHEMAS)}")
    for name in TOOL_SCHEMAS:
        emoji = "✅" if name in TOOL_FUNCS else "🔒"
        st.markdown(f"{emoji} `{name}`")
    if len(TOOL_FUNCS) < len(TOOL_SCHEMAS):
        st.info("Rellena los huecos en `paso_7.py` para activar las demás tools.")

render_cost_sidebar()

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.tool_logs = []

# Render historial
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and i // 2 < len(st.session_state.tool_logs):
            logs = st.session_state.tool_logs[i // 2]
            if logs:
                with st.expander(f"🔧 {len(logs)} tool calls"):
                    for log in logs:
                        st.markdown(f"**{log['tool']}** ← `{json.dumps(log['args'], default=str)[:120]}`")
                        st.code(json.dumps(log["result"], default=str, ensure_ascii=False, indent=2)[:600], language="json")

# Input
if prompt := st.chat_input("Pega aquí un lead o haz una pregunta…"):
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Pensando…"):
            history = st.session_state.messages.copy()
            response, logs = run_assistant(prompt, history)
        st.markdown(response)
        if logs:
            with st.expander(f"🔧 {len(logs)} tool calls"):
                for log in logs:
                    st.markdown(f"**{log['tool']}** ← `{json.dumps(log['args'], default=str)[:120]}`")
                    st.code(json.dumps(log["result"], default=str, ensure_ascii=False, indent=2)[:600], language="json")

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.tool_logs.append(logs)


# ── 🚀 Si te quedas con ganas: añade tu propia tool ─────────

st.divider()
with st.expander("🚀 Si te quedas con ganas: añade tu propia tool"):
    st.markdown("""
Tres sitios, un patrón:

1. **Define la función Python** (arriba en este archivo):
   ```python
   def draft_outreach_email(lead: dict, tone: str = "profesional") -> dict:
       resp = client.chat.completions.create(
           model=MODEL,
           messages=[{"role": "user", "content": f"Redacta un email {tone} para {lead}"}],
       )
       return {"email": resp.choices[0].message.content}
   ```

2. **Añade el schema** a `TOOL_SCHEMAS`:
   ```python
   "draft_outreach_email": {
       "type": "function",
       "function": {
           "name": "draft_outreach_email",
           "description": "Redacta un email de outreach para un lead.",
           "parameters": {
               "type": "object",
               "properties": {
                   "lead": {"type": "object"},
                   "tone": {"type": "string"},
               },
               "required": ["lead"],
           },
       },
   },
   ```

3. **Regístrala** en `TOOL_FUNCS`:
   ```python
   "draft_outreach_email": draft_outreach_email,
   ```

Recarga el navegador y pídele al bot "redacta un email para este lead". Ideas para tu empresa:
- `lookup_in_crm(lead_id)`: enriquece con histórico real de tu CRM.
- `search_web(company_name)`: verifica lo que dice la descripción.
- `analyze_competitor(industry)`: devuelve el mercado comparable.
- `score_call_priority(lead)`: combina P(convertir) + ACV + tiempo de respuesta esperado.

Function calling escala bien hasta ~20 tools por bot. Más allá, conviene partir en sub-asistentes especializados.
""")
