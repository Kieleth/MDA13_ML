# ============================================================
# dashboard_v3.py — S3 dashboard (versión completa)
# ============================================================
#
# Lo que sale al final de la S3: el dashboard de S2 + la pipeline
# de ship-it (subir batch → reentrenar → quality gates → deploy →
# auto-recarga).
#
# Tres pestañas:
#   🔍 Predicciones — los 4 modelos clásicos + 3 modos LLM
#                     (heredado de dashboard_v2.py)
#   📊 Harness     — compara clásico vs LLM zero-shot en el holdout
#   🚀 Ship it     — retrain con quality gates y deploy automático
#
# Cómo ejecutar:
#
#   streamlit run session3/reference_code/dashboard_v3.py
#
# ============================================================

import json
import os
import re
import subprocess
import sys
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

st.set_page_config(page_title="Cañadata — S3", page_icon="🚀", layout="wide")
MODEL = "gpt-4.1-mini"

RETRAIN = ROOT / "session3" / "retrain.py"
NEW_DATA_DIR = ROOT / "session3" / "new_data"
NEW_BATCH = NEW_DATA_DIR / "canadata_next_batch.csv"
FLAG = ROOT / "session1" / "models" / "_retrained_at.json"


# ── Carga ──────────────────────────────────────────────────

@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "canadata_leads_clean.csv")


@st.cache_data
def load_holdout() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "canadata_holdout.csv")


@st.cache_resource
def load_classifier():
    return joblib.load(ROOT / "session1" / "models" / "classifier.pkl")


@st.cache_resource
def load_regressor():
    return joblib.load(ROOT / "session1" / "models" / "regressor.pkl")


@st.cache_resource
def load_clusterer():
    return joblib.load(ROOT / "session1" / "models" / "clusterer.pkl")


@st.cache_resource
def load_timeseries():
    return joblib.load(ROOT / "session1" / "models" / "timeseries.pkl")


@st.cache_resource
def get_openai_client():
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        st.error("OPENAI_API_KEY no está. Crea `.env` en la raíz.")
        st.stop()
    return OpenAI(api_key=key)


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


def build_X_batch(df_subset: pd.DataFrame, training_features: list[str]) -> pd.DataFrame:
    X = pd.get_dummies(df_subset[CLF_INPUT_COLS], columns=["industry", "country", "source"])
    for col in ["demo_requested", "decision_maker_contacted"]:
        if col in X.columns:
            X[col] = X[col].astype(int)
    return X.reindex(columns=training_features, fill_value=0)


@st.cache_data(show_spinner=False)
def llm_score_lead(_client, lead_id: str, description: str) -> int:
    desc = description if isinstance(description, str) and description.strip() else "(sin descripción)"
    resp = _client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content":
            "Eres un analista comercial B2B en Cañadata. Estima la probabilidad "
            f"(0-100) de que esta empresa convierta. Sólo el número.\n\nDescripción: {desc}"}],
        max_tokens=10,
        temperature=0.0,
    )
    text = resp.choices[0].message.content.strip()
    m = re.search(r"\d+", text)
    return int(m.group(0)) if m else -1


def parse_gates(stdout: str) -> list[dict]:
    """Las líneas tienen forma:
        ✓ classifier_roc_auc: 0.845 (≥ 0.803 (95% del previo 0.845))
        ✗ timeseries_mape: 370.32 (≤ 200.0 (umbral inicial · sin previo))
    El comparador tiene paréntesis anidados; lo capturamos como string."""
    pattern = re.compile(r"([✓✗])\s+(\w+):\s+([\d.]+)\s+\((.+)\)\s*$", re.MULTILINE)
    return [
        {"métrica": name, "valor": round(float(val), 3), "comparador": comparator,
         "ok": "✓" if flag == "✓" else "✗"}
        for flag, name, val, comparator in pattern.findall(stdout)
    ]


def run_retrain(dry_run: bool, batch_path: Path | None) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(RETRAIN)]
    if dry_run:
        cmd.append("--dry-run")
    if batch_path is not None:
        cmd += ["--new-batch", str(batch_path)]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)


def load_flag() -> dict | None:
    if FLAG.exists():
        return json.loads(FLAG.read_text())
    return None


# ── App ────────────────────────────────────────────────────

df = load_data()
holdout = load_holdout()
classifier = load_classifier()
regressor = load_regressor()
clusterer = load_clusterer()
timeseries = load_timeseries()
client = get_openai_client()

st.title("🚀 Cañadata — Dashboard S3 (final)")

# Stats del último deploy en cabecera
flag = load_flag()
if flag:
    cap = (
        f"último deploy: **{flag['timestamp']}** · "
        f"{flag.get('rows_used', '?'):,} filas · "
        f"AUC clf {flag['gates']['classifier_roc_auc']['metric']:.3f}"
    )
    st.caption(cap)
else:
    st.caption("modelos del warm-up de pre-class · sin retrain registrado")


tab_pred, tab_harness, tab_ship = st.tabs(
    ["🔍 Predicciones", "📊 Harness", "🚀 Ship it"]
)


# ──────────────────────────────────────────────────────────
# Tab 1 — Predicciones por lead (heredado de dashboard_v2)
# ──────────────────────────────────────────────────────────

with tab_pred:
    with st.sidebar:
        st.subheader("Lead")
        lead_id = st.selectbox("lead_id", df["lead_id"].tolist())

    lead = df[df["lead_id"] == lead_id].iloc[0].to_dict()

    st.subheader("Datos del lead")
    visible = ["company_name", "industry", "company_size", "country", "source",
               "demo_requested", "emails_opened", "response_time_hours",
               "n_meetings", "decision_maker_contacted", "quoted_acv_eur"]
    st.json({k: lead[k] for k in visible})
    desc = lead.get("company_description")
    if isinstance(desc, str) and desc:
        st.caption(f"_{desc}_")

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

    with col_clu:
        st.subheader("🔮 Cluster")
        X_clu = build_X(lead, REG_INPUT_COLS, clusterer["feature_names"])
        cid = int(clusterer["model"].predict(clusterer["scaler"].transform(X_clu))[0])
        st.metric("Cluster", f"#{cid}")
        st.caption(f"plantado: `{lead['lead_segment_truth']}`")

    with col_llm:
        st.subheader("🤖 LLM")
        if isinstance(desc, str) and desc.strip():
            with st.spinner(""):
                score = llm_score_lead(client, lead_id, desc)
            if score >= 0:
                st.metric("P(convertir)", f"{score}%")
                st.caption("zero-shot")
        else:
            st.warning("Sin desc")

    st.divider()
    st.subheader("📈 Histórico + forecast")
    history = timeseries["history"]
    ts_m = timeseries["model"]   # Prophet
    future = ts_m.make_future_dataframe(periods=6, freq="MS")
    fc = ts_m.predict(future)
    forecast = fc.set_index("ds")["yhat"].iloc[-6:]
    chart_df = pd.DataFrame({"histórico": history, "forecast": forecast})
    if len(history) > 0 and len(forecast) > 0:
        chart_df.loc[history.index[-1], "forecast"] = history.iloc[-1]
    st.line_chart(chart_df, height=260)
    st.caption(f"MAPE en hold-out: {timeseries.get('validation_mape', 0):.1f}%")


# ──────────────────────────────────────────────────────────
# Tab 2 — Comparison harness
# ──────────────────────────────────────────────────────────

with tab_harness:
    st.subheader("Holdout: clasificador vs LLM zero-shot")
    n = st.slider("Leads a evaluar", 5, len(holdout), 20, 5, key="harness_n")

    if st.button("Correr harness", type="primary"):
        sample = holdout.head(n).copy()
        # Normaliza para que build_X no rompa
        sample["industry"] = sample["industry"].astype("string").str.lower().map({
            "saas": "SaaS", "fintech": "fintech", "retail": "retail",
            "logistics": "logistics", "healthcare": "healthcare",
        }).fillna("unknown")
        sample["country"] = sample["country"].astype("string").str.upper().str.strip()
        sample.loc[~sample["country"].isin(["ES","FR","DE","UK","IT","PT"]), "country"] = "unknown"
        sample["source"] = sample["source"].astype("string").str.lower().map({
            "organic": "organic", "paid": "paid", "referral": "referral",
            "conference": "conference", "outbound": "outbound",
        }).fillna("unknown")
        for col in ["company_size", "emails_opened", "response_time_hours", "quoted_acv_eur"]:
            sample[col] = pd.to_numeric(sample[col], errors="coerce")
        sample = sample.fillna({"emails_opened": 0, "response_time_hours": 24,
                                "quoted_acv_eur": 8000, "company_description": ""})
        sample["demo_requested"] = sample["demo_requested"].astype(str).str.lower().isin(["true", "1", "yes", "y", "sí"])
        sample["decision_maker_contacted"] = sample["decision_maker_contacted"].astype(str).str.lower().isin(["true", "1", "yes", "y", "sí"])

        t0 = time.time()
        X = build_X_batch(sample, classifier["feature_names"])
        proba_clf = classifier["model"].predict_proba(X)[:, 1]
        t_clf = time.time() - t0

        prog = st.progress(0.0, text="LLM zero-shot…")
        proba_llm = []
        t0 = time.time()
        for i, row in enumerate(sample.itertuples()):
            score = llm_score_lead(client, row.lead_id, row.company_description)
            proba_llm.append((score if score >= 0 else 50) / 100.0)
            prog.progress((i + 1) / n)
        prog.empty()
        t_llm = time.time() - t0
        proba_llm = np.array(proba_llm)

        y_true = sample["converted"].astype(int).values
        auc_clf = roc_auc_score(y_true, proba_clf) if len(set(y_true)) > 1 else float("nan")
        auc_llm = roc_auc_score(y_true, proba_llm) if len(set(y_true)) > 1 else float("nan")

        # Bootstrap CI — humildad estadística con muestras pequeñas
        def _boot_ci(y, p, n_boot=1000, seed=0):
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
            clf_lo, clf_hi = _boot_ci(y_true, proba_clf)
            llm_lo, llm_hi = _boot_ci(y_true, proba_llm)
        else:
            clf_lo = clf_hi = llm_lo = llm_hi = float("nan")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "AUC clasificador",
            f"{auc_clf:.3f}",
            help=f"95% CI: [{clf_lo:.3f}, {clf_hi:.3f}]",
        )
        c2.metric(
            "AUC LLM zero-shot",
            f"{auc_llm:.3f}",
            help=f"95% CI: [{llm_lo:.3f}, {llm_hi:.3f}]",
        )
        c3.metric("Latencia clf", f"{t_clf*1000:.0f} ms")
        c4.metric("Latencia LLM", f"{t_llm/n:.2f} s/lead")

        if not np.isnan(clf_lo):
            ci_overlap = max(clf_lo, llm_lo) <= min(clf_hi, llm_hi)
            if ci_overlap:
                st.warning(
                    f"⚠ Los CIs se solapan ({clf_lo:.3f}-{clf_hi:.3f} vs "
                    f"{llm_lo:.3f}-{llm_hi:.3f}). Con {n} leads no puedes "
                    "afirmar que uno sea mejor. Sube el slider o acepta el ruido."
                )

        st.bar_chart(pd.DataFrame(
            {"clasificador": [auc_clf], "LLM": [auc_llm]}, index=["ROC-AUC"]
        ), height=220)

        st.dataframe(pd.DataFrame({
            "lead_id": sample["lead_id"].values,
            "industry": sample["industry"].values,
            "P_clf": proba_clf.round(3),
            "P_LLM": proba_llm.round(3),
            "convirtió": y_true.astype(bool),
            "arquetipo": sample["lead_segment_truth"].values,
        }), use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────────────
# Tab 3 — Ship it
# ──────────────────────────────────────────────────────────

with tab_ship:
    st.subheader("Estado actual de los modelos")
    flag = load_flag()
    if flag:
        cols = st.columns(len(flag.get("gates", {})) + 1)
        cols[0].metric("Último deploy", flag["timestamp"])
        for i, (name, info) in enumerate(flag.get("gates", {}).items(), start=1):
            emoji = "✓" if info["passed"] else "✗"
            # Comparador exacto del JSON (lower-is-better → "≤", higher → "≥").
            comparador = info.get("comparator") or f"≥ {info['threshold']}"
            cols[i].metric(name.replace("_", " "), f"{info['metric']:.3f}",
                           f"{emoji} {comparador}")
    else:
        st.info("Sin retrain registrado. Modelos del warm-up.")

    st.divider()
    st.subheader("Sube un batch nuevo")
    uploaded = st.file_uploader("CSV de leads nuevos", type="csv")
    if uploaded is not None:
        NEW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        NEW_BATCH.write_bytes(uploaded.getvalue())
        st.success(f"Guardado en {NEW_BATCH.relative_to(ROOT)}")

    if NEW_BATCH.exists():
        df_new = pd.read_csv(NEW_BATCH)
        with st.expander(f"Batch actual ({len(df_new)} filas)"):
            st.dataframe(df_new.head(10), use_container_width=True)
    else:
        st.caption("No hay batch nuevo. Si no subes uno, retrain reentrena sólo con histórico.")

    st.divider()
    st.subheader("Reentrenar (dry-run)")

    if "ship_dry" not in st.session_state:
        st.session_state.ship_dry = None

    batch_for_retrain = NEW_BATCH if NEW_BATCH.exists() else None

    if st.button("Reentrenar y evaluar gates", type="primary"):
        with st.spinner("Reentrenando…"):
            result = run_retrain(dry_run=True, batch_path=batch_for_retrain)
        rows = parse_gates(result.stdout)
        st.session_state.ship_dry = {
            "rows": rows,
            "passed": result.returncode == 0 and all(r["ok"] == "✓" for r in rows),
            "stdout": result.stdout,
        }

    if st.session_state.ship_dry is not None:
        if st.session_state.ship_dry["rows"]:
            st.dataframe(pd.DataFrame(st.session_state.ship_dry["rows"]),
                         use_container_width=True, hide_index=True)
        if st.session_state.ship_dry["passed"]:
            st.success("Todos los gates pasan. Listo para deploy.")
        else:
            st.error("Algún gate falló.")
        with st.expander("Output completo"):
            st.code(st.session_state.ship_dry["stdout"], language="text")

    st.divider()
    st.subheader("Deploy")
    ready = st.session_state.ship_dry is not None and st.session_state.ship_dry["passed"]
    if not ready:
        st.info("Pasa primero el dry-run con todos los gates en verde.")
    else:
        confirm = st.checkbox("Confirmo: reemplazar `session1/models/`")
        if confirm and st.button("⚠ DEPLOY", type="primary"):
            with st.spinner("Deploy real…"):
                result = run_retrain(dry_run=False, batch_path=batch_for_retrain)
            if result.returncode == 0:
                st.success("✓ Deploy hecho. Recargando modelos…")
                st.cache_resource.clear()
                st.cache_data.clear()
                st.balloons()
                time.sleep(0.5)
                st.rerun()
            else:
                st.error("Falló el deploy:")
                st.code(result.stdout, language="text")
