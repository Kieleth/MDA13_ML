# ============================================================
# paso_9.py — Ship-it: nuevo batch → retrain → redeploy automático
# ============================================================
#
# ── Reto ────────────────────────────────────────────────────
#
# Imagina que es lunes y han llegado 150 leads nuevos. Quieres:
#   1. Subir el CSV.
#   2. Reentrenar contra histórico + batch.
#   3. Ver los gates.
#   4. Si pasan, desplegar.
#   5. Que el dashboard recargue los nuevos modelos sin reiniciar Streamlit.
#
# Esto es el final de la S3: la pipeline completa en una página.
#
# ── Huecos ──────────────────────────────────────────────────
#
# CINCO huecos (Independent — los más grandes):
#   - HUECO 1: file uploader que escribe el CSV subido a `session3/new_data/`.
#   - HUECO 2: lanzar retrain.py --dry-run con el nuevo batch.
#   - HUECO 3: si los gates pasan, lanzar retrain.py SIN dry-run.
#   - HUECO 4: limpiar el cache de modelos (st.cache_resource.clear()) y
#              recargar la página para que el dashboard use los nuevos.
#   - HUECO 5: vista de "antes vs después" con las métricas del JSON flag.
#
# Esta es la sesión INDEPENDIENTE: te dejo solo, miro de vez en
# cuando, y al final reveo cómo debería verse.
#
# ── Cómo ejecutar ──────────────────────────────────────────
#
#   streamlit run session3/exercises/paso_9.py
#
# ============================================================

import json
import re
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="Trying to unpickle estimator.*")

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
RETRAIN = ROOT / "session3" / "retrain.py"
NEW_DATA_DIR = ROOT / "session3" / "new_data"
NEW_BATCH = NEW_DATA_DIR / "canadata_next_batch.csv"
FLAG = ROOT / "session1" / "models" / "_retrained_at.json"

st.set_page_config(page_title="Cañadata — paso 9", page_icon="🚀", layout="wide")


# ── Sentinel para huecos sin rellenar ────────────────────
def _hueco(n: int, desc: str = ""):
    st.warning(f"👉 **HUECO {n} pendiente**: {desc}\n\nRellénalo en este archivo y refresca.")
    st.stop()


st.title("🚀 Cañadata — paso 9: ship it")
st.caption(
    "Sube un CSV con leads nuevos → retrain con quality gates → "
    "deploy automático si pasan. El dashboard se actualiza sin reiniciar."
)


# ── Helpers ────────────────────────────────────────────────

def load_flag() -> dict | None:
    if FLAG.exists():
        return json.loads(FLAG.read_text())
    return None


def parse_gates(stdout: str) -> list[dict]:
    """Parse the gate lines that retrain.py prints, in the form:
        ✓ classifier_roc_auc: 0.845 (≥ 0.803 (95% del previo 0.845))
        ✗ timeseries_mape: 370.32 (≤ 200.0 (umbral inicial · sin previo))

    El comparador entre paréntesis tiene paréntesis anidados, así que no
    parseamos un float al final; capturamos el comparador como string.
    """
    pattern = re.compile(r"([✓✗])\s+(\w+):\s+([\d.]+)\s+\((.+)\)\s*$", re.MULTILINE)
    return [
        {"métrica": name, "valor": float(val), "comparador": comparator,
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


# ── Estado ─────────────────────────────────────────────────

st.subheader("1. Estado actual de los modelos")
flag = load_flag()
if flag:
    cols = st.columns(len(flag.get("gates", {})) + 1)
    cols[0].metric("Último deploy", flag["timestamp"])
    for i, (name, info) in enumerate(flag.get("gates", {}).items(), start=1):
        emoji = "✓" if info["passed"] else "✗"
        # Usa el `comparator` del JSON (lower-is-better → "≤", higher-is-better → "≥").
        # Fallback para JSONs antiguos sin ese campo:
        comparador = info.get("comparator") or f"≥ {info['threshold']}"
        cols[i].metric(name, f"{info['metric']:.3f}", f"{emoji} {comparador}")
else:
    st.info("Aún no hay registro. Los modelos en producción son los del warm-up de pre-class.")


# ── Subir nuevo batch ──────────────────────────────────────

st.divider()
st.subheader("2. Sube un batch nuevo")

# ── HUECO 1 ────────────────────────────────────────────────
# Streamlit tiene `st.file_uploader("...", type="csv")`. Cuando el
# usuario sube un archivo, lo guardas como NEW_BATCH (sobrescribe el
# anterior). Pista:
#   uploaded = st.file_uploader("CSV de leads nuevos", type="csv")
#   if uploaded is not None:
#       NEW_DATA_DIR.mkdir(parents=True, exist_ok=True)
#       NEW_BATCH.write_bytes(uploaded.getvalue())
#       st.success(f"Guardado en {NEW_BATCH.relative_to(ROOT)}")
# ──────────────────────────────────────────────────────────
_hueco(1, "st.file_uploader y guardar bytes en NEW_BATCH (5-6 líneas)")

# Si ya hay un batch, dale opción de mostrarlo
if NEW_BATCH.exists():
    df_new = pd.read_csv(NEW_BATCH)
    with st.expander(f"Batch actual: {NEW_BATCH.name} ({len(df_new)} filas)"):
        st.dataframe(df_new.head(20), width='stretch')
else:
    st.info(f"No hay batch nuevo en `{NEW_BATCH.relative_to(ROOT)}` todavía. "
            "Sube uno arriba o usa el `canadata_next_batch.csv` que viene en el repo.")


# ── Dry-run + deploy ───────────────────────────────────────

st.divider()
st.subheader("3. Reentrena (dry-run) y mira los gates")

if "ship_dry" not in st.session_state:
    st.session_state.ship_dry = None

batch_for_retrain = NEW_BATCH if NEW_BATCH.exists() else None

if st.button("Reentrenar (dry-run)", type="primary"):
    with st.spinner("Reentrenando… (≤ 5s)"):
        # ── HUECO 2 ────────────────────────────────────────
        # Llama a run_retrain(dry_run=True, batch_path=batch_for_retrain).
        # Guárdalo en `result`.
        # ──────────────────────────────────────────────────
        result = _hueco(2, "run_retrain(dry_run=True, batch_path=batch_for_retrain)")
    rows = parse_gates(result.stdout)
    st.session_state.ship_dry = {
        "rows": rows,
        "passed": result.returncode == 0 and all(r["ok"] == "✓" for r in rows),
        "stdout": result.stdout,
    }

if st.session_state.ship_dry is not None:
    rows = st.session_state.ship_dry["rows"]
    if rows:
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
    if st.session_state.ship_dry["passed"]:
        st.success("Todos los gates pasan. Puedes desplegar.")
    else:
        st.error("Algún gate falló. Revisa antes de desplegar.")
    with st.expander("Output completo"):
        st.code(st.session_state.ship_dry["stdout"], language="text")


# ── Deploy ─────────────────────────────────────────────────

st.divider()
st.subheader("4. Deploy")

ready = (
    st.session_state.ship_dry is not None
    and st.session_state.ship_dry["passed"]
)

if not ready:
    st.info("Pasa primero el dry-run con todos los gates en verde.")
else:
    confirm = st.checkbox("Sé lo que hago: reemplazar los modelos en `session1/models/`")
    if confirm and st.button("⚠ DEPLOY", type="primary"):
        with st.spinner("Reentrenando y desplegando…"):
            # ── HUECO 3 ────────────────────────────────────
            # Igual que HUECO 2 pero `dry_run=False`. Esto SI hace swap.
            # ──────────────────────────────────────────────
            result = _hueco(3, "run_retrain(dry_run=False, batch_path=batch_for_retrain)")
        if result.returncode == 0:
            st.success("✓ Deploy hecho. Refrescando modelos…")

            # ── HUECO 4 ────────────────────────────────────
            # Limpia el cache de Streamlit para que las funciones que cargan
            # los .pkl los relean. Pista:
            #   st.cache_resource.clear()
            #   st.cache_data.clear()
            # Y luego recarga la página: st.rerun() (Streamlit ≥1.27).
            # ──────────────────────────────────────────────
            _hueco(4, "st.cache_resource.clear() + st.cache_data.clear() + st.rerun()")
        else:
            st.error("Falló el deploy:")
            st.code(result.stdout, language="text")


# ── Antes vs Después ───────────────────────────────────────

st.divider()
st.subheader("5. Antes vs después")

# ── HUECO 5 ────────────────────────────────────────────────
# Lee el JSON flag más reciente y compara con session_state.ship_dry
# (los gates DEL DRY-RUN, que son los que se acaban de aplicar tras deploy).
# Pinta una tabla de 2 columnas: "antes" (lo que había antes), "después"
# (lo recién deplegado).
#
# Si no hay "antes" (primer deploy), muestra sólo los actuales.
# ──────────────────────────────────────────────────────────
_hueco(5, "tabla comparativa antes/después leyendo flag y session_state.ship_dry")


st.divider()
st.subheader("🚀 Si te quedas con ganas")
st.markdown(
    """
- **Auto-trigger por archivo**: monitoriza `session3/new_data/` con `watchdog` y dispara el dry-run cuando aparece un archivo nuevo.
- **Programado**: scheduler (`cron` o el menú "schedule" de tu sistema) que corre `retrain.py` cada noche con los logs nuevos.
- **Notificaciones**: si los gates fallan, manda un mensaje a Slack vía webhook. Si pasan, otro mensaje (mucho más callado).
- **Endpoint REST**: convierte la pipeline en una API mínima con FastAPI para que otros servicios la puedan disparar.
- **Backwards-compatibility**: si cambias features (añades una nueva), pinta un aviso ANTES del deploy diciendo "el formato de entrada cambió, los clientes existentes pueden romper".
- **Audit log JSONL**: cada deploy escribe una línea en `session1/models/_history.jsonl`. Te queda historial completo para post-mortems.
"""
)
