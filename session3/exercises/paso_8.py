# ============================================================
# paso_8.py — Quality gates: cuándo desplegar y cuándo bloquear
# ============================================================
#
# ── Reto ────────────────────────────────────────────────────
#
# Antes de dar al botón "redeploy", queremos verificar que el modelo
# nuevo no es peor que el que tenemos. Eso son **quality gates**:
# umbrales mínimos por métrica. Si alguno falla, no despliegas.
#
# Aquí construyes la UI alrededor del script `session3/retrain.py`
# que ya hemos visto. La UI:
#   1. Lee el sello del último entrenamiento (`session1/models/_retrained_at.json`).
#   2. Permite lanzar `retrain.py --dry-run` y ver los gates.
#   3. Si todos pasan, ofrece el botón de "deploy" (sin --dry-run).
#
# ── Huecos ──────────────────────────────────────────────────
#
# CUATRO huecos (Supported — más grandes):
#   - HUECO 1: leer y mostrar el JSON del último entrenamiento.
#   - HUECO 2: subprocess.run para llamar a retrain.py --dry-run.
#   - HUECO 3: parsear el JSON resultante para mostrar los gates.
#   - HUECO 4: trigger del retrain real (sin --dry-run) tras confirmación.
#
# ── Cómo ejecutar ──────────────────────────────────────────
#
#   streamlit run session3/exercises/paso_8.py
#
# Necesitas que `session3/retrain.py` esté disponible en su ubicación.
#
# ============================================================

import json
import subprocess
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
RETRAIN = ROOT / "session3" / "retrain.py"
FLAG = ROOT / "session1" / "models" / "_retrained_at.json"
NEW_BATCH = ROOT / "session3" / "new_data" / "canadata_next_batch.csv"

st.set_page_config(page_title="Cañadata — paso 8", page_icon="🚦", layout="wide")

st.title("🚦 Cañadata — paso 8: quality gates antes de desplegar")
st.caption("Antes de cambiar los modelos en producción, exigimos que pasen umbrales mínimos.")


# ── HUECO 1 ────────────────────────────────────────────────
# Carga `session1/models/_retrained_at.json` (lo escribe retrain.py
# cuando hace el swap). Devuelve el dict, o None si no existe.
#
# Pista:
#   if FLAG.exists():
#       return json.loads(FLAG.read_text())
#   return None
# ──────────────────────────────────────────────────────────
def load_last_training_flag() -> dict | None:
    return ___


flag = load_last_training_flag()

if flag is None:
    st.warning(
        "Aún no se ha hecho un retrain con `retrain.py`. "
        "Ejecuta uno (botón abajo) o usa los modelos del warm-up de pre-class."
    )
else:
    st.subheader("Estado actual de los modelos en producción")
    c1, c2 = st.columns([1, 2])
    c1.metric("Último entrenamiento", flag.get("timestamp", "?"))
    c1.metric("Filas usadas", f"{flag.get('rows_used', 0):,}")

    with c2:
        st.markdown("**Métricas del último deploy**")
        gates = flag.get("gates", {})
        rows = []
        for name, info in gates.items():
            rows.append({
                "métrica": name,
                "valor": round(info["metric"], 3),
                "umbral": info["threshold"],
                "ok": "✓" if info["passed"] else "✗",
            })
        if rows:
            import pandas as pd
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


st.divider()
st.subheader("Lanzar un nuevo entrenamiento (dry-run)")
st.caption(
    "Combina `data/canadata_leads.csv` con el batch nuevo en "
    "`session3/new_data/canadata_next_batch.csv`, reentrena los 4 modelos, "
    "y evalúa los gates. **No** toca `session1/models/` mientras esté en dry-run."
)

if not RETRAIN.exists():
    st.error(f"No existe `{RETRAIN.relative_to(ROOT)}`. Asegúrate de estar en la rama `session-3`.")
    st.stop()

new_batch_present = NEW_BATCH.exists()
batch_arg = ["--new-batch", str(NEW_BATCH)] if new_batch_present else []
if not new_batch_present:
    st.info(f"No hay `{NEW_BATCH.relative_to(ROOT)}`. El retrain usará sólo el histórico.")

if "last_dry_run" not in st.session_state:
    st.session_state.last_dry_run = None

if st.button("Correr retrain.py --dry-run", type="primary"):
    with st.spinner("Reentrenando (dry-run)…"):
        # ── HUECO 2 ────────────────────────────────────────
        # Lanza retrain.py con --dry-run via subprocess.run.
        # Argumentos: [sys.executable, str(RETRAIN), "--dry-run", *batch_arg]
        # Captura stdout y stderr (capture_output=True, text=True), cwd=ROOT.
        # ──────────────────────────────────────────────────
        result = ___

    if result.returncode != 0:
        st.error("Algunos gates fallaron. No se haría swap si esto fuera un deploy real.")
    else:
        st.success("Todos los gates pasaron en dry-run. Listo para desplegar.")

    with st.expander("📋 Output completo"):
        st.code(result.stdout, language="text")
        if result.stderr:
            st.code(result.stderr, language="text")

    # ── HUECO 3 ────────────────────────────────────────────
    # retrain.py imprime cada gate como una línea con esta forma:
    #   "  ✓ classifier_roc_auc: 0.845 (≥ 0.803 (95% del previo 0.845))"
    #   "  ✗ timeseries_mape: 370.32 (≤ 200.0 (umbral inicial · sin previo))"
    #
    # El bloque al final entre paréntesis describe el comparador (puede tener
    # paréntesis anidados con "previo X" o "umbral inicial"). NO uses un regex
    # estricto que asuma "(umbral X.XX)" al final — el formato es más rico ahora.
    #
    # Patrón tolerante:
    #   re.findall(r"([✓✗])\s+(\w+):\s+([\d.]+)\s+\((.*?)\)\s*$", text, re.MULTILINE)
    # Te devuelve [(flag, name, valor, comparador_string), ...].
    #
    # Construye una lista de dicts {"métrica","valor","comparador","ok"}.
    # ──────────────────────────────────────────────────────
    import re
    rows = ___

    if rows:
        import pandas as pd
        st.subheader("Quality gates (dry-run)")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.session_state.last_dry_run = {"rows": rows, "passed": all(r["ok"] == "✓" for r in rows)}


st.divider()
st.subheader("Deploy real")

if st.session_state.last_dry_run is None:
    st.info("Lanza primero el dry-run para verificar que los gates pasan.")
elif not st.session_state.last_dry_run["passed"]:
    st.warning("El último dry-run falló. Arregla la causa antes de desplegar.")
else:
    confirm = st.checkbox("Confirmo que quiero reemplazar los modelos en `session1/models/`")
    if confirm and st.button("⚠ Deploy ahora", type="primary"):
        with st.spinner("Reentrenando y desplegando…"):
            # ── HUECO 4 ────────────────────────────────────
            # Igual que el HUECO 2 pero SIN --dry-run.
            # Captura stdout, parsea con la misma lógica, anuncia éxito.
            # ──────────────────────────────────────────────
            result = ___

        if result.returncode == 0:
            st.success("✓ Deploy completado. Recarga el dashboard para ver los nuevos modelos.")
            # Borrar caches para que la app vuelva a cargar los .pkl
            st.cache_resource.clear()
            st.balloons()
        else:
            st.error("El deploy falló. Mira el output:")
            st.code(result.stdout, language="text")


st.divider()
st.subheader("🚀 Si te quedas con ganas")
st.markdown(
    """
- **Más gates**: añade un gate de **estabilidad** (que la distribución de predicciones nuevas no se desvíe >X% de la antigua usando KL-divergence o Wasserstein). Útil para detectar drift.
- **Per-segment gates**: en vez de AUC global, exige AUC mínimo POR `industry`. Un modelo puede tener buen AUC global pero ser malo para fintech.
- **Ramping**: en vez de swap completo, añade un slider "% de tráfico al modelo nuevo" y úsalo en el dashboard. Canary deploy básico.
- **Audit trail**: todo deploy se registra en `session1/models/_history.jsonl`. Cada línea: timestamp, gates, rows_used, hash del modelo.
- **Rollback**: añade un botón que restaure los `.pkl` de `session1/models/_prev/`.
"""
)
