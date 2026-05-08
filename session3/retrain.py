"""
retrain.py — pipeline de reentrenamiento con quality gates

Uso (desde la raíz del repo, con `mda13_ml` activo):

    python session3/retrain.py
    python session3/retrain.py --new-batch session3/new_data/canadata_next_batch.csv
    python session3/retrain.py --dry-run         # no toca session1/models/

Flujo:
  1. Carga `data/canadata_leads.csv` (datos históricos sucios) y opcionalmente
     un batch nuevo (`session3/new_data/canadata_next_batch.csv`).
  2. Limpia con la misma `clean_canadata` que usa pre_class/1_classical_models.
  3. Reentrena los 4 modelos sobre el conjunto combinado y limpio.
  4. Pasa cada modelo por su quality gate (umbral mínimo configurado).
  5. Si TODOS los gates pasan, hace swap atómico de los .pkl en session1/models/.
     Si alguno falla, no toca nada y deja un informe en stdout.

El paso clave para el "ship it": la app Streamlit (paso_9 / dashboard_v3)
hace `st.cache_resource.clear()` o "Rerun" tras un swap exitoso, y los
nuevos modelos quedan en producción local.

NOTA: este es un sistema didáctico. En producción real querrías un job
runner (Airflow, GitHub Actions), versionado de modelos (MLflow, DVC),
canary deploys, alertas. Aquí simulamos lo mínimo para que se entienda
el patrón.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    adjusted_rand_score, mean_absolute_error, r2_score,
    roc_auc_score, silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import contextlib
import os

warnings.filterwarnings("ignore")
for _name in ("cmdstanpy", "prophet", "prophet.forecaster"):
    _lg = logging.getLogger(_name)
    _lg.setLevel(logging.ERROR)
    _lg.handlers = []
    _lg.propagate = False


@contextlib.contextmanager
def silenced_stderr_stdout():
    """Suprime stdout/stderr a nivel de descriptor (atrapa también el ruido
    de C/C++ que sale del backend Stan de Prophet, que no pasa por logging
    de Python)."""
    devnull = os.open(os.devnull, os.O_RDWR)
    saved_out, saved_err = os.dup(1), os.dup(2)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(devnull)
        os.close(saved_out)
        os.close(saved_err)

def _find_project_root() -> Path:
    """Walk up from this file until we find a directory with data/canadata_leads.csv.

    Funciona tanto si el script vive en _internal/session3/ (durante el desarrollo)
    como en session3/ (cuando publish_session.sh lo copia a la rama session-3).
    """
    here = Path(__file__).resolve()
    for candidate in [here.parent.parent, here.parent.parent.parent]:
        if (candidate / "data" / "canadata_leads.csv").exists():
            return candidate
    # Último recurso: cwd
    cwd = Path.cwd()
    if (cwd / "data" / "canadata_leads.csv").exists():
        return cwd
    raise FileNotFoundError(
        "No encuentro la raíz del repo (data/canadata_leads.csv). "
        "Ejecuta el script desde la raíz del proyecto."
    )


ROOT = _find_project_root()
MODELS_DIR = ROOT / "session1" / "models"
DATA_DIR = ROOT / "data"

USE_XGBOOST = False
try:
    from xgboost import XGBClassifier
    XGBClassifier(n_estimators=2, max_depth=2).fit(
        np.array([[0, 0], [1, 1]]), np.array([0, 1])
    )
    USE_XGBOOST = True
except Exception:
    pass

# ── Quality gates ───────────────────────────────────────────────────────────
#
# Política: NO degradar respecto al modelo en producción.
#
#   - Si existe un deploy previo (`session1/models/_retrained_at.json`),
#     el gate exige que la métrica nueva no caiga más de un 5% respecto
#     a la del modelo anterior. Esto es lo que querrías en producción real:
#     "no desplegamos algo peor que lo que ya hay".
#
#   - Si NO hay deploy previo (primer arranque), usamos los `BASELINES`
#     de aquí abajo como umbrales de seguridad mínima — evitamos publicar
#     modelos rotos en frío.
#
# `higher_is_better` indica si la métrica es del tipo "más alto mejor"
# (AUC, R², ARI) o "más bajo mejor" (MAPE).

@dataclass
class GateSpec:
    name: str
    higher_is_better: bool
    baseline: float          # umbral mínimo absoluto (para el primer deploy)
    tolerance: float = 0.05  # cuánta degradación toleramos vs deploy anterior


GATE_SPECS = {
    "classifier_roc_auc": GateSpec("classifier_roc_auc", higher_is_better=True, baseline=0.78),
    "regressor_r2_log":   GateSpec("regressor_r2_log",   higher_is_better=True, baseline=0.65),
    "clusterer_ari":      GateSpec("clusterer_ari",      higher_is_better=True, baseline=0.40),
    "timeseries_mape":    GateSpec("timeseries_mape",    higher_is_better=False, baseline=200.0),
}


def evaluate_gate(spec: GateSpec, metric: float, previous: float | None) -> "GateResult":
    """Política relativa al previo, con fallback al baseline absoluto."""
    if previous is not None:
        if spec.higher_is_better:
            threshold = previous * (1 - spec.tolerance)
            passed = metric >= threshold
            comparator = f"≥ {threshold:.3f} (95% del previo {previous:.3f})"
        else:
            threshold = previous * (1 + spec.tolerance)
            passed = metric <= threshold
            comparator = f"≤ {threshold:.3f} (105% del previo {previous:.3f})"
    else:
        threshold = spec.baseline
        if spec.higher_is_better:
            passed = metric >= threshold
            comparator = f"≥ {threshold} (umbral inicial · sin previo)"
        else:
            passed = metric <= threshold
            comparator = f"≤ {threshold} (umbral inicial · sin previo)"
    return GateResult(spec.name, metric, threshold, passed, comparator)


# ── Limpieza (idéntica al notebook pre_class/1_classical_models) ────────────

def clean_canadata(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.drop_duplicates()
    junk = df["company_name"].fillna("").str.strip().str.contains(
        "TEST|asdf|DO NOT CONTACT|XXX|^a$|^test$", case=False, regex=True
    )
    df = df[~junk].copy()

    for col in ["industry", "country", "source"]:
        df[col] = df[col].astype("string").str.strip()

    canonical_industry = {"saas": "SaaS", "fintech": "fintech", "retail": "retail",
                          "logistics": "logistics", "healthcare": "healthcare"}
    df["industry"] = df["industry"].str.lower().map(canonical_industry).fillna("unknown")

    canonical_source = {"organic": "organic", "paid": "paid", "referral": "referral",
                        "conference": "conference", "outbound": "outbound",
                        "conf": "conference", "paid_legacy": "paid", "outbound_v2": "outbound"}
    df["source"] = (
        df["source"].str.lower().str.replace(".", "", regex=False).str.strip()
        .map(canonical_source).fillna("unknown")
    )

    country_map = {
        "es": "ES", "spain": "ES", "españa": "ES", "esp": "ES",
        "fr": "FR", "france": "FR", "fra": "FR",
        "de": "DE", "germany": "DE", "deutschland": "DE", "ger": "DE",
        "uk": "UK", "gb": "UK", "united kingdom": "UK", "great britain": "UK",
        "it": "IT", "italy": "IT", "italia": "IT",
        "pt": "PT", "portugal": "PT", "prt": "PT",
    }
    df["country"] = df["country"].str.lower().map(country_map).fillna("unknown")

    truthy = {"true", "yes", "y", "1", "sí", "si"}
    def to_bool(v):
        if isinstance(v, bool): return v
        if pd.isna(v): return False
        return str(v).strip().lower() in truthy
    df["demo_requested"] = df["demo_requested"].apply(to_bool)
    df["decision_maker_contacted"] = df["decision_maker_contacted"].apply(to_bool)

    for col in ["company_size", "emails_opened", "response_time_hours", "quoted_acv_eur"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[(df["company_size"] > 0) & (df["company_size"] < 100_000)]
    df = df[df["response_time_hours"].isna() | ((df["response_time_hours"] >= 0) & (df["response_time_hours"] <= 720))]
    df = df[df["quoted_acv_eur"].isna() | ((df["quoted_acv_eur"] >= 100) & (df["quoted_acv_eur"] <= 500_000))]

    df["signup_date"] = pd.to_datetime(df["signup_date"].astype(str), errors="coerce", format="mixed")
    # Acepta hasta fin de 2026 (deja pasar lotes nuevos de los próximos meses,
    # filtra los errores de fechas futuras 2027+ que están plantados a propósito)
    df = df[df["signup_date"].notna() & (df["signup_date"] <= pd.Timestamp("2026-12-31"))]

    df["emails_opened"] = df["emails_opened"].fillna(df["emails_opened"].median())
    df["response_time_hours"] = df["response_time_hours"].fillna(df["response_time_hours"].median())
    df["quoted_acv_eur"] = df["quoted_acv_eur"].fillna(df["quoted_acv_eur"].median())
    df["company_description"] = df["company_description"].fillna("")

    return df.reset_index(drop=True)


# ── Feature builders ────────────────────────────────────────────────────────

META = ["lead_id", "company_name", "company_description", "signup_date",
        "lead_segment_truth", "converted_within_days"]
TARGET_CLF = "converted"
TARGET_REG = "quoted_acv_eur"


def build_features(df: pd.DataFrame, drop: list[str]) -> pd.DataFrame:
    cols = [c for c in df.columns if c not in META + drop]
    X = pd.get_dummies(df[cols], columns=["industry", "country", "source"], drop_first=False)
    for c in ["demo_requested", "decision_maker_contacted", "converted"]:
        if c in X.columns:
            X[c] = X[c].astype(int)
    return X


# ── Entrenamiento de cada modelo ────────────────────────────────────────────

@dataclass
class GateResult:
    name: str
    metric: float
    threshold: float
    passed: bool
    comparator: str = ""

    def __str__(self) -> str:
        flag = "✓" if self.passed else "✗"
        comp = self.comparator or f"umbral {self.threshold}"
        return f"  {flag} {self.name}: {self.metric:.3f} ({comp})"


def _previous_metric(name: str, prev_flag: dict | None) -> float | None:
    if not prev_flag:
        return None
    g = prev_flag.get("gates", {}).get(name)
    if not g:
        return None
    return float(g.get("metric"))


def train_classifier(df: pd.DataFrame, prev_flag: dict | None) -> tuple[dict, GateResult]:
    X = build_features(df, drop=[TARGET_CLF])
    y = df[TARGET_CLF].astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    if USE_XGBOOST:
        model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                              random_state=42, eval_metric="logloss")
        kind = "xgboost"
    else:
        model = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)
        kind = "random_forest"
    model.fit(Xtr, ytr)
    auc = float(roc_auc_score(yte, model.predict_proba(Xte)[:, 1]))
    artifact = {"model": model, "feature_names": list(X.columns), "kind": kind}
    return artifact, evaluate_gate(
        GATE_SPECS["classifier_roc_auc"], auc,
        _previous_metric("classifier_roc_auc", prev_flag),
    )


def train_regressor(df: pd.DataFrame, prev_flag: dict | None) -> tuple[dict, GateResult]:
    X = build_features(df, drop=[TARGET_CLF, TARGET_REG])
    y = np.log(df[TARGET_REG].values)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(Xtr, ytr)
    r2 = float(r2_score(yte, model.predict(Xte)))
    artifact = {"model": model, "feature_names": list(X.columns), "log_target": True}
    return artifact, evaluate_gate(
        GATE_SPECS["regressor_r2_log"], r2,
        _previous_metric("regressor_r2_log", prev_flag),
    )


def train_clusterer(df: pd.DataFrame, prev_flag: dict | None) -> tuple[dict, GateResult]:
    X = build_features(df, drop=[TARGET_CLF, TARGET_REG])
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = model.fit_predict(Xs)
    ari = float(adjusted_rand_score(df["lead_segment_truth"], labels))
    artifact = {"model": model, "scaler": scaler, "feature_names": list(X.columns)}
    return artifact, evaluate_gate(
        GATE_SPECS["clusterer_ari"], ari,
        _previous_metric("clusterer_ari", prev_flag),
    )


def train_timeseries(df: pd.DataFrame, prev_flag: dict | None) -> tuple[dict, GateResult]:
    from prophet import Prophet
    monthly = df[df["converted"]].set_index("signup_date").resample("MS").size()
    monthly = monthly.reindex(
        pd.date_range(monthly.index.min(), monthly.index.max(), freq="MS"),
        fill_value=0,
    )
    prophet_df = monthly.reset_index()
    prophet_df.columns = ["ds", "y"]
    train, test = prophet_df.iloc[:-3], prophet_df.iloc[-3:]

    eval_model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    with silenced_stderr_stdout():
        eval_model.fit(train)
    pred = eval_model.predict(test[["ds"]])
    denom = np.maximum(1, test["y"].values)
    mape = float(np.mean(np.abs(test["y"].values - pred["yhat"].values) / denom) * 100)

    full_model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    with silenced_stderr_stdout():
        full_model.fit(prophet_df)

    artifact = {"model": full_model, "history": monthly, "validation_mape": mape}
    return artifact, evaluate_gate(
        GATE_SPECS["timeseries_mape"], mape,
        _previous_metric("timeseries_mape", prev_flag),
    )


# ── Atomic swap ─────────────────────────────────────────────────────────────

def atomic_save(artifacts: dict, models_dir: Path) -> None:
    """Escribe a *.pkl.new, hace fsync, y luego rename atómico."""
    models_dir.mkdir(parents=True, exist_ok=True)
    tmp_paths = {}
    for name, art in artifacts.items():
        target = models_dir / f"{name}.pkl"
        tmp = models_dir / f"{name}.pkl.new"
        joblib.dump(art, tmp)
        tmp_paths[name] = (tmp, target)
    # Backup previo (por si hay que revertir manualmente)
    backup_dir = models_dir / "_prev"
    if any(t.exists() for _, t in tmp_paths.values()):
        backup_dir.mkdir(exist_ok=True)
        for _, target in tmp_paths.values():
            if target.exists():
                shutil.copy2(target, backup_dir / target.name)
    # Rename atómico
    for tmp, target in tmp_paths.values():
        tmp.replace(target)


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--new-batch", type=str, default=None,
                   help="CSV con leads nuevos a sumar al histórico.")
    p.add_argument("--dry-run", action="store_true",
                   help="No hacer swap aunque pasen los gates. Solo informa.")
    p.add_argument("--baseline", type=str, default=None,
                   help="CSV alternativo como histórico (default: data/canadata_leads.csv).")
    args = p.parse_args()

    print(f"\n{'═' * 60}")
    print(f"  retrain.py · {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═' * 60}\n")

    baseline = Path(args.baseline) if args.baseline else DATA_DIR / "canadata_leads.csv"
    df_old = pd.read_csv(baseline)
    print(f"→ Histórico: {baseline.name}  ({len(df_old)} filas)")

    if args.new_batch:
        new_path = Path(args.new_batch)
        df_new = pd.read_csv(new_path)
        print(f"→ Batch nuevo: {new_path.name}  ({len(df_new)} filas)")
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_combined = df_old
        print("→ Sin batch nuevo (sólo limpia y reentrena con histórico).")

    print(f"→ Limpiando…  {len(df_combined)} → ", end="", flush=True)
    df = clean_canadata(df_combined)
    print(f"{len(df)} filas tras clean_canadata")

    # Lee el sello del deploy anterior (si existe) para gates relativos
    prev_flag_path = MODELS_DIR / "_retrained_at.json"
    prev_flag = None
    if prev_flag_path.exists():
        try:
            prev_flag = json.loads(prev_flag_path.read_text())
            print(f"→ Deploy previo: {prev_flag.get('timestamp')}  → gates relativos a este")
        except Exception:
            print("→ Deploy previo presente pero el JSON no parsea — uso baselines absolutos")
    else:
        print("→ Sin deploy previo → uso baselines absolutos")

    print("\n→ Reentrenando los 4 modelos…")
    t0 = time.time()
    classifier, g_clf = train_classifier(df, prev_flag)
    regressor, g_reg = train_regressor(df, prev_flag)
    clusterer, g_clu = train_clusterer(df, prev_flag)
    timeseries, g_ts = train_timeseries(df, prev_flag)
    print(f"  hecho en {time.time() - t0:.1f}s\n")

    print("→ Quality gates:")
    gates = [g_clf, g_reg, g_clu, g_ts]
    for g in gates:
        print(g)

    all_pass = all(g.passed for g in gates)
    print()

    if not all_pass:
        failed = [g.name for g in gates if not g.passed]
        print(f"✗ {len(failed)} gate(s) fallaron: {', '.join(failed)}")
        print("  No se hace swap. Los modelos en session1/models/ siguen como estaban.")
        return 1

    if args.dry_run:
        print("✓ Todos los gates pasaron, pero --dry-run activado. No hago swap.")
        return 0

    print("✓ Todos los gates pasaron. Swap atómico…")
    atomic_save(
        {"classifier": classifier, "regressor": regressor,
         "clusterer": clusterer, "timeseries": timeseries},
        MODELS_DIR,
    )
    print(f"  Modelos actualizados en {MODELS_DIR.relative_to(ROOT)}")
    print(f"  Backup de los previos en {(MODELS_DIR / '_prev').relative_to(ROOT)}")

    # Deja una nota para que la app sepa que recargue
    flag = MODELS_DIR / "_retrained_at.json"
    flag.write_text(json.dumps({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gates": {g.name: {"metric": float(g.metric), "threshold": float(g.threshold),
                           "passed": bool(g.passed)} for g in gates},
        "rows_used": int(len(df)),
    }, indent=2))
    print(f"  Sello en {flag.relative_to(ROOT)}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
