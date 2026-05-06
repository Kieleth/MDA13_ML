"""
Cañadata — Streamlit verification.

Run from the project root:

    streamlit run test_app.py

If every check is green, your environment is ready and Streamlit itself works.
Use this when the CLI version (`python setup_check.py`) does not give you
enough confidence — for instance, you want to confirm Streamlit renders.
"""

import importlib
import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent

st.set_page_config(page_title="Cañadata — setup", page_icon="✅", layout="centered")
st.title("Cañadata — setup check")
st.caption("All green = ready for the workshop.")

# ── 1. Python ──────────────────────────────────────────────────────────────
st.subheader("1. Python")
version = sys.version.split()[0]
if sys.version_info >= (3, 10):
    st.success(f"Python {version}")
else:
    st.error(
        f"Python {version} (need 3.10+). "
        "Create a fresh env: `conda create -n mda13_ml python=3.12 -y`"
    )

# ── 2. Dependencies ────────────────────────────────────────────────────────
st.subheader("2. Dependencies")
required = {
    "streamlit": "pip install streamlit",
    "pandas": "pip install pandas",
    "numpy": "pip install numpy",
    "sklearn": "pip install scikit-learn",
    "xgboost": "pip install xgboost",
    "statsmodels": "pip install statsmodels",
    "openai": "pip install openai",
    "dotenv": "pip install python-dotenv",
    "joblib": "pip install joblib",
    "matplotlib": "pip install matplotlib",
    "seaborn": "pip install seaborn",
}
for mod, hint in required.items():
    try:
        m = importlib.import_module(mod)
        st.success(f"`import {mod}` OK")
    except ImportError as e:
        st.error(f"`import {mod}` failed — {e}. Fix: `{hint}`")

# ── 3. Data files ──────────────────────────────────────────────────────────
st.subheader("3. Data files")
for path in [ROOT / "data" / "canadata_leads.csv", ROOT / "data" / "canadata_holdout.csv"]:
    if path.exists():
        st.success(f"`{path.relative_to(ROOT)}` ({path.stat().st_size // 1024} KB)")
    else:
        st.error(
            f"`{path.relative_to(ROOT)}` missing. "
            "El CSV debería venir en el repo. Vuelve a clonar o pídeselo a Luis por el chat."
        )

# ── 4. Environment & API key ──────────────────────────────────────────────
st.subheader("4. Environment & API key")
env_path = ROOT / ".env"
api_key = None
if not env_path.exists():
    st.error(
        "`.env` file missing at project root. "
        "Create it with: `OPENAI_API_KEY=sk-...` (key shared by Luis)."
    )
else:
    st.success("`.env` file exists")
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            st.error(
                "`OPENAI_API_KEY` not set. "
                "Add the line `OPENAI_API_KEY=sk-...` to `.env`."
            )
        elif not api_key.startswith("sk-"):
            st.warning(
                f"Key format looks off (starts with `{api_key[:3]}`). "
                "Re-paste the key Luis shared, no quotes or spaces."
            )
        else:
            st.success(f"`OPENAI_API_KEY` set (`{api_key[:7]}...{api_key[-4:]}`)")
    except ImportError:
        st.error("python-dotenv not installed. Fix: `pip install python-dotenv`")

# ── 5. Live API ping ──────────────────────────────────────────────────────
st.subheader("5. OpenAI API ping (live, ~$0.0001)")
if api_key and api_key.startswith("sk-"):
    if st.button("Ping the API"):
        with st.spinner("Calling OpenAI..."):
            try:
                from openai import OpenAI
                client = OpenAI(api_key=api_key)
                resp = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=1,
                )
                usage = resp.usage
                st.success(
                    f"API responded ({resp.model}, "
                    f"{usage.prompt_tokens} in / {usage.completion_tokens} out tokens)"
                )
            except Exception as e:
                msg = str(e)
                if "Incorrect API key" in msg or "invalid_api_key" in msg.lower():
                    st.error("API key rejected. Re-paste from Luis (no quotes, no whitespace).")
                elif "model_not_found" in msg.lower() or "does not exist" in msg.lower():
                    st.error("`gpt-4.1-mini` not available on this account. Tell Luis.")
                elif "rate" in msg.lower():
                    st.warning("Rate-limited on the ping. Wait 30s and try again.")
                else:
                    st.error(f"API call failed: {msg.splitlines()[0]}")
else:
    st.info("API key not set, ping unavailable.")

# ── 6. Trained models ──────────────────────────────────────────────────────
st.subheader("6. Trained models (homework output)")
models_dir = ROOT / "session1" / "models"
expected = ["classifier.pkl", "regressor.pkl", "clusterer.pkl", "timeseries.pkl"]
if not models_dir.exists():
    st.warning(
        "`session1/models/` does not exist yet. "
        "Run `pre_class/1_classical_models.ipynb` (under 3 min) to produce the artifacts."
    )
else:
    missing = [name for name in expected if not (models_dir / name).exists()]
    if missing:
        st.warning(
            f"Missing artifacts: {', '.join(missing)}. "
            "Run `pre_class/1_classical_models.ipynb` to produce them."
        )
    else:
        for name in expected:
            st.success(f"`session1/models/{name}`")

st.divider()
st.markdown("If everything is green, you're ready. If not, fix the failures above and rerun.")
