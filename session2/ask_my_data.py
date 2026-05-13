# ask_my_data.py — Habla con tus datos en español
#
# Versión sin Streamlit, sin Cañadata, sin nada del taller.
# Cópialo a tu proyecto, cambia las DOS zonas marcadas, y empieza a preguntar.
#
# Requisitos:
#   pip install openai python-dotenv pandas
#   .env con OPENAI_API_KEY=sk-...

import os
import re

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


# ── 1. CAMBIA ESTAS DOS LÍNEAS POR TU DATASET ─────────────────────────
csv_path = "tu_dataset.csv"
column_hints = """
    customer_id (str), order_date (str YYYY-MM-DD), amount_eur (float),
    country (str ∈ {ES, FR, DE, IT, PT}),
    product_category (str ∈ {electronics, clothing, books}),
    is_returning (bool).
"""

# ── 2. El prompt del sistema (el corazón del patrón) ──────────────────
SYSTEM_PROMPT = f"""Eres un analista de datos. Tienes un DataFrame de pandas llamado `df` con estas columnas:
{column_hints}

Genera código Python pandas para responder la pregunta del usuario.
Termina asignando el resultado a una variable llamada `resultado`.
Devuelve SÓLO un bloque ```python ... ```, sin explicación."""


# ── 3. Bucle pregunta → código → resultado ────────────────────────────
df = pd.read_csv(csv_path)
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def ask(question: str) -> None:
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        temperature=0.0,
    )
    raw = resp.choices[0].message.content
    m = re.search(r"```(?:python)?\n(.*?)```", raw, re.DOTALL)
    code = m.group(1) if m else raw.strip()
    print("\n[code]\n" + code)

    ns = {"df": df, "pd": pd}
    try:
        exec(code, ns)
        print("\n[result]\n" + str(ns.get("resultado", "(sin `resultado`)")))
    except Exception as e:
        print(f"\n[error] {e}\n→ Si la columna no existe, añádela a column_hints.")


if __name__ == "__main__":
    print(f"Cargado {len(df)} filas de {csv_path}. Ctrl+C para salir.\n")
    try:
        while True:
            q = input("❯ ").strip()
            if q:
                ask(q)
                print()
    except (KeyboardInterrupt, EOFError):
        print()
