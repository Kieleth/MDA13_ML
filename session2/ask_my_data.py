# ============================================================
# ask_my_data.py — Habla con tus datos en español
# ============================================================
#
# Esto es lo que TE LLEVAS a tu trabajo el lunes.
#
# Es la versión sin Streamlit, sin Cañadata, sin nada del taller
# de la idea central de paso_5: una pregunta en español →
# código pandas → resultado real.
#
# Cómo usarlo en TU trabajo:
#   1. Cambia `csv_path` por la ruta de tu CSV.
#   2. Cambia `column_hints` por las columnas REALES de tu CSV
#      (con su tipo y, para categóricas, los valores que toman).
#   3. Ejecuta: python ask_my_data.py
#   4. Escribe preguntas en la terminal. Ctrl+C para salir.
#
# Requisitos: `pip install openai python-dotenv pandas`.
# Y `.env` con OPENAI_API_KEY=sk-... (mismo formato que en el taller).
#
# ============================================================

import os
import re
import sys

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── 1. CAMBIA ESTO POR TU DATASET ──────────────────────────
csv_path = "tu_dataset.csv"

# Describe tus columnas con tipo y, para categóricas, los valores.
# Cuanto más específico, mejor escribe pandas el LLM.
column_hints = """
    customer_id (str), order_date (str YYYY-MM-DD), amount_eur (float),
    country (str ∈ {ES, FR, DE, IT, PT}), product_category (str ∈
    {electronics, clothing, books}), is_returning (bool).
"""

# ── 2. PROMPT DEL SISTEMA (el corazón del patrón) ──────────
SYSTEM_PROMPT = f"""Eres un analista de datos. Tienes un DataFrame de pandas llamado `df` con estas columnas:
{column_hints}

Genera código Python pandas para responder la pregunta del usuario.
Termina asignando el resultado a una variable llamada `resultado`.
Devuelve SÓLO un bloque ```python ... ```, sin explicación."""


# ── 3. Carga datos + cliente ───────────────────────────────
df = pd.read_csv(csv_path)
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# ── 4. Bucle pregunta → código → resultado ─────────────────
def extract_code(text: str) -> str:
    """Saca el código de dentro de ```python ... ```."""
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    return m.group(1) if m else text.strip()


def ask(question: str):
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        temperature=0.0,
    )
    code = extract_code(response.choices[0].message.content)
    print("\n[code]\n" + code)

    namespace = {"df": df, "pd": pd}
    try:
        exec(code, namespace)
        print("\n[result]")
        print(namespace.get("resultado", "(no se asignó `resultado`)"))
    except Exception as e:
        print(f"\n[error] {e}\n→ Si la columna no existe, añádela a column_hints.")


if __name__ == "__main__":
    print(f"Cargado {len(df)} filas de {csv_path}. Escribe tu pregunta (Ctrl+C para salir).\n")
    try:
        while True:
            q = input("❯ ").strip()
            if q:
                ask(q)
                print()
    except (KeyboardInterrupt, EOFError):
        print("\n¡Hasta luego!")
