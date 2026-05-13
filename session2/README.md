# Sesión 2 — LLMs operando sobre las herramientas de S1

Tres modos del LLM sobre el mismo dataset y los modelos que entrenaste ayer.

## Orden

1. `exercises/paso_4.py` — LLM zero-shot, sin herramientas. 3 huecos.
2. `exercises/paso_5.py` — LLM como analista, text-to-code sobre `df`. 4 huecos.
3. `exercises/paso_6.py` — LLM como operador, text-to-code con los 4 modelos en scope. 3 huecos.

Cada `.py` se lanza con:

```sh
streamlit run session2/exercises/paso_N.py
```

## Si peta al arrancar

| Síntoma | Causa | Fix |
|---|---|---|
| `NameError: name '___' is not defined` | tienes huecos sin rellenar | búscalos en el código (`grep -n "___" session2/exercises/paso_N.py`) |
| `Falta data/canadata_leads_clean.csv` o `Falta session1/models/X.pkl` | pre-clase incompleto | corre `pre_class/1_classical_models.ipynb` |
| `OPENAI_API_KEY no está` | falta `.env` | crea `.env` en la raíz con `OPENAI_API_KEY=sk-...` |
| `No se pudo contactar OpenAI` | red rota o key inválida/sin crédito | comprueba conexión, luego que la key es válida |

## Take-home

`ask_my_data.py` — el patrón text-to-code aislado, sin Streamlit ni Cañadata. Cópialo a tu proyecto, ajusta `csv_path` y `column_hints`, ejecuta:

```sh
python session2/ask_my_data.py
```

Bucle `input()` en terminal. Pregunta en español → código pandas → resultado.

## Referencia

`reference_code/dashboard_v2.py` — solución completa con los 3 modos en un único dashboard. La lanza el profesor al final. Si te bloqueas, mira aquí, pero después de intentar tú.
