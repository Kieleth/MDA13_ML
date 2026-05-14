# Sesión 3 · Medir, validar, desplegar

S1 entrenaste herramientas. S2 las pusiste detrás de un LLM. Hoy mides si todo eso justifica su coste, montas puertas de calidad, y construyes el flujo entero del lunes ("entran 150 leads nuevos, ¿cómo redesplegamos?").

## ⚠ Si en S2 te quedaste atascado, lee esto primero

S3 hereda los conceptos de S2 (los 3 modos LLM: zero-shot, analista, operador) sin recap dentro de los pasos. Si no te quedó claro qué hace cada uno:

- **Opción rápida (5 min)**: abre `session2/exercises/paso_6.py` y mira el `SYSTEM_PROMPT_OPERADOR` pre-escrito. Eso es lo que el LLM "ve" al actuar como operador.
- **Opción visual (5 min)**: lanza `streamlit run session2/reference_code/dashboard_v2.py` y prueba la vista de comparación analista vs operador con la misma pregunta.
- **Opción cero**: hoy el "LLM zero-shot" es lo único que vas a tocar contra `paso_7` (vs el clasificador clásico). Con eso te alcanza para empezar. Los otros 2 modos llegarán al final cuando lance `dashboard_v3.py` como reveal.

## Conceptos que se asumen en S3 (warm-up rápido)

Si alguno te suena flojo, expande el bloque "🧯 Warm-up: 5 conceptos en 10 minutos" al inicio de `paso_7.py`. Resumen aquí también:

- **AUC**: mide la calidad del ranking de un modelo. Tomas dos leads cualesquiera, uno que convirtió y otro que no. Si la puntuación del modelo para el que convirtió es más alta, el par suma 1. AUC es la media sobre todos los pares. 0.5 equivale al azar, 1.0 es ordenación perfecta. No es accuracy.
- **Train / test / holdout**: train para ajustar el modelo, test para iterar durante el desarrollo, holdout es el examen final que ningún modelo ha visto. Hoy estrenamos el holdout.
- **Bootstrap**: 1000 remuestreos con reemplazo de los `n` leads te dan un intervalo de confianza al 95%. Si los IC de dos modelos se solapan, no puedes afirmar cuál es mejor.
- **Vectorización**: el clasificador procesa los 50 leads en una sola operación matricial (~24 ms). El LLM hace una llamada HTTP por lead (~650 ms cada una). Por lead la diferencia es de unos 1300×.
- **Unidades**: hoy el coste se muestra en céntimos y en euros directos. Si ves `m€` en algún sitio (heredado de S2), son milésimas de euro, no millones.

## Antes de empezar (branch hygiene)

Si tienes cambios sin commitear de S2 en tu working tree (paso_4/5/6 con huecos rellenados), commitea o stashea antes de cambiar de rama:

```sh
git status                 # ¿hay rojo en session2/?
git stash push -m "s2_test" # opción A: guardar para luego
# o:
git checkout -- session2/  # opción B: descartar
```

Los archivos `.pkl` de S1 (`session1/models/*.pkl`) están en `.gitignore`. **Persisten en tu working tree al cambiar de rama** si ya los entrenaste en S1. Si has hecho clone fresco hoy, primero corre `pre_class/1_classical_models.ipynb` para regenerarlos.

## Orden

1. `exercises/paso_7.py` · **Comparison harness** sobre 100 leads del holdout. Clasificador clásico vs LLM zero-shot, con AUC, bootstrap CI, coste y latencia. **TOGETHER**: Luis tipea contigo.
2. `exercises/paso_8.py` · **Quality gates** alrededor de `retrain.py`. Dry-run, gates relativos al modelo en producción, deploy real tras confirmación. **SUPPORTED**: huecos más grandes, Luis apoya.
3. `exercises/paso_9.py` · **Ship-it pipeline**. Upload CSV → retrain → gates → deploy → reload del dashboard. **INDEPENDENT**: lo escribes tú, Luis mira de lejos.

Cada `.py` se lanza con:

```sh
streamlit run session3/exercises/paso_N.py
```

## Si peta al arrancar

| Síntoma | Causa | Fix |
|---|---|---|
| `NameError: name '___' is not defined` | huecos sin rellenar | búscalos en el código (`grep -n "___" session3/exercises/paso_N.py`) |
| `Falta data/canadata_holdout.csv` | falta el dataset reservado | clona el repo limpio o coge `canadata_holdout.csv` del kit |
| `Falta session1/models/X.pkl` | pre-clase incompleta o `git checkout` sin S1 | corre `pre_class/1_classical_models.ipynb` |
| `No existe session3/retrain.py` | no estás en la rama `session-3` | `git checkout session-3` |
| `OPENAI_API_KEY no está` | falta `.env` | crea `.env` en la raíz con `OPENAI_API_KEY=sk-...` |
| `No se pudo contactar OpenAI` | red rota o key inválida/sin crédito | comprueba red, luego que la key es válida |
| `Algún gate falló` en dry-run | el modelo nuevo no supera el umbral | mira el output completo del expander; el comparador exacto está al final de cada línea |

## Lo que se mide en paso_7

Sobre 100 leads del holdout (que ningún modelo ha visto):
- **ROC-AUC** del clasificador clásico vs LLM zero-shot. AUC = probabilidad de que un par random (positivo, negativo) tenga el positivo con score más alto. 0.5 = aleatorio, 1.0 = perfecto.
- **Bootstrap CI 95%** (1000 remuestreos con reemplazo). Si los intervalos se solapan, NO puedes decir que un modelo gana. Es bootstrap independiente, no pareado (lo más correcto en producción sería pareado o DeLong test).
- **Coste** total y por predicción, en céntimos y € directos. Para 1000 leads, ~3 céntimos (0.03 €).
- **Latencia** media. ~650 ms por lead con el LLM, frente a ~24 ms para los 50 leads del clasificador (vectorizado). Por lead son ~1300× de diferencia.
- **Expected Value** dado un threshold y la economía del negocio (`gain_per_signing`, `cost_per_call`). El AUC mide saber, el EV mide cobrar.

Las columnas `converted` y `lead_segment_truth` del holdout vienen plantadas en pre-class (`pre_class/1_classical_models.ipynb`); son la **ground truth** contra la que se evalúan los modelos.

## Quality gates (paso_8)

`retrain.py` ya está escrito. Tu trabajo en paso_8 es construir la **UI alrededor**:

1. Leer el JSON flag del último deploy (`session1/models/_retrained_at.json`).
2. Lanzar `retrain.py --dry-run` con `subprocess.run`.
3. Parsear las líneas que imprime (regex tolerante: el comparador entre paréntesis tiene paréntesis anidados).
4. Si los gates pasan, ofrecer el botón de deploy real.

Los gates son **relativos al modelo en producción**, no absolutos. Si el AUC nuevo es <95% del AUC del modelo actual, gate falla y NO se hace swap.

## Ship-it (paso_9)

Pipeline completa en una página. File uploader → dry-run → deploy → cache reload. Al terminar, el dashboard (paso_6 o dashboard_v3) ve los modelos nuevos sin reiniciar Streamlit (`st.cache_resource.clear()` + `st.rerun()`).

## El veredicto (cierre de S3)

Al final escribes una frase:

> "Para [mi caso real / Cañadata si no tengo caso propio], desplegaría X porque Y."

Con `X ∈ {LLM solo, LLM con herramientas, herramienta sola, híbrido}`. No hay respuesta correcta; hay decisión articulada. Eso es lo que te llevas.

Si no tienes un caso real en mente, usa el de Cañadata: 700 leads/mes con scoring, ACV medio ~9K€, conversión 35-40%. La plantilla con preguntas guiadas está en [`decision_framework.md`](decision_framework.md) (mismo directorio).

## Take-home

`retrain.py` ya tiene el flujo completo (combinar histórico + batch, entrenar 4 modelos, gates, swap atómico). Lee el script con calma cuando quieras montar algo similar en tu trabajo.

## Referencia

`reference_code/dashboard_v3.py` · el dashboard integrado con los 3 modos LLM + harness de comparación + ship-it pipeline. Lo lanza el profesor al final como reveal. Mira aquí si te bloqueas, pero después de intentar tú.

## Para ver el deploy en vivo (paso_8 / paso_9)

El swap atómico del `.pkl` y el reload del dashboard sólo se aprecian si tienes OTRA pestaña abierta con el dashboard ANTES del deploy. Antes de empezar paso_8/paso_9:

```sh
streamlit run session2/exercises/paso_6.py  # o reference_code/dashboard_v2.py
```

en una ventana de Chrome aparte. Tras el deploy en paso_8/9, refrescas esa pestaña y ves las predicciones nuevas sin reiniciar Streamlit.
