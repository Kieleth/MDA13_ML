# Sesión 3 — Medir, validar, desplegar

S1 entrenaste herramientas. S2 las pusiste detrás de un LLM. Hoy mides si todo eso justifica su coste, montas puertas de calidad, y construyes el flujo entero del lunes ("entran 150 leads nuevos, ¿cómo redesplegamos?").

## Orden

1. `exercises/paso_7.py` — **Comparison harness** sobre 100 leads del holdout. Clasificador clásico vs LLM zero-shot, con AUC, bootstrap CI, coste y latencia. **TOGETHER**: Luis tipea contigo.
2. `exercises/paso_8.py` — **Quality gates** alrededor de `retrain.py`. Dry-run, gates relativos al modelo en producción, deploy real tras confirmación. **SUPPORTED**: huecos más grandes, Luis apoya.
3. `exercises/paso_9.py` — **Ship-it pipeline**. Upload CSV → retrain → gates → deploy → reload del dashboard. **INDEPENDENT**: lo escribes tú, Luis mira de lejos.

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
- **ROC-AUC** del clasificador clásico vs LLM zero-shot.
- **Bootstrap CI 95%** (1000 remuestreos). Si los intervalos se solapan, NO puedes decir que un modelo gana.
- **Coste** total y por predicción.
- **Latencia** media (LLM es 100× más lento que el clasificador).
- **Expected Value** dado un threshold y la economía del negocio (`gain_per_signing`, `cost_per_call`). El AUC mide saber, el EV mide cobrar.

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

> "Para mi caso real, desplegaría X porque Y."

Con `X ∈ {LLM solo, LLM con herramientas, herramienta sola, híbrido}`. No hay respuesta correcta; hay decisión articulada. Eso es lo que te llevas.

## Take-home

`retrain.py` ya tiene el flujo completo (combinar histórico + batch, entrenar 4 modelos, gates, swap atómico). Lee el script con calma cuando quieras montar algo similar en tu trabajo.

## Referencia

`reference_code/dashboard_v3.py` — el dashboard integrado con los 3 modos LLM + harness de comparación + ship-it pipeline. Lo lanza el profesor al final como reveal. Mira aquí si te bloqueas, pero después de intentar tú.
