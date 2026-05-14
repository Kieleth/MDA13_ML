# Cómo reentrenar y validar tus modelos cuando lleguen nuevos datos

Esta guía es lectura. No hay código que correr. Es lo que mira tu yo de dentro de 3 meses, cuando tu CRM acumule 150 leads nuevos y te preguntes si deberías reentrenar.

---

## ¿Cuándo conviene reentrenar?

Tres señales, en orden de importancia:

1. **Volumen**: tienes ≥20% más datos que cuando entrenaste la última vez. Más datos = mejores modelos casi siempre.
2. **Tiempo**: han pasado ≥3 meses desde el último entrenamiento. El negocio cambia (nuevos canales, nuevos mercados, nuevos competidores). El modelo viejo se va quedando atrás.
3. **Performance**: las predicciones empezaron a fallar más de lo esperado. Si tu calibración (P=0.8 → firman 80%) deja de cumplirse, es señal de drift.

Si dos de las tres se dan, reentrena. Si las tres, reentrena ya.

> **Aviso intelectual honesto**: estos tres umbrales (20%, 3 meses, "más de lo esperado") son **reglas de pulgar de la industria, no leyes**. No hay un paper que demuestre que son óptimas. Vienen de la experiencia operativa de equipos de ML. Si tu negocio cambia muy rápido (mercados emergentes, productos en pivote constante), baja a 1-2 meses. Si tu negocio es muy estable (B2B en sector regulado), sube a 6 meses. Calibra a tu realidad.

---

## El flujo completo

1. **Junta** histórico + leads nuevos en un solo CSV. Mismas columnas, sin duplicados.
2. **Limpia** con el mismo pipeline que `pre_class/1_classical_models.ipynb`: normaliza categorías, fillna donde toque, dropea outliers absurdos.
3. **Split**: 80% train, 20% holdout. Estratificado por `converted` para no descompensar clases.
4. **Entrena** los 4 modelos:
   - Clasificador (XGBoost o RandomForest sobre features estructuradas).
   - Regresor (RandomForest sobre log(ACV)).
   - Clusterer (KMeans k=3 con StandardScaler).
   - Series temporales (Prophet con yearly_seasonality).
5. **Calcula** las 4 métricas sobre el holdout (no sobre train).
6. **Compara** con las métricas del modelo desplegado actualmente.
7. **Decide**:
   - Si pasa los gates: swap atómico de los `.pkl`. El dashboard recarga.
   - Si no pasa: NO toques producción. El modelo viejo sigue. Investiga por qué.

---

## Las 4 métricas, una a una

### Clasificador: ROC-AUC

- **Qué mide**: la calidad del ranking. Dado un par random (positivo, negativo), ¿qué tan probable es que el modelo le dé al positivo más score que al negativo? 0.5 = aleatorio, 1.0 = perfecto.
- **No confundir con accuracy**: accuracy mide aciertos a UN threshold concreto. AUC promedia sobre todos los thresholds.
- **En Cañadata**: el modelo actual ronda **0.85-0.92** sobre holdout n=100.
- **Gate**: el modelo nuevo debe estar a ≥ 95% del valor del actual. Si actual=0.90, nuevo debe ser ≥ 0.855.

### Regresor: R² sobre log(ACV)

- **Qué mide**: qué proporción de la varianza de log(ACV) explica el modelo. 0 = no explica nada (constante), 1 = perfecto.
- **Por qué log**: el ACV está sesgado a la derecha (muchos contratos pequeños, pocos grandes). Predecir log linealiza la cosa y mejora estabilidad.
- **En Cañadata**: ronda **0.65-0.75**.
- **Gate**: nuevo ≥ 0.95 × actual.

### Clusterer: ARI (Adjusted Rand Index)

- **Qué mide**: el acuerdo entre los clusters que predice el modelo y los arquetipos plantados (`quick_mover`, `strategic`, `tire_kicker` que vienen del generador). 0 = aleatorio, 1 = acuerdo perfecto.
- **Cuándo aplica**: sólo tiene sentido si tenemos ground truth (los arquetipos). El dataset de Cañadata trae `lead_segment_truth` porque es sintético; **en tu caso real no la vas a tener**.
- **Alternativa para tu caso real (recomendada)**: usa **silhouette score** (`sklearn.metrics.silhouette_score`). Mide qué tan bien separados están los clusters entre sí, sin necesidad de ground truth. Rango -1 a 1. **Apunta a > 0.3 como "razonable"** para k-means sobre features escaladas. < 0.2 significa que los clusters se solapan tanto que probablemente no estás capturando estructura real.
- Hay otras (Davies-Bouldin, estabilidad entre re-runs) pero silhouette es la regla de pulgar más usada y la que te recomiendo aprender primero.
- **En Cañadata**: ronda **0.40-0.50**. Lower bound aceptable porque KMeans sobre features estructuradas no clava arquetipos de comportamiento al 100%.
- **Gate**: nuevo ≥ 0.95 × actual.

### Series temporales: MAPE

- **Qué mide**: Mean Absolute Percentage Error del forecast mensual. Si tu modelo predice 12 conversiones y la realidad son 15, el error es 25%. La media sobre todos los meses del holdout es el MAPE.
- **Lower is better**: a diferencia de las anteriores, aquí queremos números BAJOS.
- **En Cañadata**: ronda **80-90%**. Suena alto, sí. Es porque el dataset mensual viene de agregar diarios ruidosos. Series temporales con n bajo + estacionalidad anual fuerte = MAPE alto natural.
- **Gate**: nuevo ≤ 1.05 × actual. La regla del 95% se invierte: para lower-is-better, "no degradar" significa "no subir más de 5%".

---

## La regla del 95% (con dirección, sin trampas)

Las cuatro métricas comparan modelo nuevo vs modelo actual. La dirección cambia según si la métrica es "más alto mejor" o "más bajo mejor":

| Métrica | Dirección | Comparador | Si actual=X, nuevo debe ser… |
|---|---|---|---|
| AUC | higher is better | nuevo ≥ 0.95 × actual | actual=0.90 → nuevo ≥ 0.855 |
| R² | higher is better | nuevo ≥ 0.95 × actual | actual=0.70 → nuevo ≥ 0.665 |
| ARI | higher is better | nuevo ≥ 0.95 × actual | actual=0.45 → nuevo ≥ 0.428 |
| MAPE | lower is better | nuevo ≤ 1.05 × actual | actual=85 → nuevo ≤ 89.25 |

Las cuatro filas dicen lo mismo en lenguaje normal: **no degrades más de un 5%**. Tolera ruido pequeño (los modelos varían algo entre seeds) pero rechaza un modelo claramente peor. El cambio de operador (≥ vs ≤) y de coeficiente (0.95 vs 1.05) según la métrica es la trampa clásica al implementarlo en código.

### Por qué 95% y no 90 o 100

- **100%** bloquearía deploys legítimos por el ruido natural de re-entrenar. Te quedarías con el modelo viejo aunque el nuevo sea igual de bueno.
- **90%** permitiría aceptar un modelo claramente peor disfrazado de "casi igual". Peligroso.
- **95%** es la zona donde la mayoría de equipos de ML acaban operando.

Puedes subirlo a 100% (exigir mejora siempre) si tu producto es muy sensible a regresión y aceptas que ciertos retrainings simplemente "no se desplegarán" porque salieron peor por suerte.

---

## ¿Y si los gates fallan?

No es el fin del mundo. Significa que el modelo nuevo no es lo bastante bueno. Pasos:

1. **Investiga**: ¿qué métrica falló? ¿Una o varias? ¿Por mucho o por poco?
2. **Mira drift**: ¿hay segmentos del dataset que cambiaron mucho? (P.ej. te empezaron a entrar leads de healthcare que antes no había.)
3. **Revisa features**: ¿alguna feature nueva sería útil? ¿Alguna feature vieja se rompió?
4. **No despliegues**: el modelo actual sigue en producción, no has hecho daño. Itera y vuelve a entrenar.
5. **Considera no reentrenar**: si tu modelo actual sigue dando buenos resultados en producción, quizá no necesitas el nuevo. "No deploy" también es una decisión válida.

---

## Cómo comparar tus métricas reales contra los rangos de arriba

Las cifras (AUC 0.85-0.92, R² 0.65-0.75, etc.) son las del modelo entrenado en S1. Para ver las tuyas:

1. Abre `pre_class/1_classical_models.ipynb`.
2. Busca las celdas finales que imprimen métricas de cada modelo sobre validation/holdout (suelen tener `print(f"AUC: ...")` o un `.score(...)` al final de la celda de entrenamiento).
3. Anota tus números reales: AUC=?, R²=?, ARI=?, MAPE=?.
4. Compáralos con los rangos. Si están dentro, vas bien. Si están muy por debajo, hay probablemente un bug en feature engineering o en el split.

Cuando reentrenes, repite este ejercicio sobre el nuevo holdout y aplica la regla del 95% celda a celda.

## ⚠ Antes de cambiar a la rama session-3-advanced

Si tienes paso_7 con huecos rellenos y `.env` con TAVILY/SLACK keys, **commitea o stashea antes** de `git checkout session-3-advanced`. Sin eso, te llevas los cambios contigo y al volver a `session-3` no sabrás qué pertenece a dónde.

```sh
git status            # ¿hay rojo?
git stash push -m "s3 huecos llenos"
git checkout session-3-advanced
# ...explora...
git checkout session-3
git stash pop         # recupera tu trabajo
```

## Hacerlo automático

El track avanzado de S3 (rama `session-3-advanced`) tiene un script Python `retrain.py` que automatiza este flujo:

- Lee histórico + batch nuevo.
- Limpia.
- Entrena los 4 modelos.
- Calcula métricas.
- Aplica los gates al 95%.
- Si pasan: swap atómico de los `.pkl` con backup del previo en `_prev/`.
- Si no pasan: imprime los gates fallidos y sale con returncode != 0.
- Escribe un JSON flag (`session1/models/_retrained_at.json`) con timestamp, métricas y resultado de cada gate.

Si os interesa la versión completa con UI Streamlit (file uploader → dry-run → deploy → cache reload + tabla antes/después), está todo en `session-3-advanced` con paso_7/8/9 originales (comparison harness con bootstrap CI, quality gates con subprocess, ship-it pipeline con swap atómico). Es más densa, pero está hecha para integrarse en un pipeline de CI/CD real.

---

## Resumen para llevar al lunes

1. Reentrenar cuando: ≥20% más datos + ≥3 meses + performance bajó.
2. Sobre holdout, no sobre train.
3. 4 métricas: AUC, R², ARI, MAPE.
4. Regla del 95%: no degrades más de un 5% (con la dirección que toque por métrica).
5. Si los gates fallan: NO toques producción.
6. Si quieres el flujo automatizado: `session-3-advanced/retrain.py`.

Esto es producción mínima viable. Lo demás (audit log, rollback, canary deploy, drift monitoring continuo) sale del scope de S3 pero vive en `session-3-advanced` como referencia.
