# statsmodels en 10 minutos

Refresher rápido de `statsmodels` para llegar a S1 con la cabeza despejada. En el taller lo usamos sobre todo para series temporales (SARIMAX) y, opcionalmente, para regresión con interpretabilidad estadística (OLS).

---

## sklearn vs statsmodels — ¿cuándo cada uno?

Las dos librerías hacen modelos. La diferencia es **el énfasis**.

| | sklearn | statsmodels |
|---|---|---|
| Objetivo | predecir bien | entender relaciones estadísticas |
| Salida típica | `predict()`, `predict_proba()` | `model.summary()` con p-values, coeficientes, R², AIC |
| Series temporales | poco soporte nativo | SARIMAX, ETS, ARIMA |
| Inferencia (intervalos de confianza, p-values) | no | sí |
| Curva de aprendizaje | baja | media |

Regla del pulgar: si vas a desplegar a producción, `sklearn`. Si vas a presentar resultados a un comité o necesitas entender qué variable importa estadísticamente, `statsmodels`.

En este taller, statsmodels es el modelo de **series temporales** porque sklearn no tiene SARIMAX y este caso lo necesita.

---

## OLS — regresión lineal con inferencia

OLS = Ordinary Least Squares. Es la regresión lineal de toda la vida pero te devuelve el análisis estadístico completo.

```python
import statsmodels.api as sm
import pandas as pd
import numpy as np

df = pd.read_csv("data/canadata_leads_clean.csv")
X = df[["company_size", "n_meetings", "emails_opened"]]
X = sm.add_constant(X)             # añade intercepto (β₀)
y = np.log(df["quoted_acv_eur"])   # log target porque ACV es log-normal

model = sm.OLS(y, X).fit()
print(model.summary())
```

`model.summary()` te imprime:

- **R²**: varianza explicada.
- **Coef**: coeficiente de cada variable.
- **P>|t|**: p-value. Si < 0.05, la variable es estadísticamente significativa.
- **Conf. Int.**: intervalo de confianza al 95% del coeficiente.
- **F-statistic**: significancia global del modelo.

Esto es lo que sklearn NO te da gratis. Si la pregunta es "¿el `n_meetings` mueve el ACV de verdad o es ruido?", lees el p-value de esa fila.

---

## SARIMAX — series temporales

SARIMAX = Seasonal AutoRegressive Integrated Moving Average with eXogenous variables. Modelo clásico para predecir el siguiente valor de una serie temporal aprovechando:

- valores pasados (auto-regresivo, **AR**)
- diferencias para hacer estacionaria la serie (**I**)
- errores pasados (media móvil, **MA**)
- estacionalidad (componente **S**)
- variables explicativas externas (**X**)

### Los parámetros

`SARIMAX(serie, order=(p, d, q), seasonal_order=(P, D, Q, S))`

- `p` — número de lags AR (cuántos valores anteriores).
- `d` — número de diferenciaciones (1 si la serie tiene tendencia).
- `q` — número de lags MA.
- `P, D, Q` — los mismos pero para la componente estacional.
- `S` — periodo de la estacionalidad. 12 si tu serie es mensual y la estacionalidad es anual.

Para Cañadata usamos `(1, 1, 1)` × `(1, 1, 1, 12)` — un AR, una diferenciación, un MA, y lo mismo a nivel estacional con periodo 12 meses. Es un punto de partida razonable; en el reto puedes probar otras combinaciones.

### Ejemplo

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd

df = pd.read_csv("data/canadata_leads_clean.csv")
df["signup_date"] = pd.to_datetime(df["signup_date"])
monthly = df[df["converted"]].set_index("signup_date").resample("MS").size()

model = SARIMAX(
    monthly,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    enforce_stationarity=False,
    enforce_invertibility=False,
)
fit = model.fit(disp=False)

print(f"AIC: {fit.aic:.1f}")
forecast = fit.get_forecast(steps=6)
print(forecast.predicted_mean.round(1))
print(forecast.conf_int().round(1))
```

`AIC` (Akaike Information Criterion) es la métrica para **comparar modelos** sobre la misma serie. Más bajo es mejor. Si pruebas SARIMAX(2,1,1) y el AIC sube, el modelo extra no compensa.

`get_forecast(steps=N)` te da las próximas N predicciones. `.predicted_mean` es el valor central; `.conf_int()` es la banda de confianza al 95%.

---

## Errores típicos

### "Convergence warning"

statsmodels te dice que el optimizador no convergió. A veces es porque la serie es demasiado corta o tiene ruido extremo. Workarounds:

```python
fit = model.fit(disp=False, maxiter=200, method="lbfgs")
```

O simplifica el modelo (menos parámetros).

### Olvidar el índice temporal

```python
# MAL
SARIMAX([1, 2, 3, 4, 5], order=(1, 1, 1)).fit()
# funciona pero las predicciones no tienen fechas
```

```python
# BIEN
serie = pd.Series([1, 2, 3, 4, 5], index=pd.date_range("2024-01-01", periods=5, freq="MS"))
SARIMAX(serie, order=(1, 1, 1)).fit()
# el forecast viene con índice temporal correcto
```

### `enforce_stationarity` y `enforce_invertibility`

Por defecto `True`. Si tu serie no es estacionaria, el optimizador puede fallar. Para datos reales (no simulados), suele ser razonable ponerlos a `False`. En el taller los desactivamos.

---

## Ejemplo de 30 segundos

OLS sobre Cañadata:

```python
import statsmodels.api as sm
import pandas as pd
import numpy as np

df = pd.read_csv("data/canadata_leads_clean.csv")
X = pd.get_dummies(df[["industry", "company_size", "n_meetings"]], drop_first=True)
X = sm.add_constant(X)
y = np.log(df["quoted_acv_eur"])

print(sm.OLS(y, X).fit().summary())
```

Mira la columna **P>|t|** — los coeficientes con p-value < 0.05 son los que mueven el ACV de verdad.

---

## Para ir más allá

- [statsmodels Time Series Analysis](https://www.statsmodels.org/stable/tsa.html)
- [Forecasting: Principles and Practice (Hyndman)](https://otexts.com/fpp3/) — el libro de referencia online y gratis sobre series temporales.
- ACF / PACF para elegir `(p, q)` empíricamente: `from statsmodels.graphics.tsaplots import plot_acf, plot_pacf`.
