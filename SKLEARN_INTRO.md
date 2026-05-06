# sklearn en 10 minutos

Refresher rápido de scikit-learn para llegar a S1 con la cabeza despejada. Si has hecho EDA y modelado en otras clases, lo que sigue es repaso. Si te suena a chino, léelo dos veces y haz el ejemplo del final.

---

## Qué es sklearn

`scikit-learn` es la librería estándar de machine learning clásico en Python. Tiene una **API uniforme**: todos los modelos se entrenan, predicen y evalúan con los mismos tres métodos. Aprendido uno, aprendidos todos.

```python
from sklearn.linear_model import LogisticRegression

modelo = LogisticRegression()      # crear
modelo.fit(X_train, y_train)       # entrenar
y_pred = modelo.predict(X_test)    # predecir
```

Tres pasos. Ese es el patrón. Da igual si es regresión, clasificación, clustering: la API es la misma.

---

## Train / test split

Si entrenas y evalúas con los mismos datos, el modelo se sabe la respuesta de memoria. Para saber si **generaliza**, separas:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,        # 80% entrena, 20% evalúa
    random_state=42,      # reproducible
    stratify=y,           # mantén la proporción de clases (en clasificación)
)
```

`random_state` es el "seed" que hace que el split sea siempre el mismo. **Úsalo siempre** que quieras reproducibilidad.

---

## Los 4 modelos que usamos en el taller

### Clasificación: `RandomForestClassifier` o `XGBClassifier`

Predice una etiqueta (`converted` = True/False).

```python
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)
clf.fit(X_train, y_train)
proba = clf.predict_proba(X_test)[:, 1]   # probabilidad de la clase 1
y_pred = clf.predict(X_test)              # decisión binaria
```

`predict_proba` te da probabilidades; `predict` te da la etiqueta. Las probabilidades son más útiles porque permiten ajustar el umbral de decisión.

### Regresión: `RandomForestRegressor`

Predice un número (`quoted_acv_eur`).

```python
from sklearn.ensemble import RandomForestRegressor

reg = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)              # un float por fila
```

**Truco frecuente**: si el target tiene una distribución muy sesgada (como ACV en EUR), entrena sobre `np.log(y)` y luego haz `np.exp(predicciones)` para volver. El modelo trabaja con una distribución más simétrica y mejora el R².

### Clustering: `KMeans`

Agrupa filas similares **sin** etiquetas. Le dices cuántos clusters quieres.

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)        # KMeans necesita datos escalados

km = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = km.fit_predict(X_scaled)         # un cluster id por fila
```

**Importante**: KMeans usa distancias euclídeas. Si una columna tiene valores en miles y otra en 0-1, la primera domina. Por eso siempre se escala con `StandardScaler` antes.

### Reducción de dimensiones: `PCA`

No la usamos para predecir, sí para **visualizar** clusters en 2D.

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)        # cada fila ahora son 2 números

# scatter X_2d coloreando por labels → ves los clusters
```

---

## Evaluación

Cada familia tiene sus métricas. Las que vamos a usar:

### Clasificación

| Métrica | Qué mide | Bueno si... |
|---|---|---|
| `accuracy_score` | % de predicciones correctas | clases balanceadas |
| `roc_auc_score` | discriminación entre clases (0.5 = aleatorio, 1.0 = perfecto) | clases desbalanceadas |
| `precision_score` | de los que dije "positivo", cuántos lo eran | te importa minimizar falsos positivos |
| `recall_score` | de los positivos reales, cuántos detecté | te importa minimizar falsos negativos |
| `f1_score` | balance precision + recall | quieres una sola métrica |

```python
from sklearn.metrics import roc_auc_score, accuracy_score

print(f"AUC: {roc_auc_score(y_test, proba):.3f}")
print(f"Acc: {accuracy_score(y_test, y_pred):.3f}")
```

### Regresión

| Métrica | Qué mide |
|---|---|
| `mean_absolute_error` (MAE) | error medio absoluto, en las unidades del target (EUR aquí) |
| `mean_squared_error` (MSE) | penaliza más los errores grandes |
| `r2_score` | varianza explicada (0 = no explica nada, 1 = perfecto) |

```python
from sklearn.metrics import mean_absolute_error, r2_score

print(f"MAE: {mean_absolute_error(y_test, y_pred):,.0f} EUR")
print(f"R²:  {r2_score(y_test, y_pred):.3f}")
```

### Clustering

| Métrica | Qué mide |
|---|---|
| `silhouette_score` | qué tan compactos son los clusters (-1 a 1, más alto mejor) |
| `adjusted_rand_score` | parecido entre tus clusters y una verdad oculta (0 = aleatorio, 1 = perfecto) |

ARI solo se usa si tienes ground truth (en Cañadata sí: `lead_segment_truth`).

---

## Errores que se ven en clase

### Data leakage

```python
# MAL — usas el target para construir features
X["churn_prob_from_history"] = ...
X_train, X_test, y_train, y_test = train_test_split(X, y, ...)
# tu test ya conoce información del target
```

Regla: ninguna feature puede saber del target. El split es lo último.

### Escalado en el conjunto incorrecto

```python
# MAL — fit_transform sobre todo X antes del split
X_scaled = scaler.fit_transform(X)
X_train, X_test, ... = train_test_split(X_scaled, y, ...)
# tu test ya conoce la media/std de TODO X
```

Regla: el `fit` del scaler va sobre el train, y luego solo `transform` sobre el test.

```python
# BIEN
X_train, X_test, y_train, y_test = train_test_split(X, y, ...)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
```

### Olvidarte de `random_state`

Sin `random_state`, cada vez que ejecutas obtienes un resultado distinto. Bug pendiente. Ponlo siempre.

---

## Ejemplo de 30 segundos

Para fijar el patrón mental:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# 1. Datos
df = pd.read_csv("data/canadata_leads_clean.csv")
X = pd.get_dummies(df.drop(columns=["converted", "lead_segment_truth"]), drop_first=False)
y = df["converted"].astype(int)

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Entrena
clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# 4. Evalúa
proba = clf.predict_proba(X_test)[:, 1]
print(f"ROC AUC: {roc_auc_score(y_test, proba):.3f}")
```

Si esto te corre mentalmente sin tropiezos, llegas listo a S1.

---

## Para ir más allá

- [User Guide oficial](https://scikit-learn.org/stable/user_guide.html) (en inglés, denso pero canónico).
- [Cheat sheet de sklearn](https://scikit-learn.org/stable/tutorial/machine_learning_map.html) — qué algoritmo usar según el problema.
- En el taller usaremos `Pipeline` solo en el reto S3. Si quieres adelantarte: [docs](https://scikit-learn.org/stable/modules/compose.html#pipeline).
