# Streamlit en 10 minutos

Refresher rápido. Si has hecho el taller anterior (FitLife / MDA13_isdi), esto es repaso. Si no, léelo entero — vas a usar Streamlit desde la primera sesión.

---

## Qué es Streamlit

Una librería de Python que convierte un script `.py` en una app web. Cada función `st.algo()` añade un elemento a la página. No necesitas saber HTML, CSS ni JavaScript.

```python
# hello.py
import streamlit as st

st.title("Hola Cañadata")
st.write("Mi primera app web en 3 líneas.")
```

```sh
streamlit run hello.py
```

Abre el navegador en `http://localhost:8501` y ya está. No despliegues, no configuración, no servidor que mantener.

---

## El modelo mental: re-run top-to-bottom

**Esto es lo más importante de Streamlit.**

Cada vez que el usuario interactúa con la página (pulsa un botón, mueve un slider, escribe en un input), Streamlit **re-ejecuta TODO el archivo desde la línea 1** hasta el final. No hay callbacks, no hay event handlers. Es un re-render completo.

Implicaciones:

- Las **variables se reinicializan** en cada interacción. Si quieres recordar algo entre re-runs, usa `st.session_state` (lo veremos en S2).
- Las **operaciones costosas se repiten**. Si cargas un CSV de 5 MB en línea 10, eso se ejecuta cada vez. Por eso existen los caches `@st.cache_data` y `@st.cache_resource`.
- **El orden importa**: lo que escribes en el script es el orden en el que aparece en la página.

---

## Widgets esenciales

### Texto y formato

```python
st.title("Título grande")
st.header("Cabecera")
st.subheader("Subtítulo")
st.write("Texto normal. Acepta markdown: **negrita**, *cursiva*, `código`.")
st.caption("texto pequeño en gris")
st.divider()                       # línea horizontal
st.code("print('hola')", language="python")
```

### Mostrar datos

```python
st.dataframe(df)                   # tabla interactiva (ordenable, paginada)
st.table(df.head())                # tabla estática
st.json({"key": "value"})          # JSON formateado
st.metric("ROC AUC", "0.84", delta="+0.02")    # número grande con delta
```

### Inputs

```python
nombre = st.text_input("Tu nombre")
edad = st.slider("Edad", 0, 100, 25)                       # min, max, default
opcion = st.selectbox("Plan", ["basic", "premium", "family"])
opciones = st.multiselect("Industrias", ["SaaS", "fintech", "retail"])
activo = st.checkbox("Mostrar inactivos")
boton = st.button("Procesar")
fecha = st.date_input("Fecha")
```

Los widgets devuelven el valor actual. Cuando el usuario cambia algo, todo el script se re-ejecuta y la variable trae el nuevo valor.

### Layout

```python
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Acc", "0.78")
with col2:
    st.metric("AUC", "0.84")
with col3:
    st.metric("MAE", "1992 €")

with st.sidebar:
    st.subheader("Opciones")
    region = st.selectbox("Región", ["ES", "FR", "DE"])

with st.expander("Ver detalles"):
    st.write("Esto está oculto hasta que el usuario hace click")

tab1, tab2 = st.tabs(["Datos", "Modelo"])
with tab1:
    st.dataframe(df)
with tab2:
    st.write("...")
```

### Gráficos

```python
st.line_chart(df)                  # rápido, una línea por columna numérica
st.bar_chart(df["industry"].value_counts())
st.area_chart(df_temporal)
st.map(df_con_coordenadas)         # si tienes lat/lon

import plotly.express as px        # si quieres más control
fig = px.scatter(df, x="company_size", y="quoted_acv_eur", color="industry")
st.plotly_chart(fig)
```

### Estado y feedback

```python
st.success("Funcionó")
st.info("Información neutra")
st.warning("Cuidado")
st.error("Algo se rompió")

with st.spinner("Calculando..."):
    resultado = funcion_lenta()
```

---

## Caching: `@st.cache_data` vs `@st.cache_resource`

Recuerda: cada interacción re-ejecuta TODO. Sin cache, cargar el dataset cada vez es lento. Sin cache, recargar el modelo entrenado cada vez es lentísimo. Streamlit te da dos decoradores.

### `@st.cache_data`

Para cosas **inmutables** que no se modifican: dataframes, listas, diccionarios, arrays.

```python
@st.cache_data
def load_data():
    return pd.read_csv("data/canadata_leads_clean.csv")

df = load_data()    # primera vez tarda, siguientes son instantáneas
```

Si los argumentos cambian, se reejecuta. Si no, devuelve el resultado cacheado.

### `@st.cache_resource`

Para objetos **vivos** que mantienen estado: modelos de ML, conexiones a base de datos, sesiones HTTP.

```python
import joblib

@st.cache_resource
def load_classifier():
    return joblib.load("session1/models/classifier.pkl")

clf = load_classifier()    # se carga una vez por sesión
```

La diferencia clave: `cache_data` serializa el resultado (rápido pero el objeto es "una copia"); `cache_resource` mantiene la **misma instancia** entre re-runs.

Regla: dataframes y arrays → `cache_data`. Modelos y conexiones → `cache_resource`.

---

## `st.session_state` (lo veremos a fondo en S2)

Cuando necesitas que algo **sobreviva** entre re-runs, lo guardas en `st.session_state`:

```python
if "contador" not in st.session_state:
    st.session_state.contador = 0

if st.button("Incrementa"):
    st.session_state.contador += 1

st.write(f"Contador: {st.session_state.contador}")
```

`session_state` es un dict que vive mientras la pestaña del navegador está abierta. Cierra pestaña → empieza desde cero. Lo usaremos en S2 para guardar el historial del chat con el LLM.

---

## Ejemplo de 5 líneas

```python
import streamlit as st
import pandas as pd

df = pd.read_csv("data/canadata_leads_clean.csv")
st.title("Cañadata leads")
st.metric("Total leads", len(df))
st.dataframe(df.head(20))
```

Lánzalo: `streamlit run app.py`. Web en 5 líneas.

---

## Errores que aparecen siempre

### "Module not found" pero pip install funcionó

Estás corriendo Streamlit desde un Python distinto al que instaló las dependencias.

```sh
which streamlit       # macOS / Linux
where streamlit       # Windows
which python
```

Si `streamlit` y `python` apuntan a directorios distintos, activa tu entorno y vuelve a intentar.

### Cambios no aparecen en la web

Streamlit detecta cambios en el archivo y ofrece recargar arriba a la derecha. Si no salta:

- Guarda el archivo (`Ctrl+S` / `Cmd+S`).
- Pulsa `R` en la página para forzar rerun.
- O pulsa el icono `⋮` arriba a la derecha → "Rerun".

### "AxiosError" o WebSocket roto

Cierra y vuelve a abrir la pestaña. Si persiste, mata Streamlit (`Ctrl+C` en la terminal) y relanza.

### File path ambiguo

```python
# MAL — ruta relativa al cwd
df = pd.read_csv("data/canadata_leads_clean.csv")
```

Funciona si lanzas Streamlit desde la raíz del proyecto. Falla si lo lanzas desde otra carpeta. Patrón seguro:

```python
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent  # ajusta los .parent según tu profundidad
df = pd.read_csv(ROOT / "data" / "canadata_leads_clean.csv")
```

---

## Para ir más allá

- [Documentación oficial](https://docs.streamlit.io/) — muy buena, con ejemplos en cada widget.
- [API reference](https://docs.streamlit.io/develop/api-reference) — todos los `st.*` disponibles.
- [Cheat sheet](https://docs.streamlit.io/develop/quick-reference/cheat-sheet) — recordatorio en una página.
- [Streamlit gallery](https://streamlit.io/gallery) — apps reales para inspiración.
