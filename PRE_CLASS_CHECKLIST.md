# Pre-class checklist — MDA13 ML Tradicional vs LLM

Esta es la **única lista que necesitas**. Hay tres pistas. Lee la tabla de presupuesto y sabrás cuál te toca.

---

## Presupuesto de tiempo

| Pista | Tiempo | Para quién | Obligatorio |
|---|---|---|---|
| 🟢 **Esencial** | **~90 min** (rango 60-150 según tu velocidad) | **Todos** | **Sí — antes del 12 de mayo** |
| 🟡 **Si eres nuevo** | +20 min | Si nunca has usado terminal o VS Code | Recomendado, no obligatorio |
| 🔵 **Bonus** | +30-60 min | Si quieres llegar más fuerte a clase | No |

**Techo total**: 3 horas para alguien que arranca completamente de cero. **Suelo**: ~60 min para alguien con experiencia previa.

Si te acercas a las 3h, **salta el bonus** y entrega el esencial. La pista esencial es lo único que necesitas para arrancar S1 sin fricción.

---

## Antes de empezar

Calma. Esto no es difícil, pero tiene muchas piezas. Hazlo de un tirón si puedes, sin distracciones. Si en cualquier paso te pierdes, abre [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) y busca tu error en el índice.

---

## 🟡 Si eres totalmente nuevo (lee esto primero, ~20 min)

Si nunca has hecho desarrollo, lee estos dos antes de meterte con el camino esencial. Si ya te has manejado en una terminal y has abierto VS Code, sáltatelos.

- [`TERMINAL_BASICS.md`](TERMINAL_BASICS.md) — qué es la terminal, cómo abrirla en mac/windows, los 8 comandos que necesitas (`pwd`, `ls`/`dir`, `cd`, etc.). 10 min.
- [`VSCODE_TIPS.md`](VSCODE_TIPS.md) — abrir carpeta, terminal integrada, intérprete de Python, command palette. 10 min.

---

## 🟢 Camino esencial (~90 min)

Nueve pasos en orden. Cada uno con tiempo, criterio de éxito, y a qué documento ir si te pierdes.

### 1. Instala VS Code (5 min)

Descarga e instala desde https://code.visualstudio.com.

- **macOS**: archivo `.zip`, descomprime y arrastra a Aplicaciones.
- **Windows**: instalador `.exe`, valores por defecto.

**Cómo sé que está bien**: abro VS Code y veo la pantalla de bienvenida.

---

### 2. Instala Git (5 min)

- **macOS**: abre la app Terminal (`Cmd+Espacio` → "Terminal" → Enter), escribe `git --version`. Si no está, macOS te ofrece instalarlo. Acepta.
- **Windows**: descarga desde https://git-scm.com, instala con valores por defecto.

**Cómo sé que está bien**: en una terminal nueva, `git --version` devuelve `git version 2.x.x`.

**Si algo va mal**: [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) → "git no se reconoce".

---

### 3. Cuenta de GitHub (5 min, 0 si ya tienes)

Crea cuenta en https://github.com → Sign up. Si ya tienes, sigue al paso 4.

---

### 4. Clona el repo (5 min)

1. Abre VS Code.
2. Abre la terminal integrada: `Terminal → New Terminal` o `` Cmd+` `` / `` Ctrl+` ``.
3. Navega a donde quieras guardar el proyecto:
   ```sh
   cd ~/Desktop          # macOS
   cd Desktop            # Windows
   ```
4. Clona:
   ```sh
   git clone <URL del repo MDA13_ML>
   ```
   Luis te pasa la URL en el chat de la clase.
5. En VS Code: `File → Open Folder...`, selecciona `MDA13_ML`.

**Cómo sé que está bien**: en el panel izquierdo veo `data/`, `pre_class/`, `session1/`, `README.md`, etc.

---

### 5. Instala Python via Anaconda (15-25 min, la descarga es lo lento)

Recomendamos **Anaconda Distribution** porque trae Python + librerías científicas + el gestor `conda`.

- Descarga desde https://www.anaconda.com/download.
- **macOS**: instalador `.pkg`, asistente.
- **Windows**: instalador `.exe`. Cuando pregunte por PATH, di que sí.

Después de instalar, **cierra VS Code y vuelve a abrirlo** para que detecte conda.

**Cómo sé que está bien**: en la terminal integrada de VS Code:
```sh
conda --version            # devuelve "conda x.y.z"
python --version           # devuelve "Python 3.x.x"
```

**Si algo va mal**: [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) → "conda command not found". En PowerShell, te tocará `conda init powershell` una vez.

---

### 6. Crea el entorno e instala dependencias (10 min)

En la terminal integrada de VS Code, dentro de `MDA13_ML/`:

```sh
conda create -n mda13_ml python=3.12 -y
conda activate mda13_ml
pip install -r requirements.txt
```

Cuando veas `(mda13_ml)` al inicio de la línea, el entorno está activo. Cada vez que abras una terminal nueva, `conda activate mda13_ml` otra vez.

En VS Code, selecciona el intérprete: `Cmd+Shift+P` / `Ctrl+Shift+P` → `Python: Select Interpreter` → el que diga `mda13_ml`. Detalles en [`VSCODE_TIPS.md`](VSCODE_TIPS.md) sección 6.

**(macOS) Recomendado**: para que xgboost funcione del todo,

```sh
brew install libomp
```

Si no tienes brew, instálalo desde https://brew.sh. **No es bloqueante**: el notebook cae a RandomForest si xgboost falla. Pierdes un par de puntos de AUC, no más.

**Cómo sé que está bien**:
```sh
python -c "import streamlit, sklearn, statsmodels, openai; print('ok')"
```
Imprime `ok`. Si no, [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) → "ModuleNotFoundError después de pip install".

---

### 7. Crea el archivo `.env` con la API key (1 min)

Luis te pasa una key en el chat (formato `sk-...`).

En la raíz de `MDA13_ML/`, crea un archivo llamado **exactamente** `.env` (con punto al inicio, sin extensión). Click derecho sobre la carpeta del proyecto en el explorador → `New File` → `.env`.

Contenido:

```
OPENAI_API_KEY=sk-tu_clave_aqui_sin_comillas
```

Guarda con `Cmd+S` / `Ctrl+S`.

**Cómo sé que está bien**: `.env` aparece en el explorador (puede aparecer en gris porque está en .gitignore — bien).

> Si todavía no tienes la key, salta este paso. En el siguiente, usa `python setup_check.py --skip-api`.

---

### 8. Self-check + warm-up notebook (5-10 min)

Con el entorno activo:

```sh
python setup_check.py
```

Espera todo verde en las 8 secciones (con `.env` correcto). Si algo en rojo o amarillo, lee el "fix this by:" y aplícalo.

Luego ejecuta el warm-up:

1. Abre `pre_class/1_classical_models.ipynb` en VS Code.
2. Si te pide kernel, elige el de `mda13_ml`.
3. `Run All` arriba en la barra del notebook.

Tarda <3 min. Aparecen 4 archivos `.pkl` en `session1/models/`.

Vuelve a ejecutar `python setup_check.py`. La sección 7 (Trained models) ahora debe estar verde.

**Cómo sé que está bien**: setup_check todo en verde.

---

### 9. Lee el enunciado (5 min)

Abre `ENUNCIADO.md`. Click derecho dentro → `Open Preview` (`Cmd+Shift+V` / `Ctrl+Shift+V`) para verlo con formato.

Es el caso de Cañadata: contexto, datos, pregunta de negocio.

**Cómo sé que está bien**: tengo claro qué es Cañadata, qué hay en el dataset, y cuál es la pregunta del taller.

---

## ✅ Estás listo cuando...

- [ ] VS Code abre la carpeta `MDA13_ML/` y veo todos los archivos.
- [ ] Una terminal con `(mda13_ml)` al inicio.
- [ ] `python setup_check.py` todo en verde.
- [ ] `session1/models/` tiene los 4 `.pkl`.
- [ ] He leído `ENUNCIADO.md`.

---

## 🔵 Bonus track — opcional, si quieres llegar más fuerte (~30-60 min)

**Refreshers** (10 min cada uno, lee los que más te interesen):

- [`SKLEARN_INTRO.md`](SKLEARN_INTRO.md) — fit/predict/score, los 4 modelos del taller, métricas.
- [`STATSMODELS_INTRO.md`](STATSMODELS_INTRO.md) — OLS y SARIMAX.
- [`STREAMLIT_INTRO.md`](STREAMLIT_INTRO.md) — **muy recomendado** si no hiciste el taller anterior FitLife.
- [`GIT_INTRO.md`](GIT_INTRO.md) — git en 5 minutos (concepto).
- [`GIT_COMPANION.md`](GIT_COMPANION.md) — git deep dive con recetas de rescate (15 min).

**Notebooks de pre_class**:

- `pre_class/0_data_exploration.ipynb` — recorre la EDA del dataset Cañadata. Mira los `value_counts(dropna=False)` para ver el desorden plantado en los datos.
- Las secciones `🚀 Si te quedas con ganas` al final de cada notebook tienen retos opcionales para los más motivados.

**MDA12 invitees**:

Si no hiciste el taller anterior (FitLife, GenAI conversacional), repásalo: [`Kieleth/mda13-fitlife-workshop`](https://github.com/Kieleth/mda13-fitlife-workshop). En S2 nos apoyamos en el patrón text-to-code que se enseñó allí. Cuenta como bonus, no esencial.

---

## Si nada funciona

Avisa a Luis con:

1. Output completo de `python setup_check.py`.
2. Tu sistema operativo (mac / windows / versión).
3. Qué intentaste antes de pedir ayuda.

Lo que NO ayuda: "no me funciona". Cuanto más concreto, antes lo arreglamos.
