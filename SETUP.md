# Guía de instalación

**Taller MDA13 — ML Tradicional vs Enfoque LLM**

Preparación previa a la primera sesión. Necesitas completar todos los pasos de este documento ANTES de la primera clase. Tiempo estimado: 30-60 minutos según tu sistema y conexión.

> **¿Eres totalmente nuevo en esto?** Empieza por [`PRE_CLASS_CHECKLIST.md`](PRE_CLASS_CHECKLIST.md). Es la lista canónica con tiempos, criterios de éxito y a qué documento ir si te pierdes en cada paso. Este documento que estás leyendo es la referencia técnica detallada; el checklist es el camino guiado.

> **Nunca has usado terminal ni VS Code?** Lee [`TERMINAL_BASICS.md`](TERMINAL_BASICS.md) y [`VSCODE_TIPS.md`](VSCODE_TIPS.md) antes de seguir.

La configuración final del entorno en VS Code (intérprete, terminal, API key) la haremos en clase si haces falta. Aquí solo instalas los componentes y compruebas que todo funciona.

**API key del LLM**: no necesitas crear cuentas. Luis te facilitará la clave durante la sesión.

---

## Checklist rápido

- [ ] Visual Studio Code instalado
- [ ] Git instalado
- [ ] Cuenta en GitHub
- [ ] Proyecto clonado y abierto en VS Code
- [ ] Python instalado (Anaconda o estándar)
- [ ] Extensión Python en VS Code
- [ ] Extensión GitHub Copilot en VS Code
- [ ] Entorno `mda13_ml` creado y dependencias instaladas
- [ ] (macOS) `brew install libomp` para que xgboost funcione
- [ ] `python setup_check.py` muestra todos los checks en verde (incluido el ping a OpenAI)
- [ ] Archivo `.env` con tu `OPENAI_API_KEY`
- [ ] Notebook `pre_class/1_classical_models.ipynb` ejecutado de principio a fin (genera los `.pkl` en `session1/models/`)
- [ ] Enunciado leído (`ENUNCIADO.md`)

---

## 1. Visual Studio Code

Descarga desde https://code.visualstudio.com.

- **Windows**: instalador `.exe`. Marca "Añadir a PATH" si aparece.
- **Mac**: archivo `.zip`, descomprime y arrastra a Aplicaciones.

Abre VS Code una vez para confirmar que arranca.

---

## 2. Git

Sistema de control de versiones. Lo usamos para descargar el material y para actualizar entre sesiones.

> Si no has usado Git, lee primero [`GIT_INTRO.md`](GIT_INTRO.md) (5 min). No es imprescindible ahora, pero da contexto.

- **Windows**: descarga desde https://git-scm.com, ejecuta el instalador con valores por defecto.
- **Mac**: abre Terminal y escribe `git --version`. Si no está instalado, macOS te ofrecerá hacerlo automáticamente.

Verifica:

```
git --version
```

Debes ver `git version 2.x.x`. Si no, cierra y reabre la terminal.

---

## 3. Cuenta en GitHub

Si no tienes, crea una en https://github.com. La necesitarás para clonar el repo y para activar GitHub Copilot.

---

## 4. Clonar el proyecto

1. Abre VS Code.
2. Abre la terminal: `Terminal > New Terminal` (o `` Ctrl+` `` / `` Cmd+` ``). Ver [`VSCODE_TIPS.md`](VSCODE_TIPS.md) si no encuentras la terminal integrada.
3. Navega a donde quieras guardar el proyecto: `cd ~/Desktop` (Mac) o `cd Desktop` (Windows). Si `cd` te suena a chino, lee primero [`TERMINAL_BASICS.md`](TERMINAL_BASICS.md).
4. Clona:

   ```
   git clone <URL del repo MDA13_ML>
   ```

   La URL te la pasa Luis por el chat de la clase.
5. Abre la carpeta clonada en VS Code: `File > Open Folder`, selecciona `MDA13_ML`.

A partir de aquí, todo se hace desde VS Code.

---

## 5. Python

### Opción A: Anaconda Distribution (recomendado)

Incluye Python + librerías científicas + el gestor `conda`. Descarga desde https://www.anaconda.com/download.

- **Windows**: instalador `.exe`. Cuando pregunte por PATH, di que sí.
- **Mac**: instalador `.pkg`, asistente.

Ocupa ~3 GB. Si te falta espacio usa la Opción B.

### Opción B: Python estándar

Más ligero (~100 MB). Descarga desde https://www.python.org/downloads (3.11+).

- **Windows**: marca "Add Python to PATH" en la primera pantalla del instalador.
- **Mac**: ejecuta el `.pkg`.

### Verificación

En la terminal de VS Code:

```
python --version
```

Debe mostrar Python 3.10 o superior. En Mac quizá necesites `python3 --version`.

---

## 6. Extensión Python en VS Code

Barra lateral izquierda → icono de extensiones (cuatro cuadrados) → buscar "Python" → instalar la de Microsoft (`ms-python.python`).

---

## 7. GitHub Copilot

Mismo flujo: extensiones → buscar "GitHub Copilot" → instalar (`GitHub.copilot`). Te pedirá iniciar sesión con tu cuenta de GitHub. El plan gratuito sobra para el taller.

Durante el taller, Copilot estará desactivado por defecto. Lo usaremos en momentos concretos cuando Luis lo indique.

---

## 8. Crear el entorno y instalar dependencias

> **¿Qué es un entorno?** Un espacio aislado donde se instalan las librerías de un proyecto, separadas del resto de tu Python. Así dos proyectos pueden usar versiones distintas de pandas sin pelearse. Cuando un entorno está "activo", `python` y `pip` apuntan al de ese entorno. Lo verás indicado entre paréntesis al inicio de la línea: `(mda13_ml) $`.

### Con Anaconda

En la terminal de VS Code (estando en la carpeta `MDA13_ML`):

```
conda create -n mda13_ml python=3.12 -y
conda activate mda13_ml
pip install -r requirements.txt
```

Cuando veas `(mda13_ml)` al inicio de la línea, el entorno está activo. **Cada vez que abras una terminal nueva** tienes que re-activarlo: `conda activate mda13_ml`.

`pip install -r requirements.txt` instala de golpe todas las librerías que el taller necesita: streamlit, pandas, scikit-learn, xgboost, statsmodels, openai, etc. La lista completa está en [`requirements.txt`](requirements.txt).

### Con Python estándar

- **Mac/Linux**:
  ```
  python3 -m venv mda13_ml_env
  source mda13_ml_env/bin/activate
  pip install -r requirements.txt
  ```

- **Windows**:
  ```
  python -m venv mda13_ml_env
  mda13_ml_env\Scripts\activate
  pip install -r requirements.txt
  ```

---

## 9. macOS — instalar `libomp` para xgboost

xgboost necesita la librería de OpenMP para funcionar en macOS. Si no la tienes, xgboost importa pero falla al entrenar. Instálala con Homebrew:

```
brew install libomp
```

Si no tienes Homebrew, instala desde https://brew.sh (es el gestor de paquetes estándar de macOS).

> No bloqueante: el notebook de homework cae automáticamente a `RandomForestClassifier` si xgboost no funciona. El taller corre igual, pero con xgboost recuperas un par de puntos de AUC.

En **Windows y Linux** xgboost funciona sin pasos extra.

---

## 10. Verificar el entorno

Con el entorno activo:

```
python setup_check.py
```

Debes ver todos los checks en verde, **incluido el ping a OpenAI** (esto verifica que tu `.env` tiene una key válida). Si algo aparece en rojo o amarillo, lee el "fix this by:" de debajo y aplícalo.

> Si todavía no tienes la API key, ejecuta `python setup_check.py --skip-api` para saltarte el ping.

Versión Streamlit (más visual):

```
streamlit run test_app.py
```

---

## 11. Archivo `.env` con la API key

Crea un archivo llamado exactamente `.env` (con punto al inicio, sin extensión) en la raíz de `MDA13_ML/`. Contenido:

```
OPENAI_API_KEY=sk-...
```

La clave te la dará Luis en clase. **No** la subas a GitHub — el `.gitignore` ya la excluye.

---

## 12. Ejecutar el warm-up notebook

Antes de la primera sesión, abre y ejecuta `pre_class/1_classical_models.ipynb` de principio a fin. Te llevará menos de 3 minutos. Genera los cuatro modelos entrenados que S1 va a cargar:

- `session1/models/classifier.pkl`
- `session1/models/regressor.pkl`
- `session1/models/clusterer.pkl`
- `session1/models/timeseries.pkl`

Si te apetece, hojea también `pre_class/0_data_exploration.ipynb` para hacerte una idea del dataset Cañadata.

Re-ejecuta `python setup_check.py` después: el check de "Trained models" debe pasar a verde.

---

## 13. Lectura previa

Abre `ENUNCIADO.md` (clic derecho > "Open Preview" en VS Code) y léelo antes de la primera sesión. Es el caso de negocio que vamos a trabajar.

---

## Resumen — lo que debes tener antes de S1

- VS Code + Git + Python instalados
- Repo `MDA13_ML` clonado y abierto
- Entorno `mda13_ml` activo, dependencias instaladas
- (macOS) libomp instalado
- `.env` con `OPENAI_API_KEY` válida
- `python setup_check.py` todo en verde
- Notebook `pre_class/1_classical_models.ipynb` ejecutado, los `.pkl` generados
- `ENUNCIADO.md` leído

---

## Problemas frecuentes

**`conda: command not found`**: Anaconda no se instaló correctamente o la terminal no lo ve. En Windows usa Anaconda Prompt; en Mac cierra/reabre la terminal.

**`python no se reconoce`**: el entorno no está activado. Mira si ves `(mda13_ml)` al inicio. Si no, `conda activate mda13_ml`.

**`pip no se reconoce`**: con el entorno activo, prueba `python -m pip install -r requirements.txt`.

**`AuthenticationError` o `Incorrect API key`**: revisa `.env`. Sin espacios al inicio o final. Sin comillas alrededor de la key.

**`xgboost installed but cannot run` en macOS**: instala libomp con `brew install libomp`. Si no quieres, ignora — el notebook tiene fallback automático a RandomForest.

**Streamlit no abre el navegador**: copia la URL `http://localhost:8501` de la terminal y pégala manualmente.

**Anaconda ocupa demasiado**: usa la Opción B (Python estándar). Es más ligero y funciona igual.
