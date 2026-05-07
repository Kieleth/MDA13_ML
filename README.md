# MDA13 — ML Tradicional vs Enfoque LLM

Taller práctico online de tres sesiones impartido por **Luis Guzmán** (ISDI, Master Data Analysis & AI). Construirás un dashboard en Streamlit que sirve cuatro modelos de ML clásicos sobre un dataset B2B sintético, y los compararás contra varios enfoques LLM. Al final tendrás criterio para decidir qué desplegarías en producción y por qué.

**Fechas**: 12, 13 y 14 de mayo, online por Zoom. Detalles en la sección _Las tres sesiones_ más abajo.

---

## Cómo conseguir este repo en tu ordenador

Tienes que tener este código en local **antes** de la primera sesión. Cuatro caminos según tu nivel de comodidad. Empieza por el A, baja si te atascas.

### A · Con VS Code (recomendado, sin tocar terminal)

1. Instala **VS Code** desde https://code.visualstudio.com.
2. Instala **Git** desde https://git-scm.com (Windows) o ejecutando `git --version` en Terminal (macOS, te ofrece instalarlo).
3. Abre VS Code.
4. Abre el Command Palette: `Cmd+Shift+P` (Mac) o `Ctrl+Shift+P` (Windows).
5. Escribe `Git: Clone`, pulsa Enter.
6. Pega la URL del repo (te la pasa Luis por el chat) y elige una carpeta donde guardarlo (por ejemplo, tu Escritorio).
7. Cuando termine, VS Code te pregunta si quieres abrir la carpeta clonada. Di que sí.

Listo. Ya tienes el proyecto abierto. Si VS Code te suena a chino, lee [`VSCODE_TIPS.md`](VSCODE_TIPS.md) (10 min).

### B · Terminal en macOS

1. Abre la app **Terminal** (Spotlight: `Cmd+Espacio`, escribe "Terminal", Enter).
2. Navega a donde quieras guardar el proyecto:
   ```sh
   cd ~/Desktop
   ```
3. Clona:
   ```sh
   git clone <URL del repo>
   ```
   (URL la pasa Luis por el chat de la clase.)
4. Entra en la carpeta y ábrela en VS Code:
   ```sh
   cd MDA13_ML
   code .
   ```
   Si `code .` no funciona, abre VS Code a mano y `File → Open Folder`.

Si `cd` o `git` te resultan extraños, lee [`TERMINAL_BASICS.md`](TERMINAL_BASICS.md) (10 min).

### C · Terminal en Windows (Anaconda Prompt)

1. Instala **Git** desde https://git-scm.com con valores por defecto.
2. Instala **Anaconda Distribution** desde https://www.anaconda.com/download (lo necesitarás de todas formas para Python).
3. Abre **Anaconda Prompt** (busca en el menú inicio).
4. Navega y clona:
   ```cmd
   cd Desktop
   git clone <URL del repo>
   cd MDA13_ML
   code .
   ```

Para qué shell usar y qué hacer si algo se queja: [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md).

### D · Sin git, fallback ZIP

Solo si A, B y C te dan problemas serios:

1. Ve al repo en GitHub (la URL que te pasó Luis).
2. Botón verde **`Code`** → **Download ZIP**.
3. Descomprime en tu ordenador.
4. Abre la carpeta resultante en VS Code.

⚠ **Caveat**: sin git no podrás hacer `git pull` para recibir actualizaciones. El día 1 te tocará volver a descargar el ZIP cuando publiquemos el material de la sesión. Recomendado solo como último recurso.

---

## Después de clonar

Abre la carpeta `MDA13_ML` en VS Code y abre [`PRE_CLASS_CHECKLIST.md`](PRE_CLASS_CHECKLIST.md). Es la **única lista** que necesitas. Te lleva paso a paso desde aquí hasta llegar a clase con todo listo.

---

## Presupuesto de tiempo antes de la clase

| Pista | Tiempo | Quién | Obligatorio |
|---|---|---|---|
| 🟢 **Esencial** | ~90 min (rango 60-150) | **Todos** | **Sí, antes del 12 mayo** |
| 🟡 Si eres nuevo | +20 min | Si nunca usaste terminal o VS Code | Recomendado |
| 🔵 Bonus | +30-60 min | Si quieres llegar más fuerte | No |

**Techo total**: 3h para alguien arrancando de cero. **Suelo**: ~60 min con experiencia previa. Si te acercas a 3h, **salta el bonus** y entrega lo esencial.

---

## Las tres sesiones de un vistazo

| Sesión | Fecha | 19:00–21:30 | 21:30–22:00 |
|---|---|---|---|
| S1 | 12 May (mar) | Luis · Streamlit + 4 modelos clásicos | Miguel · automatización |
| S2 | 13 May (mié) | Luis · tres modos LLM sobre el dashboard | Miguel · automatización |
| S3 | 14 May (jue) | Luis · comparación + ship-it (hasta 20:30) | Miguel · cierre |

Q&A es fluido durante el slot de Luis. Resolvemos las dudas a medida que aparecen.

### Cadencia por sesión

Cada sesión se divide en tres bloques con la misma forma:

1. **Juntos** — yo escribo código, tú lo escribes a la vez. Huecos pequeños.
2. **Reto guiado** — huecos más grandes, te dejo intentarlo, paso por las pantallas explicando.
3. **Final independiente** — terminas tú, yo aparezco si me llamas, y al final enseño cómo debería quedar.

Cada ejercicio termina con una sección `🚀 Si te quedas con ganas` con retos opcionales para los más motivados.

---

## El día de cada sesión — cambiar de rama para recibir el material

Este repo arranca solo con el **kit de pre-clase** en la rama `main` (lo que necesitas para llegar listo). El material de cada sesión vive en una rama distinta y se publica el día que toca:

| Día | Rama | Contenido |
|---|---|---|
| Pre-clase (ahora) | `main` | Sólo el kit de pre-clase |
| 12 mayo · S1 | `session-1` | Pre-clase + ejercicios S1 |
| 13 mayo · S2 | `session-2` | Pre-clase + S1 + ejercicios S2 |
| 14 mayo · S3 | `session-3` | Pre-clase + S1 + S2 + ejercicios S3 |

En los primeros 20 minutos de cada sesión, te pediremos que cambies de rama:

```sh
git fetch
git checkout session-1     # ó session-2 / session-3 según el día
```

Y aparecerán las carpetas `session1/exercises/`, etc. con los ejercicios del día. Las ramas son **acumulativas**: cuando llegas a `session-3` tienes todo el contenido del taller.

Si te da problemas (cambios locales sin guardar, conflictos), las recetas de rescate están en [`ACTUALIZAR.md`](ACTUALIZAR.md).

> ¿Por qué ramas y no `git pull`? Dos motivos: 1) que no veas el material de los siguientes días por adelantado — la narrativa funciona mejor descubriendo en clase. 2) que aprendas a usar `git checkout` y vivas las ramas como herramienta. Es uno de los conceptos clave de Git.

---

## Estructura del repo (lo que verás al abrirlo)

| Archivo / carpeta | Para qué |
|---|---|
| `README.md` | Esto que estás leyendo |
| `PRE_CLASS_CHECKLIST.md` | **Empieza aquí.** Camino guiado paso a paso. |
| `SETUP.md` | Instalación detallada (referencia técnica) |
| `TERMINAL_BASICS.md`, `VSCODE_TIPS.md` | 🟡 Lo mínimo de terminal y VS Code para newbies |
| `TROUBLESHOOTING.md` | Pain points por plataforma (macOS / Windows) con soluciones copy-paste |
| `ENUNCIADO.md` | Caso de Cañadata: contexto, datos, pregunta de negocio |
| `GIT_INTRO.md`, `GIT_COMPANION.md` | Git en 5 / 15 minutos |
| `ACTUALIZAR.md` | Cómo hacer `git pull` y qué hacer cuando da error |
| `SKLEARN_INTRO.md`, `STATSMODELS_INTRO.md`, `STREAMLIT_INTRO.md` | 🔵 Refreshers de las librerías que usamos |
| `requirements.txt` | Dependencias de Python |
| `setup_check.py`, `test_app.py` | Verificación del entorno (CLI / Streamlit) |
| `data/` | Dataset Cañadata (CSVs sintéticos) |
| `pre_class/` | Notebooks de preparación: EDA + entrena los 4 modelos |

---

## Stack

- **VS Code** (editor)
- **Python 3.12** + entorno conda `mda13_ml`
- **Streamlit** (frontend del dashboard)
- **pandas, scikit-learn, xgboost, statsmodels, joblib** (ML)
- **OpenAI API**, modelo `gpt-4.1-mini` (LLM)

Detalle completo en [`SETUP.md`](SETUP.md).

---

## Filosofía de diseño

- **Cobertura sobre profundidad**: cuatro familias de ML en pocas horas. Cada una hasta donde da el tiempo.
- **El LLM no sustituye, complementa**: no es "LLM > sklearn" ni al revés. Es saber elegir.
- **Cero magia**: si no entiendes una línea, paramos.
- **Easy floor + deep stretch**: si te aburre el paso central, los 🚀 están ahí para profundizar.

---

## Si tienes dudas

Durante la pre-clase: pregunta a Luis por el chat de la edición. Para problemas técnicos, siempre incluye:

1. El output completo de `python setup_check.py`.
2. Tu sistema operativo y versión.
3. Qué intentaste antes de pedir ayuda.

Cuanto más concreto, antes lo arreglamos. "No me funciona" no ayuda; "el paso 8 me devuelve este error en Anaconda Prompt en Windows 11" sí.

Durante la clase: pregunta cuando quieras. Q&A fluido.

Nos vemos el 12 de mayo.
