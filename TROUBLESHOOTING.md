# Troubleshooting — macOS y Windows

Catálogo de los problemas que SIEMPRE aparecen en este tipo de talleres, con soluciones copy-paste listas. Si `setup_check.py` te marca algo en rojo o amarillo, busca aquí primero.

---

## Índice rápido

- [macOS](#macos)
  - [`brew: command not found`](#brew-command-not-found)
  - [xgboost falla con `libomp`](#xgboost-falla-con-libomp)
  - [`conda: command not found` después de instalar](#conda-command-not-found-después-de-instalar)
  - [`zsh: command not found: python`](#zsh-command-not-found-python)
  - ["Apple cannot verify this app"](#apple-cannot-verify-this-app)
  - [Permission denied al escribir](#permission-denied-al-escribir)
- [Windows](#windows)
  - [`'python' no se reconoce`](#python-no-se-reconoce)
  - [`conda activate` falla en PowerShell](#conda-activate-falla-en-powershell)
  - [PowerShell vs CMD vs Anaconda Prompt — qué uso](#powershell-vs-cmd-vs-anaconda-prompt)
  - [Caracteres raros / `UnicodeEncodeError`](#caracteres-raros)
  - [Antivirus bloquea `pip install`](#antivirus-bloquea-pip-install)
  - [Long path support](#long-path-support)
- [Ambos sistemas](#ambos-sistemas)
  - [Streamlit no abre el navegador](#streamlit-no-abre-el-navegador)
  - [`Address already in use` (puerto 8501 ocupado)](#puerto-8501-ocupado)
  - [`ModuleNotFoundError` después de `pip install`](#modulenotfounderror-después-de-pip-install)
  - [`.env` no se carga](#env-no-se-carga)
  - [`AuthenticationError` o `Incorrect API key`](#authenticationerror)
  - [`git pull` con conflictos](#git-pull-con-conflictos)
- [Last resort: reset total](#reset-total)

---

## macOS

### brew command not found

Homebrew no está instalado o no está en el PATH.

```sh
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Cuando termine, **lee la última sección de la salida**: te dice los dos comandos que tienes que copiar y pegar para añadir brew al PATH (algo como `eval "$(/opt/homebrew/bin/brew shellenv)"`). Si no lo haces, brew sigue sin estar en el PATH.

Verifica:

```sh
brew --version
```

### xgboost falla con libomp

Síntoma: `setup_check.py` te dice "xgboost installed but cannot run (OpenMP runtime missing)". O el notebook de homework cae a RandomForest con un mensaje "libomp no disponible".

Intento 1 — Homebrew:

```sh
brew install libomp
```

Si funciona, perfecto.

**Si en Apple Silicon (M1/M2/M3) el `brew install libomp` deja un Cellar vacío** o `brew --prefix libomp` apunta a una ruta que no contiene la librería: es un bug intermitente de algunas instalaciones de Homebrew con xgboost. **No te pelees con esto durante el taller**.

**Salida fácil**: ignóralo. El notebook de homework tiene fallback automático a `RandomForestClassifier`. La diferencia es ~2 puntos de ROC-AUC (xgb ~0.86 vs RF ~0.84). Para el taller es indistinguible — todos los gates pasan, todas las comparaciones funcionan, la narrativa LLM-vs-clásico no cambia.

Si quieres arreglarlo después de la clase: prueba `brew reinstall libomp`, o instala xgboost con conda (`conda install -c conda-forge xgboost` en lugar de pip), o usa Linux/WSL.

### conda command not found después de instalar

Anaconda se instaló pero el shell todavía no lo ve. Cierra y reabre la terminal entera. Si sigue sin funcionar:

```sh
source ~/miniconda3/etc/profile.d/conda.sh    # si instalaste Miniconda
source ~/anaconda3/etc/profile.d/conda.sh     # si instalaste Anaconda
```

Para hacer que se cargue en cada terminal nueva:

```sh
~/miniconda3/bin/conda init zsh    # zsh es el shell por defecto en macOS Catalina+
~/miniconda3/bin/conda init bash   # bash si usas bash
```

Cierra y reabre la terminal. Verás `(base)` al inicio de la línea — eso significa que conda funciona.

### zsh command not found: python

En macOS reciente solo existe `python3`, no `python`. Dos opciones:

- Usa `python3` y `pip3` siempre.
- O activa el entorno conda — dentro del entorno, `python` apunta a la versión correcta:

  ```sh
  conda activate mda13_ml
  python --version
  ```

### Apple cannot verify this app

Al ejecutar un instalador descargado, macOS bloquea la primera vez. Lo lanzas haciendo:

- Click derecho sobre el `.pkg` o `.dmg` → **Open** → confirma "Open" en el diálogo.
- O en `Sistema → Privacidad y seguridad`, busca el banner sobre el archivo bloqueado y pulsa "Open Anyway".

### Permission denied al escribir

Estás trabajando en una carpeta del sistema (Desktop a veces, /Applications, etc.). Mueve el proyecto a tu home:

```sh
mv ~/Desktop/MDA13_ML ~/MDA13_ML
cd ~/MDA13_ML
```

Si es un problema de Xcode Command Line Tools:

```sh
xcode-select --install
```

---

## Windows

### 'python' no se reconoce

Mensaje: `'python' is not recognized as an internal or external command, operable program or batch file.`

Causa: durante la instalación de Python NO marcaste "Add Python to PATH". Solución más limpia:

1. Desinstala Python desde `Configuración → Aplicaciones`.
2. Re-instala desde [python.org](https://www.python.org/downloads). En la **primera pantalla** del instalador, marca **"Add Python to PATH"** antes de pulsar Install.
3. Cierra y reabre la terminal.
4. `python --version`.

Alternativa rápida (sin reinstalar): usa `py` en lugar de `python`:

```powershell
py --version
py -m pip install -r requirements.txt
py -m streamlit run test_app.py
```

`py` es un launcher de Python que Windows registra siempre, aunque `python` no esté en el PATH.

### conda activate falla en PowerShell

Mensaje: `conda activate ... no se reconoce` o `CondaError: Run 'conda init' before 'conda activate'`.

Solución, **una vez**, en PowerShell como administrador (o normal):

```powershell
conda init powershell
```

Cierra PowerShell entera y vuelve a abrir. Ahora:

```powershell
conda activate mda13_ml
```

Si te da un error sobre execution policy:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Confirma con `Y` y vuelve a intentar.

### PowerShell vs CMD vs Anaconda Prompt

Tres terminales distintas vienen con Windows o con Anaconda. Para este taller:

| Terminal | Cuándo usarla |
|---|---|
| **Anaconda Prompt** | La más segura. Conda funciona sin configurar nada. Búscala en el menú inicio. |
| **PowerShell** | Funciona si has hecho `conda init powershell` (ver arriba). Es la integrada de VS Code por defecto en Windows reciente. |
| **CMD (cmd.exe)** | Evítala. Tiene encoding pobre y peor soporte de conda. |
| **Git Bash** | Cómoda para `git` pero no la mejor para conda. |

Recomendación: **Anaconda Prompt** para todo lo de conda + dependencias. **PowerShell desde VS Code** para editar y lanzar Streamlit, una vez has hecho `conda init powershell`.

### Caracteres raros

Síntoma: `UnicodeEncodeError: 'charmap' codec can't encode character` o ves `Caï¿½adata` en lugar de `Cañadata`.

En PowerShell:

```powershell
$env:PYTHONIOENCODING = "utf-8"
```

En cmd.exe (cada vez que abras una nueva):

```cmd
chcp 65001
set PYTHONIOENCODING=utf-8
```

Para hacerlo permanente en PowerShell, añádelo a tu `$PROFILE`:

```powershell
notepad $PROFILE
# añade: $env:PYTHONIOENCODING = "utf-8"
# guarda y cierra
```

### Antivirus bloquea pip install

Síntoma: `pip install` se queda colgado o aborta a mitad. Normalmente Windows Defender, Avast, Kaspersky o similar.

- Pausa temporalmente el antivirus mientras instalas dependencias.
- O añade exclusión a la carpeta del proyecto y al `python.exe` del entorno.

### Long path support

Síntoma: error `[Errno 2] No such file or directory` en rutas profundas, o `Filename too long` al clonar.

```powershell
# Como administrador, una vez:
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
  -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

Reinicia el ordenador. Esto habilita rutas de >260 caracteres en Windows 10/11.

---

## Ambos sistemas

### Streamlit no abre el navegador

Streamlit arranca pero la pestaña del navegador no se abre sola. **No es un error**. Mira en la terminal:

```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

Copia esa URL y pégala en el navegador a mano.

### Puerto 8501 ocupado

Síntoma: `Port 8501 is already in use`.

Mata la instancia anterior:

- macOS / Linux: `lsof -ti:8501 | xargs kill -9`
- Windows (PowerShell): `Get-Process | Where-Object { $_.MainWindowTitle -match "Streamlit" } | Stop-Process`

O simplemente lanza Streamlit en otro puerto:

```sh
streamlit run session1/exercises/paso_1.py --server.port 8502
```

### ModuleNotFoundError después de pip install

Síntoma: `pip install foo` parece funcionar pero `import foo` falla.

Casi siempre significa que **instalaste en un entorno y estás corriendo desde otro**. Comprueba:

```sh
which python      # macOS / Linux
where python      # Windows
which pip         # macOS / Linux
where pip         # Windows
```

Las dos rutas tienen que estar dentro de tu entorno (`mda13_ml` o `mda13_ml_env`). Si pip apunta a uno y python a otro, no se enteran. Soluciones:

```sh
conda activate mda13_ml
python -m pip install -r requirements.txt   # forzamos pip dentro del python actual
```

`python -m pip` siempre instala en el python que estás usando, sin lugar a confusión.

### .env no se carga

Síntomas: `OPENAI_API_KEY` aparece como `None`, o `setup_check.py` dice ".env file missing".

- El archivo se llama exactamente `.env` (con punto al inicio, sin extensión).
- Está en la **raíz** del proyecto (`MDA13_ML/`), no dentro de subcarpetas.
- En Windows, comprueba que no se llama `.env.txt` por accidente. Para ver extensiones reales: `Explorador → Ver → mostrar extensiones de archivo`.
- Contenido sin comillas y sin espacios:
  ```
  OPENAI_API_KEY=sk-tu_clave_aqui
  ```
  Mal: `OPENAI_API_KEY="sk-..."` o `OPENAI_API_KEY = sk-...` (con espacios).

Verifica desde Python:

```python
from dotenv import load_dotenv
import os
load_dotenv()
print(os.environ.get("OPENAI_API_KEY"))
```

### AuthenticationError

`openai.AuthenticationError: Incorrect API key provided`. La clave tiene un problema. Causas comunes:

- Comillas alrededor del valor (`OPENAI_API_KEY="sk-..."` no funciona).
- Espacios o salto de línea al copiar.
- Clave caducada — pídele a Luis una nueva.

Re-pega desde el chat, sin comillas, sin trailing whitespace.

### git pull con conflictos

Has tocado un archivo que el profesor también ha modificado. Ver `ACTUALIZAR.md` para las recetas. Resumen:

```sh
# Guarda tu trabajo en una rama y vuelve a main limpio
git checkout -b mi-trabajo
git add -A && git commit -m "guardado mi trabajo" --allow-empty
git checkout main && git fetch origin && git reset --hard origin/main
```

---

## Reset total

Si nada funciona y tienes que volver al punto cero (10-15 min):

### macOS / Linux

```sh
# 1. Mueve la carpeta vieja por si querías recuperar algo
mv ~/MDA13_ML ~/MDA13_ML_backup_$(date +%Y%m%d)

# 2. Borra el entorno conda
conda deactivate
conda env remove -n mda13_ml

# 3. Vuelve a clonar
cd ~
git clone <URL del repo>
cd MDA13_ML

# 4. Recrea el entorno
conda create -n mda13_ml python=3.12 -y
conda activate mda13_ml
pip install -r requirements.txt

# 5. Recupera tu .env del backup
cp ~/MDA13_ML_backup_*/​.env ./    # nota: tienes que escribir el punto manualmente

# 6. Verifica
python setup_check.py
```

### Windows (PowerShell)

```powershell
# 1. Backup
Move-Item ~\MDA13_ML ~\MDA13_ML_backup

# 2. Borra el env
conda deactivate
conda env remove -n mda13_ml

# 3. Re-clona
cd ~
git clone <URL del repo>
cd MDA13_ML

# 4. Recrea
conda create -n mda13_ml python=3.12 -y
conda activate mda13_ml
pip install -r requirements.txt

# 5. Recupera .env del backup (ojo, en Windows el punto inicial no se oculta como en Unix)
Copy-Item ~\MDA13_ML_backup\.env .\

# 6. Verifica
python setup_check.py
```

---

## Si nada de esto funciona

Avisa a Luis con:

1. Sistema operativo y versión (`platform` line del output de `setup_check.py`).
2. Output completo de `setup_check.py`.
3. Qué intentaste antes de pedir ayuda.

Lo que NO ayuda: "no me funciona", "se rompe". Cuanto más concreto, antes lo arreglamos.
