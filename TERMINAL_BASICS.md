# La terminal en 10 minutos

Si nunca has abierto una terminal, este es tu sitio. En 10 minutos sabrás abrir una, moverte por carpetas y ejecutar comandos. Es una herramienta, no magia.

---

## Qué es la terminal

Una **terminal** es una ventana donde tecleas comandos en lugar de hacer clic con el ratón. Todo lo que haces con la terminal también se puede hacer con botones, pero la terminal es:

- más rápida cuando sabes qué escribir,
- la única forma de hacer ciertas cosas (instalar librerías de Python, lanzar Streamlit, usar Git).

No vas a aprenderla a fondo. Vas a aprender 8 comandos. Es lo único que necesitas para el taller.

---

## Cómo abrir una terminal

### En macOS

Tres maneras, en orden de preferencia:

1. **Dentro de VS Code** (lo más cómodo cuando trabajas en el proyecto):
   - Abre VS Code, abre la carpeta del proyecto.
   - Menú `Terminal → New Terminal`, o atajo `` Ctrl+` ``.
   - Aparece un panel abajo con una línea esperando comandos.

2. **App Terminal**:
   - Pulsa `Cmd+Espacio`, escribe "Terminal", Enter.
   - Aparece una ventana negra con un cursor.

3. **iTerm2** (si lo tienes — opcional, mejor que Terminal pero requiere instalación).

### En Windows

Tres terminales distintas vienen con Windows o con Anaconda. Recomendaciones:

| Terminal | Cuándo usarla |
|---|---|
| **Anaconda Prompt** | **Recomendada** para todo lo de conda/dependencias. Búscala en el menú inicio tras instalar Anaconda. |
| **Dentro de VS Code** | Cómodo cuando ya estás en el proyecto. Menú `Terminal → New Terminal`. Si pregunta qué shell, elige PowerShell. |
| **PowerShell** | Funciona si has ejecutado `conda init powershell` una vez. |
| **CMD (cmd.exe)** | Evítala. Encoding pobre, peor soporte de conda. |

Para este taller, **abre la Anaconda Prompt** para los pasos de instalación, y luego usa **la terminal integrada de VS Code** para el día a día.

---

## El modelo mental: dónde estás

La terminal siempre está **dentro de una carpeta**. Esa carpeta se llama el "directorio actual" o "working directory".

Cada vez que ejecutas un comando, se ejecuta **desde ahí**. Si haces `streamlit run paso_1.py`, Streamlit busca `paso_1.py` en el directorio actual. Si no estás en la carpeta correcta, el comando falla con `No such file or directory`.

### El home folder y los paths

Tu **home folder** es tu carpeta personal. En macOS suele ser `/Users/tu_nombre`. En Windows, `C:\Users\tu_nombre`. La terminal abre por defecto ahí.

El símbolo `~` es un atajo que significa "mi home folder". Estos dos comandos son equivalentes:

```sh
cd /Users/luis/Desktop          # macOS, ruta completa
cd ~/Desktop                    # macOS, atajo equivalente
```

Una **ruta** es la dirección de un archivo o carpeta. Puede ser:

- **Absoluta** (empieza desde la raíz): `/Users/luis/Projects/MDA13_ML` (mac) o `C:\Users\luis\Projects\MDA13_ML` (Windows).
- **Relativa** (empieza desde donde estás): `data/canadata_leads.csv` significa "la carpeta `data` que está aquí, dentro su archivo `canadata_leads.csv`".

`..` significa "subir un nivel". Si estás en `MDA13_ML/data/` y haces `cd ..`, vas a `MDA13_ML/`.

---

## Los 8 comandos esenciales

### 1. `pwd` — ¿dónde estoy?

```sh
pwd
# /Users/luis/Projects/MDA13_ML       (mac)
# C:\Users\luis\Projects\MDA13_ML     (windows)
```

En Windows si `pwd` no funciona, prueba `cd` solo (sin argumentos).

### 2. `ls` (mac) / `dir` (windows) — qué hay aquí

```sh
ls            # mac/linux
dir           # windows
```

Ves los archivos y carpetas del directorio actual. En macOS:

```
data/  pre_class/  README.md  SETUP.md  ...
```

### 3. `cd carpeta` — entrar en una carpeta

```sh
cd Desktop                # entra en Desktop (relativo)
cd ~/Projects             # entra en ~/Projects (absoluto con atajo)
cd C:\Users\luis\Projects # entra a una ruta absoluta en Windows
```

Si la carpeta tiene espacios, ponla entre comillas: `cd "Mi Carpeta"`.

### 4. `cd ..` — subir un nivel

```sh
cd ..        # sube una carpeta
cd ../..     # sube dos
```

### 5. `cd ~` o `cd` solo — volver a home

```sh
cd ~         # mac/linux
cd           # windows o mac/linux, vuelve a home
```

### 6. Tab — autocompletar

Empieza a escribir un nombre y pulsa **Tab**. La terminal lo completa por ti. Si hay varias opciones, pulsa Tab dos veces para ver la lista.

```sh
cd Pro       # luego Tab → cd Projects/
cd Projects/MDA  # luego Tab → cd Projects/MDA13_ML/
```

Tab te ahorra escribir y te confirma que el path existe. Úsalo siempre.

### 7. Flecha arriba — historial

Pulsa la flecha arriba y reaparecen los comandos que ya ejecutaste. Es la forma rápida de repetir o corregir algo.

### 8. `Ctrl+C` — interrumpir, `clear` / `cls` — limpiar

- `Ctrl+C` para el comando que está ejecutándose. Lo necesitas para parar Streamlit (`streamlit run ...`).
- `clear` (mac/linux) o `cls` (windows) limpia la pantalla.

---

## Errores que vas a ver

### `No such file or directory` / `cannot find the path`

Estás intentando entrar en una carpeta que no existe en el directorio actual. Comprueba con `pwd` y `ls`/`dir` dónde estás. Casi siempre es:

- la carpeta tiene otro nombre (mayúsculas/minúsculas importan en mac/linux),
- o estás en un directorio padre / hijo del que crees.

### `command not found` / `'foo' no se reconoce como un comando`

El programa que invocas no está en el PATH. Causas típicas:

- te falta activar el entorno de conda: `conda activate mda13_ml`,
- no instalaste el programa,
- está instalado pero la terminal no se ha refrescado: cierra y reabre la terminal.

### `Permission denied`

Estás intentando algo que el sistema no te deja. En el taller esto pasa raramente. Si pasa, revisa que estás en una carpeta tuya (no en `/System` o `/Program Files`) y vuelve a intentar.

---

## Mini-ejercicio (5 min, para fijarlo)

Abre una terminal y ejecuta esto, comprobando con `pwd` después de cada `cd`:

```sh
pwd                    # ¿dónde estás?
cd ~                   # vuelve a home
pwd                    # confirma
ls                     # mac/linux | dir en windows
mkdir mda13_test       # crea una carpeta de prueba
cd mda13_test          # entra
pwd                    # confirma que estás dentro
cd ..                  # sale
pwd                    # confirma que saliste
rmdir mda13_test       # borra la carpeta vacía (ojo: NO usar rm -rf sin saber)
```

Si todo funcionó, ya tienes lo que necesitas para seguir [`SETUP.md`](SETUP.md).

---

## Para el taller: la jugada clásica

Cada vez que vuelvas a trabajar en el proyecto:

```sh
# 1. Abre VS Code en la carpeta del proyecto
# 2. Abre la terminal integrada (Ctrl+` o Cmd+`)
# 3. Activa el entorno
conda activate mda13_ml

# 4. Verifica que estás en la raíz del proyecto
pwd                                # debería terminar en MDA13_ML
ls                                 # ves data/, pre_class/, session1/, etc.

# 5. Lanza lo que toque, por ejemplo:
streamlit run session1/exercises/paso_1.py
```

Cuando ves `(mda13_ml)` al inicio de la línea, el entorno está activo. Si no lo ves, vuelve a hacer `conda activate mda13_ml`.

---

## Para ir más allá (opcional)

- [The Missing Semester (MIT)](https://missing.csail.mit.edu/) — curso gratis con todo lo que la universidad nunca te enseñó sobre la línea de comandos.
- En macOS: la app Terminal usa `zsh` por defecto desde Catalina. Funciona casi igual que `bash`.
- En Windows: si te enganchas, mira [WSL](https://learn.microsoft.com/en-us/windows/wsl/) — te da un Linux dentro de Windows.
