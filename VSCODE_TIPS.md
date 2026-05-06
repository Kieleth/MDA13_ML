# VS Code en 10 minutos

VS Code es el editor que vas a usar todo el taller. No necesitas dominarlo, solo conocer 8 cosas. Ahí van.

---

## 1. Abrir una carpeta (la primera y la más importante)

VS Code trabaja por **carpetas** (no por archivos sueltos).

- `File → Open Folder...` (en macOS también `Cmd+O`).
- Selecciona la carpeta `MDA13_ML` que clonaste.
- Si te pregunta "¿confías en los autores?", di que sí.

A partir de ahí, todo lo que está dentro de esa carpeta aparece en el panel izquierdo.

---

## 2. El explorador de archivos (panel izquierdo)

Es la barra lateral de la izquierda con los archivos y carpetas. Si no la ves, pulsa el primer icono de la barra ultra-izquierda (parecen dos hojas) o el atajo:

- macOS: `Cmd+Shift+E`
- Windows: `Ctrl+Shift+E`

Click en un archivo para abrirlo en el editor. Click derecho en una carpeta para crear archivos nuevos.

---

## 3. La terminal integrada

VS Code lleva una terminal dentro. Es donde vas a ejecutar `python`, `streamlit`, `git`...

- `Terminal → New Terminal`
- O atajo: `` Ctrl+` `` (Windows / Linux) o `` Cmd+` `` (macOS).
  > El acento grave `` ` `` está al lado de la tecla `1`, pulsando `Shift+ñ` o cerca según el layout.

La terminal abre **dentro de la carpeta del proyecto**, así que `pwd` te muestra el path de `MDA13_ML`. Es la forma más cómoda de trabajar.

Si no sabes manejarte en la terminal, lee [`TERMINAL_BASICS.md`](TERMINAL_BASICS.md) primero.

---

## 4. Guardar archivos

- macOS: `Cmd+S`
- Windows: `Ctrl+S`

Si un archivo tiene un punto blanco en la pestaña, hay cambios sin guardar. Streamlit y Jupyter NO ven los cambios hasta que guardas.

---

## 5. El Command Palette — el atajo a todo

Es el buscador de comandos. Abre cualquier funcionalidad de VS Code escribiendo su nombre.

- macOS: `Cmd+Shift+P`
- Windows: `Ctrl+Shift+P`

Casos en los que lo vas a necesitar:

- `Python: Select Interpreter` — para que VS Code use tu entorno `mda13_ml` (importante).
- `Markdown: Open Preview` — para ver `ENUNCIADO.md` o `README.md` con formato.
- `Terminal: Create New Terminal` — alternativa al atajo.

Si no recuerdas un atajo, ábrelo y escribe lo que quieres hacer en español o inglés.

---

## 6. Seleccionar el intérprete de Python

Después de instalar Anaconda y crear el entorno `mda13_ml`, VS Code tiene que saber **qué Python usar**.

1. Abre el Command Palette (`Cmd+Shift+P` / `Ctrl+Shift+P`).
2. Escribe `Python: Select Interpreter`, Enter.
3. En la lista, busca el que tiene `mda13_ml` en el nombre, algo como:
   ```
   Python 3.12.x ('mda13_ml': conda)  /Users/.../miniconda3/envs/mda13_ml/bin/python
   ```
4. Selecciónalo.

Sabrás que está bien cuando, en la barra inferior derecha, leas algo como `Python 3.12.x ('mda13_ml')`.

Si el entorno no aparece, recarga VS Code (`Cmd+Shift+P` → `Developer: Reload Window`) o cierra y vuelve a abrir.

---

## 7. Markdown Preview

Los archivos `.md` (como `ENUNCIADO.md`, `README.md`) se ven mucho mejor con formato.

- Abre el archivo.
- Click derecho dentro → `Open Preview` (o `Cmd+Shift+V` / `Ctrl+Shift+V`).
- O `Open Preview to the Side` para ver el original y la previa lado a lado.

---

## 8. Jupyter notebooks dentro de VS Code

Los archivos `.ipynb` (`pre_class/0_data_exploration.ipynb`, etc.) se abren directamente en VS Code.

Cosas a saber:

- **Ejecutar una celda**: pulsa `Shift+Enter` con el cursor en la celda. Avanza a la siguiente.
- **Ejecutar y quedarte**: `Ctrl+Enter` (mac/win).
- **Seleccionar el kernel**: arriba a la derecha del notebook, click en `Select Kernel`. Elige el de `mda13_ml`.
- Si una celda se queda colgada o quieres reiniciar el estado: `Restart` arriba en la barra del notebook.

Si las celdas no se ejecutan, revisa el kernel. Si todo está perdido, `Restart and Run All` empieza desde cero.

---

## Atajos que vale la pena memorizar

| Acción | macOS | Windows |
|---|---|---|
| Command Palette | `Cmd+Shift+P` | `Ctrl+Shift+P` |
| Nuevo terminal | `` Cmd+` `` | `` Ctrl+` `` |
| Guardar | `Cmd+S` | `Ctrl+S` |
| Buscar en archivo | `Cmd+F` | `Ctrl+F` |
| Buscar en todo el proyecto | `Cmd+Shift+F` | `Ctrl+Shift+F` |
| Abrir archivo por nombre | `Cmd+P` | `Ctrl+P` |
| Cerrar pestaña | `Cmd+W` | `Ctrl+W` |
| Comentar/descomentar línea | `Cmd+/` | `Ctrl+/` |
| Markdown preview | `Cmd+Shift+V` | `Ctrl+Shift+V` |

---

## Si algo va raro

- **Las extensiones se cargan despacio la primera vez**: paciencia los primeros 30 segundos al abrir el proyecto.
- **VS Code no detecta el entorno**: cierra y reabre. Si sigue, verifica con `conda env list` en la terminal que `mda13_ml` existe.
- **El Python interpreter cambia solo**: a veces VS Code se confunde si tienes varios entornos. Re-selecciónalo con `Python: Select Interpreter`.
- **El Markdown preview no aparece**: instala la extensión "Markdown All in One" (opcional, mejora la experiencia).
