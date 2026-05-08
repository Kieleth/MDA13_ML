# Git Companion — De cero a manejarte

Esta guía tiene dos partes. La primera empieza desde cero absoluto. La segunda te lleva un paso más allá. No hace falta hacer las dos de golpe.

---

# PARTE A — Tu primer repositorio

## Qué es Git (sin rodeos)

Cuando trabajas en un documento y quieres poder volver atrás, haces copias:

```
informe_final.docx
informe_final_v2.docx
informe_final_DEFINITIVO.docx
informe_final_DEFINITIVO_bueno.docx
```

Git hace lo mismo pero de forma inteligente. En vez de copiar archivos enteros, Git guarda **los cambios** que haces. Cada vez que le dices "guárdame esto", Git toma una foto del estado actual de todos tus archivos. Esa foto se llama **commit**.

La diferencia con hacer copias manuales:
- No tienes 20 versiones del mismo archivo
- Puedes ver exactamente qué cambiaste y cuándo
- Puedes volver a cualquier punto anterior con un clic
- Funciona con todos los archivos de un proyecto a la vez, no uno por uno

---

## Paso 1: Crea una carpeta y activa Git

Abre la terminal en VS Code y escribe:

```bash
mkdir mi-proyecto
cd mi-proyecto
git init
```

`mkdir` crea una carpeta. `cd` entra en ella. `git init` le dice a Git: "vigila esta carpeta."

No vas a ver nada especial. Git trabaja en silencio. Pero ahora todo lo que pase en esta carpeta queda registrado.

---

## Paso 2: Crea un archivo y mira qué pasa

Crea un archivo. Puedes hacerlo desde VS Code (File > New File) o desde la terminal:

```bash
echo "Hola, este es mi primer archivo" > notas.txt
```

Ahora pregúntale a Git qué ve:

```bash
git status
```

Git te dice algo como:

```
Untracked files:
    notas.txt
```

**Untracked** significa "veo este archivo, pero no lo estoy siguiendo todavía." Git sabe que `notas.txt` existe, pero no le has dicho que lo vigile.

---

## Paso 3: Dile a Git que vigile el archivo (staging)

```bash
git add notas.txt
```

Ahora `git status` dice:

```
Changes to be committed:
    new file: notas.txt
```

Has pasado el archivo a la **zona de preparación** (staging area). Es como poner cosas en una caja antes de sellarla. Todavía no has guardado nada — solo has dicho "esto va a ir en la próxima foto."

**¿Por qué no guarda directamente?** Porque a veces cambias 5 archivos pero solo quieres guardar 3. El staging te deja elegir qué entra en cada foto.

---

## Paso 4: Guarda la foto (commit)

```bash
git commit -m "Primer archivo del proyecto"
```

Hecho. Git ha tomado una foto de tu proyecto. El texto entre comillas es el **mensaje del commit** — una nota para tu yo del futuro que explica qué hiciste.

Comprueba:

```bash
git log
```

Verás algo como:

```
commit a1b2c3d4... (HEAD -> main)
Author: Tu Nombre <tu@email.com>
Date:   Wed Mar 4 2026

    Primer archivo del proyecto
```

Esa es tu primera foto. Puedes volver a ella en cualquier momento.

---

## Paso 5: Haz cambios y observa

Edita `notas.txt` — añade una línea, cambia algo, lo que quieras. Luego:

```bash
git status
```

```
Changes not staged for commit:
    modified: notas.txt
```

Git sabe que el archivo cambió. Si quieres ver exactamente qué cambiaste:

```bash
git diff
```

Verás las líneas antiguas (en rojo, con `-`) y las nuevas (en verde, con `+`). Esto es lo que hace Git especial: no solo sabe QUE cambió un archivo — sabe exactamente QUÉ líneas cambiaron.

Para guardar estos cambios:

```bash
git add notas.txt
git commit -m "Añadida segunda línea a notas"
```

Ahora tienes dos fotos. Puedes ver las dos con `git log`.

---

## Paso 6: Varios archivos a la vez

Crea dos archivos más:

```bash
echo "Lista de tareas" > tareas.txt
echo "Ideas para el proyecto" > ideas.txt
```

```bash
git status
```

```
Untracked files:
    ideas.txt
    tareas.txt
```

Puedes añadir los dos a la vez:

```bash
git add .
```

El punto (`.`) significa "todo lo que hay en esta carpeta." Es un atajo para no escribir cada archivo por separado.

```bash
git commit -m "Añadidos archivos de tareas e ideas"
```

Ahora tu historial tiene 3 fotos:
1. El archivo inicial
2. La edición
3. Los dos archivos nuevos

---

## Paso 7: Vuelve atrás (sin miedo)

Imagina que editas `notas.txt` y lo dejas peor de cómo estaba. Antes de hacer commit, puedes deshacerlo:

```bash
git checkout -- notas.txt
```

Eso restaura el archivo al estado del último commit. Los cambios que hiciste desaparecen. Es como pulsar Ctrl+Z pero para toda la sesión de edición.

**Importante:** esto solo funciona ANTES de hacer commit. Una vez que haces commit, el cambio queda en la foto. Pero puedes volver a fotos anteriores (eso lo vemos en la parte B).

---

## Resumen de la Parte A

Has aprendido el flujo básico de Git:

```
Editar archivos
    ↓
git status         (ver qué cambió)
    ↓
git diff           (ver los cambios exactos)
    ↓
git add .          (preparar los cambios)
    ↓
git commit -m ""   (guardar la foto)
    ↓
git log            (ver el historial de fotos)
```

Con estos comandos puedes trabajar en cualquier proyecto y tener siempre un historial de todo lo que has hecho. Si algo se rompe, puedes investigar qué cambió y cuándo.

---

---

# PARTE B — Ramas y trabajo en paralelo

## La idea de las ramas

Hasta ahora has trabajado en una sola línea: haces cambios, los guardas, sigues. Esa línea se llama `main` (la rama principal).

Pero a veces quieres probar algo sin arriesgar lo que ya funciona. Por ejemplo:

- "Quiero cambiar el diseño de la app, pero no sé si quedará bien"
- "Quiero probar otro modelo de LLM, pero si no funciona quiero volver"
- "Quiero reorganizar todo el código, pero si la lío necesito poder deshacer"

Para eso existen las **ramas**. Una rama es una copia de tu proyecto donde puedes experimentar libremente. Si el experimento sale bien, lo unes a `main`. Si sale mal, la borras y ya está.

Imagina un camino que se bifurca:

```
main:          A --- B --- C
                          \
experimento:               D --- E
```

`A`, `B`, `C` son commits en main. En el punto `C` creas una rama y haces los commits `D` y `E` ahí. `main` sigue en `C`, intacto. Si `D` y `E` salen bien, los unes a `main`. Si no, los descartas.

---

## Paso 1: Crea una rama

Desde tu proyecto (el de la Parte A o cualquier otro):

```bash
git branch experimento
```

Esto crea la rama pero no te mueve a ella. Para moverte:

```bash
git checkout experimento
```

O, en un solo paso (crear + moverte):

```bash
git checkout -b experimento
```

Para comprobar en qué rama estás:

```bash
git branch
```

Verás algo como:

```
* experimento
  main
```

El asterisco indica dónde estás.

---

## Paso 2: Trabaja en la rama

Ahora estás en la rama `experimento`. Todo lo que hagas aquí NO afecta a `main`.

Haz algún cambio:

```bash
echo "Esta es una idea experimental" > experimento.txt
git add .
git commit -m "Probando una idea nueva"
```

Edita también un archivo existente:

```bash
echo "Línea añadida desde la rama experimento" >> notas.txt
git add .
git commit -m "Modificado notas desde experimento"
```

---

## Paso 3: Vuelve a main y mira

```bash
git checkout main
```

Ahora mira tus archivos:
- `experimento.txt` **no existe** — solo vive en la rama `experimento`
- `notas.txt` **no tiene la línea que añadiste** — esa línea está en la otra rama

Esto es lo más importante de las ramas: **son mundos paralelos**. Lo que haces en uno no existe en el otro hasta que decides unirlos.

---

## Paso 4: Une la rama a main (merge)

Tu experimento funcionó. Quieres incorporar esos cambios a `main`.

Primero, asegúrate de estar en `main`:

```bash
git checkout main
```

Luego, une la otra rama:

```bash
git merge experimento
```

Si todo va bien, verás algo como:

```
Updating c3d4e5f..a1b2c3d
Fast-forward
 experimento.txt | 1 +
 notas.txt       | 1 +
 2 files changed, 2 insertions(+)
 create mode 100644 experimento.txt
```

Ahora `main` tiene todos los cambios de `experimento`. Ya no necesitas la rama:

```bash
git branch -d experimento
```

---

## Paso 5: Cuando las ramas chocan (conflictos)

A veces dos ramas modifican la misma línea del mismo archivo. Git no sabe cuál de las dos versiones elegir. Eso se llama un **conflicto**.

Vamos a provocar uno a propósito para que pierdas el miedo.

### Crea el escenario

```bash
# Asegúrate de estar en main
git checkout main

# Edita notas.txt: cambia la primera línea
# (abre el archivo y escribe "Versión de main" en la primera línea)
git add .
git commit -m "Editado notas en main"

# Crea una rama y edita la MISMA línea
git checkout -b otra-idea

# Cambia la primera línea a "Versión de otra-idea"
git add .
git commit -m "Editado notas en otra-idea"

# Vuelve a main e intenta unir
git checkout main
git merge otra-idea
```

### Git te avisa del conflicto

```
CONFLICT (content): Merge conflict in notas.txt
Automatic merge failed; fix conflicts and then commit the result.
```

No pasa nada. No se ha roto nada. Git simplemente te dice: "Hay dos versiones de la misma línea y no sé cuál quieres. Decídelo tú."

### Cómo se ve un conflicto

Abre `notas.txt`. Verás algo así:

```
<<<<<<< HEAD
Versión de main
=======
Versión de otra-idea
>>>>>>> otra-idea
```

Las marcas son de Git:
- Lo que hay entre `<<<<<<< HEAD` y `=======` es la versión de `main` (donde estás ahora)
- Lo que hay entre `=======` y `>>>>>>> otra-idea` es la versión de la otra rama

### Resuelve el conflicto

Tú decides con qué quedarte. Tienes tres opciones:

**Opción A: Quédate con la versión de main.**
Borra todo lo de la otra rama y las marcas de Git. El archivo queda:
```
Versión de main
```

**Opción B: Quédate con la versión de la otra rama.**
Borra todo lo de main y las marcas. El archivo queda:
```
Versión de otra-idea
```

**Opción C: Combina las dos.**
Escribe lo que quieras. El archivo queda:
```
Versión combinada de main y otra-idea
```

Lo importante: **borra las líneas con `<<<<<<<`, `=======`, y `>>>>>>>`**. Esas son marcas de Git, no contenido real.

### Guarda la resolución

```bash
git add notas.txt
git commit -m "Resuelto conflicto en notas"
```

Hecho. El conflicto está resuelto. La rama está unida.

---

## Casos reales: cuándo usar ramas

### Caso 1: Probar un modelo diferente

```bash
git checkout -b probar-gpt4
# Cambias el modelo en tu app de gpt-4.1-mini a gpt-4.1
# Pruebas varias preguntas, comparas resultados
# Si va mejor:
git checkout main
git merge probar-gpt4
# Si va peor:
git checkout main
git branch -D probar-gpt4    # la D mayúscula fuerza el borrado
```

### Caso 2: Cambiar el prompt sin romper lo que funciona

```bash
git checkout -b nuevo-prompt
# Experimentas con el prompt del sistema
# Pruebas, ajustas, pruebas más
# Cuando estás contento:
git checkout main
git merge nuevo-prompt
```

### Caso 3: Descargar actualizaciones del taller sin perder tus cambios

```bash
# Guarda tus cambios actuales
git add .
git commit -m "Mi trabajo hasta ahora"

# Descarga las actualizaciones
git pull
```

Si hay conflictos (porque tú cambiaste un archivo que el profesor también cambió), Git te avisa y los resuelves como vimos arriba.

---

## Referencia rápida

### Comandos del día a día

| Comando | Qué hace |
|---|---|
| `git status` | Ver qué ha cambiado |
| `git diff` | Ver las líneas exactas que cambiaron |
| `git add .` | Preparar todos los cambios |
| `git add archivo.py` | Preparar solo un archivo |
| `git commit -m "mensaje"` | Guardar una foto |
| `git log` | Ver el historial de fotos |
| `git log --oneline` | Historial compacto (una línea por commit) |

### Comandos de ramas

| Comando | Qué hace |
|---|---|
| `git branch` | Ver en qué rama estás |
| `git branch nombre` | Crear una rama |
| `git checkout nombre` | Moverte a una rama |
| `git checkout -b nombre` | Crear una rama y moverte a ella |
| `git merge nombre` | Unir una rama a la actual |
| `git branch -d nombre` | Borrar una rama (ya unida) |
| `git branch -D nombre` | Borrar una rama (sin unir, forzado) |

### Cuando algo va mal

| Situación | Qué hacer |
|---|---|
| "He cambiado un archivo y quiero deshacerlo" | `git checkout -- archivo.txt` |
| "He hecho `git add` pero no quiero incluir ese archivo" | `git reset archivo.txt` |
| "Hay un conflicto y no sé qué hacer" | Abre el archivo, busca `<<<<<<<`, decide qué versión quieres, borra las marcas, `git add` y `git commit` |
| "Todo está roto y quiero volver al último commit" | `git checkout -- .` (deshace TODOS los cambios no commiteados) |
| "Quiero ver cómo estaba un archivo en un commit anterior" | `git show HEAD~1:archivo.txt` (1 commit atrás, 2 para dos, etc.) |

---

## Resumen en una frase

Git guarda fotos de tu proyecto (commits). Las ramas te dejan experimentar sin romper nada. Los conflictos se resuelven eligiendo qué versión quieres. Con 10 comandos te manejas en el 99% de las situaciones.

---

---

# PARTE C — Rescate: cuando git pull sale mal

Cada sesión os pediré que cambiéis de rama (`git checkout session-N`) para descargar el material nuevo. A veces eso choca con cambios que habéis hecho en vuestros archivos. Aquí tenéis las recetas para salir de cualquier lío.

---

## Si todavía no habéis hecho el checkout

Antes de nada, vamos a guardar vuestro trabajo en un sitio seguro y luego descargar lo nuevo. Copiad estas 3 líneas en la terminal, una por una:

```
git checkout -b mi-trabajo
```

```
git add -A && git commit -m "guardado mi trabajo" --allow-empty
```

```
git checkout main && git fetch origin && git checkout session-1
```

Qué acaba de pasar:
- La primera línea crea una rama llamada `mi-trabajo` con todo lo que tenéis ahora. Vuestros archivos no se pierden — están ahí.
- La segunda guarda todo en esa rama.
- La tercera vuelve a `main`, descarga las novedades de GitHub, y os mueve a la rama de la sesión correspondiente. Limpio, sin conflictos, sin dramas.

Si os dice `branch 'mi-trabajo' already exists`, usad otro nombre: `mi-trabajo-2`, `mi-trabajo-sesion3`, lo que sea.

---

## Si ya habéis hecho git pull y estáis en medio de un conflicto

Sabéis que estáis en un conflicto si Git os ha dicho `CONFLICT`, o si al abrir un archivo veis líneas raras con `<<<<<<<` y `>>>>>>>`, o si `git status` dice `Unmerged paths`.

Primero, cancelad el merge que se ha quedado a medias:

```
git merge --abort
```

Eso os devuelve al estado de antes del pull. Ahora ya podéis hacer el proceso normal:

```
git checkout -b mi-trabajo
```

```
git add -A && git commit -m "guardado mi trabajo" --allow-empty
```

```
git checkout main && git fetch origin && git checkout session-1
```

---

## Si nada funciona

Opción nuclear. Funciona siempre. Renombrad vuestra carpeta y descargad el proyecto de cero:

```
cd ..
```

```
mv MDA13_ML MDA13_ML-backup
```

```
git clone <URL del repo>
```

```
cd MDA13_ML
```

Vuestro trabajo anterior está en la carpeta `MDA13_ML-backup`. La nueva carpeta es una copia limpia de GitHub.

Después de esto, recordad activar el entorno:

```
conda activate mda13_ml
```

Y verificar que todo funciona:

```
python setup_check.py
```
