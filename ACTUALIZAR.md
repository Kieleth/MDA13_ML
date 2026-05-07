# Cómo recibir el material de cada sesión

El taller publica el contenido de cada sesión en una rama distinta:

| Día | Rama |
|---|---|
| Pre-clase | `main` |
| 12 mayo (S1) | `session-1` |
| 13 mayo (S2) | `session-2` |
| 14 mayo (S3) | `session-3` |

Cada rama es **acumulativa**: `session-2` incluye lo de S1, `session-3` lo de S1 y S2.

---

## Camino feliz — cómo cambiar a la rama del día

En los primeros 20 minutos de cada sesión, ejecuta esto en la terminal integrada de VS Code (con el entorno `mda13_ml` activo):

```sh
git fetch
git checkout session-1     # ó session-2 / session-3 según el día
```

`git fetch` baja la lista de ramas nuevas que el profesor ha publicado. `git checkout session-N` te lleva a esa rama y aparecen los archivos nuevos en el explorador.

Confirma el resultado:

```sh
git status     # debería decir "On branch session-N"
ls session1/   # debería listar exercises/ y reference_code/
```

Si todo está en su sitio, listo.

---

## ¿Y si da error?

### "Already on 'session-N'"

No es un error. Ya estabas en esa rama. Sigue.

### "error: Your local changes to the following files would be overwritten by checkout"

Tienes archivos modificados (probablemente algún `paso_X.py` en el que estuviste experimentando) que chocan con lo que hay en la rama nueva. Solución segura — guarda tu trabajo en una rama propia y cambia limpio:

```sh
git checkout -b mi-trabajo                    # crea una rama tuya con tus cambios
git add -A && git commit -m "guardado mi trabajo" --allow-empty
git fetch
git checkout session-1
```

Tu trabajo queda guardado en la rama `mi-trabajo`. Para volver a verlo cualquier día: `git checkout mi-trabajo`. Para volver a la sesión: `git checkout session-1`.

Si la rama `mi-trabajo` ya existe (porque ya hiciste esto antes), usa otro nombre: `mi-trabajo-2`, `mi-trabajo-dia2`, lo que prefieras.

### "fatal: not a git repository"

Estás en la carpeta equivocada. Asegúrate de que VS Code tiene abierta la carpeta raíz `MDA13_ML/` (no una carpeta padre ni una subcarpeta). En la terminal: `pwd` debería terminar en `MDA13_ML`.

### "error: pathspec 'session-1' did not match any file(s) known to git"

La rama todavía no se ha publicado. Comprueba:

1. ¿Estamos ya en horario de la sesión? El profesor publica al inicio de cada bloque.
2. ¿Hiciste `git fetch` antes de `git checkout`? Sin fetch, tu copia local no sabe que existe la nueva rama.

```sh
git fetch
git branch -a              # te muestra todas las ramas locales y remotas
```

Deberías ver `remotes/origin/session-1` listada.

### Otros errores

Pega el error en el chat de la clase. Adjunta también el output de:

```sh
git status
git branch -a
```

Si todo lo demás falla, **opción nuclear** — reclona y rearranca:

```sh
cd ..
mv MDA13_ML MDA13_ML_backup_$(date +%Y%m%d)
git clone <URL del repo MDA13_ML>
cd MDA13_ML
# Recupera tu .env del backup:
cp ../MDA13_ML_backup_*/.env ./    # nota: en Windows escribe el punto a mano
# Activa el entorno y verifica:
conda activate mda13_ml
python setup_check.py
# Y vuelve al flujo del día:
git checkout session-1
```

Tu trabajo anterior queda en `MDA13_ML_backup_*`. La nueva carpeta empieza limpia desde GitHub.

---

## Si el profesor publica un fix durante la sesión

A veces hay que arreglar algo en directo. El profesor sube el cambio y te dice:

```sh
git fetch
git pull               # actualiza tu rama actual con los cambios remotos
```

Si el `pull` da conflicto porque tocaste el mismo archivo, usa la receta de "guardar tu trabajo" de arriba.

---

## Resumen en cuatro comandos

```sh
git fetch                   # ver qué ramas nuevas hay en remoto
git checkout session-1      # cambiar a la rama del día
git status                  # comprobar dónde estás
git pull                    # traer fixes de última hora
```

Eso es todo lo que vas a usar en clase.
