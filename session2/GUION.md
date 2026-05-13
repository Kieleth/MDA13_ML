# Guion S2 — los 5 momentos pedagógicos

> Mapa corto. La maquinaria está en los `.py`; aquí está cuándo dispararla.
> Hermano mayor de `TEACHER_GUIDE_S2.md` (que es la versión larga).

---

## Momento 1 · 19:15 — Sondeo (antes de tocar código)

**Comando**:
```sh
python -c "
import pandas as pd
df = pd.read_csv('data/canadata_leads_clean.csv')
lead = df.iloc[0]
print('Descripción:', lead['company_description'])
print('--- realidad oculta ---')
print('Convirtió:', lead['converted'], '| Arquetipo:', lead['lead_segment_truth'])
"
```

**Lo que dices**: "Vosotros. Cada uno. 0-100, ¿este lead convierte? Pegadme un número en el chat. Sin pensar."

**Esperado**: 4-5 números en el chat. Anota la media. Después al lanzar `paso_4.py`, comparas humanos / clasificador / LLM / realidad.

**El aha**: humanos y LLM solo ven UNA frase; clasificador ve 10 features. El LLM no es tonto, está ciego.

---

## Momento 2 · 20:15 — Juego de 4 minutos en paso_5

**Después de** clicar los 2 primeros ejemplos clicables (tasa por industria, ACV por arquetipo).

**Lo que dices**: "Vale. Cuatro minutos. Escribid vuestras propias preguntas en español, encontrad lo más raro que podáis. Gana la mejor sorpresa. Empezad."

**Tú te callas.** Espera 4 minutos. Después: "Pegadme las 2-3 preguntas más interesantes en el chat con su resultado."

**Lee** 2-3 en voz alta. Ejecuta la más sabrosa tú en pantalla.

**El aha**: ellos conducen, no tú. Lo que tardarían 20 min en pandas a mano, el LLM en 2 segundos.

---

## Momento 3 · 20:22 — Fallo provocado + edit en vivo

**Escribe a propósito**: `¿Cuántos leads vienen de Instagram?`

**Esperado**: el LLM inventa `df['channel']`, o filtra source==Instagram → 0 filas, o crashea.

**Lo que dices**: "El LLM no sabe los nombres de columna a menos que se los digamos. ¿Cómo lo arreglamos?"

**Edita el SYSTEM_PROMPT en vivo**, añade:
```
Si la pregunta menciona un canal o industria que no está en los valores listados, di explícitamente que no existe en los datos.
```

Guarda. Streamlit re-ejecuta. Vuelve a preguntar.

**El aha**: prompt engineering en vivo, no teoría. El prompt es texto que tú versionas.

---

## Momento 4 · 20:28 — Reveal de ask_my_data.py

**Abre** `session2/ask_my_data.py` en pantalla.

**Lo que dices**: "Esto es lo que os lleváis. paso_5 sin Streamlit, sin Cañadata. Cambiáis dos variables: `csv_path` (línea 21) y `column_hints` (líneas 22-28). Ya está. Lo pegáis en vuestro proyecto el lunes y tenéis un analista en lenguaje natural sobre vuestros datos."

**Feynman**: "Imagínate un becario carísimo que sabe pandas perfectamente pero no sabe nada de tu negocio. Cada vez que le preguntas algo tienes que recordarle cómo se llaman las columnas. A veces escribe la query mal y lo ves y le corriges. Eso es text-to-code. Eso es lo que te llevas hoy."

---

## Momento 5 · 20:58 — Punchline analista vs operador (paso_6)

**Escribe a mano** (NO clicar el botón ejemplo, escríbelo en el textarea):

> `¿Cuál es la probabilidad de conversión del lead L0001 según el clasificador?`

**Pulsa** "Comparar modos".

**Esperado**:
- **🧮 Analista**: `df[df['lead_id']=='L0001']['converted'].iloc[0]` → `False` (verdad histórica).
- **🧠 Operador**: `classifier['model'].predict_proba(...)` → `0.0964` (predicción).

**Lo que dices**: "Misma pregunta. Dos respuestas distintas. El analista, sin acceso al modelo, devuelve el dato histórico (este lead convirtió o no). Útil para auditar el pasado, inútil para un lead nuevo. El operador, con el modelo a mano, devuelve la probabilidad. **El LLM no improvisa un clasificador en pandas: usa el modelo que entrenamos ayer. Porque se lo dijimos en el prompt.**"

**Toy Einstein**: "El LLM es el recepcionista de una clínica. '¿Cuántos pacientes tenemos?' lo mira en el sistema. '¿Este paciente nuevo va a recaer?' no lo sabe, llama al especialista. **Tu modelo entrenado es el especialista. El recepcionista no diagnostica: sabe a quién pasar la llamada.**"

---

## Cierre · 21:21 — "El lunes usaría text-to-code para ___"

**Lo que dices**: "Una frase cada uno en el chat. **El lunes usaría text-to-code para ___.** Lo que sea. Datos del CRM, reports semanales, un dashboard que nadie usa. Empezad."

**Espera 60 segundos.** Lee 4-5 en voz alta sin juicio. Algunas serán flojas, no importa.

**Cierra**: "Mañana en S3 medimos sobre el holdout: cuánto acierta cada modo, cuánto cuesta, cuánto tarda. Hoy habéis construido las puertas. Mañana las medimos."

---

## El orden de sacrificio si vas justo de tiempo

| Prioridad | Sacrificar |
|---|---|
| 1 | Bonuses (siempre primero) |
| 2 | Tercera/cuarta pregunta clicable de paso_5/6 |
| 3 | Pausa intermedia (19:50) |
| **NUNCA** | Los 5 momentos de arriba |

Si tienes que comprimir paso_6 (te pillaron 10 min tarde): salta directo del walk-through (20:35) al punchline (20:58) usando el código pre-escrito del HUECO 1 y dejando huecos 2 y 3 para que los pegues a copia rápida desde la cheatsheet del guide.
