# Sesión 3 · Asistente con tools, skill compartible y retrain manual

S1 entrenaste 4 modelos. S2 le diste herramientas al LLM. Hoy tres cosas:

1. **paso_7**: un asistente comercial real en Streamlit. Pegas un lead, el LLM llama a tus modelos como tools (function calling), te da un veredicto. Tú lo activas pieza a pieza rellenando 4 huecos diminutos.
2. **paso_8**: el mismo asistente, empaquetado como un *skill* que puedes pegar en Claude Project o ChatGPT. **Sin terminal**, sin Python. Para compartir con tu equipo.
3. **paso_9**: una guía sobre qué hacer cuando dentro de 3 meses tengas leads nuevos y quieras reentrenar.

## Lo que hoy NO hacemos (la trade-off honesta)

S3 v2 cambia el foco respecto a la versión anterior. **Hoy aprendes a operar y a compartir, no a medir formalmente**. Eso significa:

- No calculas ROC-AUC con intervalos de confianza por bootstrap.
- No comparas modelo nuevo vs viejo con quality gates automáticos.
- No construyes el pipeline CI/CD que reentrene + valide + despliegue.

Todo eso está en la rama `session-3-advanced` (paso_7/8/9 originales con `retrain.py`, bootstrap CI, ship-it pipeline). Si tu rol requiere medir con rigor, ése es el siguiente paso. Hoy te llevas el patrón de **construir un asistente con tools, empaquetarlo, y saber CUÁNDO toca reentrenar**. Las métricas son lectura (paso_9), no práctica.

---

## paso_7 · El dashboard con tools

```sh
streamlit run session3/exercises/paso_7.py
```

Chat estilo ChatGPT, pero conectado a tus modelos. El bot empieza casi mudo: sólo sabe extraer features de un texto libre. Tú le enciendes capacidades una a una:

- **HUECO 1**: enciende `predict_conversion` (el clasificador).
- **HUECO 2**: enciende `predict_acv` (el regresor).
- **HUECO 3**: enciende `get_archetype` (el clusterer).
- **HUECO 4**: enciende `find_similar_leads` (KNN sobre features escaladas).

Cada hueco es una línea. Tras rellenar uno, recargas y ves que el bot tiene una capacidad nueva.

Lleva una sección final **🚀 Si te quedas con ganas: añade tu propia tool** con un ejemplo (`draft_outreach_email`) y lista de ideas para tu empresa.

### Si peta al arrancar

| Síntoma | Causa | Fix |
|---|---|---|
| `Falta data/canadata_leads_clean.csv` | pre-clase incompleta | corre `pre_class/1_classical_models.ipynb` |
| `Falta session1/models/X.pkl` | pre-clase incompleta | mismo notebook |
| `OPENAI_API_KEY no está` | falta `.env` | crea `.env` con tu key |
| `No se pudo contactar OpenAI` | red rota o key inválida | comprueba conexión + key |

---

## paso_8 · El skill compartible

Está en `skill/`:

- `canadata_lead_advisor.md`: el system prompt completo con contexto de negocio + heurísticas + 3 few-shot examples.
- `USE_IN_CLAUDE_PROJECT.md`: cómo pegarlo en claude.ai/projects.
- `USE_IN_CUSTOM_GPT.md`: cómo pegarlo en chat.openai.com/gpts.

Idea: tu colega de Marketing no tiene terminal ni los `.pkl`. Pero sí tiene claude.ai. Le mandas `canadata_lead_advisor.md`, lo pega como instrucciones de un Claude Project, y tiene un asesor decente para sus leads. **Sin sklearn, sin pickles, sin servidor.**

El skill tiene 1 hueco simbólico (HUECO 1) donde tú escribes UNA heurística de tu dominio. Eso fuerza la conversación "¿qué sabe MI empresa que un LLM puro no sabría?".

Pruébalo: pega un lead en el chat del Claude Project y compara la respuesta con la que da `paso_7`. Verás dónde el skill alcanza y dónde necesitas el dashboard local.

---

## paso_9 · La guía de retrain & validate

Está en `guide/RETRAIN_AND_VALIDATE.md`. Lectura, no código.

Dentro de 3 meses tendrás 150 leads nuevos. La guía cubre:

- Cuándo conviene reentrenar (volumen + tiempo + performance).
- El flujo completo (junta, limpia, split, entrena, mide, compara, decide).
- Las 4 métricas explicadas una a una: ROC-AUC, R², ARI, MAPE.
- La regla del 95%: no degrades más de un 5%.
- Qué hacer si los gates fallan.

Si quieres ver el flujo AUTOMATIZADO (script Python que lo hace solo + UI Streamlit con file uploader + ship-it pipeline), está todo en la rama `session-3-advanced`. Es más denso pero está pensado para integrarse en CI/CD real.

```sh
git checkout session-3-advanced
ls session3/
```

---

## La sincronía entre paso_7 y paso_8 (gotcha importante)

Los tres entregables son **complementarios, no alternativos**:

- **paso_7** (dashboard local) usa los `.pkl` entrenados. Si reentrenas (paso_9), `paso_7` mejora automáticamente al cargar los `.pkl` nuevos.
- **paso_8** (skill `.md`) **NO usa los `.pkl`**. Vive del prompt + heurísticas que escribiste a mano. Si reentrenas y el modelo aprende algo nuevo, el skill NO se entera.
- **Consecuencia**: cada vez que reentrenas y los hallazgos del modelo cambian (p.ej. ahora una nueva industria convierte mejor), tienes que **actualizar el skill `.md` a mano** y volver a pegarlo en Claude Project / Custom GPT. **Hay drift entre los dos**.

Si tu equipo va a usar el skill seriamente, marca en el calendario revisar el `.md` cada vez que retrainees. No es automatizable sin servidor.

## El veredicto (cierre de S3)

Una frase, por escrito. La pregunta NO es "¿cuál de los tres?" porque son complementarios. La pregunta es:

> "El lunes, ¿qué combinación de paso_7 + paso_8 + paso_9 usaría en mi empresa, y cuál es la primera pieza que monto?"

Variaciones aceptables:
- "Empiezo por paso_7 para mí, paso_8 para el equipo cuando paso_7 estabilice, paso_9 al trimestre."
- "Salto paso_7 porque ya tengo modelos productivizados, voy directo a paso_8 para mi equipo no-técnico."
- "Hoy solo paso_8: no tengo modelos entrenados, sólo prompt + heurísticas."

No hay respuesta correcta. Hay decisión articulada. La plantilla con preguntas guiadas está en [`decision_framework.md`](decision_framework.md).
