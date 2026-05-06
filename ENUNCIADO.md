# Cañadata | Enunciado

---

## Contexto

**Cañadata** es una empresa ficticia de software B2B (SaaS). Su producto es una plataforma para equipos comerciales: gestión de pipeline, scoring de leads, automatización de seguimiento.

Cañadata captura leads por cinco canales:

- `organic` — alguien encuentra la web por buscador
- `paid` — anuncios en LinkedIn / Google Ads
- `referral` — recomendación de un cliente o partner
- `conference` — feria, evento, charla
- `outbound` — equipo comercial proactivo

Ofrecen una versión Starter, Business y Enterprise, con cuotas anuales (`quoted_acv_eur`) que dependen del tamaño de la empresa, sector e intensidad de uso.

Un lead "convierte" cuando firma contrato dentro de los seis meses siguientes a la primera interacción. Históricamente la tasa de conversión global ronda el 35-40%. Hay variación enorme entre tipos de empresa.

---

## La pregunta de negocio

> **¿Cómo de bien podemos predecir qué leads van a convertir, y qué nos cuesta cada enfoque?**

Tres enfoques posibles, todos sobre los mismos datos:

1. **Modelo clásico de ML**: entrenar un clasificador (xgboost / RandomForest) sobre features estructuradas. Esto es lo que ha hecho la industria los últimos 20 años.
2. **LLM directo**: enviar la descripción de la empresa a un modelo grande y pedirle que puntúe. Cero entrenamiento, cero pipeline, cero data scientist.
3. **LLM + modelo entrenado**: el LLM hace de interfaz; el modelo entrenado hace los números.

El taller compara los tres en cuatro tareas distintas (clasificación, regresión, clustering, series temporales), y termina con la pregunta que importa: ¿cuál de las tres usarías en producción y por qué?

---

## Datos

### `data/canadata_leads.csv`

Tabla principal. 700 filas, una por lead. (Hay además un `canadata_holdout.csv` de 100 filas que se reserva para la comparación LLM-vs-clásico.)

| Columna | Descripción |
|---------|-------------|
| `lead_id` | Identificador único del lead |
| `company_name` | Nombre de la empresa (ficticio, generado) |
| `industry` | Sector (`SaaS`, `fintech`, `retail`, `logistics`, `healthcare`) |
| `company_size` | Empleados estimados |
| `country` | País (`ES`, `FR`, `DE`, `UK`, `IT`, `PT`) |
| `signup_date` | Fecha de primera interacción (YYYY-MM-DD) |
| `source` | Canal por el que entró el lead |
| `demo_requested` | ¿Pidió demo? (True/False) |
| `emails_opened` | Número de emails abiertos del seguimiento |
| `response_time_hours` | Horas hasta primera respuesta del lead |
| `n_meetings` | Número de reuniones celebradas |
| `decision_maker_contacted` | ¿Hemos hablado con el decision maker? (True/False) |
| `quoted_acv_eur` | Cuota anual cotizada (EUR) |
| `company_description` | Descripción libre en español de la empresa |
| `converted` | **Target binario**: ¿firmó contrato? |
| `converted_within_days` | Días hasta firmar (NULL si no firmó) |
| `lead_segment_truth` | **Verdad oculta**: arquetipo plantado por el generador. Solo para evaluar clustering — NO uses esta columna como feature. |

### Las cuatro tareas de ML sobre los mismos datos

| Tarea | Target | Modelo clásico de partida |
|---|---|---|
| Clasificación | `converted` | xgboost / RandomForest |
| Regresión | `quoted_acv_eur` | RandomForest sobre log-target |
| Clustering | recuperar `lead_segment_truth` | KMeans (k=3) + StandardScaler |
| Series temporales | conversiones mensuales | SARIMAX(1,1,1)(1,1,1,12) |

`company_description` está pensada para el modo LLM zero-shot. El clásico usa las features estructuradas; el LLM se las apaña con el texto.

---

## Lo que vas a entregar

Al final del taller tendrás:

- Un dashboard de Streamlit que sirve predicciones de los cuatro modelos sobre cualquier lead.
- Tres modos LLM operando sobre el mismo dataset (zero-shot, analista vía text-to-code, operador del modelo entrenado).
- Una matriz de comparación (4 tareas × 3 modos LLM + clásico) con accuracy, coste y latencia.
- Un veredicto razonado: para cada tarea, qué desplegarías y por qué.
- Bonus en S3: un flujo para reentrenar con datos nuevos respetando puertas de calidad y redesplegar a Streamlit semi-automáticamente.

---

## Lo que NO vamos a hacer

- Optimización exhaustiva de hiperparámetros. Los modelos clásicos son razonables, no campeones.
- Despliegue en producción real. Todo corre en local en tu Streamlit.
- Modelos de frontera. Usamos `gpt-4.1-mini` (o equivalente). El punto del taller es el patrón, no el último modelo.
- Análisis estadístico exhaustivo. El warm-up notebook ancla el dataset; no hay tiempo para EDA profundo en clase.
