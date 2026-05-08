# Veredicto — qué desplegarías y por qué

Plantilla para tu cierre del taller. Rellénala antes de la ronda final. No tiene que ser largo: claro y argumentado.

---

## 1. Mi caso (elige UNO de los cuatro)

- [ ] Clasificación: scoring de leads (¿este lead va a convertir?)
- [ ] Regresión: estimación de ACV para cotización inicial
- [ ] Clustering: segmentación automática para campañas de marketing
- [ ] Series temporales: forecast mensual para planificar capacidad comercial

**Mi caso elegido**: ____________________

---

## 2. Lo que desplegaría

**Approach que enviaría a producción**:

- [ ] Modelo clásico (sklearn/xgboost/Prophet) entrenado con nuestro dataset
- [ ] LLM zero-shot directo
- [ ] LLM como analista (text-to-code sobre los datos)
- [ ] LLM operando el modelo entrenado (lo mejor de los dos)
- [ ] Híbrido específico: ____________________

**Por qué este y no otro** (2-4 frases):

```
________________________________________________________________
________________________________________________________________
________________________________________________________________
```

---

## 3. Lo que NO desplegaría

**Approach que descarto**:

```
________________________________________________________________
```

**Por qué (la razón concreta, no genérica)**:

```
________________________________________________________________
________________________________________________________________
```

---

## 4. Plan de monitoreo

¿Qué métrica vigilaría día a día y a partir de qué umbral me preocuparía?

| Métrica | Cómo la mido | Umbral |
|---|---|---|
| | | |
| | | |

¿Cómo me daría cuenta de **drift**? (la población de leads cambia)

```
________________________________________________________________
```

---

## 5. Coste y latencia

| Dimensión | Estimación |
|---|---|
| Coste por predicción (€) | |
| Latencia por predicción | |
| Coste mensual a 1.000 leads/mes | |
| Coste mensual a 10.000 leads/mes | |

¿Cuándo deja de tener sentido el LLM por coste? ¿Cuándo el clásico por mantenimiento?

```
________________________________________________________________
________________________________________________________________
```

---

## 6. Lo que arreglaría primero si tuviera una semana más

```
1. ________________________________________________________________
2. ________________________________________________________________
3. ________________________________________________________________
```

---

## 7. Una frase para vendérselo a tu jefe

Escríbela como si fuera el primer mensaje de Slack. Incluye: qué propones, qué se gana, qué hace falta.

```
________________________________________________________________
________________________________________________________________
```

---

## Para la ronda final

Cuando te toque hablar (un minuto cada uno), prepara estas tres frases:

1. **Lo que envío**: `_______________`
2. **Lo que vigilo**: `_______________`
3. **Lo que reviso en 90 días**: `_______________`
