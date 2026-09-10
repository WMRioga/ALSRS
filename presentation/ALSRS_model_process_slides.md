# ALSRS — Guion de presentación (7 slides, versión ampliada)

Guion para exponer el notebook `ml/04_alsrs_model_process.ipynb` en 7
diapositivas (tiempo corto). Cada slide trae:

- 📄 **Texto en slide** (lo que se ve, conciso)
- ❓ **Qué significa** (explicación en lenguaje llano, para explicar de viva voz)
- 💻 **Código + qué hace** (recuadro breve)
- ✅ **¿Usamos modelo aquí?** (respuesta explícita a "¿en este paso usamos modelo o solo observamos?")
- 🎤 **Notas del orador**
- 📚 **Bibliografía** (pie de slide)

Los números citados son los reales (239 fincas, run completo).

---

## Slide 1 — Título y objetivo

**📄 Texto en slide**

> **ALSRS — Predicción de necesidad de riego (Cacao CCN-51)**
>
> *Del dataset al modelo final: el proceso de entrenamiento*
>
> - 52,341 filas × 33 columnas · 240 fincas · 24 quincenas/año
> - Target: déficit de agua acumulado a **1, 3 y 6 meses**
> - Pregunta central: **"¿necesito riego en las próximas ventanas?"**

**🎤 Notas del orador**

- Presenta el objetivo: "voy a mostrar cómo pasamos del dataset crudo al modelo
  que predice la necesidad de riego".
- ALSRS = Agricultural Land Suitability Recommendation System. Hoy nos enfocamos
  **solo en la fase de modelado** (la extracción satelital y el balance hídrico
  son fases previas).
- Transición: "Todo empieza con un problema: la variable que queremos predecir
  casi siempre es cero."

**📚 Bibliografía (pie)** — FAO (1976), *A Framework for Land Evaluation*.

---

## Slide 2 — El problema (target zero-inflated) + baseline

**📄 Texto en slide**

> **El target es "zero-inflated"**
>
> - **73% / 56% / 39%** de ceros (1m / 3m / 6m)
> - Un solo regresor debe aprender dos cosas a la vez: **"¿hay déficit?"** y **"¿cuánto?"**
>
> **Baseline: Random Forest único** → MAE ≈ **11**, inventa un "déficit fantasma" (~11.8 en filas con déficit real = 0)

**❓ Qué significa**

- **Zero-inflated** = la variable casi siempre vale 0 (no hay estrés hídrico),
  pero cuando NO es 0 puede ser grande. Es una distribución "con un pico enorme
  en 0 y una cola larga".
- **Déficit fantasma** = el modelo predice ~11.8 de déficit en quincenas donde
  en realidad NO hubo déficit. Es una falsa alarma sistemática.

**💻 Código + qué hace**

```python
# Un solo modelo: Random Forest de regresión
m = RandomForestRegressor(n_estimators=300, random_state=42)
m.fit(Xtr, ytr, sample_weight=deficit)   # las filas con 0 pesan poco
pred = m.predict(Xte)
# MAE ≈ 11  →  en filas con déficit real = 0, predice ~11.8
```

→ Entrena **un solo** Random Forest que predice el déficit directamente, y
medimos su error (MAE).

**✅ ¿Usamos modelo aquí?**

**Sí, las dos cosas a la vez:**
1. **Usamos un modelo** (el Random Forest único de baseline).
2. **Al ejecutarlo descubrimos una limitación** (el déficit fantasma).

El baseline es un modelo real que sirve de punto de comparación, y su fallo es
lo que motiva todo lo que sigue.

**🎤 Notas del orador**

- Muestra el histograma: pico enorme en 0 + cola larga.
- El hallazgo clave: en las filas donde el déficit real es 0 (la mayoría), el
  baseline predice ~11.8. Ese es el "déficit fantasma".
- Transición: "El problema de fondo es que un solo modelo hace dos trabajos
  distintos. La solución clásica es dividirlo en dos."

**📚 Bibliografía (pie)** — Breiman (2001), *Random Forests*.

---

## Slide 3 — El hurdle (two-stage)

**📄 Texto en slide**

> **Dividir el problema en dos etapas**
>
> - **Etapa 1 (clasificador):** `P(déficit > 0)` → *"¿habrá déficit?"*
> - **Etapa 2 (regresor):** entrenado **solo** en filas positivas → *"¿cuánto?"*
>
> *(diagrama: [features] → clasificador → regresor → déficit)*

**❓ Qué significa (explicación llana)**

- **`P(déficit > 0)`** = la **probabilidad** (un número entre 0 y 1) que da el
  clasificador de que el déficit va a ser **mayor que 0**. Responde a la pregunta
  "¿va a haber déficit o no?".
- **Hurdle / two-stage** = "hurdle" significa **valla/obstáculo**. Es un modelo
  estadístico clásico para variables *semicontinuas* (mucho 0 + valores
  positivos): primero hay que "saltar la valla" de si hay evento (etapa 1), y
  solo entonces se estima la magnitud (etapa 2).
- **Por qué lo traemos a la mesa:** porque nuestro target es exactamente ese
  caso (73% de ceros). No lo inventamos — es literatura estándar desde los años
  70–90 (ver pie de slide).

**💻 Código + qué hace**

```python
# Etapa 1: clasificador -> P(deficit > 0)
clf = RandomForestClassifier(class_weight="balanced")
clf.fit(X, y > 0)
p_pos = clf.predict_proba(X)[:, 1]   # probabilidad 0..1

# Etapa 2: regresor SOLO en las filas positivas -> magnitud
reg = RandomForestRegressor()
reg.fit(X[y > 0], y[y > 0])
mu_pos = reg.predict(X)
```

→ La etapa 1 aprende **cuándo** hay déficit; la etapa 2 aprende **cuánto**, dado
que ya hay déficit. Cada modelo se especializa en un solo trabajo.

**✅ ¿Usamos modelo aquí?**

**Sí, dos modelos:**
1. `RandomForestClassifier` (etapa 1) → decide "¿hay déficit?".
2. `RandomForestRegressor` (etapa 2) → decide "¿cuánto déficit?".

Aquí ya **tomamos la decisión de arquitectura**: en vez de un solo modelo, dos.

**🎤 Notas del orador**

- Enfatiza: "no es idea nuestra — es el **two-part / hurdle model**, literatura
  clásica (Cragg 1971, Mullahy 1986, Lambert 1992)".
- Transición: "Pero al combinar las dos etapas surge un problema matemático sutil."

**📚 Bibliografía (pie) — ⭐ clave para "no lo inventamos"**

- Cragg (1971), *Some statistical models for limited dependent variables*, Econometrica 39(5).
- Mullahy (1986), *Specification and testing of some modified count data models*, J. Econometrics 33(3).
- Lambert (1992), *Zero-inflated Poisson regression*, Technometrics 34(1).

---

## Slide 4 — El gate (fix del shrinkage) + umbral

**📄 Texto en slide**

> **El problema:** `esperado = P(déficit>0) × magnitud` → la P (<1) **encoge** los casos severos
>
> **El fix (gate):** si P ≥ 0.5 → magnitud **completa**; si no → 0
>
> **Barrido del umbral:** trade-off recall vs falsas alarmas → **elegimos 0.5**

**❓ Qué significa (explicación llana)**

- **`esperado = P × magnitud`** = la forma "natural" de combinar las dos etapas:
  si hay 80% de probabilidad de déficit y la magnitud predicha es 50, el valor
  esperado es `0.8 × 50 = 40`. El problema: multiplicar por una probabilidad
  (<1) **siempre achica** el número, así que los casos severos se degradan a
  moderados (a eso se le llama **shrinkage**).
- **Gate (compuerta)** = una regla que evita el encogimiento: si la probabilidad
  supera un umbral (0.5), usamos la **magnitud completa** (sin multiplicar); si
  no lo supera, predecimos 0.

**💻 Código + qué hace**

```python
# Combinación "esperada": P × magnitud  ->  encoge
expected = p_pos * mu_pos

# Gate: si P >= 0.5 usa la magnitud completa; si no, 0
gated = np.where(p_pos >= 0.5, mu_pos, 0.0)
```

→ La primera línea es lo que **no** queremos (encoge); la segunda es el fix que
conserva la magnitud de los casos severos.

**✅ ¿Usamos RF classifier? ¿La clasificación viene antes?**

**Sí y sí:**
1. El **RF classifier** ya se usó en la **etapa 1** (slide 3) → produce `P(déficit > 0)`.
2. El **regresor** (etapa 2) produce la magnitud.
3. El **gate** solo **combina** ambos resultados.

**El orden es: clasificador → regresor → gate.** La clasificación viene **antes**
(es la etapa 1); el gate es el paso final de combinación.

**🎤 Notas del orador**

- Explica el shrinkage con un número concreto: `0.8 × 50 = 40` (perdimos 10
  puntos solo por multiplicar).
- El gate es la decisión de diseño que conserva la magnitud de los casos graves.
- Muestra el barrido: bajar el umbral sube recall pero aumenta falsas alarmas.
  **0.5** es el punto dulce.
- Transición: "Ya tenemos el modelo. Ahora: ¿cómo lo evaluamos, y en qué métrica?"

**📚 Bibliografía (pie)** — Two-Stage ML Precipitation Framework (MDPI, 2025).

---

## Slide 5 — Recall + clasificación flexible ⭐

**📄 Texto en slide**

> **Recall (la métrica que importa):** MODERATE **0.83** · SEVERE **0.67**
>
> **Clasificación flexible:** como el modelo es **regresión**, re-bineamos **sin re-entrenar**
>
> - 4 → 3 → 2 clases: accuracy **0.69 → 0.78 → 0.85**
> - Adoptamos **3 clases**: LOW / MODERATE / SEVERE (umbrales 15 / 50)

**❓ Qué significa (explicación llana)**

- **¿Ya hicimos regresión?** **Sí** — la regresión se hizo en la **etapa 2 del
  hurdle** (slide 3). Su resultado es un **número continuo** de déficit, por
  ejemplo `38.96` o `61.20`.
- **¿Por qué clasificamos aquí?** Porque la salida final debe ser una
  **recomendación** (LOW/MODERATE/SEVERE), no un número crudo. Clasificar = poner
  una etiqueta al número.
- **¿Volvimos a usar un RF classifier para clasificar?** **No.** La
  "clasificación" aquí **no es un modelo nuevo**: es solo aplicar **umbrales**
  (15 y 50) al número continuo que ya salió de la regresión.

**💻 Código + qué hace**

```python
# La regresión YA dio un número continuo (ej. 38.96, 61.20)
# Clasificar = aplicar umbrales (NO es un modelo nuevo)
classify_deficit(38.96, [15, 50])  # -> "MODERATE"  (15 ≤ 38.96 < 50)
classify_deficit(61.20, [15, 50])  # -> "SEVERE"    (≥ 50)
```

→ Esa es la **flexibilidad**: los mismos números se pueden re-agrupar en 4, 3 o
2 clases **sin re-entrenar**, solo cambiando los umbrales.

**✅ ¿Usamos modelo aquí?**

**No entrenamos nada nuevo.** Aquí solo **re-etiquetamos** la salida continua de
la regresión con umbrales. (En el notebook también se compara contra un
"clasificador directo de 4 clases", pero **no** es lo que adoptamos: el resultado
final es regresión → umbrales.)

**🎤 Notas del orador**

- Aclara por qué **recall** (no accuracy pura): nos importa no dejar escapar los
  casos que **sí** necesitan riego.
- El hallazgo estrella: como elegimos **regresión**, la clasificación es un paso
  posterior que se ajusta sin tocar el modelo.
- Muestra la progresión: **menos clases = menos fronteras = menos errores**.
- Transición: "Todo esto fue en un holdout. Para el número oficial validamos con
  cross-validation."

**📚 Bibliografía (pie)** — Torgo & Ribeiro (2009), *Precision and Recall for
Regression*.

---

## Slide 6 — Cross-validation (métrica oficial)

**📄 Texto en slide**

> **5-fold GroupKFold por finca** (ninguna finca en train y test a la vez → sin leakage)
>
> | horizonte | baseline | hurdle | mejora |
> |---|---|---|---|
> | 1m | 11.21 | **4.01** | **−64%** |
> | 3m | 11.00 | **4.65** | **−58%** |
> | 6m | 10.60 | **7.03** | **−34%** |

**❓ Qué significa (explicación llana)**

- **Cross-validation (CV)** = en vez de evaluar una sola vez, partimos los datos
  en **5 pliegues (folds)**; entrenamos en 4, evaluamos en el 1 restante, y
  **rotamos** hasta que cada pliegue fue evaluado una vez. El promedio de las 5
  evaluaciones es la métrica **oficial** (no depende de un único corte "con
  suerte").
- **¿Por qué GroupKFold?** Porque agrupamos por **finca** (`point_id`). Una finca
  tiene una serie temporal completa; si sus filas se repartieran entre train y
  test, el modelo "memorizaría" esa finca y parecería mejor de lo que es
  (**data leakage**). GroupKFold garantiza que **cada finca está solo en train o
  solo en test**.

**💻 Código + qué hace**

```python
gkf = GroupKFold(n_splits=5)
for tr_idx, va_idx in gkf.split(df, groups=df["point_id"]):
    # entrena en 4 pliegues, evalúa en 1, rota 5 veces
    ...
# -> promedio de las 5 evaluaciones = métrica oficial
```

→ Repite "entrenar + evaluar" 5 veces, cada finca siempre completa en un solo
lado, y promedia el MAE/RMSE.

**✅ ¿Usamos modelo aquí?**

**Sí, se entrena de nuevo en cada pliegue** (5 veces). No es un modelo nuevo de
arquitectura — es el **mismo** hurdle re-entrenado 5 veces para medir su error
de forma confiable.

**🎤 Notas del orador**

- Explica GroupKFold con un ejemplo: "la finca X nunca aparece partida".
- El número oficial: el hurdle reduce el MAE entre **34% y 64%**.
- Hallazgo importante: la mejora viene de **eliminar el "déficit fantasma" en
  los ceros**, no de predecir mejor los extremos (en SEVERE, baseline y hurdle
  empatan en recall 0.67).
- Transición: "Con la métrica oficial confirmada, entrenamos el modelo final con
  todos los datos."

**📚 Bibliografía (pie)** — Breiman (2001) · scikit-learn (`GroupKFold`,
`RandomForest`).

---

## Slide 7 — Modelo final + conclusiones

**📄 Texto en slide**

> **Modelo final:** hurdle (clasificador + regresor) + gate (0.5) + 3 clases
>
> Guardado en `hurdle_models/` + `manifest.json` (6 modelos joblib)
>
> **Conclusiones**
>
> 1. La **regresión** da flexibilidad: re-binear sin re-entrenar
> 2. El **hurdle** corta el MAE **−34% a −64%** y empata el recall de los casos graves
> 3. **Menos clases → más accuracy** (cada frontera quitada elimina errores)

**💻 Código + qué hace (el despliegue)**

```python
def predict_hurdle(X):
    # 17 features en -> déficit (gate) + sugerencia (clases configurables)
    ...
    return out   # columnas: future_deficit_1m/3m/6m + suggestion
```

→ La función final carga los 6 modelos guardados y devuelve la recomendación.

**🎤 Notas del orador**

- Resume el entregable: 6 modelos (clasificador + regresor × 3 horizontes),
  configuración versionada en `manifest.json`.
- Repite los 3 hallazgos como cierre.
- Frase final: *"El valor no está solo en el modelo, sino en la flexibilidad: la
  clasificación es una decisión de negocio que se ajusta sin re-entrenar."*
- Abre a preguntas.

**📚 Bibliografía (pie)** — Cragg (1971) · Mullahy (1986) · Lambert (1992) ·
Breiman (2001) · Torgo & Ribeiro (2009).

---

## Mapa de bibliografía por slide (para marcar el pie)

| Slide | Tema | Cita(s) en el pie |
|---|---|---|
| 1 | Marco de evaluación de tierras | FAO (1976) |
| 2 | Random Forest baseline | Breiman (2001) |
| 3 | Hurdle / two-part model | Cragg (1971) · Mullahy (1986) · Lambert (1992) |
| 4 | Gate + two-stage ML | Two-Stage ML Precipitation (MDPI, 2025) |
| 5 | Métrica recall para regresión | Torgo & Ribeiro (2009) |
| 6 | CV / GroupKFold | Breiman (2001) · scikit-learn |
| 7 | Resumen | Cragg · Mullahy · Lambert · Breiman · Torgo & Ribeiro |

### Referencias completas (para una lámina final opcional de "Referencias")

- Breiman, L. (2001). Random Forests. *Machine Learning, 45*(1), 5–32.
- Cragg, J.G. (1971). Some statistical models for limited dependent variables
  with application to the demand for durable goods. *Econometrica, 39*(5), 829–844.
- Lambert, D. (1992). Zero-inflated Poisson regression, with an application to
  defects in manufacturing. *Technometrics, 34*(1), 1–14.
- Mullahy, J. (1986). Specification and testing of some modified count data
  models. *Journal of Econometrics, 33*(3), 341–365.
- Torgo, L., & Ribeiro, R.P. (2009). Precision and Recall for Regression.
  *Discovery Science, LNCS 5808*, 332–346.
- FAO (1976). *A Framework for Land Evaluation.* Soils Bulletin No. 32. Rome.
- A Two-Stage Machine Learning Framework for High-Resolution Multi-Source
  Precipitation Fusion in Complex Terrain (2025). *Atmosphere (MDPI).*
  https://www.mdpi.com/2073-4433/17/8/762
