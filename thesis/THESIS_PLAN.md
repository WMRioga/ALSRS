# ALSRS — Mapa metodológico de tesis

Plan de trabajo para estructurar la tesis de maestría a partir del proyecto
ALSRS ya construido. Cada capítulo mapea los artefactos existentes (código,
notebooks, documentación, bibliografía) hacia la redacción académica.

---

## 1. Mapa conceptual (progresión de la tesis)

```mermaid
flowchart LR
    CH1["Ch1 · Introduction<br/>(problema y RQs)"] --> CH2["Ch2 · Literature Review<br/>(temas y gap)"]
    CH2 --> CH3["Ch3 · Methodology<br/>(pipeline y modelo)"]
    CH3 --> CH4["Ch4 · Analysis & Results<br/>(evidencia)"]
    CH4 --> CH5["Ch5 · Discussion<br/>(interpretación)"]
    CH5 --> CH6["Ch6 · Conclusion<br/>(contribución)"]
```

## 2. Mapa del pipeline ALSRS → capítulos

```mermaid
flowchart LR
    A["GEE · datos satelitales<br/>(gratuitos)"] --> B["Balance hídrico<br/>(etiquetas WRSI)"]
    B --> C["Dataset<br/>52k filas × 33 cols"]
    C --> D["Hurdle model<br/>(clasificador + regresor)"]
    D --> E["Pronóstico déficit<br/>1 / 3 / 6 meses"]
    E --> F["Recomendación<br/>LOW / MODERATE / SEVERE"]

    A -.-> CH3["Ch3"]
    B -.-> CH3
    C -.-> CH3
    D -.-> CH3
    E -.-> CH4["Ch4"]
    F -.-> CH4
```

## 3. Mapeo capítulo × lo que ya tienes × lo que falta

| Capítulo | Contenido | Ya tienes | Falta escribir |
|---|---|---|---|
| **1. Intro** | contexto, problema, RQs, aporte | motivación + pipeline + contexto agrícola | RQs/objetivos formales + redacción |
| **2. Lit Review** | 7 temas + gap | `BIBLIOGRAPHY.md` (organizada por tema) | síntesis crítica + gap argumentado |
| **3. Metodología** | pipeline, datos, modelo, evaluación | `ml/ML_MODEL.md` + `ml/DATASET_DICTIONARY.md` + `ml/collect_training.py` | redacción formal (casi listo) |
| **4. Resultados** | cero-inflated, baseline, hurdle, gate, clases, CV | `ml/01` / `ml/02` / `ml/04` notebooks | presentación objetiva (tablas/figuras) |
| **5. Discusión** | interpretar hallazgos | conversaciones (phantom deficit, flexibilidad, complementariedad) | redacción crítica |
| **6. Conclusión** | resumen, contribución, futuro | híbrido, `predict_point`, otros cultivos | redacción breve |
| **7. Referencias** | — | `BIBLIOGRAPHY.md` ✅ | formatear al estilo de la universidad |

**Tres puntos de trabajo intelectual real (no mecánico):**
1. Articular el **research gap** (Ch1.2 + Ch2.3).
2. Formalizar las **research questions y objectives** (Ch1.3).
3. La **discusión** (Ch5).

---

## 4. Research gap (recomendación)

**Lo que ya existe (y sus límites):**
- El ML agrícola se enfoca en **predicción de rendimiento** o **clasificación de
  sequía**, no en la **decisión de riego** (el déficit) como target explícito.
- Muchos usan datos **de pago/cerrados**, o ignoran que el déficit es
  **zero-inflated**, lo que un regresor único maneja mal.
- El **two-part/hurdle model** (Cragg 1971; Mullahy 1986; Lambert 1992) es
  clásico en econometría, pero **rara vez se aplica al pronóstico de déficit
  hídrico agrícola con features satelitales**.

**El gap que atacas:**
> Combinar **etiquetas débilmente supervisadas con base física (balance
> hídrico/WRSI)** + un **modelo hurdle (clasificador + regresor)** para
> pronosticar necesidad de riego, usando **datos satelitales gratuitos**, en un
> marco **transferible entre cultivos**.

**El aporte (3 patas):**
1. **Metodológico** — aplicar el hurdle a un contexto nuevo y demostrar *por qué*
   funciona (elimina el "déficit fantasma" → MAE −34% a −64%).
2. **Empírico** — evidencia real en cacao colombiano (239 fincas, 52k registros).
3. **Práctico** — sistema transferible (crop es un parámetro) con clasificación
   ajustable al negocio sin re-entrenar.

---

## 5. Research questions y objectives

**RQ principal:**
> ¿Cómo pueden datos satelitales y climáticos **gratuitos**, combinados con
> **etiquetas de base física** (balance hídrico), usarse para **pronosticar la
> necesidad de riego** (déficit hídrico) mediante machine learning?

**Sub-preguntas:**
- **RQ1 (etiquetado):** ¿Puede un balance hídrico quincenal (WRSI) generar
  etiquetas de déficit con base física a partir de datos GEE gratuitos?
- **RQ2 (modelado):** ¿Mejora un modelo hurdle (clasificador + regresor) el
  pronóstico de déficit frente a un regresor único, dado el carácter
  zero-inflated del target?
- **RQ3 (decisión/transferencia):** ¿Cómo afecta el esquema de clases (4/3/2) al
  recall de la decisión, y es el enfoque transferible entre cultivos?

**Objetivos (mapeados a las RQ):**
- **O1** → construir el pipeline ALSRS (GEE + balance hídrico + AHP). *(soporta RQ1)*
- **O2** → construir y evaluar el modelo hurdle vs baseline. *(soporta RQ2)*
- **O3** → evaluar la clasificación flexible y la transferibilidad. *(soporta RQ3)*

---

## 6. ⚠️ Punto honesto: transferibilidad (afecta RQ3)

La transferibilidad hoy es **arquitectónica** (el crop es un parámetro), pero
**no está demostrada empíricamente** (solo se validó cacao).

- **(A) Demostrarla** — correr `sugarcane` (cambiar `CROP` y el CSV de puntos,
  re-entrenar) y mostrar que el mismo pipeline + hurdle funciona. **Fortalece
  RQ3 y el aporte.**
- **(B) Acotarla** — presentar la transferibilidad como propiedad de diseño +
  trabajo futuro (Ch6), y dejar RQ3 como "clasificación flexible".

**Recomendación: hacer (A)** — es barato y convierte "prometo que funciona" en
"demuestro que funciona".

---

## 7. Orden de trabajo

1. Fijar RQs + objetivos + gap (Ch1.3 y Ch2.3) — el ancla de todo.
2. Decidir el punto 6 (¿demostrar transferibilidad con caña o acotarla?).
3. Escribir Ch2 (Literature Review) con `BIBLIOGRAPHY.md` ya organizada.
4. Escribir Ch3 (Metodología) — casi transcribir `ML_MODEL.md` + `DATASET_DICTIONARY.md`.
5. Ch4 (Resultados) — exportar tablas/figuras de los notebooks `01`/`02`/`04`.
6. Ch5 (Discusión) — convertir los hallazgos en argumento.
7. Ch1 (Intro) al final + Ch6.
