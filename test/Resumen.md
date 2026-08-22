# Summary 260822 ---------
# Resumen — agri_land_suitability_pipeline

Pipeline de Data Science (Python + Google Earth Engine) para tesis de maestría:
clasificación de necesidad de riego (Bajo/Medio/Alto/No recomendado) por parcela,
combinando terreno, suelo, clima y teledetección.

**Coordenadas de prueba:** Finca Matanza (7.300921, -73.009794, Colombia, cacao) ·
El Playón (7.4584221918243045, -73.222052853104) · Sugarcane_COL
(3.580109040361371, -76.31299479308868) · Sugarcane_QLD (-19.689669877950884,
147.22717515914223, Queensland)

**Entorno:** conda env `agri_land_env`. Scripts en `scripts/`, salidas CSV en
`../databases/`, imágenes en `../img/maps/`.

**Convenciones del pipeline:** resolución quincenal (1-15 / 16-fin de mes),
10 años de historia (2016-2026), cálculo server-side en Earth Engine
(`ee.List.map()` + una sola `.getInfo()`), nombres de archivo con timestamp
(`{prefix}-vYYMMDDHHMMSS.csv`), NaN explícito (nunca se elimina en el script,
se avisa por consola), docstring con sección "CÓMO LEER EL RESULTADO".

---

## Scripts completados

### Pilares estáticos
- **`terrain_profile.py`** — Copernicus DEM: elevación, pendiente, orientación
  (+ desviación estándar de cada una; orientación promediada con estadística
  circular seno/coseno, no promedio directo).
- **`soil_profile.py` + `soil_hydraulics.py`** — propiedades de SoilGrids-ISRIC
  en 5 profundidades + capacidad de campo / punto de marchitez / AWC vía
  Saxton & Rawls (2006). Se corrigió un bug de conversión de unidades (los
  factores fijos de SoilGrids se aplicaban condicionalmente por magnitud en
  vez de siempre).

### Series temporales de clima (quincenal, 10 años)
- **`temperature_profile.py`** — ERA5-Land.
- **`precipitation_profile.py`** — CHIRPS (total, días de lluvia, máximo
  diario — se **suma**, no se promedia, por ser variable acumulativa).
- **`evapotranspiration_profile.py`** — MOD16A2/MOD16A2GF con doble fuente
  (GF + fallback no-GF, porque GF es "year-end gap-filled" y puede faltar
  el año en curso completo). Refactorizado en una función genérica
  `_get_mod16_band_biweekly()` reusada para ET y PET.
- **`spei_profile.py`** — SPEI en escalas 1/3/6/12 meses. Versión final:
  PET por Thornthwaite corregido (índice de calor anual climatológico real +
  corrección de horas de luz por geometría solar) + agrupación explícita por
  quincena-calendario (paquete `standard_precip`).

### Teledetección — usados como VALIDACIÓN, no como feature principal
- **`soil_moisture_profile.py`** — Sentinel-1 SAR, índice de change detection
  vs. referencia histórica p5/p95 propia del punto, con fallback de doble
  órbita (ASC/DESC, la cobertura de ESA varía por región). La mayoría de
  quincenas tenían solo 1 imagen — muy ruidoso para usar como feature
  quincenal; se reservó para validación cruzada puntual contra el balance
  hídrico.
- **`sentinel2_indices_profile.py`** — NDVI, EVI, NDMI, BSI, NDRE
  (`COPERNICUS/S2_SR_HARMONIZED`, enmascarado de nubes con SCL — QA60 dejó
  de funcionar bien desde 2022 —, compuesto por mediana). El dataset solo
  arranca el 28-marzo-2017, así que 2016 y principios de 2017 salen en NaN
  por diseño, no por error.

### Visualización
- **`point_map.py`** — foto satelital (ESRI) o relieve (OpenTopoMap)
  centrada en un punto, con marcador y cuadro del área de buffer.
- **`regional_elevation_map.py`** — mapa de elevación de todo un
  state/país (límites FAO GAUL).

### Módulos compartidos
`tile_utils.py` (mapas), `period_utils.py` (quincenas), `soil_hydraulics.py`
(Saxton-Rawls, reusable aparte).

---

## Diseño del modelo objetivo (pendiente de implementar)

**Etiquetas (WRSI-style):** balance hídrico quincenal secuencial —
`almacenamiento = min(AWC, almacenamiento_anterior + precipitación)`,
`requerimiento = requerimiento_anual_GAEZ / 24`, `% déficit` clasificado en
Bajo (0-15%) / Medio (15-30%) / Alto (30-50%) / No recomendado (>50%).
ET real se usa como **feature** del modelo, no en el cálculo del déficit
(PET se usa aparte, para el SPEI, evitando circularidad). El almacenamiento
inicial de la serie se decidió tomar como la media de precipitación de la
primera quincena disponible.

**Clasificador:** Random Forest (Breiman, 2001) sobre las etiquetas WRSI,
usando variables de los 4 pilares. Encuadre metodológico: **weak
supervision** (Ratner et al., 2017) / *physics-informed machine learning* —
respuesta a la pregunta del comité de "¿dónde está la ciencia de datos?".

---

## Bibliografía armada (tabla comparativa completa disponible)

**Marco general:** FAO (1976), Fischer et al. (2021, GAEZ v4)
**Suelo:** Poggio et al. (2021, SoilGrids 2.0), Saxton & Rawls (2006)
**Clima:** Muñoz-Sabater et al. (2021, ERA5-Land), Funk et al. (2015, CHIRPS),
Running et al. (2021, MOD16A2/GF)
**Teledetección:** Bauer-Marschallinger et al. (2019, Sentinel-1); Rouse et al.
1974 (NDVI), Huete et al. 2002 (EVI), Gao 1996 (NDMI), Rikimaru et al. 2002
(BSI), Barnes et al. 2000 (NDRE); Baetens et al. 2019 (SCL), White et al.
2014 (mediana)
**WRSI:** Frère & Popov (1979), Doorenbos & Kassam (1979), Verdin & Klaver
(2002)
**Modelo/ML:** Breiman (2001), Ratner et al. (2017), Wei et al. (2022, el
precedente más directo)
**Contexto de campo activo:** Cao & Jiang (2024), Salehi Hikouei et al.
(2021) — citados con sus debilidades explícitas, como evidencia de área
activa, no como precedentes centrales.

---

## Entregables de presentación

- Guión de 12 slides para presentación de avances (problema, marco FAO/GAEZ,
  estado de cada pilar, diseño del modelo objetivo, respuesta a "dónde está
  la ciencia de datos", referencias).
- Tabla comparativa de papers (Article / Date / Dataset / Model / Application
  / Strengths / Weaknesses).

---

## Pendiente

1. Implementar el script del balance hídrico secuencial / etiquetado WRSI.
2. Entrenar el Random Forest.
3. Unificar todas las tablas generadas en un solo dataset por punto/quincena.