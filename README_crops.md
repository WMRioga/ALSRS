# Crop Parameters Documentation

## Overview

This file documents the crop parameters used for land suitability assessment. The parameters define the physical and climatic conditions required for viable crop growth, as well as the water requirements used in the Water Requirement Satisfaction Index (WRSI) calculation.

## File Description

The parameters are stored in `crop_parameters.csv` with the following structure:

| Column | Description | Units |
|--------|-------------|-------|
| `crop` | Short name identifier for the crop | — |
| `common_name` | Common name of the crop | — |
| `type` | `perennial` or `annual` | — |
| `temp_min_tolerable_c` | Minimum tolerable temperature | °C |
| `temp_opt_min_c` | Minimum optimal temperature | °C |
| `temp_opt_max_c` | Maximum optimal temperature | °C |
| `temp_max_tolerable_c` | Maximum tolerable temperature | °C |
| `elev_min_m` | Minimum elevation | meters above sea level |
| `elev_max_m` | Maximum elevation | meters above sea level |
| `slope_max_deg` | Maximum slope for viable cultivation | degrees |
| `ph_min` | Minimum soil pH for optimal growth | pH units |
| `ph_max` | Maximum soil pH for optimal growth | pH units |
| `clay_max_pct` | Maximum clay content (universal physical limit) | % |
| `sand_max_pct` | Maximum sand content (universal physical limit) | % |
| `soc_min_pct` | Minimum soil organic carbon | % |
| `water_requirement_mm` | Total crop water requirement per cycle | mm |
| `cycle_quincenas` | Crop cycle duration in biweekly periods | quincenas (14 days each) |
| `root_depth_cm` | Effective root depth for AWC calculation | cm |

## Filter Logic

### Temperature

For each biweekly period (24 per year), the historical mean temperature is compared against crop ranges:

- `temp_quincenal < temp_min_tolerable_c` → **NOT RECOMMENDED** ("too cold for crop")
- `temp_quincenal > temp_max_tolerable_c` → **NOT RECOMMENDED** ("too hot for crop")
- `temp_min_tolerable_c ≤ temp_quincenal < temp_opt_min_c` → **VIABLE WITH RESERVE** ("cool for crop, validate with agronomist")
- `temp_opt_max_c < temp_quincenal ≤ temp_max_tolerable_c` → **VIABLE WITH RESERVE** ("warm for crop, validate with agronomist")
- `temp_opt_min_c ≤ temp_quincenal ≤ temp_opt_max_c` → **VIABLE** ("optimal temperature range")

**Final temperature status:**
- Any biweekly period outside tolerable range → **NOT RECOMMENDED**
- All periods within tolerable range but some outside optimal → **VIABLE WITH RESERVE**
- All periods within optimal range → **VIABLE**

### Elevation

- `elevation_m < elev_min_m` → **NOT RECOMMENDED** ("elevation too low")
- `elevation_m > elev_max_m` → **NOT RECOMMENDED** ("elevation too high")
- `elev_min_m ≤ elevation_m ≤ elev_max_m` → **VIABLE**

### Slope

- `slope_deg > slope_max_deg` → **NOT RECOMMENDED** ("slope exceeds crop limit")
- `slope_deg ≤ slope_max_deg` and `slope_deg + slope_std_deg > slope_max_deg` → **VIABLE WITH ADVISORY** ("some zones exceed slope limit, validate in field")
- `slope_deg + slope_std_deg ≤ slope_max_deg` → **VIABLE**

### Soil Texture

Universal physical limits based on FAO (1976) and USDA Soil Taxonomy:

- `sand_0_60cm > sand_max_pct (70%)` → **NOT RECOMMENDED** ("soil too sandy, does not retain water")
- `clay_0_60cm > clay_max_pct (50%)` → **NOT RECOMMENDED** ("soil too clayey, poor drainage and root penetration")
- Otherwise → **VIABLE**

### Soil pH

- `ph_min ≤ ph_0_60cm ≤ ph_max` → **VIABLE**
- `|ph_0_60cm - ph_min| ≤ 1.0` or `|ph_0_60cm - ph_max| ≤ 1.0` → **VIABLE WITH ADVISORY** ("requires pH correction with lime/sulfur")
- `|ph_0_60cm - ph_min| > 1.0` or `|ph_0_60cm - ph_max| > 1.0` → **VIABLE WITH ADVISORY** ("pH correction is costly; validate resources for amendments")

### Soil Organic Carbon (SOC)

- `soc_0_60cm < soc_min_pct (0.5%)` → **VIABLE WITH ADVISORY** ("soil poor in organic matter, requires amendments")
- `soc_0_60cm ≥ soc_min_pct` → **VIABLE**

## Crop Parameters

### Sugarcane (sugarcane)
- **Type:** Perennial
- **Temperature optimal:** 20-35°C (tolerable: 15-38°C)
- **Elevation:** 0-1500 m
- **Slope max:** 8°
- **pH:** 5.5-7.5
- **Water requirement:** 2000 mm/cycle
- **Cycle:** 24 quincenas (12 months)
- **Root depth:** 100 cm
- **Sources:** Doorenbos & Kassam (1979); Inman-Bamber & Smith (2005)

### CCN-51 Cocoa (cacao_ccn51)
- **Type:** Perennial
- **Temperature optimal:** 18-32°C (tolerable: 15-35°C)
- **Elevation:** 0-800 m
- **Slope max:** 8°
- **pH:** 5.5-7.0
- **Water requirement:** 1500 mm/cycle
- **Cycle:** 24 quincenas (12 months)
- **Root depth:** 70 cm
- **Sources:** Carr & Lockwood (2011); ICCO (2017)

### Arabica Coffee (arabica_coffee)
- **Type:** Perennial
- **Temperature optimal:** 15-24°C (tolerable: 10-30°C)
- **Elevation:** 800-2000 m
- **Slope max:** 15°
- **pH:** 5.0-6.5
- **Water requirement:** 1400 mm/cycle
- **Cycle:** 24 quincenas (12 months)
- **Root depth:** 80 cm
- **Sources:** DaMatta & Ramalho (2006); Cenicafé (2016)

### Plantain (plantain)
- **Type:** Perennial
- **Temperature optimal:** 18-32°C (tolerable: 10-38°C)
- **Elevation:** 0-1000 m
- **Slope max:** 8°
- **pH:** 5.5-7.0
- **Water requirement:** 1500 mm/cycle
- **Cycle:** 24 quincenas (12 months)
- **Root depth:** 60 cm
- **Sources:** Doorenbos & Kassam (1979); FAO 56 (Allen et al., 1998)

### Wheat (wheat)
- **Type:** Annual
- **Temperature optimal:** 10-24°C (tolerable: 5-30°C)
- **Elevation:** 0-1000 m
- **Slope max:** 8°
- **pH:** 5.5-7.5
- **Water requirement:** 550 mm/cycle
- **Cycle:** 9 quincenas (135 days)
- **Root depth:** 110 cm
- **Sources:** Doorenbos & Kassam (1979); FAO 56 (Allen et al., 1998)

### Sorghum (sorghum)
- **Type:** Annual
- **Temperature optimal:** 15-35°C (tolerable: 10-40°C)
- **Elevation:** 0-1000 m
- **Slope max:** 8°
- **pH:** 5.5-8.0
- **Water requirement:** 550 mm/cycle
- **Cycle:** 8 quincenas (110 days)
- **Root depth:** 110 cm
- **Sources:** Doorenbos & Kassam (1979); FAO 56 (Allen et al., 1998)

### Canola (canola)
- **Type:** Annual
- **Temperature optimal:** 10-25°C (tolerable: 5-30°C)
- **Elevation:** 0-800 m
- **Slope max:** 8°
- **pH:** 5.5-7.5
- **Water requirement:** 475 mm/cycle
- **Cycle:** 9 quincenas (125 days)
- **Root depth:** 100 cm
- **Sources:** Doorenbos & Kassam (1979); FAO 56 (Allen et al., 1998)

## Universal Soil Limits (All Crops)

| Parameter | Threshold | Justification |
|-----------|-----------|---------------|
| Sand max | 70% | Above this, soil does not retain water (USDA: sand/loamy sand) |
| Clay max | 50% | Above this, soil has poor drainage and root penetration (USDA: clay) |
| SOC min | 0.5% | Below this, soil is degraded with poor fertility and structure |

## References

- Allen, R.G., Pereira, L.S., Raes, D., & Smith, M. (1998). Crop evapotranspiration: Guidelines for computing crop water requirements. FAO Irrigation and Drainage Paper 56. Rome.
- Carr, M.K.V., & Lockwood, R. (2011). The water relations and irrigation requirements of cocoa (Theobroma cacao L.): A review. Experimental Agriculture, 47(4), 653-676.
- Cenicafé (2016). Guía para el cultivo de café en Colombia. Centro Nacional de Investigaciones de Café.
- DaMatta, F.M., & Ramalho, J.D.C. (2006). Impacts of drought and temperature stress on coffee physiology and production: A review. Brazilian Journal of Plant Physiology, 18(1), 55-81.
- Doorenbos, J., & Kassam, A.H. (1979). Yield response to water. FAO Irrigation and Drainage Paper 33. Rome.
- FAO (1976). A Framework for Land Evaluation. Soils Bulletin No. 32. Rome.
- ICCO (2017). Growing Cocoa. International Cocoa Organization.
- Inman-Bamber, N.G., & Smith, D.M. (2005). Water relations in sugarcane and response to water deficits. Field Crops Research, 92(2-3), 185-202.
- USDA Soil Taxonomy. Soil Survey Staff. Natural Resources Conservation Service.

## How to Add a New Crop

1. Open `crop_parameters.csv` in a text editor or spreadsheet software
2. Add a new row with the crop parameters
3. Use lowercase with underscores for the `crop` identifier (e.g., `maize_grain`)
4. For `type`: use `perennial` if the crop lives multiple years, `annual` if it completes its cycle in one season
5. For `cycle_quincenas`: divide the cycle duration in days by 14 and round to nearest integer
6. For `water_requirement_mm`: use FAO 56 or crop-specific literature
7. For `root_depth_cm`: use effective root depth for mature plants
8. Save the CSV and update this README with the new crop's documentation

## How to Modify Existing Parameters

1. Open `crop_parameters.csv`
2. Locate the crop row by `crop` identifier
3. Modify the value in the appropriate column
4. Update this README if the change affects the crop's documentation
5. Re-run the viability filter to validate the new values