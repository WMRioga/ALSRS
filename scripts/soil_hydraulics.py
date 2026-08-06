"""
Cálculo de propiedades hídricas del suelo (capacidad de campo, punto de
marchitez, agua disponible) usando las funciones de pedotransferencia
de Saxton & Rawls (2006), a partir de propiedades crudas de SoilGrids.

CÓMO LEER EL RESULTADO
------------------------
Field_Capacity_CC_%       Capacidad de campo: % de agua (volumétrico)
                           que el suelo retiene después de drenar el
                           exceso por gravedad. El "lleno" del balde.
Wilting_Point_PMP_%       Punto de marchitez permanente: % de agua por
                           debajo del cual la planta ya no puede
                           extraer más humedad del suelo. El "vacío"
                           real del balde (nunca llega a 0% absoluto).
Available_Water_Capacity_AWC_%   CC - PMP: el rango de agua realmente
                           utilizable por la planta.
AWC_layer_mm               Lo mismo pero en mm de agua para el espesor
                           real de esa capa -> sumando todas las capas
                           se obtiene la lámina total disponible en el
                           perfil radicular (0-100cm), el dato que se
                           cruza con precipitation_profile.py para el
                           balance hídrico.

NOTA: los coeficientes de Saxton & Rawls (2006) usados acá conviene
verificarlos contra la Tabla 1 del paper original antes de usarlos
para conclusiones finales de tesis.
"""
import pandas as pd


DEPTHS = ['0-5cm', '5-15cm', '15-30cm', '30-60cm', '60-100cm']
LAYER_THICKNESS_CM = {
    '0-5cm': 5, '5-15cm': 10, '15-30cm': 15, '30-60cm': 30, '60-100cm': 40,
}

# Factores de conversión FIJOS de SoilGrids (mapped units -> unidades
# convencionales). No dependen de la magnitud del valor -> siempre se
# aplican igual, sin importar qué tan alto o bajo sea el dato.
SOILGRIDS_CONVERSION = {
    'clay': 10.0,   # g/kg -> %
    'sand': 10.0,   # g/kg -> %
    'silt': 10.0,   # g/kg -> %
    'soc': 100.0,   # dg/kg -> % (factor combinado: /10 a g/kg, /10 más a %)
    'bdod': 100.0,  # cg/cm3 -> g/cm3
}


def calculate_hydraulic_properties(soil_data):
    """
    Calcula capacidad de campo, punto de marchitez y agua disponible
    por capa, usando las funciones de Saxton & Rawls (2006).

    soil_data: dict con la forma {'clay': {depth: val}, 'sand': {...}, ...}
               tal como lo devuelve get_soil_profile_area() en soil_profile.py
    """
    hydric_results = {
        'Field_Capacity_CC_%': {},
        'Wilting_Point_PMP_%': {},
        'Available_Water_Capacity_AWC_%': {},
        'AWC_layer_mm': {},
    }

    for depth in DEPTHS:
        # Conversión de unidades SoilGrids -> convencionales (factores fijos)
        c_clay = float(soil_data['clay'][depth]) / SOILGRIDS_CONVERSION['clay']
        c_sand = float(soil_data['sand'][depth]) / SOILGRIDS_CONVERSION['sand']
        c_silt = float(soil_data['silt'][depth]) / SOILGRIDS_CONVERSION['silt']

        # Normalizamos para que la textura sume exactamente 100%
        total_text = c_clay + c_sand + c_silt
        if total_text > 0:
            c_clay = (c_clay / total_text) * 100
            c_sand = (c_sand / total_text) * 100
            c_silt = (c_silt / total_text) * 100

        # Materia orgánica: %SOC * 1.724 (factor de Van Bemmelen)
        c_soc_pct = float(soil_data['soc'][depth]) / SOILGRIDS_CONVERSION['soc']
        c_om = max(0.1, min(c_soc_pct * 1.724, 15.0))  # límite realista suelos minerales

        # Saxton & Rawls (2006) - verificar coeficientes contra el paper original
        wp_33 = (-0.024 * c_sand) + (0.487 * c_clay) + (0.006 * c_om) \
                - (0.005 * c_sand * c_om) + (0.013 * c_clay * c_om) \
                + (0.068 * c_sand * c_clay) + 0.031
        wp = wp_33 + (0.14 * wp_33) - 0.02

        fc_33 = (-0.251 * c_sand) + (0.195 * c_clay) + (0.011 * c_om) \
                + (0.006 * c_sand * c_om) - (0.027 * c_clay * c_om) \
                + (0.045 * c_sand * c_clay) + 0.297
        fc = fc_33 + (1.28 * (fc_33 ** 2)) - (0.38 * fc_33) \
             - (0.03 * c_sand * fc_33) + 0.02

        # Límites físicos de sanidad (no forman parte de la ecuación original,
        # son un resguardo ante extrapolaciones fuera de rango)
        fc_pct = max(5.0, min(fc * 100, 55.0))
        wp_pct = max(2.0, min(wp * 100, 40.0))
        if fc_pct < wp_pct:
            fc_pct = wp_pct + 5.0

        awc_pct = fc_pct - wp_pct
        thickness_mm = LAYER_THICKNESS_CM[depth] * 10
        awc_mm = (awc_pct / 100.0) * thickness_mm

        hydric_results['Field_Capacity_CC_%'][depth] = round(fc_pct, 2)
        hydric_results['Wilting_Point_PMP_%'][depth] = round(wp_pct, 2)
        hydric_results['Available_Water_Capacity_AWC_%'][depth] = round(awc_pct, 2)
        hydric_results['AWC_layer_mm'][depth] = round(awc_mm, 2)

    return pd.DataFrame(hydric_results)