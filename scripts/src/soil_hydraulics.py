"""
Soil Hydraulics Module
======================

Calculates soil hydraulic properties (field capacity, wilting point, and
available water capacity) from raw SoilGrids texture properties, using the
pedotransfer functions of Saxton & Rawls (2006).

Reference
---------
Saxton, K. E., & Rawls, W. J. (2006). Soil water characteristic estimates
by texture and organic matter for hydrologic solutions. Soil Science
Society of America Journal, 70(5), 1569-1578. doi:10.2136/sssaj2005.0117

The regression coefficients were verified against the USDA-ARS reference
implementation (``saxpar.for``) and the HydroTools.R SPAW implementation.

HOW TO READ THE RESULT
-----------------------
Field_Capacity_CC_%       Field capacity: volumetric water percentage
                           that the soil retains after excess water drains
                           by gravity. The "full" state of the bucket.
Wilting_Point_PMP_%       Permanent wilting point: water percentage below
                           which the plant can no longer extract moisture
                           from the soil. The actual "empty" state of the
                           bucket (never reaches absolute 0%).
Available_Water_Capacity_AWC_%   CC - PMP: the range of water actually
                           usable by the plant.
AWC_layer_mm              Same but in mm of water for the actual thickness
                           of that layer -> summing all layers gives the
                           total available water depth in the root profile
                           (0-100cm), the value used with
                           precipitation_profile.py for water balance.
"""
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Standard SoilGrids depth intervals for soil profiles
DEPTHS = ['0-5cm', '5-15cm', '15-30cm', '30-60cm', '60-100cm']

# Layer thickness in centimeters for each standard depth interval
LAYER_THICKNESS_CM = {
    '0-5cm': 5, '5-15cm': 10, '15-30cm': 15, '30-60cm': 30, '60-100cm': 40,
}

# FIXED conversion factors for SoilGrids (mapped units -> conventional
# units). These do not depend on the value magnitude -> they are always
# applied the same way, regardless of how high or low the data is.
SOILGRIDS_CONVERSION = {
    'clay': 10.0,   # g/kg -> % (divide by 10 to get percentage)
    'sand': 10.0,   # g/kg -> % (divide by 10 to get percentage)
    'silt': 10.0,   # g/kg -> % (divide by 10 to get percentage)
    'soc': 100.0,   # dg/kg -> % (combined factor: /10 to g/kg, /10 more to %)
    'bdod': 100.0,  # cg/cm³ -> g/cm³ (divide by 100 to get g/cm³)
}


# ---------------------------------------------------------------------------
# Main Function
# ---------------------------------------------------------------------------

def calculate_hydraulic_properties(soil_data: dict) -> pd.DataFrame:
    """
    Calculates field capacity, wilting point, and available water
    per layer, using the Saxton & Rawls (2006) pedotransfer functions.

    Args:
        soil_data: dictionary with the form
                   {'clay': {depth: val}, 'sand': {...}, ...}
                   as returned by get_soil_profile_area() in soil_profile.py

    Returns:
        pandas.DataFrame with columns for each hydraulic property
        and rows for each depth layer
    """
    # Initialize results dictionary with empty dictionaries for each property
    hydric_results = {
        'Field_Capacity_CC_%': {},
        'Wilting_Point_PMP_%': {},
        'Available_Water_Capacity_AWC_%': {},
        'AWC_layer_mm': {},
    }

    for depth in DEPTHS:
        # Convert SoilGrids units to conventional units using fixed factors
        c_clay = float(soil_data['clay'][depth]) / SOILGRIDS_CONVERSION['clay']
        c_sand = float(soil_data['sand'][depth]) / SOILGRIDS_CONVERSION['sand']
        c_silt = float(soil_data['silt'][depth]) / SOILGRIDS_CONVERSION['silt']

        # Normalize texture fractions so they sum exactly to 100%
        # This corrects for any rounding or measurement inconsistencies in SoilGrids data
        total_text = c_clay + c_sand + c_silt
        if total_text > 0:
            c_clay = (c_clay / total_text) * 100
            c_sand = (c_sand / total_text) * 100
            c_silt = (c_silt / total_text) * 100

        s_frac = c_sand / 100.0   # Sand fraction (0-1), as required by S&R 2006
        c_frac = c_clay / 100.0   # Clay fraction (0-1), as required by S&R 2006

        # Organic matter: %SOC * 1.724 (Van Bemmelen factor)
        # Converts soil organic carbon to soil organic matter.
        # NOTE: in S&R 2006, OM is a percentage (not a fraction), unlike S and C.
        c_soc_pct = float(soil_data['soc'][depth]) / SOILGRIDS_CONVERSION['soc']
        c_om = max(0.1, min(c_soc_pct * 1.724, 15.0))  # Realistic limits for mineral soils

        # -------------------------------------------------------------------
        # Wilting point at 1500 kPa (Permanent Wilting Point, theta_1500)
        # -------------------------------------------------------------------

        # First-stage estimate (theta_1500t). Cross-product terms verified
        # against USDA-ARS saxpar.for and HydroTools.R:
        #   + 0.005*(S*OM)  and  - 0.013*(C*OM)
        wp_33 = (-0.024 * s_frac) + (0.487 * c_frac) + (0.006 * c_om) \
                + (0.005 * s_frac * c_om) - (0.013 * c_frac * c_om) \
                + (0.068 * s_frac * c_frac) + 0.031

        # Final wilting point (theta_1500)
        wp = wp_33 + (0.14 * wp_33) - 0.02

        # -------------------------------------------------------------------
        # Field capacity at 33 kPa (theta_33)
        # -------------------------------------------------------------------

        # First-stage estimate (theta_33t)
        fc_33 = (-0.251 * s_frac) + (0.195 * c_frac) + (0.011 * c_om) \
                + (0.006 * s_frac * c_om) - (0.027 * c_frac * c_om) \
                + (0.452 * s_frac * c_frac) + 0.299

        # Final field capacity (theta_33)
        fc = fc_33 + (1.283 * (fc_33 ** 2)) - (0.374 * fc_33) - 0.015

        # -------------------------------------------------------------------
        # Physical sanity limits
        # -------------------------------------------------------------------
        # Not part of the original equation: safeguards against out-of-range
        # extrapolations. Field capacity constrained between 5% and 55%
        # volumetric water content; wilting point between 2% and 40%.
        fc_pct = max(5.0, min(fc * 100, 55.0))
        wp_pct = max(2.0, min(wp * 100, 40.0))

        # Ensure field capacity is always greater than wilting point by at least 5%
        if fc_pct < wp_pct:
            fc_pct = wp_pct + 5.0

        # Calculate available water capacity as the difference between FC and WP
        awc_pct = fc_pct - wp_pct

        # Convert layer thickness from cm to mm and calculate AWC in mm
        thickness_mm = LAYER_THICKNESS_CM[depth] * 10
        awc_mm = (awc_pct / 100.0) * thickness_mm

        # Store rounded results for this depth layer
        hydric_results['Field_Capacity_CC_%'][depth] = round(fc_pct, 2)
        hydric_results['Wilting_Point_PMP_%'][depth] = round(wp_pct, 2)
        hydric_results['Available_Water_Capacity_AWC_%'][depth] = round(awc_pct, 2)
        hydric_results['AWC_layer_mm'][depth] = round(awc_mm, 2)

    return pd.DataFrame(hydric_results)
