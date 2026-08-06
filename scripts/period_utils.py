"""
Generación de periodos quincenales (1-15, 16-fin de mes), compartida
entre temperature_profile.py y precipitation_profile.py, para que ambas
series usen exactamente los mismos periodo_inicio/periodo_fin y se
puedan cruzar directo por esa columna.
"""
from datetime import date, timedelta


def build_biweekly_periods(start_date, end_date):
    """
    Genera la lista de periodos quincenales entre start_date y end_date
    (objetos date de Python). Solo incluye quincenas ya completadas
    (period_end <= end_date + 1 día), para no meter periodos parciales.
    """
    periods = []
    current = date(start_date.year, start_date.month, 1)

    while current <= end_date:
        year, month = current.year, current.month

        q1_start = date(year, month, 1)
        q1_end = date(year, month, 16)  # exclusivo en filterDate -> cubre días 1-15

        q2_start = date(year, month, 16)
        if month == 12:
            q2_end = date(year + 1, 1, 1)
        else:
            q2_end = date(year, month + 1, 1)

        if q1_end <= end_date + timedelta(days=1):
            periods.append((f'{year}-{month:02d}_Q1', q1_start, q1_end))
        if q2_end <= end_date + timedelta(days=1):
            periods.append((f'{year}-{month:02d}_Q2', q2_start, q2_end))

        current = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)

    return periods
