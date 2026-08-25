"""
Biweekly Period Utilities
=========================

Generation of biweekly periods (1-15, 16-end of month), shared between
temperature_profile.py and precipitation_profile.py, so that both series
use exactly the same period_start/period_end and can be joined directly
by those columns.
"""
from datetime import date, timedelta
from typing import List, Tuple


def build_biweekly_periods(
    start_date: date,
    end_date: date,
) -> List[Tuple[str, date, date]]:
    """
    Generates a list of biweekly periods between start_date and end_date
    (Python date objects). Only includes already completed biweekly periods
    (period_end <= end_date + 1 day), to avoid including partial periods.

    Each period is a tuple containing:
        - period_label: string like '2024-01_Q1' or '2024-01_Q2'
        - period_start: date object marking the first day of the period
        - period_end: date object marking the day AFTER the last day of the period
                      (exclusive end, used for filterDate in Earth Engine)

    Biweekly breakdown:
        Q1: days 1-15 of the month (period_end = 16th, exclusive)
        Q2: days 16 to end of month (period_end = 1st of next month, exclusive)

    Args:
        start_date: datetime.date object for the beginning of the date range
        end_date: datetime.date object for the end of the date range

    Returns:
        list of tuples: [(label, period_start, period_end), ...]
                        ordered chronologically by month and quarter
    """
    periods = []

    # Start iterating from the first day of the start_date's month
    current = date(start_date.year, start_date.month, 1)

    while current <= end_date:
        year, month = current.year, current.month

        # First biweekly period (Q1): days 1-15
        q1_start = date(year, month, 1)
        q1_end = date(year, month, 16)  # Exclusive in filterDate -> covers days 1-15

        # Second biweekly period (Q2): days 16 to end of month
        q2_start = date(year, month, 16)
        # Handle December specially: Q2 ends on January 1st of next year
        if month == 12:
            q2_end = date(year + 1, 1, 1)
        else:
            q2_end = date(year, month + 1, 1)

        # Only include Q1/Q2 if their end date falls within the allowed range.
        # timedelta(days=1) allows the period to be considered complete even
        # on the exact end date (inclusive check).
        if q1_end <= end_date + timedelta(days=1):
            periods.append((f'{year}-{month:02d}_Q1', q1_start, q1_end))
        if q2_end <= end_date + timedelta(days=1):
            periods.append((f'{year}-{month:02d}_Q2', q2_start, q2_end))

        # Move to the next month (December -> January of the next year)
        current = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)

    return periods
