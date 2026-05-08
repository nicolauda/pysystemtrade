"""Regression: short calendar spans must not assume two yearly boundaries."""

import datetime as dt

from sysquant.fitting_dates import (
    EXPANDING,
    generate_fitting_dates_given_start_and_end_date,
)


def test_expanding_dummy_period_when_single_boundary():
    """Less than one interval between start and end leaves one boundary only."""
    start = dt.datetime(2026, 3, 1)
    end = dt.datetime(2026, 5, 1)
    out = generate_fitting_dates_given_start_and_end_date(
        start_date=start,
        end_date=end,
        date_method=EXPANDING,
        rollyears=20,
        interval_frequency="12M",
    )
    assert len(out) >= 1
