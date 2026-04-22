import pandas as pd

from syscore.constants import arg_not_supplied
from sysproduction.reporting.data.price_quality import (
    DEFAULT_DB_PRICE_QUALITY_MIN_BAD_DAYS,
    QUALITY_STATUS_ACTION,
    QUALITY_STATUS_CRITICAL,
    QUALITY_STATUS_MINOR,
    QUALITY_STATUS_ORDER,
    assess_instrument_db_price_quality,
    assess_adjusted_price_alignment,
    drop_partial_gap_rows_from_multiple_prices,
    normalise_price_quality_window,
    resolve_db_price_quality_dates,
    summarise_missing_current_price_gaps,
    summarise_db_price_quality,
)


def test_assess_instrument_db_price_quality_flags_actionable_bad_days():
    index = pd.DatetimeIndex(
        [
            "2026-01-02 23:00:00",
            "2026-01-05 23:00:00",
            "2026-01-06 23:00:00",
            "2026-01-07 23:00:00",
        ]
    )
    multiple_prices = pd.DataFrame(
        {
            "PRICE": [None, None, None, 10.5],
            "FORWARD": [9.9, 10.0, 10.1, 10.2],
            "CARRY": [9.7, 9.8, 9.9, 10.0],
            "PRICE_CONTRACT": ["20260300", "20260300", "20260300", "20260300"],
            "FORWARD_CONTRACT": ["20260400", "20260400", "20260400", "20260400"],
            "CARRY_CONTRACT": ["20260200", "20260200", "20260200", "20260200"],
        },
        index=index,
    )
    adjusted_prices = pd.Series([100.0, 101.0, 102.0, 103.0], index=index)

    quality = assess_instrument_db_price_quality(
        instrument_code="TEST",
        multiple_prices=multiple_prices,
        adjusted_prices=adjusted_prices,
        min_bad_days=DEFAULT_DB_PRICE_QUALITY_MIN_BAD_DAYS,
    )

    assert quality["status"] == QUALITY_STATUS_ACTION
    assert quality["bad_days"] == 3
    assert quality["longest_bad_day_streak"] == 3
    assert str(quality["streak_start"].date()) == "2026-01-02"
    assert str(quality["streak_end"].date()) == "2026-01-06"


def test_assess_instrument_db_price_quality_flags_minor_for_two_bad_days():
    index = pd.DatetimeIndex(
        ["2026-01-05 23:00:00", "2026-01-06 23:00:00", "2026-01-07 23:00:00"]
    )
    multiple_prices = pd.DataFrame(
        {
            "PRICE": [None, None, 10.5],
            "FORWARD": [9.9, 10.0, 10.2],
            "CARRY": [9.7, 9.8, 10.0],
            "PRICE_CONTRACT": ["20260300", "20260300", "20260300"],
            "FORWARD_CONTRACT": ["20260400", "20260400", "20260400"],
            "CARRY_CONTRACT": ["20260200", "20260200", "20260200"],
        },
        index=index,
    )
    adjusted_prices = pd.Series([100.0, 101.0, 102.0], index=index)

    quality = assess_instrument_db_price_quality(
        instrument_code="TEST",
        multiple_prices=multiple_prices,
        adjusted_prices=adjusted_prices,
        min_bad_days=DEFAULT_DB_PRICE_QUALITY_MIN_BAD_DAYS,
    )

    assert quality["status"] == QUALITY_STATUS_MINOR
    assert quality["bad_days"] == 2
    assert quality["contract_regressions"] == 0


def test_assess_instrument_db_price_quality_flags_critical_for_regression():
    index = pd.DatetimeIndex(
        ["2026-02-10 00:00:00", "2026-02-10 01:00:00", "2026-02-10 02:00:00"]
    )
    multiple_prices = pd.DataFrame(
        {
            "PRICE": [10.0, 10.1, 10.2],
            "FORWARD": [10.2, 10.3, 10.4],
            "CARRY": [9.9, 10.0, 10.1],
            "PRICE_CONTRACT": ["20260400", "20260300", "20260300"],
            "FORWARD_CONTRACT": ["20260500", "20260400", "20260400"],
            "CARRY_CONTRACT": ["20260500", "20260400", "20260400"],
        },
        index=index,
    )
    adjusted_prices = pd.Series([100.0, 100.1, 100.2], index=index)

    quality = assess_instrument_db_price_quality(
        instrument_code="TEST",
        multiple_prices=multiple_prices,
        adjusted_prices=adjusted_prices,
        min_bad_days=DEFAULT_DB_PRICE_QUALITY_MIN_BAD_DAYS,
    )

    assert quality["status"] == QUALITY_STATUS_CRITICAL
    assert quality["contract_regressions"] == 1
    assert pd.Timestamp("2026-02-10 01:00:00") == quality["last_regression"]


def test_resolve_db_price_quality_dates_handles_full_history():
    start_date, end_date, label = resolve_db_price_quality_dates(
        start_date=arg_not_supplied,
        end_date=arg_not_supplied,
        calendar_days_back=arg_not_supplied,
    )

    assert start_date is None
    assert end_date is None
    assert label == "full history"


def test_normalise_price_quality_window_expands_to_full_days():
    start_date, end_date = normalise_price_quality_window(
        start_date=pd.Timestamp("2026-01-05 12:34:56"),
        end_date=pd.Timestamp("2026-01-06 08:00:00"),
    )

    assert start_date == pd.Timestamp("2026-01-05 00:00:00")
    assert end_date == pd.Timestamp("2026-01-06 23:59:59.999999999")


def test_summarise_db_price_quality_respects_status_order():
    quality_df = pd.DataFrame(
        {"status": ["minor", "critical", "clean", "action", "critical"]},
        index=["A", "B", "C", "D", "E"],
    )

    summary_df = summarise_db_price_quality(quality_df)

    assert list(summary_df.index) == QUALITY_STATUS_ORDER
    assert summary_df.loc["critical", "count"] == 2
    assert summary_df.loc["action", "count"] == 1
    assert summary_df.loc["minor", "count"] == 1
    assert summary_df.loc["clean", "count"] == 1


def test_summarise_missing_current_price_gaps_distinguishes_partial_and_full_days():
    index = pd.DatetimeIndex(
        [
            "2026-01-05 09:00:00",
            "2026-01-05 10:00:00",
            "2026-01-06 09:00:00",
            "2026-01-06 10:00:00",
        ]
    )
    multiple_prices = pd.DataFrame(
        {
            "PRICE": [10.0, None, None, None],
            "FORWARD": [10.2, 10.3, 10.4, 10.5],
            "CARRY": [9.9, 10.0, 10.1, 10.2],
            "PRICE_CONTRACT": ["20260300"] * 4,
            "FORWARD_CONTRACT": ["20260400"] * 4,
            "CARRY_CONTRACT": ["20260200"] * 4,
        },
        index=index,
    )

    summary = summarise_missing_current_price_gaps(multiple_prices)

    assert summary["bad_rows"] == 3
    assert summary["bad_days"] == 2
    assert summary["partial_gap_rows"] == 1
    assert summary["partial_gap_days"] == 1
    assert summary["full_gap_rows"] == 2
    assert summary["full_gap_days"] == 1


def test_drop_partial_gap_rows_from_multiple_prices_keeps_full_gap_rows():
    index = pd.DatetimeIndex(
        [
            "2026-01-05 09:00:00",
            "2026-01-05 10:00:00",
            "2026-01-06 09:00:00",
        ]
    )
    multiple_prices = pd.DataFrame(
        {
            "PRICE": [10.0, None, None],
            "FORWARD": [10.2, 10.3, 10.4],
            "CARRY": [9.9, 10.0, 10.1],
            "PRICE_CONTRACT": ["20260300"] * 3,
            "FORWARD_CONTRACT": ["20260400"] * 3,
            "CARRY_CONTRACT": ["20260200"] * 3,
        },
        index=index,
    )

    filtered = pd.DataFrame(drop_partial_gap_rows_from_multiple_prices(multiple_prices))

    assert list(filtered.index) == [
        pd.Timestamp("2026-01-05 09:00:00"),
        pd.Timestamp("2026-01-06 09:00:00"),
    ]


def test_assess_adjusted_price_alignment_counts_missing_adjusted_rows():
    index = pd.DatetimeIndex(
        [
            "2026-01-05 09:00:00",
            "2026-01-05 10:00:00",
            "2026-01-06 09:00:00",
        ]
    )
    multiple_prices = pd.DataFrame(
        {
            "PRICE": [10.0, 10.2, 10.4],
            "FORWARD": [10.1, 10.3, 10.5],
            "CARRY": [9.9, 10.0, 10.2],
            "PRICE_CONTRACT": ["20260300"] * 3,
            "FORWARD_CONTRACT": ["20260400"] * 3,
            "CARRY_CONTRACT": ["20260200"] * 3,
        },
        index=index,
    )
    adjusted_prices = pd.Series(
        [100.0, 100.4],
        index=pd.DatetimeIndex(["2026-01-05 09:00:00", "2026-01-06 09:00:00"]),
    )

    alignment = assess_adjusted_price_alignment(
        multiple_prices=multiple_prices,
        adjusted_prices=adjusted_prices,
    )

    assert alignment["multiple_price_rows"] == 3
    assert alignment["adjusted_rows"] == 2
    assert alignment["missing_adjusted_rows"] == 1
