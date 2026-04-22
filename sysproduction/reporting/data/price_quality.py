from __future__ import annotations

import datetime

import pandas as pd

from syscore.constants import arg_not_supplied
from syscore.dateutils import calculate_start_and_end_dates
from sysdata.data_blob import dataBlob
from sysobjects.adjusted_prices import futuresAdjustedPrices
from sysobjects.multiple_prices import futuresMultiplePrices
from sysproduction.data.prices import diagPrices

ALL_DB_PRICE_QUALITY_INSTRUMENTS = "ALL"
DEFAULT_DB_PRICE_QUALITY_CALENDAR_DAYS = 365
DEFAULT_DB_PRICE_QUALITY_MIN_BAD_DAYS = 3

QUALITY_STATUS_CRITICAL = "critical"
QUALITY_STATUS_ACTION = "action"
QUALITY_STATUS_MINOR = "minor"
QUALITY_STATUS_CLEAN = "clean"

QUALITY_STATUS_ORDER = [
    QUALITY_STATUS_CRITICAL,
    QUALITY_STATUS_ACTION,
    QUALITY_STATUS_MINOR,
    QUALITY_STATUS_CLEAN,
]


def resolve_db_price_quality_dates(
    start_date: datetime.datetime | object = arg_not_supplied,
    end_date: datetime.datetime | object = arg_not_supplied,
    calendar_days_back: int | object = DEFAULT_DB_PRICE_QUALITY_CALENDAR_DAYS,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None, str]:
    """Resolve the diagnostic date window and a readable label.

    Args:
        start_date: Explicit lower bound for the analysis window. If
            `arg_not_supplied`, the value is derived from `calendar_days_back`.
        end_date: Explicit upper bound for the analysis window. If
            `arg_not_supplied`, the current datetime is used when
            `calendar_days_back` is set.
        calendar_days_back: Rolling lookback window used when explicit dates are
            not supplied. If this is also `arg_not_supplied`, full history is
            analysed.

    Returns:
        A tuple `(resolved_start, resolved_end, window_label)`. For full-history
        scans, the first two items are `None`.
    """

    if (
        start_date is arg_not_supplied
        and end_date is arg_not_supplied
        and calendar_days_back is arg_not_supplied
    ):
        return None, None, "full history"

    resolved_start, resolved_end = calculate_start_and_end_dates(
        calendar_days_back=calendar_days_back,
        start_date=start_date,
        end_date=end_date,
    )
    start_ts, end_ts = normalise_price_quality_window(
        start_date=resolved_start,
        end_date=resolved_end,
    )

    return start_ts, end_ts, f"{start_ts.date()} to {end_ts.date()}"


def normalise_price_quality_window(
    start_date: datetime.datetime | pd.Timestamp | None,
    end_date: datetime.datetime | pd.Timestamp | None,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """Expand a datetime window to inclusive whole-day bounds.

    Args:
        start_date: Optional lower bound. If provided, the returned timestamp is
            normalised to the start of that calendar day.
        end_date: Optional upper bound. If provided, the returned timestamp is
            normalised to the inclusive end of that calendar day.

    Returns:
        Tuple `(start_ts, end_ts)` with normalised inclusive day bounds.
    """

    start_ts = None
    if start_date is not None:
        start_ts = pd.Timestamp(start_date).normalize()

    end_ts = None
    if end_date is not None:
        end_ts = pd.Timestamp(end_date).normalize()
        end_ts = end_ts + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)

    return start_ts, end_ts


def assess_instrument_db_price_quality(
    instrument_code: str,
    multiple_prices: pd.DataFrame,
    adjusted_prices: pd.Series | pd.DataFrame | None,
    start_date: datetime.datetime | pd.Timestamp | None = None,
    end_date: datetime.datetime | pd.Timestamp | None = None,
    min_bad_days: int = DEFAULT_DB_PRICE_QUALITY_MIN_BAD_DAYS,
) -> dict[str, object]:
    """Assess derived-price quality for one instrument.

    Args:
        instrument_code: Instrument to evaluate.
        multiple_prices: Multiple-price dataframe for the instrument.
        adjusted_prices: Adjusted-price series or single-column dataframe.
        start_date: Inclusive lower bound for the evaluation window.
        end_date: Inclusive upper bound for the evaluation window.
        min_bad_days: Minimum number of bad business days required before the
            instrument is classified as actionable.

    Returns:
        A dictionary containing instrument-level quality statistics and a
        severity `status`.
    """

    multiple_prices_as_df = pd.DataFrame(multiple_prices).copy()
    multiple_prices_as_df.index = pd.DatetimeIndex(multiple_prices_as_df.index)
    multiple_prices_as_df = multiple_prices_as_df.sort_index()

    adjusted_series = _coerce_adjusted_prices_to_series(adjusted_prices)

    start_ts, end_ts = normalise_price_quality_window(
        start_date=start_date,
        end_date=end_date,
    )

    if start_ts is not None:
        multiple_prices_as_df = multiple_prices_as_df.loc[
            multiple_prices_as_df.index >= start_ts
        ]
        adjusted_series = adjusted_series.loc[adjusted_series.index >= start_ts]
    if end_ts is not None:
        multiple_prices_as_df = multiple_prices_as_df.loc[
            multiple_prices_as_df.index <= end_ts
        ]
        adjusted_series = adjusted_series.loc[adjusted_series.index <= end_ts]

    gap_summary = summarise_missing_current_price_gaps(multiple_prices_as_df)
    bad_mask = gap_summary["bad_row_mask"]
    bad_days = pd.DatetimeIndex(multiple_prices_as_df.index[bad_mask].normalize())
    unique_bad_days = bad_days.unique().sort_values()
    longest_streak, streak_start, streak_end = _longest_business_day_streak(
        unique_bad_days
    )
    last_bad_day = unique_bad_days.max() if len(unique_bad_days) else pd.NaT

    price_contract = multiple_prices_as_df["PRICE_CONTRACT"].dropna().astype(float)
    regression_mask = price_contract.diff() < 0
    contract_regressions = int(regression_mask.sum())
    last_regression = (
        multiple_prices_as_df.index[regression_mask].max()
        if contract_regressions > 0
        else pd.NaT
    )

    adjusted_non_na = adjusted_series.dropna()
    adjusted_missing = bool(
        multiple_prices_as_df.shape[0] > 0 and adjusted_non_na.empty
    )
    adjusted_alignment = assess_adjusted_price_alignment(
        multiple_prices=multiple_prices_as_df,
        adjusted_prices=adjusted_series,
    )

    return {
        "instrument": instrument_code,
        "status": _classify_quality_status(
            bad_days=len(unique_bad_days),
            contract_regressions=contract_regressions,
            adjusted_missing=adjusted_missing,
            min_bad_days=min_bad_days,
        ),
        "bad_rows": int(gap_summary["bad_rows"]),
        "bad_days": len(unique_bad_days),
        "partial_gap_rows": int(gap_summary["partial_gap_rows"]),
        "partial_gap_days": int(gap_summary["partial_gap_days"]),
        "full_gap_rows": int(gap_summary["full_gap_rows"]),
        "full_gap_days": int(gap_summary["full_gap_days"]),
        "longest_bad_day_streak": longest_streak,
        "streak_start": streak_start,
        "streak_end": streak_end,
        "last_bad_day": last_bad_day,
        "contract_regressions": contract_regressions,
        "last_regression": last_regression,
        "adjusted_missing": adjusted_missing,
        "adjusted_missing_rows": int(adjusted_alignment["missing_adjusted_rows"]),
        "adjusted_mismatch_rows": int(adjusted_alignment["adjusted_mismatch_rows"]),
        "multiple_rows": len(multiple_prices_as_df),
        "adjusted_rows": len(adjusted_non_na),
        "multiple_start": (
            multiple_prices_as_df.index.min()
            if not multiple_prices_as_df.empty
            else pd.NaT
        ),
        "multiple_end": (
            multiple_prices_as_df.index.max()
            if not multiple_prices_as_df.empty
            else pd.NaT
        ),
        "adjusted_start": adjusted_non_na.index.min()
        if not adjusted_non_na.empty
        else pd.NaT,
        "adjusted_end": adjusted_non_na.index.max()
        if not adjusted_non_na.empty
        else pd.NaT,
    }


def get_db_price_quality_df(
    data: dataBlob,
    instrument_code: str | object = arg_not_supplied,
    start_date: datetime.datetime | object = arg_not_supplied,
    end_date: datetime.datetime | object = arg_not_supplied,
    calendar_days_back: int | object = DEFAULT_DB_PRICE_QUALITY_CALENDAR_DAYS,
    min_bad_days: int = DEFAULT_DB_PRICE_QUALITY_MIN_BAD_DAYS,
) -> tuple[pd.DataFrame, str]:
    """Return DB price-quality diagnostics for one or more instruments.

    Args:
        data: Production data blob.
        instrument_code: Optional instrument filter. Use
            `ALL_DB_PRICE_QUALITY_INSTRUMENTS` or `arg_not_supplied` for all
            instruments in DB multiple prices.
        start_date: Inclusive lower bound for the analysis window.
        end_date: Inclusive upper bound for the analysis window.
        calendar_days_back: Rolling lookback window used when explicit dates are
            not supplied.
        min_bad_days: Minimum number of bad business days required before an
            instrument is classified as actionable.

    Returns:
        A tuple `(quality_df, window_label)` where `quality_df` is indexed by
        instrument code and sorted by severity.
    """

    resolved_start, resolved_end, window_label = resolve_db_price_quality_dates(
        start_date=start_date,
        end_date=end_date,
        calendar_days_back=calendar_days_back,
    )

    prices = diagPrices(data)
    instrument_list = prices.get_list_of_instruments_in_multiple_prices()
    if (
        instrument_code is not arg_not_supplied
        and instrument_code != ALL_DB_PRICE_QUALITY_INSTRUMENTS
    ):
        instrument_list = [instrument_code]

    records = []
    for current_instrument in sorted(instrument_list):
        multiple_prices = prices.get_multiple_prices(current_instrument)
        adjusted_prices = prices.get_adjusted_prices(current_instrument)
        records.append(
            assess_instrument_db_price_quality(
                instrument_code=current_instrument,
                multiple_prices=pd.DataFrame(multiple_prices),
                adjusted_prices=pd.Series(adjusted_prices),
                start_date=resolved_start,
                end_date=resolved_end,
                min_bad_days=min_bad_days,
            )
        )

    if not records:
        return pd.DataFrame(), window_label

    quality_df = pd.DataFrame(records).set_index("instrument")
    quality_df["status_rank"] = quality_df["status"].apply(QUALITY_STATUS_ORDER.index)
    quality_df = quality_df.sort_values(
        by=[
            "status_rank",
            "bad_days",
            "contract_regressions",
            "longest_bad_day_streak",
        ],
        ascending=[True, False, False, False],
    ).drop(columns=["status_rank"])

    return quality_df, window_label


def summarise_db_price_quality(quality_df: pd.DataFrame) -> pd.DataFrame:
    """Summarise instrument counts by quality status."""

    if quality_df.empty:
        return pd.DataFrame(columns=["count"])

    return (
        quality_df["status"]
        .value_counts()
        .reindex(QUALITY_STATUS_ORDER, fill_value=0)
        .rename_axis("status")
        .to_frame("count")
    )


def get_missing_current_price_row_mask(multiple_prices: pd.DataFrame) -> pd.Series:
    """Return mask of rows where current price is missing but forward exists."""

    multiple_prices_as_df = pd.DataFrame(multiple_prices).copy()
    if len(multiple_prices_as_df) == 0:
        return pd.Series(dtype=bool)

    return (
        multiple_prices_as_df["PRICE"].isna() & multiple_prices_as_df["FORWARD"].notna()
    )


def summarise_missing_current_price_gaps(
    multiple_prices: pd.DataFrame,
) -> dict[str, object]:
    """Summarise partial-day and full-day current-price gaps.

    A partial-gap day contains at least one missing-current row but also at
    least one valid `PRICE` observation on the same calendar day. A full-gap
    day contains missing-current rows and no valid `PRICE` observations while
    forward prices are still available that day.
    """

    multiple_prices_as_df = pd.DataFrame(multiple_prices).copy()
    multiple_prices_as_df.index = pd.DatetimeIndex(multiple_prices_as_df.index)
    multiple_prices_as_df = multiple_prices_as_df.sort_index()

    if multiple_prices_as_df.empty:
        return dict(
            bad_row_mask=pd.Series(dtype=bool),
            partial_gap_row_mask=pd.Series(dtype=bool),
            full_gap_row_mask=pd.Series(dtype=bool),
            bad_rows=0,
            bad_days=0,
            partial_gap_rows=0,
            partial_gap_days=0,
            full_gap_rows=0,
            full_gap_days=0,
        )

    bad_row_mask = get_missing_current_price_row_mask(multiple_prices_as_df)
    if not bad_row_mask.any():
        empty_mask = pd.Series(False, index=multiple_prices_as_df.index)
        return dict(
            bad_row_mask=empty_mask,
            partial_gap_row_mask=empty_mask.copy(),
            full_gap_row_mask=empty_mask.copy(),
            bad_rows=0,
            bad_days=0,
            partial_gap_rows=0,
            partial_gap_days=0,
            full_gap_rows=0,
            full_gap_days=0,
        )

    with_day = multiple_prices_as_df.copy()
    with_day["__day"] = with_day.index.normalize()
    day_has_price = with_day.groupby("__day")["PRICE"].transform(
        lambda series: series.notna().any()
    )
    day_has_forward = with_day.groupby("__day")["FORWARD"].transform(
        lambda series: series.notna().any()
    )

    partial_gap_row_mask = bad_row_mask & day_has_price
    full_gap_row_mask = bad_row_mask & (~day_has_price) & day_has_forward

    return dict(
        bad_row_mask=bad_row_mask,
        partial_gap_row_mask=partial_gap_row_mask,
        full_gap_row_mask=full_gap_row_mask,
        bad_rows=int(bad_row_mask.sum()),
        bad_days=int(with_day.loc[bad_row_mask, "__day"].nunique()),
        partial_gap_rows=int(partial_gap_row_mask.sum()),
        partial_gap_days=int(with_day.loc[partial_gap_row_mask, "__day"].nunique()),
        full_gap_rows=int(full_gap_row_mask.sum()),
        full_gap_days=int(with_day.loc[full_gap_row_mask, "__day"].nunique()),
    )


def drop_partial_gap_rows_from_multiple_prices(
    multiple_prices: pd.DataFrame | futuresMultiplePrices,
) -> futuresMultiplePrices:
    """Drop rows that are missing current price only due to same-day misalignment."""

    multiple_prices_as_df = pd.DataFrame(multiple_prices).copy()
    multiple_prices_as_df.index = pd.DatetimeIndex(multiple_prices_as_df.index)
    multiple_prices_as_df = multiple_prices_as_df.sort_index()
    if multiple_prices_as_df.empty:
        return futuresMultiplePrices.create_empty()

    gap_summary = summarise_missing_current_price_gaps(multiple_prices_as_df)
    filtered_multiple_prices = multiple_prices_as_df.loc[
        ~gap_summary["partial_gap_row_mask"]
    ].copy()
    if filtered_multiple_prices.empty:
        return futuresMultiplePrices.create_empty()

    return futuresMultiplePrices(filtered_multiple_prices)


def assess_adjusted_price_alignment(
    multiple_prices: pd.DataFrame | futuresMultiplePrices,
    adjusted_prices: pd.Series | pd.DataFrame | None,
) -> dict[str, int]:
    """Measure whether adjusted prices are aligned with stitched multiple prices."""

    multiple_prices_as_df = pd.DataFrame(multiple_prices).copy()
    multiple_prices_as_df.index = pd.DatetimeIndex(multiple_prices_as_df.index)
    multiple_prices_as_df = multiple_prices_as_df.sort_index()
    adjusted_series = _coerce_adjusted_prices_to_series(adjusted_prices)

    if multiple_prices_as_df.empty:
        return dict(
            multiple_price_rows=0,
            adjusted_rows=int(adjusted_series.dropna().shape[0]),
            rebuilt_adjusted_rows=0,
            missing_adjusted_rows=0,
            adjusted_mismatch_rows=0,
        )

    multiple_prices_object = futuresMultiplePrices(multiple_prices_as_df)
    rebuilt_adjusted = futuresAdjustedPrices.stitch_multiple_prices(
        multiple_prices_object,
        forward_fill=True,
    )
    rebuilt_adjusted_series = pd.Series(rebuilt_adjusted).sort_index()

    price_index = multiple_prices_as_df.index[multiple_prices_as_df["PRICE"].notna()]
    adjusted_index = adjusted_series.dropna().index
    missing_adjusted_rows = int((~price_index.isin(adjusted_index)).sum())

    common_index = adjusted_index.intersection(rebuilt_adjusted_series.dropna().index)
    adjusted_mismatch_rows = 0
    if len(common_index) > 0:
        adjusted_mismatch_rows = int(
            (
                rebuilt_adjusted_series.loc[common_index]
                .sub(adjusted_series.loc[common_index])
                .abs()
                > 1e-9
            ).sum()
        )

    return dict(
        multiple_price_rows=int(multiple_prices_as_df["PRICE"].notna().sum()),
        adjusted_rows=int(adjusted_series.dropna().shape[0]),
        rebuilt_adjusted_rows=int(rebuilt_adjusted_series.dropna().shape[0]),
        missing_adjusted_rows=missing_adjusted_rows,
        adjusted_mismatch_rows=adjusted_mismatch_rows,
    )


def _coerce_adjusted_prices_to_series(
    adjusted_prices: pd.Series | pd.DataFrame | None,
) -> pd.Series:
    if adjusted_prices is None:
        return pd.Series(dtype=float)

    adjusted_as_series = pd.Series(adjusted_prices).copy()
    adjusted_as_series.index = pd.DatetimeIndex(adjusted_as_series.index)

    return adjusted_as_series.sort_index()


def _longest_business_day_streak(
    business_days: pd.DatetimeIndex,
) -> tuple[int, pd.Timestamp | pd.NaT, pd.Timestamp | pd.NaT]:
    if len(business_days) == 0:
        return 0, pd.NaT, pd.NaT

    current_start = business_days[0]
    current_end = business_days[0]
    current_length = 1
    longest_length = 1
    longest_start = business_days[0]
    longest_end = business_days[0]

    for day in business_days[1:]:
        if day == current_end + pd.offsets.BDay(1):
            current_end = day
            current_length += 1
            continue

        if current_length > longest_length:
            longest_length = current_length
            longest_start = current_start
            longest_end = current_end

        current_start = day
        current_end = day
        current_length = 1

    if current_length > longest_length:
        longest_length = current_length
        longest_start = current_start
        longest_end = current_end

    return longest_length, longest_start, longest_end


def _classify_quality_status(
    bad_days: int,
    contract_regressions: int,
    adjusted_missing: bool,
    min_bad_days: int,
) -> str:
    if adjusted_missing or contract_regressions > 0:
        return QUALITY_STATUS_CRITICAL
    if bad_days >= min_bad_days:
        return QUALITY_STATUS_ACTION
    if bad_days > 0:
        return QUALITY_STATUS_MINOR

    return QUALITY_STATUS_CLEAN
