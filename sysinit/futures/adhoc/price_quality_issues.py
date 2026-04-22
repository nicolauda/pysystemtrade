from __future__ import annotations

import io
from contextlib import redirect_stdout
from dataclasses import dataclass

import pandas as pd

from syscore.constants import arg_not_supplied
from sysdata.csv.csv_roll_calendars import csvRollCalendarData
from sysdata.data_blob import dataBlob
from sysinit.futures.adjustedprices_from_db_multiple_to_db import (
    process_adjusted_prices_single_instrument,
)
from sysinit.futures.adhoc.repair_from_ib_to_db_non_destructive import (
    _contract_dates_with_price_data_for_instrument_code,
    _merge_multiple_prices_tail_only,
    _trim_unavailable_leading_roll_calendar_rows,
)
from sysinit.futures.multipleprices_from_db_prices_and_csv_calendars_to_db import (
    process_multiple_prices_single_instrument,
)
from sysobjects.adjusted_prices import futuresAdjustedPrices
from sysobjects.multiple_prices import futuresMultiplePrices
from sysproduction.data.prices import diagPrices
from sysproduction.reporting.data.price_quality import (
    DEFAULT_DB_PRICE_QUALITY_MIN_BAD_DAYS,
    assess_adjusted_price_alignment,
    assess_instrument_db_price_quality,
    drop_partial_gap_rows_from_multiple_prices,
    summarise_missing_current_price_gaps,
)


@dataclass
class PriceQualityCandidate:
    """Describe one candidate multiple-price series for repair scoring."""

    label: str
    multiple_prices: futuresMultiplePrices
    quality: dict[str, object]
    gap_summary: dict[str, object]


@dataclass
class PriceQualityRepairPlan:
    """Summarise the proposed repair for one instrument."""

    instrument_code: str
    current_multiple_prices: futuresMultiplePrices
    selected_multiple_prices: futuresMultiplePrices
    current_adjusted_prices: futuresAdjustedPrices
    rebuilt_adjusted_prices: futuresAdjustedPrices
    current_candidate: PriceQualityCandidate
    filtered_candidate: PriceQualityCandidate
    selected_candidate: PriceQualityCandidate
    tail_candidate: PriceQualityCandidate | None
    current_adjusted_alignment: dict[str, int]
    selected_adjusted_alignment: dict[str, int]
    write_multiple: bool
    write_adjusted: bool


def build_price_quality_repair_plan(
    data: dataBlob,
    instrument_code: str,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
    min_bad_days: int = DEFAULT_DB_PRICE_QUALITY_MIN_BAD_DAYS,
    include_tail_candidate: bool = True,
) -> PriceQualityRepairPlan:
    """Build a conservative repair plan for one instrument.

    The plan evaluates three candidates in order of increasing intervention:
    current DB multiple prices, the same series with partial-gap rows removed,
    and an optional tail rebuild from the current roll calendar followed by the
    same partial-gap filtering. The best candidate is selected by quality score
    without touching per-contract source prices or roll-calendar CSV files.
    """

    prices = diagPrices(data)
    current_multiple_prices = prices.get_multiple_prices(instrument_code)
    current_adjusted_prices = prices.get_adjusted_prices(instrument_code)

    current_candidate = _build_candidate(
        instrument_code=instrument_code,
        label="current",
        multiple_prices=current_multiple_prices,
        adjusted_prices=current_adjusted_prices,
        start_date=start_date,
        end_date=end_date,
        min_bad_days=min_bad_days,
    )

    filtered_multiple_prices = drop_partial_gap_rows_from_multiple_prices(
        current_multiple_prices
    )
    filtered_candidate = _build_candidate(
        instrument_code=instrument_code,
        label="drop_partial_gap_rows",
        multiple_prices=filtered_multiple_prices,
        adjusted_prices=current_adjusted_prices,
        start_date=start_date,
        end_date=end_date,
        min_bad_days=min_bad_days,
    )

    tail_candidate = None
    filtered_still_problematic = (
        int(filtered_candidate.quality["bad_rows"]) > 0
        or int(filtered_candidate.quality["contract_regressions"]) > 0
    )
    if include_tail_candidate and filtered_still_problematic:
        tail_rebuilt_multiple_prices = build_tail_rebuild_multiple_prices_candidate(
            instrument_code=instrument_code,
            current_multiple_prices=current_multiple_prices,
        )
        if tail_rebuilt_multiple_prices is not None:
            tail_rebuilt_filtered = drop_partial_gap_rows_from_multiple_prices(
                tail_rebuilt_multiple_prices
            )
            tail_candidate = _build_candidate(
                instrument_code=instrument_code,
                label="tail_rebuild_then_drop_partial_gap_rows",
                multiple_prices=tail_rebuilt_filtered,
                adjusted_prices=current_adjusted_prices,
                start_date=start_date,
                end_date=end_date,
                min_bad_days=min_bad_days,
            )

    selected_candidate = choose_best_price_quality_candidate(
        [current_candidate, filtered_candidate, tail_candidate]
    )
    selected_multiple_prices = selected_candidate.multiple_prices
    rebuilt_adjusted_prices = rebuild_adjusted_prices_from_multiple(
        instrument_code=instrument_code,
        multiple_prices=selected_multiple_prices,
    )

    current_adjusted_alignment = assess_adjusted_price_alignment(
        multiple_prices=current_multiple_prices,
        adjusted_prices=current_adjusted_prices,
    )
    selected_adjusted_alignment = assess_adjusted_price_alignment(
        multiple_prices=selected_multiple_prices,
        adjusted_prices=current_adjusted_prices,
    )

    write_multiple = not _multiple_prices_equal(
        current_multiple_prices,
        selected_multiple_prices,
    )
    write_adjusted = not _adjusted_prices_equal(
        current_adjusted_prices,
        rebuilt_adjusted_prices,
    )

    return PriceQualityRepairPlan(
        instrument_code=instrument_code,
        current_multiple_prices=current_multiple_prices,
        selected_multiple_prices=selected_multiple_prices,
        current_adjusted_prices=current_adjusted_prices,
        rebuilt_adjusted_prices=rebuilt_adjusted_prices,
        current_candidate=current_candidate,
        filtered_candidate=filtered_candidate,
        selected_candidate=selected_candidate,
        tail_candidate=tail_candidate,
        current_adjusted_alignment=current_adjusted_alignment,
        selected_adjusted_alignment=selected_adjusted_alignment,
        write_multiple=write_multiple,
        write_adjusted=write_adjusted,
    )


def build_tail_rebuild_multiple_prices_candidate(
    instrument_code: str,
    current_multiple_prices: futuresMultiplePrices,
) -> futuresMultiplePrices | None:
    """Build a tail-only multiple-price candidate from the current roll calendar."""

    available_contract_codes = _contract_dates_with_price_data_for_instrument_code(
        instrument_code
    )
    if len(available_contract_codes) == 0:
        return None

    roll_calendar = csvRollCalendarData(arg_not_supplied).get_roll_calendar(
        instrument_code
    )
    roll_calendar_for_multiple_prices = _trim_unavailable_leading_roll_calendar_rows(
        roll_calendar,
        available_contract_codes=available_contract_codes,
    )
    if len(roll_calendar_for_multiple_prices) == 0:
        return None

    with redirect_stdout(io.StringIO()):
        candidate_multiple_prices = process_multiple_prices_single_instrument(
            instrument_code=instrument_code,
            roll_calendar=roll_calendar_for_multiple_prices,
            adjust_calendar_to_prices=False,
            ADD_TO_DB=False,
            ADD_TO_CSV=False,
        )

    if len(candidate_multiple_prices) == 0:
        return None

    return _merge_multiple_prices_tail_only(
        existing_multiple_prices=current_multiple_prices,
        candidate_multiple_prices=candidate_multiple_prices,
    )


def rebuild_adjusted_prices_from_multiple(
    instrument_code: str,
    multiple_prices: futuresMultiplePrices,
) -> futuresAdjustedPrices:
    """Rebuild adjusted prices from a multiple-price series without DB writes."""

    with redirect_stdout(io.StringIO()):
        adjusted_prices = process_adjusted_prices_single_instrument(
            instrument_code=instrument_code,
            multiple_prices=multiple_prices,
            ADD_TO_DB=False,
            ADD_TO_CSV=False,
        )

    return futuresAdjustedPrices(pd.Series(adjusted_prices).sort_index())


def choose_best_price_quality_candidate(
    candidates: list[PriceQualityCandidate | None],
) -> PriceQualityCandidate:
    """Pick the best candidate by residual quality issues.

    Earlier candidates win ties so the selection remains conservative.
    """

    valid_candidates = [candidate for candidate in candidates if candidate is not None]
    if len(valid_candidates) == 0:
        raise ValueError("At least one price-quality candidate is required.")

    best_candidate = valid_candidates[0]
    best_score = _price_quality_score(best_candidate.quality)

    for candidate in valid_candidates[1:]:
        candidate_score = _price_quality_score(candidate.quality)
        if candidate_score < best_score:
            best_candidate = candidate
            best_score = candidate_score

    return best_candidate


def _build_candidate(
    instrument_code: str,
    label: str,
    multiple_prices: futuresMultiplePrices,
    adjusted_prices: futuresAdjustedPrices,
    start_date: pd.Timestamp | None,
    end_date: pd.Timestamp | None,
    min_bad_days: int,
) -> PriceQualityCandidate:
    quality = assess_instrument_db_price_quality(
        instrument_code=instrument_code,
        multiple_prices=pd.DataFrame(multiple_prices),
        adjusted_prices=pd.Series(adjusted_prices),
        start_date=start_date,
        end_date=end_date,
        min_bad_days=min_bad_days,
    )
    gap_summary = summarise_missing_current_price_gaps(pd.DataFrame(multiple_prices))

    return PriceQualityCandidate(
        label=label,
        multiple_prices=futuresMultiplePrices(
            pd.DataFrame(multiple_prices).sort_index()
        )
        if len(multiple_prices) > 0
        else futuresMultiplePrices.create_empty(),
        quality=quality,
        gap_summary=gap_summary,
    )


def _price_quality_score(quality: dict[str, object]) -> tuple[int, int, int, int, int]:
    return (
        int(quality["contract_regressions"]),
        int(quality["full_gap_days"]),
        int(quality["bad_days"]),
        int(quality["bad_rows"]),
        int(quality["longest_bad_day_streak"]),
    )


def _multiple_prices_equal(
    first: futuresMultiplePrices,
    second: futuresMultiplePrices,
) -> bool:
    first_df = pd.DataFrame(first).sort_index()
    second_df = pd.DataFrame(second).sort_index()
    return first_df.equals(second_df)


def _adjusted_prices_equal(
    first: futuresAdjustedPrices,
    second: futuresAdjustedPrices,
) -> bool:
    first_series = pd.Series(first).sort_index()
    second_series = pd.Series(second).sort_index()
    return first_series.equals(second_series)
