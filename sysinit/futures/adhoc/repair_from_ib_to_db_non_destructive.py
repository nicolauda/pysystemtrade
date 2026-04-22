import argparse
from dataclasses import dataclass

import pandas as pd

from syscore.constants import arg_not_supplied
from syscore.dateutils import DAILY_PRICE_FREQ, MIXED_FREQ, Frequency
from syscore.exceptions import missingData
from syscore.interactive.input import true_if_answer_is_yes
from syscore.pandas.frequency import merge_data_with_different_freq
from sysdata.csv.csv_roll_calendars import csvRollCalendarData
from sysdata.data_blob import dataBlob
from sysinit.futures.adjustedprices_from_db_multiple_to_db import (
    process_adjusted_prices_single_instrument,
)
from sysinit.futures.multipleprices_from_db_prices_and_csv_calendars_to_db import (
    process_multiple_prices_single_instrument,
)
from sysinit.futures.rollcalendars_from_db_prices_to_csv import (
    build_and_write_roll_calendar,
)
from sysobjects.contracts import futuresContract
from sysobjects.adjusted_prices import futuresAdjustedPrices
from sysobjects.futures_per_contract_prices import futuresContractPrices
from sysobjects.multiple_prices import futuresMultiplePrices
from sysobjects.roll_calendars import rollCalendar
from sysproduction.data.broker import dataBroker
from sysproduction.data.prices import diagPrices, updatePrices


@dataclass
class MergeOutcome:
    """Summarise one non-destructive merge operation.

    Args:
        existing_rows: Number of rows currently in the database.
        incoming_rows: Number of rows fetched from IB (or rebuilt for mixed freq).
        merged_rows: Number of rows after non-destructive merge.
        rows_added: Number of rows effectively added to the database.
    """

    existing_rows: int
    incoming_rows: int
    merged_rows: int
    rows_added: int


@dataclass
class RollCalendarMergeOutcome:
    """Summarise conservative roll calendar update results.

    Args:
        existing_rows: Number of rows in existing calendar CSV.
        candidate_rows: Number of rows in newly built candidate calendar.
        merged_rows: Number of rows after conservative merge.
        rows_added: Number of new roll dates added from candidate calendar.
        rows_replaced: Number of existing roll dates updated from candidate.
    """

    existing_rows: int
    candidate_rows: int
    merged_rows: int
    rows_added: int
    rows_replaced: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repair per-contract futures prices from IB without overwriting existing "
            "values on overlapping timestamps."
        )
    )
    parser.add_argument(
        "--instrument",
        dest="instrument_code",
        default=None,
        help="Instrument code to repair (if omitted, prompted interactively).",
    )
    parser.add_argument(
        "--rebuild-derived",
        action="store_true",
        help=(
            "Rebuild roll calendar (CSV), multiple prices, and adjusted prices "
            "after repair."
        ),
    )
    parser.add_argument(
        "--skip-derived",
        action="store_true",
        help=(
            "Skip roll calendar/multiple/adjusted rebuild after repair. "
            "If neither flag is set, the script asks interactively."
        ),
    )
    return parser.parse_args()


def _normalise_prices(price_data: futuresContractPrices) -> futuresContractPrices:
    """Return sorted and de-duplicated price data.

    Args:
        price_data: Price series to clean.

    Returns:
        `futuresContractPrices` sorted by timestamp with duplicate index rows removed.
    """

    as_dataframe = price_data.sort_index()
    as_dataframe = as_dataframe[~as_dataframe.index.duplicated(keep="first")]
    return futuresContractPrices(as_dataframe)


def _merge_prices_non_destructive(
    existing_prices: futuresContractPrices,
    incoming_prices: futuresContractPrices,
) -> tuple[futuresContractPrices, int]:
    """Merge prices preserving existing rows on timestamp overlap.

    The merge is intentionally strict at row level: if a timestamp is already
    present in DB, the existing row is kept as-is (including any NaN fields).
    Incoming data is used only for timestamps not already present.

    Args:
        existing_prices: Existing series in DB.
        incoming_prices: New series fetched from IB.

    Returns:
        Tuple `(merged_prices, rows_added)`.
    """

    existing_as_dataframe = pd.DataFrame(existing_prices).copy()
    incoming_as_dataframe = pd.DataFrame(incoming_prices).copy()

    if len(existing_as_dataframe) == 0:
        merged_as_dataframe = incoming_as_dataframe
    else:
        merged_as_dataframe = pd.concat(
            [existing_as_dataframe, incoming_as_dataframe], axis=0
        )
        merged_as_dataframe = merged_as_dataframe[
            ~merged_as_dataframe.index.duplicated(keep="first")
        ]

    merged_as_dataframe = merged_as_dataframe.sort_index()
    merged_prices = futuresContractPrices(merged_as_dataframe)
    rows_added = len(merged_prices) - len(existing_prices)
    return merged_prices, rows_added


def _merge_and_write_prices_for_frequency(
    contract: futuresContract,
    incoming_prices: futuresContractPrices,
    frequency: Frequency,
    diag_prices: diagPrices,
    prices_updater: updatePrices,
) -> MergeOutcome:
    """Merge incoming prices into DB and write only when rows are added.

    Args:
        contract: Contract being updated.
        incoming_prices: New prices for the specified frequency.
        frequency: Target frequency in DB.
        diag_prices: Price diagnostics/data reader layer.
        prices_updater: Price updater/data writer layer.

    Returns:
        `MergeOutcome` with row counts and number of added rows.
    """

    normalised_incoming = _normalise_prices(incoming_prices)
    existing_prices = diag_prices.get_prices_at_frequency_for_contract_object(
        contract, frequency=frequency
    )
    normalised_existing = _normalise_prices(existing_prices)
    merged_prices, rows_added = _merge_prices_non_destructive(
        existing_prices=normalised_existing,
        incoming_prices=normalised_incoming,
    )

    if rows_added > 0:
        prices_updater.overwrite_prices_at_frequency_for_contract(
            contract_object=contract,
            new_prices=merged_prices,
            frequency=frequency,
        )

    return MergeOutcome(
        existing_rows=len(normalised_existing),
        incoming_rows=len(normalised_incoming),
        merged_rows=len(merged_prices),
        rows_added=rows_added,
    )


def _mixed_prices_candidate_from_db(
    contract: futuresContract,
    intraday_frequency: Frequency,
    diag_prices: diagPrices,
) -> futuresContractPrices:
    """Build mixed-frequency candidate prices from repaired intraday+daily data.

    Args:
        contract: Contract to rebuild.
        intraday_frequency: Intraday frequency configured for historical updates.
        diag_prices: Price diagnostics/data reader layer.

    Returns:
        A `futuresContractPrices` object for mixed frequency.
    """

    frequencies_to_merge = [intraday_frequency, DAILY_PRICE_FREQ]
    if intraday_frequency == DAILY_PRICE_FREQ:
        frequencies_to_merge = [DAILY_PRICE_FREQ]

    list_of_data = []
    for frequency in frequencies_to_merge:
        price_data = diag_prices.get_prices_at_frequency_for_contract_object(
            contract, frequency=frequency
        )
        price_data = _normalise_prices(price_data)
        if len(price_data) > 0:
            list_of_data.append(price_data)

    if len(list_of_data) == 0:
        return futuresContractPrices.create_empty()
    if len(list_of_data) == 1:
        return list_of_data[0]

    merged_prices = merge_data_with_different_freq(list_of_data)
    return futuresContractPrices(merged_prices)


def _resolve_instrument_code(instrument_code: str | None) -> str:
    """Resolve instrument code from arg or interactive prompt.

    Args:
        instrument_code: Optional code passed by CLI.

    Returns:
        Non-empty instrument code.

    Raises:
        ValueError: If no instrument code is provided.
    """

    if instrument_code is None:
        instrument_code = input("Instrument code to repair? <return to abort> ").strip()

    if instrument_code == "":
        raise ValueError("No instrument code provided. Aborting.")

    return instrument_code


def _resolve_rebuild_choice(args: argparse.Namespace) -> bool:
    """Resolve whether to rebuild roll/multiple/adjusted data.

    Args:
        args: Parsed CLI arguments.

    Returns:
        `True` when derived data should be rebuilt.

    Raises:
        ValueError: If conflicting flags are provided.
    """

    if args.rebuild_derived and args.skip_derived:
        raise ValueError("Use only one of --rebuild-derived or --skip-derived.")
    if args.rebuild_derived:
        return True
    if args.skip_derived:
        return False

    answer = true_if_answer_is_yes(
        "Rebuild roll calendar + multiple + adjusted prices after repair? [y/N] ",
        allow_empty_to_return_none=True,
    )
    if answer is None:
        return False
    return answer


def _rebuild_derived_data_for_instrument(instrument_code: str):
    """Rebuild roll calendar and derived series for one instrument.

    Args:
        instrument_code: Instrument code to rebuild.

    Notes:
        Multiple prices rebuild keeps the calendar dates already written by the
        conservative roll-calendar step (`adjust_calendar_to_prices=False`).
        This avoids accidental calendar truncation on instruments with sparse
        concurrent contract timestamps.
    """

    calendar_outcome = _build_and_write_roll_calendar_conservative(
        instrument_code=instrument_code
    )
    print(
        "Roll calendar conservative update: "
        f"existing={calendar_outcome.existing_rows}, "
        f"candidate={calendar_outcome.candidate_rows}, "
        f"added={calendar_outcome.rows_added}, "
        f"replaced={calendar_outcome.rows_replaced}"
    )
    csv_roll_calendars = csvRollCalendarData(arg_not_supplied)
    roll_calendar = csv_roll_calendars.get_roll_calendar(instrument_code)
    available_contract_codes = _contract_dates_with_price_data_for_instrument_code(
        instrument_code
    )
    roll_calendar_for_multiple_prices = _trim_unavailable_leading_roll_calendar_rows(
        roll_calendar,
        available_contract_codes=available_contract_codes,
    )
    candidate_multiple_prices = process_multiple_prices_single_instrument(
        instrument_code=instrument_code,
        roll_calendar=roll_calendar_for_multiple_prices,
        adjust_calendar_to_prices=False,
        ADD_TO_DB=False,
        ADD_TO_CSV=False,
    )
    with dataBlob(log_name=f"Repair-Derived-{instrument_code}") as data:
        diag_prices = diagPrices(data)
        existing_multiple_prices = diag_prices.get_multiple_prices(instrument_code)
        merged_multiple_prices = _merge_multiple_prices_tail_only(
            existing_multiple_prices=existing_multiple_prices,
            candidate_multiple_prices=candidate_multiple_prices,
        )

        diag_prices.db_futures_multiple_prices_data.add_multiple_prices(
            instrument_code,
            merged_multiple_prices,
            ignore_duplication=True,
        )
        candidate_adjusted_prices = process_adjusted_prices_single_instrument(
            instrument_code=instrument_code,
            multiple_prices=merged_multiple_prices,
            ADD_TO_DB=False,
            ADD_TO_CSV=False,
        )
        existing_adjusted_prices = diag_prices.get_adjusted_prices(instrument_code)
        merged_adjusted_prices = _merge_adjusted_prices_tail_only(
            existing_adjusted_prices=existing_adjusted_prices,
            candidate_adjusted_prices=candidate_adjusted_prices,
        )

        diag_prices.db_futures_adjusted_prices_data.add_adjusted_prices(
            instrument_code,
            merged_adjusted_prices,
            ignore_duplication=True,
        )


def _merge_multiple_prices_tail_only(
    existing_multiple_prices: futuresMultiplePrices,
    candidate_multiple_prices: futuresMultiplePrices,
) -> futuresMultiplePrices:
    """Replace only the rebuildable tail of multiple prices.

    Args:
        existing_multiple_prices: Current DB multiple prices.
        candidate_multiple_prices: Freshly rebuilt recent tail.

    Returns:
        A merged series that preserves existing history strictly before the
        candidate start timestamp, overwrites the candidate window, and keeps any
        existing tail strictly after the candidate end timestamp.
    """

    existing_as_dataframe = pd.DataFrame(existing_multiple_prices).sort_index()
    candidate_as_dataframe = pd.DataFrame(candidate_multiple_prices).sort_index()
    if len(candidate_as_dataframe) == 0:
        return futuresMultiplePrices(existing_as_dataframe)
    if len(existing_as_dataframe) == 0:
        return futuresMultiplePrices(candidate_as_dataframe)

    candidate_start = candidate_as_dataframe.index.min()
    candidate_end = candidate_as_dataframe.index.max()
    preserved_prefix = existing_as_dataframe[existing_as_dataframe.index < candidate_start]
    preserved_suffix = existing_as_dataframe[existing_as_dataframe.index > candidate_end]
    merged_as_dataframe = pd.concat(
        [preserved_prefix, candidate_as_dataframe, preserved_suffix], axis=0
    )
    merged_as_dataframe = merged_as_dataframe.sort_index()
    merged_as_dataframe = merged_as_dataframe[
        ~merged_as_dataframe.index.duplicated(keep="last")
    ]

    return futuresMultiplePrices(merged_as_dataframe)


def _merge_adjusted_prices_tail_only(
    existing_adjusted_prices: futuresAdjustedPrices,
    candidate_adjusted_prices: futuresAdjustedPrices,
) -> futuresAdjustedPrices:
    """Replace only the rebuildable tail of adjusted prices."""

    existing_as_series = pd.Series(existing_adjusted_prices).sort_index()
    candidate_as_series = pd.Series(candidate_adjusted_prices).sort_index()
    if len(candidate_as_series) == 0:
        return futuresAdjustedPrices(existing_as_series)
    if len(existing_as_series) == 0:
        return futuresAdjustedPrices(candidate_as_series)

    candidate_start = candidate_as_series.index.min()
    candidate_end = candidate_as_series.index.max()
    preserved_prefix = existing_as_series[existing_as_series.index < candidate_start]
    preserved_suffix = existing_as_series[existing_as_series.index > candidate_end]
    merged_as_series = pd.concat(
        [preserved_prefix, candidate_as_series, preserved_suffix], axis=0
    )
    merged_as_series = merged_as_series.sort_index()
    merged_as_series = merged_as_series[
        ~merged_as_series.index.duplicated(keep="last")
    ]

    return futuresAdjustedPrices(merged_as_series)


def _empty_roll_calendar() -> rollCalendar:
    """Return an empty roll-calendar object with expected columns."""

    empty_dataframe = pd.DataFrame(
        columns=["current_contract", "next_contract", "carry_contract"]
    )
    return rollCalendar(empty_dataframe)


def _normalise_roll_calendar(calendar: rollCalendar) -> rollCalendar:
    """Return sorted, de-duplicated roll calendar.

    Args:
        calendar: Calendar to normalise.

    Returns:
        `rollCalendar` sorted by index and deduplicated on datetime index.
    """

    if len(calendar) == 0:
        return _empty_roll_calendar()

    as_dataframe = pd.DataFrame(calendar).sort_index()
    as_dataframe = as_dataframe[~as_dataframe.index.duplicated(keep="first")]
    return rollCalendar(as_dataframe)


def _calendar_rows_equal(existing_row: pd.Series, candidate_row: pd.Series) -> bool:
    """Return `True` if two roll-calendar rows are equivalent.

    Args:
        existing_row: Row from existing calendar.
        candidate_row: Row from candidate calendar.

    Returns:
        `True` when current/next/carry contracts match.
    """

    columns = ["current_contract", "next_contract", "carry_contract"]
    return all(
        str(existing_row[column]) == str(candidate_row[column]) for column in columns
    )


def _normalise_contract_code(raw_contract_code: object) -> str | None:
    """Normalise contract code string and validate expected format."""

    if pd.isna(raw_contract_code):
        return None

    contract_code = str(raw_contract_code).strip()
    if contract_code == "":
        return None
    if contract_code.endswith(".0") and contract_code[:-2].isdigit():
        contract_code = contract_code[:-2]

    if contract_code.isdigit() and len(contract_code) in (6, 8):
        return contract_code

    return None


def _roll_transition_key(calendar_row: pd.Series) -> tuple[str, str, str] | None:
    """Return a normalised transition key `(current, next, carry)` for one row."""

    required_columns = ("current_contract", "next_contract", "carry_contract")
    if any(column not in calendar_row for column in required_columns):
        return None

    current_contract = _normalise_contract_code(calendar_row["current_contract"])
    next_contract = _normalise_contract_code(calendar_row["next_contract"])
    carry_contract = _normalise_contract_code(calendar_row["carry_contract"])
    if current_contract is None or next_contract is None or carry_contract is None:
        return None

    return (current_contract, next_contract, carry_contract)


def _drop_duplicate_transition_rows_keep_first(calendar: rollCalendar) -> rollCalendar:
    """Drop duplicate transition rows, keeping the earliest timestamp.

    Two rows are considered duplicates when `(current, next, carry)` contracts
    are equal after normalisation. Rows with unparseable contract codes are
    preserved.
    """

    calendar_norm = _normalise_roll_calendar(calendar)
    if len(calendar_norm) == 0:
        return _empty_roll_calendar()

    calendar_dataframe = pd.DataFrame(calendar_norm).sort_index()
    seen_transition_keys: set[tuple[str, str, str]] = set()
    kept_row_indices = []
    for roll_date, calendar_row in calendar_dataframe.iterrows():
        transition_key = _roll_transition_key(calendar_row)
        if transition_key is None:
            kept_row_indices.append(roll_date)
            continue
        if transition_key in seen_transition_keys:
            continue

        seen_transition_keys.add(transition_key)
        kept_row_indices.append(roll_date)

    deduplicated_calendar = calendar_dataframe.loc[kept_row_indices].copy()
    deduplicated_calendar = deduplicated_calendar.sort_index()
    return rollCalendar(deduplicated_calendar)


def _roll_calendar_row_is_buildable_from_available_contracts(
    calendar_row: pd.Series,
    *,
    is_last_row: bool,
    available_contract_codes: set[str],
) -> bool:
    """Return whether one roll-calendar row can be rebuilt from current DB prices.

    Args:
        calendar_row: Row to validate.
        is_last_row: Whether the row is the final row in the calendar.
        available_contract_codes: Contract date strings that currently have price
            data in DB.

    Returns:
        `True` if the row has the contract coverage required by the multiple-price
        rebuild. The last row only requires the current contract because the builder
        already tolerates missing next/carry there.
    """

    current_contract = _normalise_contract_code(calendar_row.get("current_contract"))
    next_contract = _normalise_contract_code(calendar_row.get("next_contract"))
    carry_contract = _normalise_contract_code(calendar_row.get("carry_contract"))

    if current_contract is None or current_contract not in available_contract_codes:
        return False
    if is_last_row:
        return True

    required_contracts = (next_contract, carry_contract)
    return all(
        contract is not None and contract in available_contract_codes
        for contract in required_contracts
    )


def _trim_unavailable_leading_roll_calendar_rows(
    calendar: rollCalendar,
    *,
    available_contract_codes: set[str],
) -> rollCalendar:
    """Drop the stale leading prefix that current DB prices cannot rebuild.

    Conservative calendar merge intentionally keeps non-regenerated dates, but this
    can preserve an old prefix whose contracts no longer have any per-contract price
    data in DB. That prefix is harmless until multiple prices are rebuilt, where it
    can surface as a "missing contract in middle of roll calendar" failure. Trim only
    the leading unavailable rows and keep the remaining manual/conservative rows.

    Args:
        calendar: Roll calendar to trim.
        available_contract_codes: Contract date strings with price data in DB.

    Returns:
        A `rollCalendar` starting from the first row that the current DB data can
        rebuild. If no row is rebuildable, returns an empty roll calendar.
    """

    calendar_norm = _drop_duplicate_transition_rows_keep_first(calendar)
    if len(calendar_norm) == 0:
        return _empty_roll_calendar()
    if len(available_contract_codes) == 0:
        return _empty_roll_calendar()

    calendar_dataframe = pd.DataFrame(calendar_norm).sort_index()
    total_rows = len(calendar_dataframe.index)
    first_valid_position = None

    for position, (_, calendar_row) in enumerate(calendar_dataframe.iterrows()):
        is_last_row = position == total_rows - 1
        if _roll_calendar_row_is_buildable_from_available_contracts(
            calendar_row,
            is_last_row=is_last_row,
            available_contract_codes=available_contract_codes,
        ):
            first_valid_position = position
            break

    if first_valid_position is None:
        return _empty_roll_calendar()

    trimmed_calendar = calendar_dataframe.iloc[first_valid_position:].copy()
    return rollCalendar(trimmed_calendar)


def _contract_dates_with_price_data_for_instrument_code(
    instrument_code: str,
) -> set[str]:
    """Return the set of contract dates that currently have DB price data."""

    with dataBlob(log_name=f"Repair-RollCalendar-{instrument_code}") as data:
        diag_prices = diagPrices(data)
        contract_dates = diag_prices.contract_dates_with_price_data_for_instrument_code(
            instrument_code
        )

    return set(contract_dates)


def _merge_roll_calendars_conservative(
    existing_calendar: rollCalendar,
    candidate_calendar: rollCalendar,
) -> tuple[rollCalendar, RollCalendarMergeOutcome]:
    """Conservatively merge roll calendars using candidate dates only.

    Existing rows are preserved by default. Candidate rows are applied only on
    roll dates present in candidate data (overlay by index):
    - if the date exists, the row is updated from candidate;
    - if the date is new, it is appended;
    - dates not present in candidate are left untouched.

    Args:
        existing_calendar: Current calendar stored on disk.
        candidate_calendar: Newly rebuilt calendar from available prices.

    Returns:
        Tuple of `(merged_calendar, outcome)`.
    """

    existing_norm = _drop_duplicate_transition_rows_keep_first(existing_calendar)
    candidate_norm = _drop_duplicate_transition_rows_keep_first(candidate_calendar)

    if len(existing_norm) == 0:
        return candidate_norm, RollCalendarMergeOutcome(
            existing_rows=0,
            candidate_rows=len(candidate_norm),
            merged_rows=len(candidate_norm),
            rows_added=len(candidate_norm),
            rows_replaced=0,
        )
    if len(candidate_norm) == 0:
        return existing_norm, RollCalendarMergeOutcome(
            existing_rows=len(existing_norm),
            candidate_rows=0,
            merged_rows=len(existing_norm),
            rows_added=0,
            rows_replaced=0,
        )

    overlapping_dates = existing_norm.index.intersection(candidate_norm.index)
    rows_replaced = sum(
        not _calendar_rows_equal(existing_norm.loc[date], candidate_norm.loc[date])
        for date in overlapping_dates
    )
    rows_added = len(candidate_norm.index.difference(existing_norm.index))

    merged_calendar = pd.concat([existing_norm, candidate_norm], axis=0)
    merged_calendar = merged_calendar[~merged_calendar.index.duplicated(keep="last")]
    merged_calendar = merged_calendar.sort_index()
    merged_calendar = rollCalendar(merged_calendar)
    merged_calendar = _drop_duplicate_transition_rows_keep_first(merged_calendar)

    return merged_calendar, RollCalendarMergeOutcome(
        existing_rows=len(existing_norm),
        candidate_rows=len(candidate_norm),
        merged_rows=len(merged_calendar),
        rows_added=rows_added,
        rows_replaced=rows_replaced,
    )


def _build_and_write_roll_calendar_conservative(
    instrument_code: str,
) -> RollCalendarMergeOutcome:
    """Build and write roll calendar in conservative available-date mode.

    Args:
        instrument_code: Instrument code to update.

    Returns:
        `RollCalendarMergeOutcome` with row counts and merge updates.

    """

    csv_roll_calendars = csvRollCalendarData(arg_not_supplied)
    has_existing_calendar = csv_roll_calendars.is_code_in_data(instrument_code)
    existing_rows_raw = 0
    if has_existing_calendar:
        existing_calendar = csv_roll_calendars.get_roll_calendar(instrument_code)
        existing_rows_raw = len(existing_calendar)
    else:
        existing_calendar = _empty_roll_calendar()

    candidate_calendar = build_and_write_roll_calendar(
        instrument_code=instrument_code,
        output_datapath=arg_not_supplied,
        write=False,
    )

    merged_calendar, merge_outcome = _merge_roll_calendars_conservative(
        existing_calendar=existing_calendar,
        candidate_calendar=candidate_calendar,
    )
    if (
        merge_outcome.rows_added == 0
        and merge_outcome.rows_replaced == 0
        and len(merged_calendar) == existing_rows_raw
        and has_existing_calendar
    ):
        return merge_outcome

    csv_roll_calendars.add_roll_calendar(
        instrument_code=instrument_code,
        roll_calendar=merged_calendar,
        ignore_duplication=True,
    )

    return merge_outcome


def repair_from_ib_non_destructive(
    instrument_code: str,
    rebuild_derived_data: bool = False,
):
    """Repair IB contract prices without overwriting existing DB values.

    The routine loads available contract dates from IB (including expired),
    merges intraday and daily prices with `keep_older=True`, and updates mixed
    frequency prices from repaired intraday+daily data.

    Args:
        instrument_code: Instrument code to repair.
        rebuild_derived_data: If `True`, rebuild roll calendar CSV and refresh
            multiple/adjusted prices after contract-level repairs.
    """

    rows_added_by_frequency = {
        "intraday": 0,
        "daily": 0,
        "mixed": 0,
    }
    contracts_with_updates: list[str] = []

    with dataBlob(log_name=f"Repair-IB-{instrument_code}") as data:
        broker_data = dataBroker(data)
        diag_prices = diagPrices(data)
        prices_updater = updatePrices(data)
        intraday_frequency = (
            diag_prices.get_intraday_frequency_for_historical_download()
        )
        frequencies_to_repair = [intraday_frequency]
        if DAILY_PRICE_FREQ not in frequencies_to_repair:
            frequencies_to_repair.append(DAILY_PRICE_FREQ)

        list_of_contract_dates = (
            broker_data.get_list_of_contract_dates_for_instrument_code(
                instrument_code=instrument_code,
                allow_expired=True,
            )
        )
        list_of_contract_months = sorted(
            {date_str[:6] for date_str in list_of_contract_dates}
        )

        if len(list_of_contract_months) == 0:
            print(f"No contracts found in IB for {instrument_code}. Nothing to repair.")
            return

        print(
            f"Found {len(list_of_contract_months)} contracts in IB for {instrument_code}."
        )
        print(f"Configured intraday frequency: {intraday_frequency}")

        for contract_month in list_of_contract_months:
            contract = futuresContract(instrument_code, contract_month)
            contract_rows_added = 0
            print(f"\nProcessing {contract}")

            for frequency in frequencies_to_repair:
                try:
                    incoming_prices = broker_data.get_prices_at_frequency_for_potentially_expired_contract_object(
                        contract_object=contract,
                        frequency=frequency,
                    )
                except missingData:
                    print(f"  {frequency}: no data from IB")
                    continue

                if len(incoming_prices) == 0:
                    print(f"  {frequency}: empty data from IB")
                    continue

                outcome = _merge_and_write_prices_for_frequency(
                    contract=contract,
                    incoming_prices=incoming_prices,
                    frequency=frequency,
                    diag_prices=diag_prices,
                    prices_updater=prices_updater,
                )
                print(
                    f"  {frequency}: existing={outcome.existing_rows}, "
                    f"incoming={outcome.incoming_rows}, added={outcome.rows_added}"
                )

                if frequency == DAILY_PRICE_FREQ:
                    rows_added_by_frequency["daily"] += outcome.rows_added
                else:
                    rows_added_by_frequency["intraday"] += outcome.rows_added

                contract_rows_added += outcome.rows_added

            mixed_candidate = _mixed_prices_candidate_from_db(
                contract=contract,
                intraday_frequency=intraday_frequency,
                diag_prices=diag_prices,
            )
            if len(mixed_candidate) > 0:
                mixed_outcome = _merge_and_write_prices_for_frequency(
                    contract=contract,
                    incoming_prices=mixed_candidate,
                    frequency=MIXED_FREQ,
                    diag_prices=diag_prices,
                    prices_updater=prices_updater,
                )
                rows_added_by_frequency["mixed"] += mixed_outcome.rows_added
                contract_rows_added += mixed_outcome.rows_added
                print(
                    f"  {MIXED_FREQ}: existing={mixed_outcome.existing_rows}, "
                    f"incoming={mixed_outcome.incoming_rows}, added={mixed_outcome.rows_added}"
                )

            if contract_rows_added > 0:
                contracts_with_updates.append(contract_month)

    print("\nRepair summary")
    print(f"Instrument: {instrument_code}")
    print(f"Contracts scanned: {len(list_of_contract_months)}")
    print(f"Contracts updated: {len(contracts_with_updates)}")
    print(f"Rows added (intraday): {rows_added_by_frequency['intraday']}")
    print(f"Rows added (daily): {rows_added_by_frequency['daily']}")
    print(f"Rows added (mixed): {rows_added_by_frequency['mixed']}")

    if rebuild_derived_data:
        print("\nRebuilding roll calendar + multiple + adjusted prices...")
        _rebuild_derived_data_for_instrument(instrument_code=instrument_code)
        print("Derived data rebuild completed.")
    else:
        print(
            "\nSkipped roll calendar/multiple/adjusted rebuild. "
            "Run it separately if you repaired historical gaps that affect rolls."
        )


if __name__ == "__main__":
    cli_args = _parse_args()
    resolved_instrument_code = _resolve_instrument_code(cli_args.instrument_code)
    rebuild_derived = _resolve_rebuild_choice(cli_args)
    repair_from_ib_non_destructive(
        instrument_code=resolved_instrument_code,
        rebuild_derived_data=rebuild_derived,
    )
