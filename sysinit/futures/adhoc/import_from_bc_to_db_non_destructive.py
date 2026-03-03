import math
import pandas as pd
from dataclasses import dataclass
from contextlib import redirect_stdout
from io import StringIO

from syscore.constants import arg_not_supplied
from syscore.dateutils import DAILY_PRICE_FREQ, HOURLY_FREQ, MIXED_FREQ, Frequency
from syscore.fileutils import resolve_path_and_filename_for_package
from syscore.pandas.frequency import merge_data_with_different_freq
from sysdata.config.production_config import get_production_config
from sysdata.csv.csv_roll_calendars import csvRollCalendarData
from sysdata.csv.csv_roll_parameters import csvRollParametersData
from sysdata.csv.csv_futures_contract_prices import (
    ConfigCsvFuturesPrices,
    csvFuturesContractPriceData,
)
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
from sysobjects.dict_of_futures_per_contract_prices import (
    dictFuturesContractFinalPrices,
)
from sysobjects.futures_per_contract_prices import futuresContractPrices
from sysobjects.adjusted_prices import futuresAdjustedPrices
from sysobjects.multiple_prices import futuresMultiplePrices
from sysobjects.dict_of_named_futures_per_contract_prices import (
    list_of_contract_column_names,
    list_of_price_column_names,
)
from sysobjects.roll_calendars import rollCalendar
from sysproduction.data.prices import diagPrices

diag_prices = diagPrices()
db_prices = diag_prices.db_futures_contract_price_data

BARCHART_FINAL_PRICE_COLUMNS = ("Latest", "Close", "Last")
ROLL_CALENDAR_MAX_STALENESS_DAYS = 180
MIN_CONTRACTS_FOR_ROLL_CALENDAR = 3
FALLBACK_PREFER_JANUARY_CUTOFF = False


def _build_barchart_csv_config(final_column_name: str) -> ConfigCsvFuturesPrices:
    return ConfigCsvFuturesPrices(
        input_date_index_name="Time",
        input_skiprows=0,
        input_skipfooter=0,
        input_date_format="ISO8601",
        input_column_mapping=dict(
            OPEN="Open",
            HIGH="High",
            LOW="Low",
            FINAL=final_column_name,
            VOLUME="Volume",
        ),
    )


BARCHART_CONFIG = _build_barchart_csv_config("Latest")


@dataclass
class MergeOutcome:
    """Summarize the result of a non-destructive merge for one contract/frequency.

    Args:
        existing_rows: Number of rows already present in the database.
        incoming_rows: Number of rows read from CSV for the same contract/frequency.
        merged_rows: Number of rows after applying the non-destructive merge.
        rows_added: Number of rows effectively added to the database.
        rows_replaced: Number of existing rows replaced because they were
            incoherent and incoming rows were coherent.
        incoming_rows_skipped: Number of incoming rows ignored because they
            were incoherent.
    """

    existing_rows: int
    incoming_rows: int
    merged_rows: int
    rows_added: int
    rows_replaced: int
    incoming_rows_skipped: int


@dataclass
class RollCalendarMergeOutcome:
    """Summarize one conservative roll-calendar merge.

    Args:
        existing_rows: Number of rows already present in the calendar CSV.
        candidate_rows: Number of rows in the newly built candidate calendar.
        merged_rows: Number of rows after applying conservative merge.
        rows_added: Number of candidate rows added at new dates.
        rows_replaced: Number of existing rows replaced due incoherence.
        candidate_rows_skipped: Number of candidate rows ignored as incoherent.
    """

    existing_rows: int
    candidate_rows: int
    merged_rows: int
    rows_added: int
    rows_replaced: int
    candidate_rows_skipped: int


@dataclass
class DerivedMergeOutcome:
    """Summarize conservative merge outcome for derived series.

    Args:
        existing_rows: Number of rows currently stored in DB.
        candidate_rows: Number of rows in rebuilt candidate series.
        merged_rows: Number of rows after conservative merge.
        rows_added: Number of new rows added from candidate.
        rows_replaced: Number of existing incoherent rows replaced.
        candidate_rows_skipped: Number of candidate rows ignored as incoherent.
    """

    existing_rows: int
    candidate_rows: int
    merged_rows: int
    rows_added: int
    rows_replaced: int
    candidate_rows_skipped: int


@dataclass
class RollCalendarBuildResult:
    """Describe how a roll calendar was built for one instrument.

    Args:
        roll_calendar: Resulting roll calendar.
        mode: Build mode used (`full_history` or `recent_to_past`).
        cutoff_contract: Earliest contract retained when using
            `recent_to_past` fallback.
    """

    roll_calendar: rollCalendar
    mode: str
    cutoff_contract: str | None = None


def _copy_config_with_new_final_column(
    config: ConfigCsvFuturesPrices, final_column_name: str
) -> ConfigCsvFuturesPrices:
    input_column_mapping = dict(config.input_column_mapping)
    input_column_mapping["FINAL"] = final_column_name

    return ConfigCsvFuturesPrices(
        input_date_index_name=config.input_date_index_name,
        input_date_format=config.input_date_format,
        input_column_mapping=input_column_mapping,
        input_skiprows=config.input_skiprows,
        input_skipfooter=config.input_skipfooter,
        apply_multiplier=config.apply_multiplier,
        apply_inverse=config.apply_inverse,
    )


def _get_candidate_csv_configs(
    csv_config: ConfigCsvFuturesPrices | object = arg_not_supplied,
) -> list[ConfigCsvFuturesPrices]:
    """Build candidate CSV configs trying known Barchart final-price columns.

    Args:
        csv_config: Optional custom config to prioritise.

    Returns:
        Ordered list of configs to try, starting from provided/default config
        and then adding fallback final-price columns.
    """

    if csv_config is arg_not_supplied:
        base_config = BARCHART_CONFIG
    else:
        base_config = csv_config

    configured_final = base_config.input_column_mapping.get("FINAL")
    ordered_finals = [
        configured_final,
        *BARCHART_FINAL_PRICE_COLUMNS,
    ]

    candidate_configs = []
    seen_final_columns = set()
    for final_column_name in ordered_finals:
        if final_column_name is None:
            continue
        if final_column_name in seen_final_columns:
            continue
        seen_final_columns.add(final_column_name)
        candidate_configs.append(
            _copy_config_with_new_final_column(base_config, final_column_name)
        )

    return candidate_configs


def _coalesce_final_values_from_fallback(
    primary_prices: futuresContractPrices,
    fallback_prices: futuresContractPrices,
) -> tuple[futuresContractPrices, int, int]:
    """Fill missing `FINAL` values from fallback prices on the same timestamps.

    The function is conservative:
    - existing non-null `FINAL` values in `primary_prices` are preserved;
    - rows missing in `primary_prices` are appended only when fallback `FINAL`
      is non-null;
    - on backfilled timestamps, OHLCV fields are filled from fallback only
      where currently null in the primary dataset.

    Args:
        primary_prices: Parsed prices from the preferred final-price column.
        fallback_prices: Parsed prices from a fallback final-price column.

    Returns:
        Tuple `(coalesced_prices, rows_backfilled, rows_added_from_fallback)`.
    """

    primary_dataframe = pd.DataFrame(primary_prices).copy()
    fallback_dataframe = pd.DataFrame(fallback_prices).copy()

    if len(primary_dataframe) == 0:
        fallback_dataframe = fallback_dataframe[fallback_dataframe["FINAL"].notna()]
        fallback_dataframe = fallback_dataframe.sort_index()
        fallback_dataframe = fallback_dataframe[
            ~fallback_dataframe.index.duplicated(keep="first")
        ]
        return futuresContractPrices(fallback_dataframe), 0, len(fallback_dataframe)
    if len(fallback_dataframe) == 0:
        return futuresContractPrices(primary_dataframe), 0, 0

    overlapping_index = primary_dataframe.index.intersection(fallback_dataframe.index)
    backfillable_index = overlapping_index[
        primary_dataframe.loc[overlapping_index, "FINAL"].isna()
        & fallback_dataframe.loc[overlapping_index, "FINAL"].notna()
    ]
    rows_backfilled = len(backfillable_index)

    if rows_backfilled > 0:
        primary_dataframe.loc[backfillable_index, "FINAL"] = fallback_dataframe.loc[
            backfillable_index, "FINAL"
        ]
        for column_name in ("OPEN", "HIGH", "LOW", "VOLUME"):
            primary_dataframe.loc[
                backfillable_index, column_name
            ] = primary_dataframe.loc[backfillable_index, column_name].where(
                primary_dataframe.loc[backfillable_index, column_name].notna(),
                fallback_dataframe.loc[backfillable_index, column_name],
            )

    missing_index = fallback_dataframe.index.difference(primary_dataframe.index)
    additional_rows = fallback_dataframe.loc[missing_index]
    additional_rows = additional_rows[additional_rows["FINAL"].notna()]
    rows_added_from_fallback = len(additional_rows)
    if rows_added_from_fallback > 0:
        primary_dataframe = pd.concat([primary_dataframe, additional_rows], axis=0)

    primary_dataframe = primary_dataframe.sort_index()
    primary_dataframe = primary_dataframe[
        ~primary_dataframe.index.duplicated(keep="first")
    ]

    return (
        futuresContractPrices(primary_dataframe),
        rows_backfilled,
        rows_added_from_fallback,
    )


def _load_split_freq_prices_with_fallback(
    instrument_code: str,
    datapath: str,
    csv_config: ConfigCsvFuturesPrices | object = arg_not_supplied,
) -> tuple[dict, dict, set[str], set[str]]:
    """Load day/hour dictionaries trying fallback final-price column names.

    Args:
        instrument_code: Instrument code to load.
        datapath: Path containing Barchart CSV files.
        csv_config: Optional primary config to try first.

    Returns:
        Tuple of `(hourly_dict, daily_dict, hourly_final_columns, daily_final_columns)`.

    Raises:
        ValueError: If no candidate configuration can parse the input files.
    """

    candidate_configs = _get_candidate_csv_configs(csv_config)
    csv_readers = [
        (candidate, csvFuturesContractPriceData(datapath, config=candidate))
        for candidate in candidate_configs
    ]
    list_reader_for_contract_discovery = csv_readers[0][1]

    def _load_frequency_dict(
        frequency: Frequency,
    ) -> tuple[dict, set[str]]:
        contract_dates = list_reader_for_contract_discovery.contract_dates_with_price_data_at_frequency_for_instrument_code(
            instrument_code=instrument_code,
            frequency=frequency,
        )
        contract_dates = sorted(contract_dates)

        loaded_prices_by_contract = {}
        used_final_columns = set()
        total_backfilled_rows = 0
        total_added_rows = 0

        for contract_date_str in contract_dates:
            contract = futuresContract(instrument_code, contract_date_str)
            parse_errors = []
            parsed_prices_by_final_column: list[tuple[str, futuresContractPrices]] = []
            for candidate_config, csv_reader in csv_readers:
                final_column_name = candidate_config.input_column_mapping.get("FINAL")
                try:
                    parsed_prices = _normalise_prices(
                        csv_reader.get_prices_at_frequency_for_contract_object(
                            contract,
                            frequency=frequency,
                        )
                    )
                except KeyError as exception:
                    parse_errors.append(f"FINAL={final_column_name}: {exception}")
                    continue

                parsed_prices_by_final_column.append((final_column_name, parsed_prices))
                used_final_columns.add(final_column_name)

            if len(parsed_prices_by_final_column) == 0:
                attempted_final_columns = [
                    candidate.input_column_mapping.get("FINAL")
                    for candidate, _ in csv_readers
                ]
                raise ValueError(
                    "Failed to parse Barchart CSV for "
                    f"{instrument_code} contract {contract_date_str} at frequency "
                    f"{frequency}. Tried FINAL columns {attempted_final_columns}. "
                    f"Errors: {parse_errors}"
                )

            coalesced_prices = parsed_prices_by_final_column[0][1]
            rows_backfilled_for_contract = 0
            rows_added_for_contract = 0
            for _, fallback_prices in parsed_prices_by_final_column[1:]:
                (
                    coalesced_prices,
                    rows_backfilled,
                    rows_added_from_fallback,
                ) = _coalesce_final_values_from_fallback(
                    primary_prices=coalesced_prices,
                    fallback_prices=fallback_prices,
                )
                rows_backfilled_for_contract += rows_backfilled
                rows_added_for_contract += rows_added_from_fallback

            loaded_prices_by_contract[contract_date_str] = coalesced_prices
            total_backfilled_rows += rows_backfilled_for_contract
            total_added_rows += rows_added_for_contract

        if total_backfilled_rows > 0 or total_added_rows > 0:
            print(
                f"{frequency}: FINAL coalescing from fallback columns backfilled "
                f"{total_backfilled_rows} rows and added {total_added_rows} rows."
            )

        return loaded_prices_by_contract, used_final_columns

    hourly_dict, hourly_final_columns = _load_frequency_dict(HOURLY_FREQ)
    daily_dict, daily_final_columns = _load_frequency_dict(DAILY_PRICE_FREQ)

    print(
        f"Using FINAL column(s) for hourly CSV parsing: {sorted(hourly_final_columns)}"
    )
    print(f"Using FINAL column(s) for daily CSV parsing: {sorted(daily_final_columns)}")

    return hourly_dict, daily_dict, hourly_final_columns, daily_final_columns


def _format_sorted_column_set(column_set: set[str]) -> str:
    if len(column_set) == 0:
        return "[]"
    return str(sorted(column_set))


def _resolve_csv_mapping_message(
    hourly_final_columns: set[str], daily_final_columns: set[str]
) -> str:
    return (
        "Resolved CSV mapping: "
        f"hourly FINAL -> {_format_sorted_column_set(hourly_final_columns)}; "
        f"daily FINAL -> {_format_sorted_column_set(daily_final_columns)}"
    )


def _normalise_prices(price_data: futuresContractPrices) -> futuresContractPrices:
    """Return a sorted and de-duplicated `futuresContractPrices` object.

    Args:
        price_data: Raw or already-clean price data.

    Returns:
        Price data sorted by timestamp with duplicated index values removed.
    """

    as_dataframe = pd.DataFrame(price_data).sort_index()
    as_dataframe = as_dataframe[~as_dataframe.index.duplicated(keep="first")]

    return futuresContractPrices(as_dataframe)


def _is_finite_number(value: object) -> bool:
    """Return `True` when value is numeric and finite."""

    if pd.isna(value):
        return False
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return False

    return math.isfinite(numeric_value)


def _price_row_is_coherent(price_row: pd.Series) -> bool:
    """Return `True` if one OHLCV row is internally coherent.

    The check is intentionally conservative: existing rows are considered
    incoherent only when clearly broken (missing/non-numeric OHLC values,
    inverted high/low range, or open/final outside the high-low range).

    Args:
        price_row: One row from a `futuresContractPrices` dataframe.

    Returns:
        `True` if row is coherent, otherwise `False`.
    """

    required_columns = ("OPEN", "HIGH", "LOW", "FINAL")
    for column_name in required_columns:
        if column_name not in price_row or not _is_finite_number(
            price_row[column_name]
        ):
            return False

    open_price = float(price_row["OPEN"])
    high_price = float(price_row["HIGH"])
    low_price = float(price_row["LOW"])
    final_price = float(price_row["FINAL"])

    if high_price < low_price:
        return False
    if open_price < low_price or open_price > high_price:
        return False
    if final_price < low_price or final_price > high_price:
        return False

    if "VOLUME" in price_row and pd.notna(price_row["VOLUME"]):
        if not _is_finite_number(price_row["VOLUME"]):
            return False
        if float(price_row["VOLUME"]) < 0.0:
            return False

    return True


def _merge_prices_non_destructive(
    existing_prices: futuresContractPrices,
    incoming_prices: futuresContractPrices,
) -> tuple[futuresContractPrices, int, int, int]:
    """Merge prices without overwriting coherent existing rows.

    Overlapping rows are replaced only when existing rows are incoherent and
    incoming rows are coherent.

    Args:
        existing_prices: Data already present in the database.
        incoming_prices: Data read from CSV to be merged in.

    Returns:
        Tuple `(merged_prices, rows_added, rows_replaced, incoming_rows_skipped)`.

    Raises:
        RuntimeError: If merged data shrinks compared to existing data.
    """

    existing_as_dataframe = pd.DataFrame(existing_prices).copy()
    incoming_as_dataframe = pd.DataFrame(incoming_prices).copy()
    merged_as_dataframe = existing_as_dataframe.copy()

    rows_added = 0
    rows_replaced = 0
    incoming_rows_skipped = 0

    if len(existing_as_dataframe) == 0:
        coherent_mask = incoming_as_dataframe.apply(_price_row_is_coherent, axis=1)
        merged_as_dataframe = incoming_as_dataframe[coherent_mask].copy()
        rows_added = int(coherent_mask.sum())
        incoming_rows_skipped = int((~coherent_mask).sum())
    else:
        for timestamp, incoming_row in incoming_as_dataframe.iterrows():
            incoming_is_coherent = _price_row_is_coherent(incoming_row)
            if timestamp not in merged_as_dataframe.index:
                if incoming_is_coherent:
                    merged_as_dataframe.loc[timestamp] = incoming_row
                    rows_added += 1
                else:
                    incoming_rows_skipped += 1
                continue

            existing_row = merged_as_dataframe.loc[timestamp]
            existing_is_coherent = _price_row_is_coherent(existing_row)
            if (not existing_is_coherent) and incoming_is_coherent:
                merged_as_dataframe.loc[timestamp] = incoming_row
                rows_replaced += 1
                continue

            if not incoming_is_coherent:
                incoming_rows_skipped += 1

    merged_as_dataframe = merged_as_dataframe.sort_index()
    merged_prices = futuresContractPrices(merged_as_dataframe)
    if len(merged_prices) < len(existing_prices):
        raise RuntimeError("Merged prices have fewer rows than existing prices")

    return merged_prices, rows_added, rows_replaced, incoming_rows_skipped


def _merge_and_write_prices_for_frequency(
    contract: futuresContract,
    incoming_prices: futuresContractPrices,
    frequency: Frequency,
) -> MergeOutcome:
    """Merge CSV prices into DB for one contract/frequency without overwriting history.

    Args:
        contract: Contract to update.
        incoming_prices: CSV prices for the contract and frequency.
        frequency: Target frequency (`Hour`, `Day`, or mixed).

    Returns:
        `MergeOutcome` with row counts and added rows.
    """

    normalised_incoming = _normalise_prices(incoming_prices)
    existing_prices = db_prices.get_prices_at_frequency_for_contract_object(
        contract, frequency=frequency
    )
    normalised_existing = _normalise_prices(existing_prices)
    (
        merged_prices,
        rows_added,
        rows_replaced,
        incoming_rows_skipped,
    ) = _merge_prices_non_destructive(
        existing_prices=normalised_existing,
        incoming_prices=normalised_incoming,
    )

    if rows_added > 0 or rows_replaced > 0:
        db_prices.write_prices_at_frequency_for_contract_object(
            contract,
            merged_prices,
            frequency=frequency,
            ignore_duplication=True,
        )

    return MergeOutcome(
        existing_rows=len(normalised_existing),
        incoming_rows=len(normalised_incoming),
        merged_rows=len(merged_prices),
        rows_added=rows_added,
        rows_replaced=rows_replaced,
        incoming_rows_skipped=incoming_rows_skipped,
    )


def _mixed_prices_candidate_from_db(contract: futuresContract) -> futuresContractPrices:
    """Build candidate mixed-frequency prices from DB day/hour data for a contract.

    Args:
        contract: Contract to rebuild as mixed-frequency prices.

    Returns:
        Mixed prices produced from available hourly/daily DB data. If only one
        frequency exists, it is returned directly. If none exists, an empty
        `futuresContractPrices` object is returned.
    """

    has_hourly = db_prices.has_price_data_for_contract_at_frequency(
        contract, frequency=HOURLY_FREQ
    )
    has_daily = db_prices.has_price_data_for_contract_at_frequency(
        contract, frequency=DAILY_PRICE_FREQ
    )

    if has_hourly and has_daily:
        hourly_prices = _normalise_prices(
            db_prices.get_prices_at_frequency_for_contract_object(
                contract, frequency=HOURLY_FREQ
            )
        )
        daily_prices = _normalise_prices(
            db_prices.get_prices_at_frequency_for_contract_object(
                contract, frequency=DAILY_PRICE_FREQ
            )
        )
        return futuresContractPrices(
            merge_data_with_different_freq([hourly_prices, daily_prices])
        )

    if has_daily:
        return _normalise_prices(
            db_prices.get_prices_at_frequency_for_contract_object(
                contract, frequency=DAILY_PRICE_FREQ
            )
        )

    if has_hourly:
        return _normalise_prices(
            db_prices.get_prices_at_frequency_for_contract_object(
                contract, frequency=HOURLY_FREQ
            )
        )

    return futuresContractPrices.create_empty()


def _latest_available_timestamp_in_prices_dict(
    prices_dict: dictFuturesContractFinalPrices,
) -> pd.Timestamp | None:
    """Return the latest timestamp available across all contract prices.

    Args:
        prices_dict: Dictionary of final prices keyed by contract date.

    Returns:
        Latest timestamp if any prices exist, otherwise `None`.
    """

    latest_timestamp = None
    for contract_prices in prices_dict.values():
        if len(contract_prices) == 0:
            continue
        contract_latest = contract_prices.index.max()
        if latest_timestamp is None or contract_latest > latest_timestamp:
            latest_timestamp = contract_latest

    return latest_timestamp


def _calendar_is_recent_enough(
    calendar: rollCalendar,
    latest_available_timestamp: pd.Timestamp | None,
    max_staleness_days: int = ROLL_CALENDAR_MAX_STALENESS_DAYS,
) -> bool:
    """Check if a calendar reaches sufficiently recent history.

    Args:
        calendar: Candidate roll calendar.
        latest_available_timestamp: Most recent timestamp in available prices.
        max_staleness_days: Maximum accepted lag between latest price timestamp
            and the last roll date.

    Returns:
        `True` if calendar is non-empty and recent enough, else `False`.
    """

    if len(calendar) == 0:
        return False
    if latest_available_timestamp is None:
        return True

    last_roll_date = calendar.index.max()
    staleness = latest_available_timestamp - last_roll_date

    return staleness <= pd.Timedelta(days=max_staleness_days)


def _build_roll_calendar_from_prices_dict(
    prices_dict: dictFuturesContractFinalPrices, roll_parameters
) -> rollCalendar:
    """Build a roll calendar from a dictionary of contract prices.

    Args:
        prices_dict: Dictionary of final prices keyed by contract date.
        roll_parameters: Roll parameters for the instrument.

    Returns:
        A generated `rollCalendar`.
    """

    return rollCalendar.create_from_prices(prices_dict, roll_parameters)


def _is_roll_calendar_valid(
    calendar: rollCalendar, prices_dict: dictFuturesContractFinalPrices
) -> bool:
    """Validate date/price consistency for a calendar.

    Args:
        calendar: Roll calendar to validate.
        prices_dict: Contract prices used for validation.

    Returns:
        `True` if calendar passes date/price consistency checks, otherwise
        `False`.
    """

    # Keep behaviour aligned with `build_and_write_roll_calendar`: monotonic
    # warnings are informative but not blocking.
    calendar.check_if_date_index_monotonic()
    return calendar.check_dates_are_valid_for_prices(prices_dict)


def _build_recent_to_past_roll_calendar(
    instrument_code: str,
    prices_dict: dictFuturesContractFinalPrices,
    latest_available_timestamp: pd.Timestamp | None,
) -> RollCalendarBuildResult:
    """Build a roll calendar by expanding from recent contracts backwards.

    The function starts from the most recent contracts and progressively
    includes older contracts, keeping the oldest cutoff that still yields a
    valid and recent calendar.

    Args:
        instrument_code: Instrument code to process.
        prices_dict: Dictionary of final prices keyed by contract date.
        latest_available_timestamp: Most recent timestamp available in prices.

    Returns:
        `RollCalendarBuildResult` built with `recent_to_past` mode.

    Raises:
        RuntimeError: If no valid fallback calendar can be built.
    """

    contract_dates = sorted(prices_dict.keys())
    if len(contract_dates) < MIN_CONTRACTS_FOR_ROLL_CALENDAR:
        raise RuntimeError(
            "Not enough contracts to build a fallback roll calendar for "
            f"{instrument_code}"
        )

    print(
        "Recent-to-past fallback: scanning contract cutoffs from newest to oldest "
        f"for {instrument_code} ({len(contract_dates)} contracts)"
    )

    candidate_cutoffs = contract_dates[
        : len(contract_dates) - MIN_CONTRACTS_FOR_ROLL_CALENDAR + 1
    ]
    candidate_cutoffs_desc = list(reversed(candidate_cutoffs))
    total_candidates = len(candidate_cutoffs_desc)
    if total_candidates == 0:
        raise RuntimeError(
            "Not enough contract cutoffs to build a fallback roll calendar for "
            f"{instrument_code}"
        )

    roll_parameters = csvRollParametersData().get_roll_parameters(instrument_code)
    evaluation_cache: dict[int, tuple[bool, rollCalendar | None]] = {}
    evaluated_attempts = 0

    def _evaluate_candidate_index(
        candidate_index: int,
    ) -> tuple[bool, rollCalendar | None]:
        nonlocal evaluated_attempts
        if candidate_index in evaluation_cache:
            return evaluation_cache[candidate_index]

        cutoff_contract = candidate_cutoffs_desc[candidate_index]
        retained_contract_dates = [
            contract_date
            for contract_date in contract_dates
            if contract_date >= cutoff_contract
        ]
        evaluated_attempts += 1
        print(
            "  Fallback attempt "
            f"{evaluated_attempts}: cutoff >= {cutoff_contract} "
            f"({len(retained_contract_dates)} contracts)"
        )

        filtered_prices = dictFuturesContractFinalPrices(
            {
                contract_date: prices_dict[contract_date]
                for contract_date in retained_contract_dates
            }
        )

        try:
            with redirect_stdout(StringIO()):
                candidate_calendar = _build_roll_calendar_from_prices_dict(
                    filtered_prices, roll_parameters
                )
        except Exception:
            evaluation_cache[candidate_index] = (False, None)
            return evaluation_cache[candidate_index]

        is_recent_enough = _calendar_is_recent_enough(
            candidate_calendar,
            latest_available_timestamp=latest_available_timestamp,
        )
        if not is_recent_enough:
            evaluation_cache[candidate_index] = (False, None)
            return evaluation_cache[candidate_index]

        is_valid = _is_roll_calendar_valid(candidate_calendar, filtered_prices)
        if not is_valid:
            evaluation_cache[candidate_index] = (False, None)
            return evaluation_cache[candidate_index]

        evaluation_cache[candidate_index] = (True, candidate_calendar)
        print(
            "  Valid fallback cutoff accepted: "
            f"{cutoff_contract} (rows={len(candidate_calendar)})"
        )

        return evaluation_cache[candidate_index]

    first_valid_index = None
    selected_calendar = None
    for candidate_index in range(total_candidates):
        is_valid_candidate, candidate_calendar = _evaluate_candidate_index(
            candidate_index
        )
        if is_valid_candidate:
            first_valid_index = candidate_index
            selected_calendar = candidate_calendar
            break

    if first_valid_index is None or selected_calendar is None:
        raise RuntimeError(
            "Failed to generate a recent-to-past roll calendar for "
            f"{instrument_code}"
        )

    selected_index = first_valid_index
    probe_step = 1
    failing_index = None

    while True:
        probe_index = selected_index + probe_step
        if probe_index >= total_candidates:
            break

        is_valid_candidate, candidate_calendar = _evaluate_candidate_index(probe_index)
        if is_valid_candidate and candidate_calendar is not None:
            selected_index = probe_index
            selected_calendar = candidate_calendar
            probe_step *= 2
            continue

        failing_index = probe_index
        break

    if failing_index is not None:
        left_index = selected_index + 1
        right_index = failing_index - 1
        while left_index <= right_index:
            middle_index = (left_index + right_index) // 2
            is_valid_candidate, candidate_calendar = _evaluate_candidate_index(
                middle_index
            )
            if is_valid_candidate and candidate_calendar is not None:
                selected_index = middle_index
                selected_calendar = candidate_calendar
                left_index = middle_index + 1
            else:
                right_index = middle_index - 1

        next_index = selected_index + 1
        while next_index < total_candidates:
            is_valid_candidate, candidate_calendar = _evaluate_candidate_index(
                next_index
            )
            if not is_valid_candidate or candidate_calendar is None:
                break
            selected_index = next_index
            selected_calendar = candidate_calendar
            next_index += 1
    else:
        oldest_index = total_candidates - 1
        is_valid_candidate, candidate_calendar = _evaluate_candidate_index(oldest_index)
        if is_valid_candidate and candidate_calendar is not None:
            selected_index = oldest_index
            selected_calendar = candidate_calendar

    if FALLBACK_PREFER_JANUARY_CUTOFF:
        january_indices = [
            index
            for index in range(selected_index + 1)
            if candidate_cutoffs_desc[index][4:6] == "01"
        ]
        january_indices.sort(reverse=True)
        for january_index in january_indices:
            is_valid_candidate, candidate_calendar = _evaluate_candidate_index(
                january_index
            )
            if not is_valid_candidate or candidate_calendar is None:
                continue
            if january_index == selected_index:
                break

            selected_index = january_index
            selected_calendar = candidate_calendar
            january_cutoff = candidate_cutoffs_desc[selected_index]
            print(
                "Adjusted fallback cutoff to January boundary "
                f"{january_cutoff} for cleaner restart."
            )
            break

    selected_cutoff = candidate_cutoffs_desc[selected_index]
    print(
        "Recent-to-past fallback selected cutoff "
        f"{selected_cutoff} for {instrument_code}"
    )

    return RollCalendarBuildResult(
        roll_calendar=selected_calendar,
        mode="recent_to_past",
        cutoff_contract=selected_cutoff,
    )


def _build_roll_calendar_with_recent_to_past_fallback(
    instrument_code: str,
) -> RollCalendarBuildResult:
    """Build roll calendar with fallback from recent contracts to older ones.

    Args:
        instrument_code: Instrument code to process.

    Returns:
        `RollCalendarBuildResult` describing the chosen build mode.
    """

    all_prices_dict = db_prices.get_merged_prices_for_instrument(instrument_code)
    final_prices_dict = all_prices_dict.final_prices()
    if len(final_prices_dict) == 0:
        raise RuntimeError(f"No merged prices available for {instrument_code}")

    latest_available_timestamp = _latest_available_timestamp_in_prices_dict(
        final_prices_dict
    )

    try:
        standard_calendar = build_and_write_roll_calendar(
            instrument_code,
            write=False,
            check_before_writing=False,
        )
    except Exception as exception:
        print(
            "Standard roll calendar build failed, switching to recent-to-past "
            f"fallback: {exception}"
        )
        return _build_recent_to_past_roll_calendar(
            instrument_code=instrument_code,
            prices_dict=final_prices_dict,
            latest_available_timestamp=latest_available_timestamp,
        )

    if _calendar_is_recent_enough(
        standard_calendar,
        latest_available_timestamp=latest_available_timestamp,
    ):
        return RollCalendarBuildResult(
            roll_calendar=standard_calendar,
            mode="full_history",
        )

    print(
        "Standard roll calendar ended too far in the past. "
        "Switching to recent-to-past fallback."
    )

    return _build_recent_to_past_roll_calendar(
        instrument_code=instrument_code,
        prices_dict=final_prices_dict,
        latest_available_timestamp=latest_available_timestamp,
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


def _empty_roll_calendar() -> rollCalendar:
    """Return an empty roll-calendar object with expected columns."""

    empty_dataframe = pd.DataFrame(
        columns=["current_contract", "next_contract", "carry_contract"]
    )
    return rollCalendar(empty_dataframe)


def _normalise_roll_calendar(calendar: rollCalendar) -> rollCalendar:
    """Return sorted, de-duplicated roll calendar."""

    if len(calendar) == 0:
        return _empty_roll_calendar()

    as_dataframe = pd.DataFrame(calendar).sort_index()
    as_dataframe = as_dataframe[~as_dataframe.index.duplicated(keep="first")]
    return rollCalendar(as_dataframe)


def _roll_calendar_row_is_coherent(calendar_row: pd.Series) -> bool:
    """Return `True` when one roll-calendar row is structurally coherent."""

    required_columns = ("current_contract", "next_contract", "carry_contract")
    normalised_codes = {}
    for column_name in required_columns:
        if column_name not in calendar_row:
            return False
        normalised_codes[column_name] = _normalise_contract_code(
            calendar_row[column_name]
        )
        if normalised_codes[column_name] is None:
            return False

    # A roll row where current and next contract are equal is clearly broken.
    if normalised_codes["current_contract"] == normalised_codes["next_contract"]:
        return False

    return True


def _roll_calendar_transition_key(
    calendar_row: pd.Series,
) -> tuple[str, str, str] | None:
    """Return a normalised transition key `(current, next, carry)` for a row."""

    required_columns = ("current_contract", "next_contract", "carry_contract")
    if any(column_name not in calendar_row for column_name in required_columns):
        return None

    current_contract = _normalise_contract_code(calendar_row["current_contract"])
    next_contract = _normalise_contract_code(calendar_row["next_contract"])
    carry_contract = _normalise_contract_code(calendar_row["carry_contract"])
    if current_contract is None or next_contract is None or carry_contract is None:
        return None

    return (current_contract, next_contract, carry_contract)


def _extract_roll_calendar_transition_keys(
    calendar_dataframe: pd.DataFrame,
) -> set[tuple[str, str, str]]:
    """Extract all normalised transition keys from a roll-calendar dataframe."""

    transition_keys = set()
    for _, calendar_row in calendar_dataframe.iterrows():
        row_key = _roll_calendar_transition_key(calendar_row)
        if row_key is None:
            continue
        transition_keys.add(row_key)

    return transition_keys


def _merge_roll_calendars_non_destructive(
    existing_calendar: rollCalendar,
    candidate_calendar: rollCalendar,
) -> tuple[rollCalendar, RollCalendarMergeOutcome]:
    """Merge roll calendar conservatively by available dates.

    Candidate rows are considered only on dates available in candidate data.
    Existing rows are preserved unless they are incoherent and the candidate
    row on the same date is coherent.

    Args:
        existing_calendar: Roll calendar currently stored on disk.
        candidate_calendar: Freshly rebuilt roll calendar.

    Returns:
        Tuple `(merged_calendar, merge_outcome)`.
    """

    existing_norm = _normalise_roll_calendar(existing_calendar)
    candidate_norm = _normalise_roll_calendar(candidate_calendar)
    merged_as_dataframe = pd.DataFrame(existing_norm).copy()
    existing_transition_keys = _extract_roll_calendar_transition_keys(
        merged_as_dataframe
    )

    rows_added = 0
    rows_replaced = 0
    candidate_rows_skipped = 0

    if len(existing_norm) == 0:
        coherent_mask = pd.DataFrame(candidate_norm).apply(
            _roll_calendar_row_is_coherent, axis=1
        )
        filtered_candidate = pd.DataFrame(candidate_norm)[coherent_mask].copy()
        if len(filtered_candidate) > 0:
            kept_rows = []
            kept_index = []
            for roll_date, candidate_row in filtered_candidate.iterrows():
                candidate_key = _roll_calendar_transition_key(candidate_row)
                if candidate_key in existing_transition_keys:
                    candidate_rows_skipped += 1
                    continue
                kept_rows.append(candidate_row)
                kept_index.append(roll_date)
                if candidate_key is not None:
                    existing_transition_keys.add(candidate_key)
            if kept_rows:
                merged_as_dataframe = pd.DataFrame(kept_rows, index=kept_index)
            else:
                merged_as_dataframe = pd.DataFrame(columns=filtered_candidate.columns)
        rows_added = len(merged_as_dataframe)
        candidate_rows_skipped += int((~coherent_mask).sum())
    else:
        for roll_date, candidate_row in pd.DataFrame(candidate_norm).iterrows():
            candidate_is_coherent = _roll_calendar_row_is_coherent(candidate_row)
            candidate_key = _roll_calendar_transition_key(candidate_row)
            if roll_date not in merged_as_dataframe.index:
                if candidate_is_coherent:
                    if candidate_key in existing_transition_keys:
                        candidate_rows_skipped += 1
                        continue
                    merged_as_dataframe.loc[roll_date] = candidate_row
                    rows_added += 1
                    if candidate_key is not None:
                        existing_transition_keys.add(candidate_key)
                else:
                    candidate_rows_skipped += 1
                continue

            existing_row = merged_as_dataframe.loc[roll_date]
            existing_is_coherent = _roll_calendar_row_is_coherent(existing_row)
            existing_key = _roll_calendar_transition_key(existing_row)
            if (not existing_is_coherent) and candidate_is_coherent:
                if (
                    candidate_key in existing_transition_keys
                    and candidate_key != existing_key
                ):
                    candidate_rows_skipped += 1
                    continue
                merged_as_dataframe.loc[roll_date] = candidate_row
                rows_replaced += 1
                existing_transition_keys = _extract_roll_calendar_transition_keys(
                    merged_as_dataframe
                )
                continue

            if not candidate_is_coherent:
                candidate_rows_skipped += 1

    merged_as_dataframe = merged_as_dataframe.sort_index()
    merged_calendar = rollCalendar(merged_as_dataframe)
    merge_outcome = RollCalendarMergeOutcome(
        existing_rows=len(existing_norm),
        candidate_rows=len(candidate_norm),
        merged_rows=len(merged_calendar),
        rows_added=rows_added,
        rows_replaced=rows_replaced,
        candidate_rows_skipped=candidate_rows_skipped,
    )

    return merged_calendar, merge_outcome


def _write_roll_calendar_to_csv(
    instrument_code: str,
    output_datapath: str,
    roll_calendar_to_write: rollCalendar,
) -> RollCalendarMergeOutcome:
    """Write roll calendar to CSV with conservative non-destructive merge.

    Args:
        instrument_code: Instrument code to write.
        output_datapath: Destination directory for roll calendar CSV files.
        roll_calendar_to_write: Candidate roll calendar to merge and persist.

    Returns:
        `RollCalendarMergeOutcome` for persisted merge operation.
    """

    csv_roll_calendars = csvRollCalendarData(output_datapath)
    has_existing_calendar = csv_roll_calendars.is_code_in_data(instrument_code)
    if has_existing_calendar:
        existing_calendar = csv_roll_calendars.get_roll_calendar(instrument_code)
    else:
        existing_calendar = _empty_roll_calendar()

    merged_calendar, merge_outcome = _merge_roll_calendars_non_destructive(
        existing_calendar=existing_calendar,
        candidate_calendar=roll_calendar_to_write,
    )
    should_write = merge_outcome.rows_added > 0 or merge_outcome.rows_replaced > 0
    should_write = should_write or (
        not has_existing_calendar and len(merged_calendar) > 0
    )
    if should_write:
        csv_roll_calendars.add_roll_calendar(
            instrument_code,
            merged_calendar,
            ignore_duplication=True,
        )

    return merge_outcome


def _normalise_multiple_prices(
    multiple_prices: futuresMultiplePrices,
) -> futuresMultiplePrices:
    """Return sorted, de-duplicated multiple prices."""

    as_dataframe = pd.DataFrame(multiple_prices).sort_index()
    as_dataframe = as_dataframe[~as_dataframe.index.duplicated(keep="first")]
    if len(as_dataframe) == 0:
        return futuresMultiplePrices.create_empty()

    return futuresMultiplePrices(as_dataframe)


def _multiple_prices_row_is_coherent(multiple_row: pd.Series) -> bool:
    """Return `True` when one multiple-prices row is coherent."""

    if "PRICE" not in multiple_row or not _is_finite_number(multiple_row["PRICE"]):
        return False
    if "PRICE_CONTRACT" not in multiple_row:
        return False
    if _normalise_contract_code(multiple_row["PRICE_CONTRACT"]) is None:
        return False

    for price_column_name in list_of_price_column_names:
        if price_column_name not in multiple_row:
            continue
        if pd.isna(multiple_row[price_column_name]):
            continue
        if not _is_finite_number(multiple_row[price_column_name]):
            return False

    for contract_column_name in list_of_contract_column_names:
        if contract_column_name not in multiple_row:
            continue
        if pd.isna(multiple_row[contract_column_name]):
            continue
        if _normalise_contract_code(multiple_row[contract_column_name]) is None:
            return False

    return True


def _contract_code_to_month_number(raw_contract_code: object) -> int | None:
    """Convert a contract code to a comparable month number (`YYYY * 12 + MM`)."""

    normalised_contract = _normalise_contract_code(raw_contract_code)
    if normalised_contract is None:
        return None

    contract_yyyymm = normalised_contract[:6]
    try:
        year = int(contract_yyyymm[:4])
        month = int(contract_yyyymm[4:6])
    except ValueError:
        return None

    if month < 1 or month > 12:
        return None

    return year * 12 + month


def _count_price_contract_regressions(multiple_prices: futuresMultiplePrices) -> int:
    """Count timestamps where `PRICE_CONTRACT` goes backwards in time."""

    normalised_multiple = _normalise_multiple_prices(multiple_prices)
    if len(normalised_multiple) <= 1:
        return 0

    multiple_dataframe = pd.DataFrame(normalised_multiple)
    if "PRICE_CONTRACT" not in multiple_dataframe:
        return 0

    month_numbers = multiple_dataframe["PRICE_CONTRACT"].apply(
        _contract_code_to_month_number
    )
    previous_month_numbers = month_numbers.shift(1)
    regression_mask = (
        month_numbers.notna()
        & previous_month_numbers.notna()
        & (month_numbers < previous_month_numbers)
    )

    return int(regression_mask.sum())


def _should_prefer_candidate_multiple_prices(
    merged_multiple_prices: futuresMultiplePrices,
    candidate_multiple_prices: futuresMultiplePrices,
) -> bool:
    """Return `True` when candidate should replace merged multiple prices.

    We only switch to candidate when the merged output still contains
    contract-sequence regressions but candidate is clean and reaches at least
    the same latest timestamp.
    """

    merged_regressions = _count_price_contract_regressions(merged_multiple_prices)
    if merged_regressions == 0:
        return False

    candidate_regressions = _count_price_contract_regressions(candidate_multiple_prices)
    if candidate_regressions > 0:
        return False

    merged_norm = _normalise_multiple_prices(merged_multiple_prices)
    candidate_norm = _normalise_multiple_prices(candidate_multiple_prices)
    if len(candidate_norm) == 0:
        return False
    if len(merged_norm) == 0:
        return True

    merged_last_timestamp = pd.DataFrame(merged_norm).index.max()
    candidate_last_timestamp = pd.DataFrame(candidate_norm).index.max()

    return candidate_last_timestamp >= merged_last_timestamp


def _merge_multiple_prices_non_destructive(
    existing_multiple_prices: futuresMultiplePrices,
    candidate_multiple_prices: futuresMultiplePrices,
) -> tuple[futuresMultiplePrices, DerivedMergeOutcome]:
    """Merge multiple prices without overwriting coherent existing rows."""

    existing_norm = _normalise_multiple_prices(existing_multiple_prices)
    candidate_norm = _normalise_multiple_prices(candidate_multiple_prices)
    merged_as_dataframe = pd.DataFrame(existing_norm).copy()

    rows_added = 0
    rows_replaced = 0
    candidate_rows_skipped = 0

    if len(existing_norm) == 0:
        candidate_as_dataframe = pd.DataFrame(candidate_norm)
        coherent_mask = candidate_as_dataframe.apply(
            _multiple_prices_row_is_coherent, axis=1
        )
        merged_as_dataframe = candidate_as_dataframe[coherent_mask].copy()
        rows_added = int(coherent_mask.sum())
        candidate_rows_skipped = int((~coherent_mask).sum())
    else:
        for timestamp, candidate_row in pd.DataFrame(candidate_norm).iterrows():
            candidate_is_coherent = _multiple_prices_row_is_coherent(candidate_row)
            if timestamp not in merged_as_dataframe.index:
                if candidate_is_coherent:
                    merged_as_dataframe.loc[timestamp] = candidate_row
                    rows_added += 1
                else:
                    candidate_rows_skipped += 1
                continue

            existing_row = merged_as_dataframe.loc[timestamp]
            existing_is_coherent = _multiple_prices_row_is_coherent(existing_row)
            if (not existing_is_coherent) and candidate_is_coherent:
                merged_as_dataframe.loc[timestamp] = candidate_row
                rows_replaced += 1
                continue

            if not candidate_is_coherent:
                candidate_rows_skipped += 1

    merged_as_dataframe = merged_as_dataframe.sort_index()
    if len(merged_as_dataframe) == 0:
        merged_multiple_prices = futuresMultiplePrices.create_empty()
    else:
        merged_multiple_prices = futuresMultiplePrices(merged_as_dataframe)

    merge_outcome = DerivedMergeOutcome(
        existing_rows=len(existing_norm),
        candidate_rows=len(candidate_norm),
        merged_rows=len(merged_multiple_prices),
        rows_added=rows_added,
        rows_replaced=rows_replaced,
        candidate_rows_skipped=candidate_rows_skipped,
    )

    return merged_multiple_prices, merge_outcome


def _normalise_adjusted_prices(
    adjusted_prices: futuresAdjustedPrices,
) -> futuresAdjustedPrices:
    """Return sorted, de-duplicated adjusted prices."""

    as_series = pd.Series(adjusted_prices).sort_index()
    as_series = as_series[~as_series.index.duplicated(keep="first")]
    return futuresAdjustedPrices(as_series)


def _merge_adjusted_prices_non_destructive(
    existing_adjusted_prices: futuresAdjustedPrices,
    candidate_adjusted_prices: futuresAdjustedPrices,
) -> tuple[futuresAdjustedPrices, DerivedMergeOutcome]:
    """Merge adjusted prices without overwriting coherent existing values."""

    existing_norm = _normalise_adjusted_prices(existing_adjusted_prices)
    candidate_norm = _normalise_adjusted_prices(candidate_adjusted_prices)
    merged_as_series = pd.Series(existing_norm).copy()

    rows_added = 0
    rows_replaced = 0
    candidate_rows_skipped = 0

    if len(existing_norm) == 0:
        coherent_mask = pd.Series(candidate_norm).apply(_is_finite_number)
        merged_as_series = pd.Series(candidate_norm)[coherent_mask].copy()
        rows_added = int(coherent_mask.sum())
        candidate_rows_skipped = int((~coherent_mask).sum())
    else:
        for timestamp, candidate_value in pd.Series(candidate_norm).items():
            candidate_is_coherent = _is_finite_number(candidate_value)
            if timestamp not in merged_as_series.index:
                if candidate_is_coherent:
                    merged_as_series.loc[timestamp] = candidate_value
                    rows_added += 1
                else:
                    candidate_rows_skipped += 1
                continue

            existing_value = merged_as_series.loc[timestamp]
            existing_is_coherent = _is_finite_number(existing_value)
            if (not existing_is_coherent) and candidate_is_coherent:
                merged_as_series.loc[timestamp] = candidate_value
                rows_replaced += 1
                continue

            if not candidate_is_coherent:
                candidate_rows_skipped += 1

    merged_as_series = merged_as_series.sort_index()
    merged_adjusted_prices = futuresAdjustedPrices(merged_as_series)
    merge_outcome = DerivedMergeOutcome(
        existing_rows=len(existing_norm),
        candidate_rows=len(candidate_norm),
        merged_rows=len(merged_adjusted_prices),
        rows_added=rows_added,
        rows_replaced=rows_replaced,
        candidate_rows_skipped=candidate_rows_skipped,
    )

    return merged_adjusted_prices, merge_outcome


def _rebuild_and_merge_multiple_prices_non_destructive(
    instrument_code: str,
) -> tuple[futuresMultiplePrices, DerivedMergeOutcome, bool]:
    """Rebuild and store multiple prices with a monotonicity safety fallback.

    Args:
        instrument_code: Instrument to rebuild.

    Returns:
        Tuple `(multiple_prices, merge_outcome, force_candidate_overwrite)`.
        The boolean is `True` when the function bypasses conservative
        timestamp-level merge and stores the rebuilt candidate directly to
        eliminate detected `PRICE_CONTRACT` regressions.
    """

    candidate_multiple_prices = process_multiple_prices_single_instrument(
        instrument_code=instrument_code,
        ADD_TO_DB=False,
        ADD_TO_CSV=False,
    )
    existing_multiple_prices = diag_prices.get_multiple_prices(instrument_code)
    merged_multiple_prices, merge_outcome = _merge_multiple_prices_non_destructive(
        existing_multiple_prices=existing_multiple_prices,
        candidate_multiple_prices=candidate_multiple_prices,
    )

    force_candidate_overwrite = _should_prefer_candidate_multiple_prices(
        merged_multiple_prices=merged_multiple_prices,
        candidate_multiple_prices=candidate_multiple_prices,
    )
    if force_candidate_overwrite:
        merged_multiple_prices = _normalise_multiple_prices(candidate_multiple_prices)
        diag_prices.db_futures_multiple_prices_data.add_multiple_prices(
            instrument_code,
            merged_multiple_prices,
            ignore_duplication=True,
        )
        return merged_multiple_prices, merge_outcome, True

    if merge_outcome.rows_added > 0 or merge_outcome.rows_replaced > 0:
        diag_prices.db_futures_multiple_prices_data.add_multiple_prices(
            instrument_code,
            merged_multiple_prices,
            ignore_duplication=True,
        )

    return merged_multiple_prices, merge_outcome, False


def _rebuild_and_merge_adjusted_prices_non_destructive(
    instrument_code: str,
    multiple_prices: futuresMultiplePrices,
    force_candidate_overwrite: bool = False,
) -> DerivedMergeOutcome:
    """Rebuild adjusted prices and merge or overwrite into DB.

    Args:
        instrument_code: Instrument to rebuild.
        multiple_prices: Multiple-price series used to rebuild adjusted prices.
        force_candidate_overwrite: If `True`, write rebuilt adjusted prices as
            authoritative output instead of conservative row-level merge.

    Returns:
        Merge summary for adjusted prices.
    """

    candidate_adjusted_prices = process_adjusted_prices_single_instrument(
        instrument_code=instrument_code,
        multiple_prices=multiple_prices,
        ADD_TO_DB=False,
        ADD_TO_CSV=False,
    )
    existing_adjusted_prices = diag_prices.get_adjusted_prices(instrument_code)
    existing_norm = _normalise_adjusted_prices(existing_adjusted_prices)
    candidate_norm = _normalise_adjusted_prices(candidate_adjusted_prices)
    if force_candidate_overwrite:
        diag_prices.db_futures_adjusted_prices_data.add_adjusted_prices(
            instrument_code,
            candidate_norm,
            ignore_duplication=True,
        )
        return DerivedMergeOutcome(
            existing_rows=len(existing_norm),
            candidate_rows=len(candidate_norm),
            merged_rows=len(candidate_norm),
            rows_added=max(len(candidate_norm) - len(existing_norm), 0),
            rows_replaced=0,
            candidate_rows_skipped=0,
        )

    merged_adjusted_prices, merge_outcome = _merge_adjusted_prices_non_destructive(
        existing_adjusted_prices=existing_norm,
        candidate_adjusted_prices=candidate_norm,
    )

    if merge_outcome.rows_added > 0 or merge_outcome.rows_replaced > 0:
        diag_prices.db_futures_adjusted_prices_data.add_adjusted_prices(
            instrument_code,
            merged_adjusted_prices,
            ignore_duplication=True,
        )

    return merge_outcome


def init_db_with_split_freq_csv_prices_for_code_non_destructive(
    instrument_code: str,
    datapath: str,
    csv_config=arg_not_supplied,
):
    """Merge Barchart day/hour CSV prices into DB without destructive overwrite.

    Existing values are kept on overlapping timestamps. The script adds missing
    rows and replaces overlaps only when the existing row is incoherent and the
    incoming row is coherent.

    Args:
        instrument_code: Instrument code to process.
        datapath: Directory containing Barchart CSV files.
        csv_config: CSV parsing configuration. If omitted, default parser rules
            from `ConfigCsvFuturesPrices` are used.
    """
    (
        hourly_dict,
        daily_dict,
        hourly_final_columns,
        daily_final_columns,
    ) = _load_split_freq_prices_with_fallback(
        instrument_code=instrument_code,
        datapath=datapath,
        csv_config=csv_config,
    )
    print(_resolve_csv_mapping_message(hourly_final_columns, daily_final_columns))
    contract_date_list = sorted(set(hourly_dict.keys()) | set(daily_dict.keys()))

    print(
        f"Found {len(contract_date_list)} contracts for {instrument_code} in CSV path "
        f"{datapath}"
    )
    rows_added_by_frequency = {
        HOURLY_FREQ: 0,
        DAILY_PRICE_FREQ: 0,
        MIXED_FREQ: 0,
    }
    rows_replaced_by_frequency = {
        HOURLY_FREQ: 0,
        DAILY_PRICE_FREQ: 0,
        MIXED_FREQ: 0,
    }
    rows_skipped_by_frequency = {
        HOURLY_FREQ: 0,
        DAILY_PRICE_FREQ: 0,
        MIXED_FREQ: 0,
    }
    contracts_with_updates = []

    for contract_date_str in contract_date_list:
        contract = futuresContract(instrument_code, contract_date_str)
        contract_rows_changed = 0

        print(f"\nProcessing {contract}")
        if contract_date_str in hourly_dict:
            hourly_outcome = _merge_and_write_prices_for_frequency(
                contract=contract,
                incoming_prices=hourly_dict[contract_date_str],
                frequency=HOURLY_FREQ,
            )
            rows_added_by_frequency[HOURLY_FREQ] += hourly_outcome.rows_added
            rows_replaced_by_frequency[HOURLY_FREQ] += hourly_outcome.rows_replaced
            rows_skipped_by_frequency[
                HOURLY_FREQ
            ] += hourly_outcome.incoming_rows_skipped
            contract_rows_changed += (
                hourly_outcome.rows_added + hourly_outcome.rows_replaced
            )
            print(
                f"  Hourly rows: existing={hourly_outcome.existing_rows}, "
                f"incoming={hourly_outcome.incoming_rows}, "
                f"added={hourly_outcome.rows_added}, "
                f"replaced={hourly_outcome.rows_replaced}, "
                f"skipped={hourly_outcome.incoming_rows_skipped}"
            )

        if contract_date_str in daily_dict:
            daily_outcome = _merge_and_write_prices_for_frequency(
                contract=contract,
                incoming_prices=daily_dict[contract_date_str],
                frequency=DAILY_PRICE_FREQ,
            )
            rows_added_by_frequency[DAILY_PRICE_FREQ] += daily_outcome.rows_added
            rows_replaced_by_frequency[DAILY_PRICE_FREQ] += daily_outcome.rows_replaced
            rows_skipped_by_frequency[
                DAILY_PRICE_FREQ
            ] += daily_outcome.incoming_rows_skipped
            contract_rows_changed += (
                daily_outcome.rows_added + daily_outcome.rows_replaced
            )
            print(
                f"  Daily rows: existing={daily_outcome.existing_rows}, "
                f"incoming={daily_outcome.incoming_rows}, "
                f"added={daily_outcome.rows_added}, "
                f"replaced={daily_outcome.rows_replaced}, "
                f"skipped={daily_outcome.incoming_rows_skipped}"
            )

        mixed_candidate = _mixed_prices_candidate_from_db(contract)
        if len(mixed_candidate) > 0:
            mixed_outcome = _merge_and_write_prices_for_frequency(
                contract=contract,
                incoming_prices=mixed_candidate,
                frequency=MIXED_FREQ,
            )
            rows_added_by_frequency[MIXED_FREQ] += mixed_outcome.rows_added
            rows_replaced_by_frequency[MIXED_FREQ] += mixed_outcome.rows_replaced
            rows_skipped_by_frequency[MIXED_FREQ] += mixed_outcome.incoming_rows_skipped
            contract_rows_changed += (
                mixed_outcome.rows_added + mixed_outcome.rows_replaced
            )
            print(
                f"  Mixed rows: existing={mixed_outcome.existing_rows}, "
                f"incoming={mixed_outcome.incoming_rows}, "
                f"added={mixed_outcome.rows_added}, "
                f"replaced={mixed_outcome.rows_replaced}, "
                f"skipped={mixed_outcome.incoming_rows_skipped}"
            )

        if contract_rows_changed > 0:
            contracts_with_updates.append(contract_date_str)

    print("\nNon-destructive import summary")
    print(f"Instrument: {instrument_code}")
    print(f"Contracts in CSV: {len(contract_date_list)}")
    print(f"Contracts updated in DB: {len(contracts_with_updates)}")
    print(f"Rows added (hourly): {rows_added_by_frequency[HOURLY_FREQ]}")
    print(f"Rows replaced (hourly): {rows_replaced_by_frequency[HOURLY_FREQ]}")
    print(f"Rows skipped (hourly): {rows_skipped_by_frequency[HOURLY_FREQ]}")
    print(f"Rows added (daily): {rows_added_by_frequency[DAILY_PRICE_FREQ]}")
    print(f"Rows replaced (daily): {rows_replaced_by_frequency[DAILY_PRICE_FREQ]}")
    print(f"Rows skipped (daily): {rows_skipped_by_frequency[DAILY_PRICE_FREQ]}")
    print(f"Rows added (mixed): {rows_added_by_frequency[MIXED_FREQ]}")
    print(f"Rows replaced (mixed): {rows_replaced_by_frequency[MIXED_FREQ]}")
    print(f"Rows skipped (mixed): {rows_skipped_by_frequency[MIXED_FREQ]}")


if __name__ == "__main__":
    input(
        "Will run full Barchart import process in non-destructive mode. "
        "Existing data is preserved, except incoherent rows that can be replaced "
        "with coherent values. CTRL-C to abort."
    )
    production_config = get_production_config()
    instrument_code = input("Enter the futures symbol (e.g., 'BITCOIN'): ")

    barchart_path = production_config.get_element_or_default("barchart_path", None)
    if barchart_path is None:
        raise ValueError("Missing `barchart_path` in production/private config")

    roll_calendars_path = production_config.get_element_or_default(
        "csv_roll_calendars_directory", None
    )
    if roll_calendars_path is None:
        raise ValueError(
            "Missing `csv_roll_calendars_directory` in production/private config"
        )

    datapath = resolve_path_and_filename_for_package(barchart_path)
    calendar_output_datapath = resolve_path_and_filename_for_package(
        roll_calendars_path
    )

    init_db_with_split_freq_csv_prices_for_code_non_destructive(
        instrument_code=instrument_code,
        datapath=datapath,
        csv_config=BARCHART_CONFIG,
    )

    rebuild_derived_input = input(
        "Rebuild roll calendar + multiple + adjusted after contract import? [y/N] "
    ).strip()
    rebuild_derived = rebuild_derived_input.lower() in {"y", "yes"}
    if not rebuild_derived:
        print(
            "Skipped roll calendar/multiple/adjusted rebuild. "
            "Contract prices import completed in conservative mode."
        )
        raise SystemExit(0)

    calendar_build_result = _build_roll_calendar_with_recent_to_past_fallback(
        instrument_code=instrument_code,
    )
    calendar_merge_outcome = _write_roll_calendar_to_csv(
        instrument_code=instrument_code,
        output_datapath=calendar_output_datapath,
        roll_calendar_to_write=calendar_build_result.roll_calendar,
    )
    if calendar_build_result.mode == "recent_to_past":
        print(
            "Applied recent-to-past fallback for roll calendar generation; "
            "earliest retained contract: "
            f"{calendar_build_result.cutoff_contract}"
        )
    else:
        print("Roll calendar generated with full-history build.")
    print(
        "Roll calendar merge: "
        f"existing={calendar_merge_outcome.existing_rows}, "
        f"candidate={calendar_merge_outcome.candidate_rows}, "
        f"added={calendar_merge_outcome.rows_added}, "
        f"replaced={calendar_merge_outcome.rows_replaced}, "
        f"skipped={calendar_merge_outcome.candidate_rows_skipped}"
    )

    _ = input(
        f"Roll calendar for {instrument_code} written to "
        f"{calendar_output_datapath}. Press Enter to continue."
    )

    (
        merged_multiple_prices,
        multiple_merge_outcome,
        multiple_was_force_overwritten,
    ) = _rebuild_and_merge_multiple_prices_non_destructive(
        instrument_code=instrument_code
    )
    print(
        "Multiple prices merge: "
        f"existing={multiple_merge_outcome.existing_rows}, "
        f"candidate={multiple_merge_outcome.candidate_rows}, "
        f"added={multiple_merge_outcome.rows_added}, "
        f"replaced={multiple_merge_outcome.rows_replaced}, "
        f"skipped={multiple_merge_outcome.candidate_rows_skipped}"
    )
    if multiple_was_force_overwritten:
        print(
            "Detected `PRICE_CONTRACT` regressions after conservative merge; "
            "stored the freshly rebuilt multiple prices to keep contract "
            "transitions monotonic."
        )
    adjusted_merge_outcome = _rebuild_and_merge_adjusted_prices_non_destructive(
        instrument_code=instrument_code,
        multiple_prices=merged_multiple_prices,
        force_candidate_overwrite=multiple_was_force_overwritten,
    )
    print(
        "Adjusted prices merge: "
        f"existing={adjusted_merge_outcome.existing_rows}, "
        f"candidate={adjusted_merge_outcome.candidate_rows}, "
        f"added={adjusted_merge_outcome.rows_added}, "
        f"replaced={adjusted_merge_outcome.rows_replaced}, "
        f"skipped={adjusted_merge_outcome.candidate_rows_skipped}"
    )
