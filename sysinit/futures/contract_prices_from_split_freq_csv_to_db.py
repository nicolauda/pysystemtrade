from syscore.constants import arg_not_supplied
from syscore.dateutils import MIXED_FREQ, HOURLY_FREQ, DAILY_PRICE_FREQ
from syscore.pandas.frequency import merge_data_with_different_freq
from sysdata.csv.csv_futures_contract_prices import ConfigCsvFuturesPrices
from sysdata.csv.csv_futures_contract_prices import csvFuturesContractPriceData
from sysobjects.contracts import futuresContract
from sysobjects.futures_per_contract_prices import futuresContractPrices
from sysproduction.data.prices import diagPrices, get_valid_instrument_code_from_user
import pandas as pd

diag_prices = diagPrices()
db_prices = diag_prices.db_futures_contract_price_data

BARCHART_CONFIG = ConfigCsvFuturesPrices(
    input_date_index_name="Time",
    input_skiprows=0,
    input_skipfooter=0,
    input_date_format="%Y-%m-%dT%H:%M:%S",
    input_column_mapping=dict(
        OPEN="Open", HIGH="High", LOW="Low", FINAL="Close", VOLUME="Volume"
    ),
)
BARCHART_FINAL_PRICE_COLUMNS = ("Latest", "Close", "Last")


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
    if csv_config is arg_not_supplied:
        base_config = BARCHART_CONFIG
    else:
        base_config = csv_config

    configured_final = base_config.input_column_mapping.get("FINAL")
    ordered_finals = [configured_final, *BARCHART_FINAL_PRICE_COLUMNS]

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


def _normalise_prices(price_data: futuresContractPrices) -> futuresContractPrices:
    as_dataframe = pd.DataFrame(price_data).sort_index()
    as_dataframe = as_dataframe[~as_dataframe.index.duplicated(keep="first")]
    return futuresContractPrices(as_dataframe)


def _coalesce_final_values_from_fallback(
    primary_prices: futuresContractPrices,
    fallback_prices: futuresContractPrices,
) -> tuple[futuresContractPrices, int, int]:
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


def _load_frequency_dict_with_fallback(
    instrument_code: str,
    frequency,
    candidate_readers: list[tuple[ConfigCsvFuturesPrices, csvFuturesContractPriceData]],
) -> tuple[dict, set[str]]:
    list_reader_for_contract_discovery = candidate_readers[0][1]
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
        parsed_prices_by_final_column: list[tuple[str, futuresContractPrices]] = []
        parse_errors = []

        for candidate_config, csv_reader in candidate_readers:
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
                for candidate, _ in candidate_readers
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


def _load_split_freq_prices_with_fallback(
    instrument_code: str,
    datapath: str,
    csv_config: ConfigCsvFuturesPrices | object = arg_not_supplied,
) -> tuple[dict, dict, set[str], set[str]]:
    candidate_configs = _get_candidate_csv_configs(csv_config)
    candidate_readers = [
        (candidate, csvFuturesContractPriceData(datapath, config=candidate))
        for candidate in candidate_configs
    ]

    hourly_dict, hourly_final_columns = _load_frequency_dict_with_fallback(
        instrument_code=instrument_code,
        frequency=HOURLY_FREQ,
        candidate_readers=candidate_readers,
    )
    daily_dict, daily_final_columns = _load_frequency_dict_with_fallback(
        instrument_code=instrument_code,
        frequency=DAILY_PRICE_FREQ,
        candidate_readers=candidate_readers,
    )

    print(
        f"Using FINAL column(s) for hourly CSV parsing: {sorted(hourly_final_columns)}"
    )
    print(f"Using FINAL column(s) for daily CSV parsing: {sorted(daily_final_columns)}")

    return hourly_dict, daily_dict, hourly_final_columns, daily_final_columns


def init_db_with_split_freq_csv_prices_for_code(
    instrument_code: str,
    datapath: str,
    csv_config=arg_not_supplied,
    ignore_duplication: bool = True,
):
    same_length = []
    too_short = []
    print(f"Importing split freq csv prices for {instrument_code}")
    print("Getting split freq .csv prices may take some time")
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
    print(
        "Resolved CSV mapping: "
        f"hourly FINAL -> {sorted(hourly_final_columns)}; "
        f"daily FINAL -> {sorted(daily_final_columns)}"
    )

    hourly_and_daily = sorted(hourly_dict.keys() & daily_dict.keys())
    daily_only = sorted(set(daily_dict.keys()) - set(hourly_dict.keys()))
    hourly_only = sorted(set(hourly_dict.keys()) - set(daily_dict.keys()))

    print(f"hourly_and_daily: {sorted(hourly_and_daily)}")
    print(f"daily_only: {sorted(daily_only)}")
    print(f"hourly_only: {sorted(hourly_only)}")

    print(f"Have hourly and daily .csv prices for: {str(hourly_and_daily)}")
    for contract_date_str in hourly_and_daily:
        print(f"Processing {contract_date_str}")

        contract = futuresContract(instrument_code, contract_date_str)
        print(f"Contract object is {str(contract)}")

        hourly = hourly_dict[contract_date_str]
        write_prices_for_contract_at_frequency(
            contract, hourly, HOURLY_FREQ, ignore_duplication=ignore_duplication
        )

        daily = daily_dict[contract_date_str]
        write_prices_for_contract_at_frequency(
            contract, daily, DAILY_PRICE_FREQ, ignore_duplication=ignore_duplication
        )

        merged = futuresContractPrices(merge_data_with_different_freq([hourly, daily]))
        write_prices_for_contract_at_frequency(
            contract, merged, MIXED_FREQ, ignore_duplication=ignore_duplication
        )

        if len(hourly) == len(daily):
            same_length.append(contract_date_str)

    print(f"Have daily only .csv prices for: {str(daily_only)}")
    for contract_date_str in daily_only:
        print(f"Processing {contract_date_str}")

        contract = futuresContract(instrument_code, contract_date_str)
        print(f"Contract object is {str(contract)}")

        daily = daily_dict[contract_date_str]
        write_prices_for_contract_at_frequency(
            contract, daily, DAILY_PRICE_FREQ, ignore_duplication=ignore_duplication
        )

        # if we already have hourly data in the db, get it and merge with daily
        if db_prices.has_price_data_for_contract_at_frequency(
            contract, frequency=HOURLY_FREQ
        ):
            hourly = db_prices.get_prices_at_frequency_for_contract_object(
                contract, frequency=HOURLY_FREQ
            )
            merged = futuresContractPrices(
                merge_data_with_different_freq([hourly, daily])
            )
        else:
            merged = daily
        write_prices_for_contract_at_frequency(
            contract, merged, MIXED_FREQ, ignore_duplication=ignore_duplication
        )

        if len(daily) < 65:
            too_short.append(contract_date_str)

    print(f"Have hourly only .csv prices for: {str(hourly_only)}")
    for contract_date_str in hourly_only:
        print(f"Processing {contract_date_str}")

        contract = futuresContract(instrument_code, contract_date_str)
        print(f"Contract object is {str(contract)}")

        hourly = hourly_dict[contract_date_str]
        write_prices_for_contract_at_frequency(
            contract, hourly, HOURLY_FREQ, ignore_duplication=ignore_duplication
        )

        # if we already have daily data in the db, get it and merge with hourly
        if db_prices.has_price_data_for_contract_at_frequency(
            contract, frequency=DAILY_PRICE_FREQ
        ):
            daily = db_prices.get_prices_at_frequency_for_contract_object(
                contract, frequency=DAILY_PRICE_FREQ
            )
            merged = futuresContractPrices(
                merge_data_with_different_freq([hourly, daily])
            )
        else:
            merged = hourly
        write_prices_for_contract_at_frequency(
            contract, merged, MIXED_FREQ, ignore_duplication=ignore_duplication
        )

    print(f"These contracts have the same length for daily and hourly: {same_length}")
    print(f"These daily contracts are short: {too_short}")


def write_prices_for_contract_at_frequency(
    contract, prices, frequency, ignore_duplication=False
):
    print(f"{frequency} .csv prices are \n{str(prices)}")
    print("Writing to db")
    db_prices.write_prices_at_frequency_for_contract_object(
        contract,
        prices,
        frequency=frequency,
        ignore_duplication=ignore_duplication,
    )
    print("Reading back prices from db to check")
    written_prices = db_prices.get_prices_at_frequency_for_contract_object(
        contract, frequency=frequency
    )
    print(f"Read back prices ({frequency}) are \n{str(written_prices)}")


if __name__ == "__main__":
    input("Will overwrite existing prices are you sure?! CTL-C to abort")
    # modify flags as required
    datapath = "*** NEED TO DEFINE A DATAPATH***"
    do_another = True
    while do_another:
        EXIT_STR = "Finished: Exit"
        instrument_code = get_valid_instrument_code_from_user(
            source="single", allow_exit=True, exit_code=EXIT_STR
        )
        if instrument_code is EXIT_STR:
            do_another = False
        else:
            init_db_with_split_freq_csv_prices_for_code(
                instrument_code, datapath, csv_config=BARCHART_CONFIG
            )
