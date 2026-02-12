import pandas as pd

from sysdata.csv.csv_futures_contract_prices import ConfigCsvFuturesPrices
from sysinit.futures import (
    contract_prices_from_split_freq_csv_to_db as split_import,
)
from sysobjects.futures_per_contract_prices import futuresContractPrices


def _make_prices(rows: list[tuple[str, float]]) -> futuresContractPrices:
    index = pd.to_datetime([date_str for date_str, _ in rows])
    final_values = [final for _, final in rows]
    dataframe = pd.DataFrame(
        {
            "OPEN": final_values,
            "HIGH": [value + 1.0 for value in final_values],
            "LOW": [value - 1.0 for value in final_values],
            "FINAL": final_values,
            "VOLUME": [100.0] * len(final_values),
        },
        index=index,
    )
    return futuresContractPrices(dataframe)


class _FakeCsvPriceDataWithRowLevelCoalesce:
    def __init__(self, datapath, config):
        self._final_column = config.input_column_mapping.get("FINAL")

    def contract_dates_with_price_data_at_frequency_for_instrument_code(
        self, instrument_code, frequency
    ):
        return ["20200300"]

    def get_prices_at_frequency_for_contract_object(self, contract, frequency):
        if self._final_column == "Latest":
            return _make_prices(
                [
                    ("2020-01-01 00:00:00", 100.0),
                    ("2020-01-02 00:00:00", float("nan")),
                ]
            )
        if self._final_column == "Close":
            return _make_prices(
                [
                    ("2020-01-01 00:00:00", 101.0),
                    ("2020-01-02 00:00:00", 102.0),
                    ("2020-01-03 00:00:00", 103.0),
                ]
            )
        raise KeyError(self._final_column)


def test_load_split_freq_prices_with_fallback_coalesces_final_row_wise(monkeypatch):
    monkeypatch.setattr(
        split_import,
        "csvFuturesContractPriceData",
        _FakeCsvPriceDataWithRowLevelCoalesce,
    )
    starting_config = ConfigCsvFuturesPrices(
        input_date_index_name="Time",
        input_skiprows=0,
        input_skipfooter=0,
        input_date_format="ISO8601",
        input_column_mapping=dict(
            OPEN="Open", HIGH="High", LOW="Low", FINAL="Latest", VOLUME="Volume"
        ),
    )

    (
        hourly_dict,
        daily_dict,
        hourly_final_columns,
        daily_final_columns,
    ) = split_import._load_split_freq_prices_with_fallback(
        instrument_code="TEST",
        datapath="unused",
        csv_config=starting_config,
    )

    for merged_prices in (hourly_dict["20200300"], daily_dict["20200300"]):
        assert merged_prices.loc[pd.Timestamp("2020-01-01 00:00:00"), "FINAL"] == 100.0
        assert merged_prices.loc[pd.Timestamp("2020-01-02 00:00:00"), "FINAL"] == 102.0
        assert merged_prices.loc[pd.Timestamp("2020-01-03 00:00:00"), "FINAL"] == 103.0

    assert hourly_final_columns == {"Latest", "Close"}
    assert daily_final_columns == {"Latest", "Close"}
