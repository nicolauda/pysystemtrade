import pandas as pd

from syscore.dateutils import DAILY_PRICE_FREQ
from sysdata.csv.csv_futures_contract_prices import ConfigCsvFuturesPrices
from sysinit.futures.adhoc import import_from_bc_to_db_non_destructive as nd_import
from sysobjects.contracts import futuresContract
from sysobjects.adjusted_prices import futuresAdjustedPrices
from sysobjects.futures_per_contract_prices import futuresContractPrices
from sysobjects.multiple_prices import futuresMultiplePrices


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


def _make_roll_calendar(
    roll_rows: list[tuple[str, str, str, str]]
) -> nd_import.rollCalendar:
    calendar = pd.DataFrame(
        [
            {
                "current_contract": current_contract,
                "next_contract": next_contract,
                "carry_contract": carry_contract,
            }
            for _, current_contract, next_contract, carry_contract in roll_rows
        ],
        index=pd.to_datetime([roll_date for roll_date, _, _, _ in roll_rows]),
    )
    return nd_import.rollCalendar(calendar)


def _make_multiple_prices(
    rows: list[tuple[str, float, float, float, str, str, str]]
) -> futuresMultiplePrices:
    index = pd.to_datetime([date_str for date_str, *_ in rows])
    dataframe = pd.DataFrame(
        {
            "PRICE": [price for _, price, *_ in rows],
            "CARRY": [carry for _, _, carry, *_ in rows],
            "FORWARD": [forward for _, _, _, forward, *_ in rows],
            "PRICE_CONTRACT": [
                price_contract for _, _, _, _, price_contract, *_ in rows
            ],
            "CARRY_CONTRACT": [
                carry_contract for _, _, _, _, _, carry_contract, _ in rows
            ],
            "FORWARD_CONTRACT": [
                forward_contract for _, _, _, _, _, _, forward_contract in rows
            ],
        },
        index=index,
    )
    return futuresMultiplePrices(dataframe)


def _make_adjusted_prices(rows: list[tuple[str, float]]) -> futuresAdjustedPrices:
    index = pd.to_datetime([date_str for date_str, _ in rows])
    values = [value for _, value in rows]
    return futuresAdjustedPrices(pd.Series(values, index=index))


class _FakeDbPrices:
    def __init__(self):
        self._data = {}
        self.writes: list[tuple[tuple, futuresContractPrices]] = []

    def _key(self, contract: futuresContract, frequency):
        return contract.key, frequency

    def set_prices(self, contract: futuresContract, frequency, prices):
        self._data[self._key(contract, frequency)] = prices

    def get_prices_at_frequency_for_contract_object(self, contract, frequency):
        key = self._key(contract, frequency)
        return self._data.get(key, futuresContractPrices.create_empty())

    def has_price_data_for_contract_at_frequency(self, contract, frequency):
        key = self._key(contract, frequency)
        if key not in self._data:
            return False
        return len(self._data[key]) > 0

    def write_prices_at_frequency_for_contract_object(
        self, contract, prices, frequency, ignore_duplication=True
    ):
        key = self._key(contract, frequency)
        self._data[key] = prices
        self.writes.append((key, prices))


class _FakeMergedPrices(dict):
    def final_prices(self):
        return self


class _FakeDbPricesForRollCalendar:
    def __init__(self, merged_prices: _FakeMergedPrices):
        self._merged_prices = merged_prices

    def get_merged_prices_for_instrument(self, instrument_code):
        return self._merged_prices


class _FakeRollParametersData:
    def get_roll_parameters(self, instrument_code):
        return object()


class _FakeCsvPriceDataWithFinalFallback:
    def __init__(self, datapath, config):
        self._final_column = config.input_column_mapping.get("FINAL")

    def contract_dates_with_price_data_at_frequency_for_instrument_code(
        self, instrument_code, frequency
    ):
        return ["20200300"]

    def get_prices_at_frequency_for_contract_object(self, contract, frequency):
        if self._final_column != "Latest":
            raise KeyError(self._final_column)
        return _make_prices([("2020-01-01 00:00:00", 100.0)])


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


def test_merge_prices_non_destructive_preserves_existing_values_on_overlap():
    existing = _make_prices(
        [
            ("2020-01-01 00:00:00", 100.0),
            ("2020-01-02 00:00:00", 101.0),
        ]
    )
    incoming = _make_prices(
        [
            ("2020-01-02 00:00:00", 999.0),
            ("2020-01-03 00:00:00", 102.0),
        ]
    )

    (
        merged,
        rows_added,
        rows_replaced,
        rows_skipped,
    ) = nd_import._merge_prices_non_destructive(existing, incoming)

    assert rows_added == 1
    assert rows_replaced == 0
    assert rows_skipped == 0
    assert len(merged) == 3
    assert merged.loc[pd.Timestamp("2020-01-02 00:00:00"), "FINAL"] == 101.0
    assert merged.loc[pd.Timestamp("2020-01-03 00:00:00"), "FINAL"] == 102.0


def test_merge_prices_non_destructive_replaces_incoherent_existing_on_overlap():
    existing = _make_prices([("2020-01-02 00:00:00", 101.0)])
    existing_as_dataframe = pd.DataFrame(existing)
    existing_as_dataframe.loc[pd.Timestamp("2020-01-02 00:00:00"), "HIGH"] = 100.0
    existing_as_dataframe.loc[pd.Timestamp("2020-01-02 00:00:00"), "LOW"] = 99.0
    existing_as_dataframe.loc[pd.Timestamp("2020-01-02 00:00:00"), "FINAL"] = 120.0
    incoherent_existing = futuresContractPrices(existing_as_dataframe)

    incoming = _make_prices([("2020-01-02 00:00:00", 101.0)])
    (
        merged,
        rows_added,
        rows_replaced,
        rows_skipped,
    ) = nd_import._merge_prices_non_destructive(incoherent_existing, incoming)

    assert rows_added == 0
    assert rows_replaced == 1
    assert rows_skipped == 0
    assert merged.loc[pd.Timestamp("2020-01-02 00:00:00"), "FINAL"] == 101.0


def test_merge_and_write_only_writes_when_new_rows_are_added(monkeypatch):
    fake_db = _FakeDbPrices()
    monkeypatch.setattr(nd_import, "db_prices", fake_db)

    contract = futuresContract("TEST", "20200300")
    base_existing = _make_prices(
        [
            ("2020-01-01 00:00:00", 100.0),
            ("2020-01-02 00:00:00", 101.0),
        ]
    )
    fake_db.set_prices(contract, DAILY_PRICE_FREQ, base_existing)

    overlap_only = _make_prices(
        [
            ("2020-01-01 00:00:00", 500.0),
            ("2020-01-02 00:00:00", 600.0),
        ]
    )
    no_add_outcome = nd_import._merge_and_write_prices_for_frequency(
        contract=contract,
        incoming_prices=overlap_only,
        frequency=DAILY_PRICE_FREQ,
    )
    assert no_add_outcome.rows_added == 0
    assert no_add_outcome.rows_replaced == 0
    assert len(fake_db.writes) == 0

    with_backfill = _make_prices(
        [
            ("2019-12-31 00:00:00", 90.0),
            ("2020-01-01 00:00:00", 700.0),
            ("2020-01-02 00:00:00", 800.0),
        ]
    )
    add_outcome = nd_import._merge_and_write_prices_for_frequency(
        contract=contract,
        incoming_prices=with_backfill,
        frequency=DAILY_PRICE_FREQ,
    )
    assert add_outcome.rows_added == 1
    assert add_outcome.rows_replaced == 0
    assert len(fake_db.writes) == 1

    written = fake_db.get_prices_at_frequency_for_contract_object(
        contract, DAILY_PRICE_FREQ
    )
    assert len(written) == 3
    assert written.loc[pd.Timestamp("2020-01-01 00:00:00"), "FINAL"] == 100.0
    assert written.loc[pd.Timestamp("2019-12-31 00:00:00"), "FINAL"] == 90.0


def test_merge_and_write_writes_when_incoherent_existing_row_is_replaced(monkeypatch):
    fake_db = _FakeDbPrices()
    monkeypatch.setattr(nd_import, "db_prices", fake_db)

    contract = futuresContract("TEST", "20200300")
    existing = _make_prices([("2020-01-01 00:00:00", 100.0)])
    existing_as_dataframe = pd.DataFrame(existing)
    existing_as_dataframe.loc[pd.Timestamp("2020-01-01 00:00:00"), "HIGH"] = 95.0
    existing_as_dataframe.loc[pd.Timestamp("2020-01-01 00:00:00"), "LOW"] = 94.0
    existing_as_dataframe.loc[pd.Timestamp("2020-01-01 00:00:00"), "FINAL"] = 100.0
    fake_db.set_prices(
        contract, DAILY_PRICE_FREQ, futuresContractPrices(existing_as_dataframe)
    )

    incoming = _make_prices([("2020-01-01 00:00:00", 100.0)])
    outcome = nd_import._merge_and_write_prices_for_frequency(
        contract=contract,
        incoming_prices=incoming,
        frequency=DAILY_PRICE_FREQ,
    )

    assert outcome.rows_added == 0
    assert outcome.rows_replaced == 1
    assert len(fake_db.writes) == 1
    written = fake_db.get_prices_at_frequency_for_contract_object(
        contract, DAILY_PRICE_FREQ
    )
    assert written.loc[pd.Timestamp("2020-01-01 00:00:00"), "HIGH"] == 101.0


def test_merge_roll_calendar_preserves_coherent_existing_overlap():
    existing = _make_roll_calendar(
        [
            ("2020-01-10 00:00:00", "20200300", "20200600", "20200300"),
            ("2020-03-10 00:00:00", "20200600", "20200900", "20200600"),
        ]
    )
    candidate = _make_roll_calendar(
        [
            ("2020-03-10 00:00:00", "20200600", "20201000", "20200600"),
            ("2020-06-10 00:00:00", "20200900", "20201200", "20200900"),
        ]
    )

    merged_calendar, outcome = nd_import._merge_roll_calendars_non_destructive(
        existing_calendar=existing,
        candidate_calendar=candidate,
    )

    assert outcome.rows_added == 1
    assert outcome.rows_replaced == 0
    assert str(
        merged_calendar.loc[pd.Timestamp("2020-03-10 00:00:00"), "next_contract"]
    ) == ("20200900")


def test_merge_roll_calendar_replaces_incoherent_existing_overlap():
    existing = _make_roll_calendar(
        [
            ("2020-03-10 00:00:00", "20200600", "20200600", "20200600"),
        ]
    )
    candidate = _make_roll_calendar(
        [
            ("2020-03-10 00:00:00", "20200600", "20200900", "20200600"),
        ]
    )

    merged_calendar, outcome = nd_import._merge_roll_calendars_non_destructive(
        existing_calendar=existing,
        candidate_calendar=candidate,
    )

    assert outcome.rows_added == 0
    assert outcome.rows_replaced == 1
    assert str(
        merged_calendar.loc[pd.Timestamp("2020-03-10 00:00:00"), "next_contract"]
    ) == ("20200900")


def test_merge_roll_calendar_skips_duplicate_transition_on_new_timestamp():
    existing = _make_roll_calendar(
        [
            ("2025-09-11 05:00:00", "20250900", "20251200", "20251200"),
        ]
    )
    candidate = _make_roll_calendar(
        [
            ("2025-09-11 01:45:00", "20250900", "20251200", "20251200"),
        ]
    )

    merged_calendar, outcome = nd_import._merge_roll_calendars_non_destructive(
        existing_calendar=existing,
        candidate_calendar=candidate,
    )

    assert outcome.rows_added == 0
    assert outcome.rows_replaced == 0
    assert outcome.candidate_rows_skipped == 1
    assert len(merged_calendar) == 1
    assert merged_calendar.index[0] == pd.Timestamp("2025-09-11 05:00:00")


def test_merge_multiple_prices_replaces_incoherent_overlap_and_adds_missing_rows():
    existing = _make_multiple_prices(
        [
            (
                "2020-01-02 00:00:00",
                101.0,
                100.5,
                101.5,
                "20200300",
                "20200300",
                "20200600",
            ),
            (
                "2020-01-03 00:00:00",
                float("nan"),
                101.5,
                102.5,
                "20200600",
                "20200600",
                "20200900",
            ),
        ]
    )
    candidate = _make_multiple_prices(
        [
            (
                "2020-01-01 00:00:00",
                100.0,
                99.5,
                100.5,
                "20200300",
                "20200300",
                "20200600",
            ),
            (
                "2020-01-03 00:00:00",
                102.0,
                101.5,
                102.5,
                "20200600",
                "20200600",
                "20200900",
            ),
        ]
    )

    merged, outcome = nd_import._merge_multiple_prices_non_destructive(
        existing, candidate
    )

    assert outcome.rows_added == 1
    assert outcome.rows_replaced == 1
    assert merged.loc[pd.Timestamp("2020-01-03 00:00:00"), "PRICE"] == 102.0
    assert merged.loc[pd.Timestamp("2020-01-01 00:00:00"), "PRICE"] == 100.0


def test_merge_adjusted_prices_replaces_nan_and_keeps_coherent_existing():
    existing = _make_adjusted_prices(
        [
            ("2020-01-01 00:00:00", 100.0),
            ("2020-01-02 00:00:00", float("nan")),
        ]
    )
    candidate = _make_adjusted_prices(
        [
            ("2020-01-02 00:00:00", 101.0),
            ("2020-01-03 00:00:00", 102.0),
        ]
    )

    merged, outcome = nd_import._merge_adjusted_prices_non_destructive(
        existing, candidate
    )

    assert outcome.rows_added == 1
    assert outcome.rows_replaced == 1
    assert merged.loc[pd.Timestamp("2020-01-01 00:00:00")] == 100.0
    assert merged.loc[pd.Timestamp("2020-01-02 00:00:00")] == 101.0


def test_mixed_prices_candidate_daily_only_returns_daily_from_db(monkeypatch):
    fake_db = _FakeDbPrices()
    monkeypatch.setattr(nd_import, "db_prices", fake_db)

    contract = futuresContract("TEST", "20200600")
    daily_prices = _make_prices(
        [
            ("2020-01-03 00:00:00", 103.0),
            ("2020-01-01 00:00:00", 101.0),
            ("2020-01-02 00:00:00", 102.0),
        ]
    )
    fake_db.set_prices(contract, DAILY_PRICE_FREQ, daily_prices)

    mixed_candidate = nd_import._mixed_prices_candidate_from_db(contract)

    assert len(mixed_candidate) == 3
    assert list(mixed_candidate.index) == sorted(mixed_candidate.index)
    assert mixed_candidate.loc[pd.Timestamp("2020-01-01 00:00:00"), "FINAL"] == 101.0


def test_load_split_freq_prices_with_fallback_uses_latest(monkeypatch):
    monkeypatch.setattr(
        nd_import, "csvFuturesContractPriceData", _FakeCsvPriceDataWithFinalFallback
    )
    starting_config = ConfigCsvFuturesPrices(
        input_date_index_name="Time",
        input_skiprows=0,
        input_skipfooter=0,
        input_date_format="ISO8601",
        input_column_mapping=dict(
            OPEN="Open", HIGH="High", LOW="Low", FINAL="Close", VOLUME="Volume"
        ),
    )

    (
        hourly_dict,
        daily_dict,
        hourly_final_columns,
        daily_final_columns,
    ) = nd_import._load_split_freq_prices_with_fallback(
        instrument_code="TEST",
        datapath="unused",
        csv_config=starting_config,
    )

    assert "20200300" in hourly_dict
    assert "20200300" in daily_dict
    assert hourly_final_columns == {"Latest"}
    assert daily_final_columns == {"Latest"}


def test_load_split_freq_prices_with_fallback_coalesces_final_row_wise(monkeypatch):
    monkeypatch.setattr(
        nd_import, "csvFuturesContractPriceData", _FakeCsvPriceDataWithRowLevelCoalesce
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
    ) = nd_import._load_split_freq_prices_with_fallback(
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


def test_build_roll_calendar_with_recent_to_past_fallback_uses_recent_path(monkeypatch):
    merged_prices = _FakeMergedPrices(
        {
            "20160100": _make_prices([("2016-01-04 00:00:00", 95.0)]),
            "20170100": _make_prices([("2026-02-07 00:00:00", 101.0)]),
            "20180100": _make_prices([("2026-02-08 00:00:00", 102.0)]),
            "20190100": _make_prices([("2026-02-09 00:00:00", 103.0)]),
        }
    )
    monkeypatch.setattr(
        nd_import, "db_prices", _FakeDbPricesForRollCalendar(merged_prices)
    )
    monkeypatch.setattr(nd_import, "csvRollParametersData", _FakeRollParametersData)

    stale_standard_calendar = _make_roll_calendar(
        [("2015-01-16 17:00:00", "20150300", "20150400", "20150200")]
    )
    monkeypatch.setattr(
        nd_import,
        "build_and_write_roll_calendar",
        lambda *args, **kwargs: stale_standard_calendar,
    )

    def _fake_create_calendar(prices_dict, roll_parameters):
        earliest_contract = min(prices_dict.keys())
        if earliest_contract < "20170100":
            return _make_roll_calendar(
                [("2015-01-16 17:00:00", "20150300", "20150400", "20150200")]
            )
        return _make_roll_calendar(
            [("2026-02-03 00:00:00", "20260400", "20260500", "20260300")]
        )

    monkeypatch.setattr(
        nd_import, "_build_roll_calendar_from_prices_dict", _fake_create_calendar
    )
    monkeypatch.setattr(nd_import, "_is_roll_calendar_valid", lambda *_: True)

    result = nd_import._build_roll_calendar_with_recent_to_past_fallback("BRENT-LAST")

    assert result.mode == "recent_to_past"
    assert result.cutoff_contract == "20170100"
    assert result.roll_calendar.index.max() == pd.Timestamp("2026-02-03 00:00:00")


def test_recent_to_past_fallback_stops_after_first_failing_older_cutoff(monkeypatch):
    merged_prices = _FakeMergedPrices(
        {
            "20160100": _make_prices([("2016-01-04 00:00:00", 95.0)]),
            "20170100": _make_prices([("2026-02-07 00:00:00", 101.0)]),
            "20180100": _make_prices([("2026-02-08 00:00:00", 102.0)]),
            "20190100": _make_prices([("2026-02-09 00:00:00", 103.0)]),
        }
    )
    monkeypatch.setattr(nd_import, "csvRollParametersData", _FakeRollParametersData)

    called_cutoffs: list[str] = []

    def _fake_create_calendar(prices_dict, roll_parameters):
        earliest_contract = min(prices_dict.keys())
        called_cutoffs.append(earliest_contract)
        if earliest_contract == "20160100":
            return _make_roll_calendar(
                [("2015-01-16 17:00:00", "20150300", "20150400", "20150200")]
            )
        return _make_roll_calendar(
            [("2026-02-03 00:00:00", "20260400", "20260500", "20260300")]
        )

    monkeypatch.setattr(
        nd_import, "_build_roll_calendar_from_prices_dict", _fake_create_calendar
    )
    monkeypatch.setattr(nd_import, "_is_roll_calendar_valid", lambda *_: True)

    latest_timestamp = nd_import._latest_available_timestamp_in_prices_dict(
        merged_prices
    )
    result = nd_import._build_recent_to_past_roll_calendar(
        instrument_code="BRENT-LAST",
        prices_dict=merged_prices,
        latest_available_timestamp=latest_timestamp,
    )

    assert result.cutoff_contract == "20170100"
    assert called_cutoffs == ["20170100", "20160100"]


def test_build_roll_calendar_with_recent_to_past_fallback_keeps_full_history(
    monkeypatch,
):
    merged_prices = _FakeMergedPrices(
        {
            "20170100": _make_prices([("2026-02-08 00:00:00", 101.0)]),
            "20180100": _make_prices([("2026-02-09 00:00:00", 102.0)]),
            "20190100": _make_prices([("2026-02-10 00:00:00", 103.0)]),
        }
    )
    monkeypatch.setattr(
        nd_import, "db_prices", _FakeDbPricesForRollCalendar(merged_prices)
    )

    full_history_calendar = _make_roll_calendar(
        [("2026-02-03 00:00:00", "20260400", "20260500", "20260300")]
    )
    monkeypatch.setattr(
        nd_import,
        "build_and_write_roll_calendar",
        lambda *args, **kwargs: full_history_calendar,
    )
    monkeypatch.setattr(
        nd_import,
        "_build_recent_to_past_roll_calendar",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Fallback should not be called")
        ),
    )

    result = nd_import._build_roll_calendar_with_recent_to_past_fallback("BRENT-LAST")

    assert result.mode == "full_history"
    assert result.cutoff_contract is None
    assert result.roll_calendar.index.max() == pd.Timestamp("2026-02-03 00:00:00")


def test_recent_to_past_fallback_prefers_january_cutoff(monkeypatch):
    merged_prices = _FakeMergedPrices(
        {
            "20160800": _make_prices([("2026-01-01 00:00:00", 101.0)]),
            "20160900": _make_prices([("2026-01-02 00:00:00", 102.0)]),
            "20161000": _make_prices([("2026-01-03 00:00:00", 103.0)]),
            "20161100": _make_prices([("2026-01-04 00:00:00", 104.0)]),
            "20161200": _make_prices([("2026-01-05 00:00:00", 105.0)]),
            "20170100": _make_prices([("2026-01-06 00:00:00", 106.0)]),
            "20170200": _make_prices([("2026-01-07 00:00:00", 107.0)]),
            "20170300": _make_prices([("2026-01-08 00:00:00", 108.0)]),
        }
    )
    monkeypatch.setattr(nd_import, "csvRollParametersData", _FakeRollParametersData)
    monkeypatch.setattr(nd_import, "FALLBACK_PREFER_JANUARY_CUTOFF", True)

    def _fake_create_calendar(prices_dict, roll_parameters):
        earliest_contract = min(prices_dict.keys())
        return _make_roll_calendar(
            [("2026-01-06 00:00:00", earliest_contract, "20990100", "20981200")]
        )

    monkeypatch.setattr(
        nd_import, "_build_roll_calendar_from_prices_dict", _fake_create_calendar
    )
    monkeypatch.setattr(nd_import, "_is_roll_calendar_valid", lambda *_: True)

    latest_timestamp = nd_import._latest_available_timestamp_in_prices_dict(
        merged_prices
    )
    result = nd_import._build_recent_to_past_roll_calendar(
        instrument_code="TEST",
        prices_dict=merged_prices,
        latest_available_timestamp=latest_timestamp,
    )

    assert result.cutoff_contract == "20170100"
