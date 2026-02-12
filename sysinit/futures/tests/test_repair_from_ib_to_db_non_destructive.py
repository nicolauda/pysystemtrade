import pandas as pd

from syscore.dateutils import DAILY_PRICE_FREQ, HOURLY_FREQ
from sysinit.futures.adhoc import repair_from_ib_to_db_non_destructive as ib_repair
from sysobjects.contracts import futuresContract
from sysobjects.futures_per_contract_prices import futuresContractPrices
from sysobjects.roll_calendars import rollCalendar


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


class _FakeDiagPrices:
    def __init__(self):
        self._data = {}

    def set_prices(
        self,
        contract: futuresContract,
        frequency,
        prices: futuresContractPrices,
    ):
        self._data[(contract.key, frequency)] = prices

    def get_prices_at_frequency_for_contract_object(
        self, contract: futuresContract, frequency
    ) -> futuresContractPrices:
        return self._data.get(
            (contract.key, frequency), futuresContractPrices.create_empty()
        )


def test_merge_prices_non_destructive_keeps_existing_values_on_overlap():
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

    merged, rows_added = ib_repair._merge_prices_non_destructive(existing, incoming)

    assert rows_added == 1
    assert len(merged) == 3
    assert merged.loc[pd.Timestamp("2020-01-02 00:00:00"), "FINAL"] == 101.0
    assert merged.loc[pd.Timestamp("2020-01-03 00:00:00"), "FINAL"] == 102.0


def test_merge_prices_non_destructive_does_not_fill_existing_nan_on_overlap():
    existing = _make_prices(
        [
            ("2020-01-02 00:00:00", 100.0),
        ]
    )
    existing_as_dataframe = pd.DataFrame(existing)
    existing_as_dataframe.loc[pd.Timestamp("2020-01-02 00:00:00"), "HIGH"] = float(
        "nan"
    )
    existing_with_nan = futuresContractPrices(existing_as_dataframe)

    incoming = _make_prices(
        [
            ("2020-01-02 00:00:00", 999.0),
            ("2020-01-03 00:00:00", 101.0),
        ]
    )

    merged, rows_added = ib_repair._merge_prices_non_destructive(
        existing_with_nan, incoming
    )

    assert rows_added == 1
    assert len(merged) == 2
    assert pd.isna(merged.loc[pd.Timestamp("2020-01-02 00:00:00"), "HIGH"])
    assert merged.loc[pd.Timestamp("2020-01-02 00:00:00"), "FINAL"] == 100.0
    assert merged.loc[pd.Timestamp("2020-01-03 00:00:00"), "FINAL"] == 101.0


def test_mixed_prices_candidate_uses_intraday_and_daily_data():
    contract = futuresContract("TEST", "20200300")
    fake_diag_prices = _FakeDiagPrices()
    fake_diag_prices.set_prices(
        contract,
        HOURLY_FREQ,
        _make_prices(
            [
                ("2020-01-02 10:00:00", 101.5),
                ("2020-01-02 11:00:00", 101.7),
            ]
        ),
    )
    fake_diag_prices.set_prices(
        contract,
        DAILY_PRICE_FREQ,
        _make_prices(
            [
                ("2020-01-02 00:00:00", 101.0),
                ("2020-01-03 00:00:00", 102.0),
            ]
        ),
    )

    mixed_candidate = ib_repair._mixed_prices_candidate_from_db(
        contract=contract,
        intraday_frequency=HOURLY_FREQ,
        diag_prices=fake_diag_prices,
    )

    assert len(mixed_candidate) >= 3
    assert mixed_candidate.index.is_monotonic_increasing
    assert mixed_candidate.loc[pd.Timestamp("2020-01-03 00:00:00"), "FINAL"] == 102.0


def _make_roll_calendar(
    rows: list[tuple[str, str, str, str]],
) -> rollCalendar:
    dataframe = pd.DataFrame(
        [
            {
                "current_contract": current_contract,
                "next_contract": next_contract,
                "carry_contract": carry_contract,
            }
            for _, current_contract, next_contract, carry_contract in rows
        ],
        index=pd.to_datetime([roll_date for roll_date, _, _, _ in rows]),
    )
    return rollCalendar(dataframe)


def _empty_roll_calendar() -> rollCalendar:
    return rollCalendar(
        pd.DataFrame(columns=["current_contract", "next_contract", "carry_contract"])
    )


def test_merge_roll_calendars_conservative_updates_available_dates_only():
    existing = _make_roll_calendar(
        [
            ("2020-01-10 00:00:00", "20200300", "20200600", "20200300"),
            ("2020-03-10 00:00:00", "20200600", "20200900", "20200600"),
            ("2020-06-10 00:00:00", "20200900", "20201200", "20200900"),
        ]
    )
    candidate = _make_roll_calendar(
        [
            ("2020-03-10 00:00:00", "20200600", "20201000", "20200600"),
            ("2020-09-10 00:00:00", "20201000", "20210100", "20201000"),
        ]
    )

    merged_calendar, outcome = ib_repair._merge_roll_calendars_conservative(
        existing, candidate
    )

    assert outcome.existing_rows == 3
    assert outcome.candidate_rows == 2
    assert outcome.rows_added == 1
    assert outcome.rows_replaced == 1
    assert outcome.merged_rows == 4

    # Existing row on a date not present in candidate stays unchanged
    assert (
        str(merged_calendar.loc[pd.Timestamp("2020-01-10 00:00:00"), "next_contract"])
        == "20200600"
    )
    # Existing row on an available date gets refreshed from candidate
    assert (
        str(merged_calendar.loc[pd.Timestamp("2020-03-10 00:00:00"), "next_contract"])
        == "20201000"
    )
    # New available date is added
    assert (
        str(merged_calendar.loc[pd.Timestamp("2020-09-10 00:00:00"), "next_contract"])
        == "20210100"
    )


def test_merge_roll_calendars_conservative_drops_duplicate_transitions():
    existing = _make_roll_calendar(
        [
            ("2020-01-10 00:00:00", "20200300", "20200600", "20200300"),
            ("2020-01-11 00:00:00", "20200300", "20200600", "20200300"),
            ("2020-03-10 00:00:00", "20200600", "20200900", "20200600"),
        ]
    )
    candidate = _empty_roll_calendar()

    merged_calendar, outcome = ib_repair._merge_roll_calendars_conservative(
        existing, candidate
    )

    assert outcome.existing_rows == 2
    assert outcome.candidate_rows == 0
    assert outcome.rows_added == 0
    assert outcome.rows_replaced == 0
    assert outcome.merged_rows == 2

    assert list(merged_calendar.index) == [
        pd.Timestamp("2020-01-10 00:00:00"),
        pd.Timestamp("2020-03-10 00:00:00"),
    ]


def test_build_roll_calendar_writes_when_only_duplicate_transitions_are_removed(
    monkeypatch,
):
    class _FakeCsvRollCalendars:
        def __init__(self):
            self.written_calendar = None
            self._existing = _make_roll_calendar(
                [
                    ("2020-01-10 00:00:00", "20200300", "20200600", "20200300"),
                    ("2020-01-11 00:00:00", "20200300", "20200600", "20200300"),
                    ("2020-03-10 00:00:00", "20200600", "20200900", "20200600"),
                ]
            )

        def is_code_in_data(self, instrument_code: str) -> bool:
            return True

        def get_roll_calendar(self, instrument_code: str) -> rollCalendar:
            return self._existing

        def add_roll_calendar(
            self,
            instrument_code: str,
            roll_calendar: rollCalendar,
            ignore_duplication: bool = True,
        ):
            self.written_calendar = roll_calendar

    fake_csv = _FakeCsvRollCalendars()

    monkeypatch.setattr(ib_repair, "csvRollCalendarData", lambda *_: fake_csv)
    monkeypatch.setattr(
        ib_repair,
        "build_and_write_roll_calendar",
        lambda **_: _empty_roll_calendar(),
    )

    outcome = ib_repair._build_and_write_roll_calendar_conservative("KOSDAQ")

    assert outcome.existing_rows == 2
    assert outcome.candidate_rows == 0
    assert outcome.rows_added == 0
    assert outcome.rows_replaced == 0
    assert outcome.merged_rows == 2

    assert fake_csv.written_calendar is not None
    written_calendar = pd.DataFrame(fake_csv.written_calendar)
    assert list(written_calendar.index) == [
        pd.Timestamp("2020-01-10 00:00:00"),
        pd.Timestamp("2020-03-10 00:00:00"),
    ]


def test_rebuild_derived_data_uses_existing_calendar_dates(monkeypatch):
    captured_call = {}

    def _fake_build_and_write_roll_calendar_conservative(instrument_code: str):
        return ib_repair.RollCalendarMergeOutcome(
            existing_rows=1,
            candidate_rows=1,
            merged_rows=1,
            rows_added=0,
            rows_replaced=0,
        )

    def _fake_process_multiple_prices_single_instrument(
        instrument_code: str,
        adjust_calendar_to_prices: bool = True,
        **_,
    ):
        captured_call["instrument_code"] = instrument_code
        captured_call["adjust_calendar_to_prices"] = adjust_calendar_to_prices

    monkeypatch.setattr(
        ib_repair,
        "_build_and_write_roll_calendar_conservative",
        _fake_build_and_write_roll_calendar_conservative,
    )
    monkeypatch.setattr(
        ib_repair,
        "process_multiple_prices_single_instrument",
        _fake_process_multiple_prices_single_instrument,
    )
    monkeypatch.setattr(
        ib_repair,
        "process_adjusted_prices_single_instrument",
        lambda instrument_code: None,
    )

    ib_repair._rebuild_derived_data_for_instrument("KOSDAQ")

    assert captured_call == {
        "instrument_code": "KOSDAQ",
        "adjust_calendar_to_prices": False,
    }
