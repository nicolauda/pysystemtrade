import pytest
from pandas import DataFrame, DatetimeIndex, Series

from sysproduction.info_on_sampled_contracts import (
    OUTPUT_MODE_REPORT,
    OUTPUT_MODE_TERMINAL,
    _extract_instrument_metadata_from_df,
    _filter_adjusted_prices_to_date_window,
    _format_contract_report,
    _prompt_for_headless_plot_dir,
    _prompt_for_instrument_selection,
    _prompt_for_output_mode,
    _select_date_window,
    _select_instruments_for_report,
    _select_output_mode,
)


def test_prompt_selection_returns_all_instruments_on_enter(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "")

    selected = _prompt_for_instrument_selection(["SOFR", "US10"])

    assert selected == ["SOFR", "US10"]


def test_prompt_selection_shows_available_instruments_on_invalid_input(
    monkeypatch, capsys
):
    answers = iter(["BADCODE", "us10"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    selected = _prompt_for_instrument_selection(["SOFR", "US10"])

    assert selected == ["US10"]
    captured = capsys.readouterr()
    assert "Available instruments: SOFR, US10" in captured.out


def test_prompt_selection_exits_cleanly_on_keyboard_interrupt(monkeypatch, capsys):
    def _raise_keyboard_interrupt(_):
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", _raise_keyboard_interrupt)

    with pytest.raises(SystemExit) as exc:
        _prompt_for_instrument_selection(["SOFR", "US10"])

    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "Keyboard interrupt received, exiting." in captured.out


def test_select_instruments_rejects_invalid_requested_instrument():
    with pytest.raises(ValueError, match="Available instruments: SOFR, US10"):
        _select_instruments_for_report(
            available_instruments=["SOFR", "US10"],
            requested_instrument="BADCODE",
            prompt_for_instrument=False,
        )


def test_select_date_window_accepts_enter_as_all_available():
    selected_start_date, selected_end_date = _select_date_window(
        start_date=None,
        end_date=None,
        prompt_for_date_window=False,
    )

    assert selected_start_date is None
    assert selected_end_date is None


def test_select_date_window_rejects_inverted_range():
    with pytest.raises(ValueError, match="start date"):
        _select_date_window(
            start_date="2026-02-10",
            end_date="2026-02-01",
            prompt_for_date_window=False,
        )


def test_filter_adjusted_prices_to_date_window_is_inclusive():
    series = Series(
        [1.0, 2.0, 3.0, 4.0],
        index=DatetimeIndex(["2026-01-01", "2026-01-10", "2026-01-20", "2026-02-01"]),
    )

    filtered = _filter_adjusted_prices_to_date_window(
        series,
        start_date=_select_date_window("2026-01-10", None, False)[0],
        end_date=_select_date_window(None, "2026-01-20", False)[1],
    )

    assert list(filtered.index.strftime("%Y-%m-%d")) == ["2026-01-10", "2026-01-20"]


def test_select_output_mode_rejects_invalid_value():
    with pytest.raises(ValueError, match="Invalid output mode"):
        _select_output_mode(output_mode="pdf_only", prompt_for_output_mode=False)


def test_prompt_for_output_mode_accepts_terminal_shortcut(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "t")

    selected_mode = _prompt_for_output_mode()

    assert selected_mode == OUTPUT_MODE_TERMINAL


def test_select_output_mode_defaults_to_report_without_prompt():
    selected_mode = _select_output_mode(output_mode=None, prompt_for_output_mode=False)

    assert selected_mode == OUTPUT_MODE_REPORT


def test_prompt_for_headless_plot_dir_returns_none_on_enter(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "")

    selected_dir = _prompt_for_headless_plot_dir()

    assert selected_dir is None


def test_prompt_for_headless_plot_dir_returns_user_path(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "/tmp/plots")

    selected_dir = _prompt_for_headless_plot_dir()

    assert str(selected_dir) == "/tmp/plots"


def test_extract_instrument_metadata_from_df_returns_all_columns_as_strings():
    instrument_metadata_df = DataFrame(
        {
            "Description": ["Brent"],
            "Pointsize": [1000],
            "Currency": ["USD"],
            "AssetClass": ["OilGas"],
        },
        index=["BRENT-LAST"],
    )

    metadata = _extract_instrument_metadata_from_df(
        "BRENT-LAST", instrument_metadata_df
    )

    assert metadata is not None
    assert metadata["Description"] == "Brent"
    assert metadata["Pointsize"] == "1000"
    assert metadata["Currency"] == "USD"
    assert metadata["AssetClass"] == "OilGas"


def test_extract_instrument_metadata_from_df_returns_none_for_missing_instrument():
    instrument_metadata_df = DataFrame(
        {"Description": ["Brent"]},
        index=["BRENT-LAST"],
    )

    metadata = _extract_instrument_metadata_from_df("BRE", instrument_metadata_df)

    assert metadata is None


def test_format_contract_report_does_not_include_total_instruments_in_db():
    report_lines = _format_contract_report(
        instrument_code="US10",
        sampled_contracts=[],
        data_quality=None,
        selected_window_label="all available data",
    )

    assert all(
        "Instruments in DB (multiple prices):" not in line for line in report_lines
    )
