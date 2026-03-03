import argparse
import logging
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from syscore.exceptions import missingData
from syscore.fileutils import (
    get_resolved_pathname,
    resolve_path_and_filename_for_package,
)
from sysdata.csv.csv_instrument_data import csvFuturesInstrumentData
from sysdata.data_blob import dataBlob
from sysproduction.data.contracts import dataContracts
from sysproduction.data.prices import diagPrices

DEFAULT_DASHBOARD_FILENAME = "sampled_contracts_quality_report.pdf"
OUTPUT_MODE_REPORT = "report"
OUTPUT_MODE_TERMINAL = "terminal"


def _normalise_datetime_index(index: pd.Index) -> pd.DatetimeIndex:
    """
    Return a timezone-naive DatetimeIndex.
    """
    index_as_dt = pd.DatetimeIndex(index)
    if index_as_dt.tz is not None:
        index_as_dt = index_as_dt.tz_convert(None)
    return index_as_dt


def _calculate_data_quality(adjusted_prices: pd.Series) -> Optional[dict]:
    """
    Return basic data quality stats for an adjusted price series.
    """
    if adjusted_prices is None or adjusted_prices.empty:
        return None

    series_without_na = adjusted_prices.dropna()
    if series_without_na.empty:
        return None

    index_as_dt = _normalise_datetime_index(series_without_na.index)
    unique_days = index_as_dt.normalize().unique()

    start_date = index_as_dt.min().date()
    end_date = index_as_dt.max().date()
    business_day_index = pd.bdate_range(start=start_date, end=end_date)
    business_days = len(business_day_index)
    available_points = series_without_na.shape[0]
    available_business_days = len(
        pd.DatetimeIndex(unique_days).intersection(business_day_index)
    )
    coverage_pct = (
        (available_business_days / business_days) * 100 if business_days > 0 else 0.0
    )

    return {
        "start": start_date,
        "end": end_date,
        "business_days": business_days,
        "available_points": available_points,
        "available_days": available_business_days,
        "coverage_pct": coverage_pct,
    }


def _plot_adjusted_prices(
    instrument_code: str,
    adjusted_prices: pd.Series,
    plot_dir: Optional[Path],
    show: bool = False,
    pdf_writer=None,
    info_lines: Optional[list[str]] = None,
):
    """
    Plot adjusted prices for an instrument, optionally saving or adding to a dashboard.
    """
    if adjusted_prices is None or adjusted_prices.empty:
        print(f"No adjusted prices available to plot for {instrument_code}.")
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "matplotlib is required to plot adjusted prices. "
            "Install it or provide --plot-dir to save charts where available."
        )
        return

    plt.figure(figsize=(10, 4))
    ax = plt.gca()
    adjusted_prices.dropna().plot(ax=ax, title=f"{instrument_code} adjusted prices")
    ax.set_xlabel("Date")
    ax.set_ylabel("Adjusted price")

    if info_lines:
        info_text = "\n".join(info_lines)
        ax.text(
            0.01,
            0.99,
            info_text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "grey"},
        )

    plt.tight_layout()

    if plot_dir:
        plot_dir.mkdir(parents=True, exist_ok=True)
        output_path = plot_dir / f"{instrument_code}_adjusted_prices.png"
        plt.savefig(output_path)
        print(f"Saved adjusted price plot to {output_path}")
    if pdf_writer is not None:
        pdf_writer.savefig()
    if show:
        plt.show()
    plt.close()


def _plot_all_instruments_scaled(
    adjusted_by_instrument: dict[str, pd.Series],
    output_path: Optional[Path],
    pdf_writer=None,
    show: bool = False,
):
    """
    Plot all instruments on a single chart after scaling each series by its own max
    to highlight coverage length rather than level.
    """
    if not adjusted_by_instrument:
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "matplotlib is required to plot the combined chart. "
            "Install it or provide --plot-dir to save charts where available."
        )
        return

    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    for inst, series in adjusted_by_instrument.items():
        if series is None:
            continue
        series = pd.Series(series).dropna()
        if series.empty:
            continue
        max_abs = series.abs().max()
        if max_abs == 0 or pd.isna(max_abs):
            continue
        scaled = series / max_abs
        scaled.sort_index().plot(ax=ax, label=inst)

    if not ax.lines:
        plt.close()
        return

    ax.set_title("Adjusted prices scaled by own max (coverage by instrument)")
    ax.set_ylabel("Scaled price (max = 1)")
    ax.set_xlabel("Date")
    ax.legend(ncol=3, fontsize=8)
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        print(f"Saved combined scaled plot to {output_path}")
    if pdf_writer is not None:
        pdf_writer.savefig()
    if show:
        plt.show()
    plt.close()


def _resolve_dashboard_pdf_path(
    config, dashboard_pdf_arg: Optional[str]
) -> Optional[Path]:
    """
    Prefer CLI argument, otherwise fall back to configuration.
    """
    if dashboard_pdf_arg:
        return _resolve_pdf_filename(dashboard_pdf_arg)

    try:
        configured_path = config.get_element("reporting_directory")
        configured_path = Path(configured_path) / DEFAULT_DASHBOARD_FILENAME
    except missingData:
        return None

    if not configured_path:
        return None

    return _resolve_pdf_filename(configured_path)


def _resolve_pdf_filename(path: str) -> Path:
    """
    Resolve a PDF path while preserving the filename rather than treating it as a folder.
    """
    path_as_str = str(path)
    if Path(path_as_str).suffix:
        resolved = resolve_path_and_filename_for_package(path_as_str)
    else:
        resolved = get_resolved_pathname(path_as_str)

    return Path(resolved)


def _silence_pil_debug_output():
    for logger_name in ("PIL", "PIL.PngImagePlugin"):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.ERROR)


def _can_display_figures_in_terminal() -> bool:
    """
    Return whether this session can show matplotlib figures interactively.
    """
    try:
        import matplotlib
    except ImportError:
        return False

    backend_name = matplotlib.get_backend().lower()
    non_interactive_tokens = ("agg", "pdf", "ps", "svg", "template", "cairo")
    if any(token in backend_name for token in non_interactive_tokens):
        return False

    if sys.platform.startswith("linux"):
        has_display = bool(
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        )
        if not has_display:
            return False

    return True


def _prompt_for_headless_plot_dir() -> Optional[Path]:
    """
    Ask user where to save figures when interactive display is unavailable.

    Returns:
        Selected directory path, or `None` to skip figure output.

    Raises:
        SystemExit: If `KeyboardInterrupt` is received.
    """
    prompt = (
        "No interactive display detected. Enter directory to save figures "
        "(ENTER to skip figures): "
    )
    try:
        response = input(prompt).strip()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received, exiting.")
        raise SystemExit(0)

    if not response:
        return None
    return Path(response)


def _add_text_page(instrument_code: str, info_lines: list[str], pdf_writer) -> None:
    """
    Add a text-only page to the dashboard PDF with contract info.
    """
    if pdf_writer is None:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(
        0.02,
        0.98,
        "\n".join(info_lines),
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
    )
    fig.suptitle(
        f"{instrument_code} contracts", y=0.995, fontsize=12, fontweight="bold"
    )
    pdf_writer.savefig(fig)
    plt.close(fig)


def _add_summary_pages(
    all_reports: list[list[str]], pdf_writer, lines_per_page: int = 45
) -> None:
    """
    Add a summary section with all instruments to the start of the PDF.
    """
    if pdf_writer is None or not all_reports:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # Flatten with blank lines between instruments
    combined_lines: list[str] = []
    for report in all_reports:
        if combined_lines:
            combined_lines.append("")  # blank separator
        combined_lines.extend(report)

    for start in range(0, len(combined_lines), lines_per_page):
        page_lines = combined_lines[start : start + lines_per_page]
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(
            0.02,
            0.98,
            "\n".join(page_lines),
            va="top",
            ha="left",
            fontsize=10,
            family="monospace",
        )
        title = "Sampled contracts summary"
        if len(combined_lines) > lines_per_page:
            page_num = (start // lines_per_page) + 1
            title = f"{title} (page {page_num})"
        fig.suptitle(title, y=0.995, fontsize=12, fontweight="bold")
        pdf_writer.savefig(fig)
        plt.close(fig)


def _add_sampled_index_pages(
    collected_reports: list[dict], pdf_writer, lines_per_page: int = 45
) -> None:
    """
    Add an index page listing instruments and their sampled contracts.
    """
    if pdf_writer is None or not collected_reports:
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    lines: list[str] = []
    for record in collected_reports:
        contracts = record.get("sampled_contracts") or []
        contract_str = ", ".join(str(c) for c in contracts) if contracts else "None"
        lines.append(f"{record['instrument_code']}: {contract_str}")

    for start in range(0, len(lines), lines_per_page):
        page_lines = lines[start : start + lines_per_page]
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(
            0.02,
            0.98,
            "\n".join(page_lines),
            va="top",
            ha="left",
            fontsize=10,
            family="monospace",
        )
        title = "Sampled contracts index"
        if len(lines) > lines_per_page:
            page_num = (start // lines_per_page) + 1
            title = f"{title} (page {page_num})"
        fig.suptitle(title, y=0.995, fontsize=12, fontweight="bold")
        pdf_writer.savefig(fig)
        plt.close(fig)


def _format_contract_report(
    instrument_code: str,
    sampled_contracts: list,
    data_quality: Optional[dict],
    selected_window_label: str,
    instrument_metadata: Optional[dict[str, str]] = None,
    metadata_source: Optional[str] = None,
) -> list[str]:
    lines = [f"--- {instrument_code} ---"]
    if sampled_contracts:
        lines.extend([str(contract) for contract in sampled_contracts])
    else:
        lines.append("No sampled contracts found.")

    lines.append(f"Selected data window: {selected_window_label}")

    if data_quality:
        lines.append(
            f"Data window: {data_quality['start']} -> {data_quality['end']} "
            f"({data_quality['business_days']} business days)"
        )
        lines.append(
            f"Available observations: {data_quality['available_points']} rows "
            f"across {data_quality['available_days']} days "
            f"({data_quality['coverage_pct']:.1f}% coverage of window)"
        )
    else:
        lines.append("No adjusted price data available to assess coverage.")

    if instrument_metadata:
        lines.append("Instrument config metadata:")
        for field_name, field_value in instrument_metadata.items():
            lines.append(f"{field_name}: {field_value}")
        if metadata_source:
            lines.append(f"Instrument config source: {metadata_source}")
    else:
        lines.append("Instrument config metadata not found.")

    return lines


def _available_instruments_text(available_instruments: list[str]) -> str:
    """
    Return a readable list of available instruments for console messages.
    """
    return ", ".join(available_instruments)


def _stringify_metadata_value(raw_value: object) -> str:
    """
    Convert one metadata field value to a readable string.
    """
    if pd.isna(raw_value):
        return "N/A"
    return str(raw_value)


def _extract_instrument_metadata_from_df(
    instrument_code: str, instrument_metadata_df: Optional[pd.DataFrame]
) -> Optional[dict[str, str]]:
    """
    Extract metadata for one instrument from instrument config dataframe.
    """
    if instrument_metadata_df is None or instrument_metadata_df.empty:
        return None
    if instrument_code not in instrument_metadata_df.index:
        return None

    metadata_row = instrument_metadata_df.loc[instrument_code]
    return {
        str(field_name): _stringify_metadata_value(metadata_row[field_name])
        for field_name in instrument_metadata_df.columns
    }


def _load_instrument_metadata_from_csv_config() -> (
    tuple[Optional[pd.DataFrame], Optional[str]]
):
    """
    Load instrument metadata from pysystemtrade CSV config.

    Returns:
        Tuple `(metadata_df, source_path)` or `(None, None)` on failure.
    """
    try:
        instrument_config = csvFuturesInstrumentData()
        instrument_metadata_df = instrument_config.get_all_instrument_data_as_df()
        return instrument_metadata_df, str(instrument_config.config_file)
    except Exception as exception:
        print(f"Could not load instrument config metadata: {exception}")
        return None, None


def _parse_optional_date(raw_value: str, field_name: str) -> Optional[date]:
    """
    Parse an optional date string in YYYY-MM-DD format.

    Args:
        raw_value: User-provided date string, possibly empty.
        field_name: Human-readable field label for errors.

    Returns:
        A `date` if provided, otherwise `None`.

    Raises:
        ValueError: If the value is not empty and cannot be parsed.
    """
    cleaned = raw_value.strip()
    if not cleaned:
        return None
    try:
        return datetime.strptime(cleaned, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(
            f"Invalid {field_name} '{raw_value}'. Use YYYY-MM-DD format."
        ) from exc


def _validate_date_window(start_date: Optional[date], end_date: Optional[date]) -> None:
    """
    Validate chronological consistency of a date window.

    Raises:
        ValueError: If both dates are provided and start_date is after end_date.
    """
    if start_date is not None and end_date is not None and start_date > end_date:
        raise ValueError(
            f"Invalid date window: start date {start_date} is after end date "
            f"{end_date}."
        )


def _format_selected_window_label(
    start_date: Optional[date], end_date: Optional[date]
) -> str:
    """
    Return a readable label describing the selected date window.
    """
    if start_date is None and end_date is None:
        return "all available data"
    start_label = start_date.isoformat() if start_date is not None else "earliest"
    end_label = end_date.isoformat() if end_date is not None else "latest"
    return f"{start_label} -> {end_label}"


def _filter_adjusted_prices_to_date_window(
    adjusted_prices: pd.Series,
    start_date: Optional[date],
    end_date: Optional[date],
) -> pd.Series:
    """
    Filter adjusted prices to the selected date window (inclusive bounds).
    """
    if adjusted_prices is None or adjusted_prices.empty:
        return adjusted_prices
    if start_date is None and end_date is None:
        return adjusted_prices

    series = pd.Series(adjusted_prices).sort_index()
    index_as_dt = _normalise_datetime_index(series.index).normalize()
    mask = pd.Series(True, index=series.index, dtype=bool)
    if start_date is not None:
        mask &= pd.Series(index_as_dt >= pd.Timestamp(start_date), index=series.index)
    if end_date is not None:
        mask &= pd.Series(index_as_dt <= pd.Timestamp(end_date), index=series.index)

    return series.loc[mask]


def _prompt_for_date_window() -> tuple[Optional[date], Optional[date]]:
    """
    Interactively request start and end dates for diagnostics filtering.

    Returns:
        Tuple `(start_date, end_date)`. Empty inputs map to `None`.

    Raises:
        SystemExit: If `KeyboardInterrupt` is received.
    """
    start_prompt = "Start date YYYY-MM-DD (press ENTER for earliest available): "
    end_prompt = "End date YYYY-MM-DD (press ENTER for latest available): "

    while True:
        try:
            start_raw = input(start_prompt)
            end_raw = input(end_prompt)
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received, exiting.")
            raise SystemExit(0)

        try:
            start_date = _parse_optional_date(start_raw, "start date")
            end_date = _parse_optional_date(end_raw, "end date")
            _validate_date_window(start_date, end_date)
            return start_date, end_date
        except ValueError as exc:
            print(exc)


def _select_date_window(
    start_date: Optional[str],
    end_date: Optional[str],
    prompt_for_date_window: bool,
) -> tuple[Optional[date], Optional[date]]:
    """
    Resolve date-window selection from args or interactive prompt.
    """
    if start_date is not None or end_date is not None:
        parsed_start = _parse_optional_date(start_date or "", "start date")
        parsed_end = _parse_optional_date(end_date or "", "end date")
        _validate_date_window(parsed_start, parsed_end)
        return parsed_start, parsed_end

    if prompt_for_date_window and sys.stdin.isatty():
        return _prompt_for_date_window()

    return None, None


def _normalise_output_mode(output_mode: str) -> Optional[str]:
    """
    Normalize output-mode aliases to canonical values.
    """
    cleaned = output_mode.strip().lower()
    if cleaned in {"", "r", OUTPUT_MODE_REPORT}:
        return OUTPUT_MODE_REPORT
    if cleaned in {"t", OUTPUT_MODE_TERMINAL}:
        return OUTPUT_MODE_TERMINAL
    return None


def _prompt_for_output_mode() -> str:
    """
    Interactively request output mode.

    Returns:
        `report` or `terminal`.

    Raises:
        SystemExit: If `KeyboardInterrupt` is received.
    """
    prompt = "Output mode ([R]eport PDF / [T]erminal with figures, ENTER=report): "
    while True:
        try:
            response = input(prompt)
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received, exiting.")
            raise SystemExit(0)

        mode = _normalise_output_mode(response)
        if mode is not None:
            return mode

        print("Invalid output mode. Valid options: report (or R), " "terminal (or T).")


def _select_output_mode(
    output_mode: Optional[str], prompt_for_output_mode: bool
) -> str:
    """
    Resolve output mode from args or interactive prompt.

    Raises:
        ValueError: If an explicit output mode is invalid.
    """
    if output_mode is not None:
        mode = _normalise_output_mode(output_mode)
        if mode is None:
            raise ValueError("Invalid output mode. Valid options: report, terminal.")
        return mode

    if prompt_for_output_mode and sys.stdin.isatty():
        return _prompt_for_output_mode()

    return OUTPUT_MODE_REPORT


def _resolve_requested_instrument(
    requested_instrument: str, available_instruments: list[str]
) -> Optional[str]:
    """
    Resolve a user-provided instrument code using case-insensitive matching.

    Args:
        requested_instrument: Instrument code entered by the user.
        available_instruments: Valid instrument codes.

    Returns:
        The canonical instrument code if a match is found; otherwise `None`.
    """
    cleaned = requested_instrument.strip()
    if not cleaned:
        return None

    lookup = {
        instrument_code.upper(): instrument_code
        for instrument_code in available_instruments
    }
    return lookup.get(cleaned.upper())


def _prompt_for_instrument_selection(available_instruments: list[str]) -> list[str]:
    """
    Interactively select one instrument, or all instruments on empty input.

    Args:
        available_instruments: Instrument codes available for selection.

    Returns:
        A list containing either one selected instrument or all instruments.

    Raises:
        SystemExit: If `KeyboardInterrupt` is received.
    """
    prompt = "Instrument code (press ENTER for all instruments, Ctrl+C to exit): "
    available_text = _available_instruments_text(available_instruments)

    while True:
        try:
            requested_instrument = input(prompt).strip()
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received, exiting.")
            raise SystemExit(0)

        if not requested_instrument:
            return available_instruments

        resolved = _resolve_requested_instrument(
            requested_instrument, available_instruments
        )
        if resolved is not None:
            return [resolved]

        print(
            f"Instrument '{requested_instrument}' not found. "
            f"Available instruments: {available_text}"
        )


def _select_instruments_for_report(
    available_instruments: list[str],
    requested_instrument: Optional[str],
    prompt_for_instrument: bool,
) -> list[str]:
    """
    Return the list of instruments to report on.

    Selection priority:
    1) explicit `requested_instrument`
    2) interactive prompt when `prompt_for_instrument` and stdin is a TTY
    3) all available instruments

    Args:
        available_instruments: All instruments in multiple prices.
        requested_instrument: Optional instrument provided programmatically/CLI.
        prompt_for_instrument: Whether to prompt the user in interactive sessions.

    Returns:
        Selected instrument list.

    Raises:
        ValueError: If `requested_instrument` is provided but not valid.
    """
    if requested_instrument is not None:
        requested_instrument = requested_instrument.strip()
        if not requested_instrument:
            return available_instruments
        resolved = _resolve_requested_instrument(
            requested_instrument, available_instruments
        )
        if resolved is None:
            available_text = _available_instruments_text(available_instruments)
            raise ValueError(
                f"Instrument '{requested_instrument}' not found. "
                f"Available instruments: {available_text}"
            )
        return [resolved]

    if prompt_for_instrument and sys.stdin.isatty():
        return _prompt_for_instrument_selection(available_instruments)

    return available_instruments


def list_sampled_contracts(
    plot: bool = False,
    plot_dir: str = None,
    dashboard_pdf: str = None,
    instrument: str = None,
    prompt_for_instrument: bool = True,
    start_date: str = None,
    end_date: str = None,
    output_mode: str = None,
    prompt_for_date_window: bool = True,
    prompt_for_output_mode: bool = True,
):
    """
    List sampled contracts and adjusted-price quality statistics.

    Args:
        plot: Whether to show charts interactively.
        plot_dir: Optional directory where PNG plots are written.
        dashboard_pdf: Optional dashboard PDF output path.
        instrument: Optional instrument code filter. If omitted, selection is
            interactive when running in a TTY.
        prompt_for_instrument: If `True`, ask for instrument selection in TTY
            sessions when `instrument` is not explicitly provided.
        start_date: Optional start date (YYYY-MM-DD) for diagnostics filtering.
        end_date: Optional end date (YYYY-MM-DD) for diagnostics filtering.
        output_mode: Optional output mode (`report` or `terminal`).
        prompt_for_date_window: If `True`, ask for date window in TTY sessions
            when dates are not explicitly provided.
        prompt_for_output_mode: If `True`, ask for output mode in TTY sessions
            when `output_mode` is not explicitly provided.
    """
    _silence_pil_debug_output()
    plot_output_dir = Path(plot_dir) if plot_dir else None
    pdf_writer = None
    dashboard_path = None

    with dataBlob(log_name="info_on_sampled_contracts") as data:
        diag_prices = diagPrices(data)
        data_contracts = dataContracts(data)
        available_instruments = diag_prices.get_list_of_instruments_in_multiple_prices()
        if not available_instruments:
            print("No instruments found in multiple prices.")
            return
        print(f"Instruments in DB (multiple prices): {len(available_instruments)}")
        (
            instrument_metadata_df,
            instrument_metadata_source,
        ) = _load_instrument_metadata_from_csv_config()
        try:
            instrument_list = _select_instruments_for_report(
                available_instruments=available_instruments,
                requested_instrument=instrument,
                prompt_for_instrument=prompt_for_instrument,
            )
            selected_start_date, selected_end_date = _select_date_window(
                start_date=start_date,
                end_date=end_date,
                prompt_for_date_window=prompt_for_date_window,
            )
            selected_output_mode = _select_output_mode(
                output_mode=output_mode,
                prompt_for_output_mode=prompt_for_output_mode,
            )
        except ValueError as exc:
            print(exc)
            return
        selected_window_label = _format_selected_window_label(
            selected_start_date, selected_end_date
        )

        should_build_report = selected_output_mode == OUTPUT_MODE_REPORT
        should_show_plots = plot or selected_output_mode == OUTPUT_MODE_TERMINAL

        if selected_output_mode == OUTPUT_MODE_TERMINAL and should_show_plots:
            if not _can_display_figures_in_terminal():
                should_show_plots = False
                if plot_output_dir is None and sys.stdin.isatty():
                    selected_dir = _prompt_for_headless_plot_dir()
                    if selected_dir is not None:
                        plot_output_dir = selected_dir
                        print(f"Saving figures to {plot_output_dir}")
                    else:
                        print(
                            "Skipping figures (no display and no directory selected)."
                        )
                elif plot_output_dir is not None:
                    print(
                        "Interactive figure display is unavailable in this terminal "
                        "session. Saving figures to the provided --plot-dir."
                    )
                else:
                    print(
                        "Interactive figure display is unavailable in this terminal "
                        "session. Provide --plot-dir to save figures."
                    )

        if should_build_report:
            dashboard_path = _resolve_dashboard_pdf_path(data.config, dashboard_pdf)
        else:
            if dashboard_pdf:
                print("Output mode 'terminal' selected: --dashboard-pdf ignored.")
            dashboard_path = None

        if dashboard_path:
            dashboard_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                from matplotlib.backends.backend_pdf import PdfPages

                pdf_writer = PdfPages(dashboard_path)
                print(f"Building dashboard PDF at {dashboard_path}")
            except ImportError:
                print(
                    "matplotlib is required to build the dashboard PDF. "
                    "Install it or omit --dashboard-pdf."
                )
                pdf_writer = None

        print("--- Sampled Contracts ---")
        collected_reports = []
        adjusted_by_instrument: dict[str, pd.Series] = {}

        for instrument_code in instrument_list:
            try:
                sampled_contracts = data_contracts.get_all_sampled_contracts(
                    instrument_code
                )
                adjusted_prices = diag_prices.get_adjusted_prices(instrument_code)
                adjusted_prices = _filter_adjusted_prices_to_date_window(
                    adjusted_prices,
                    selected_start_date,
                    selected_end_date,
                )
                adjusted_by_instrument[instrument_code] = adjusted_prices
                data_quality = _calculate_data_quality(adjusted_prices)
                instrument_metadata = _extract_instrument_metadata_from_df(
                    instrument_code, instrument_metadata_df
                )
                report_lines = _format_contract_report(
                    instrument_code,
                    sampled_contracts,
                    data_quality,
                    selected_window_label,
                    instrument_metadata=instrument_metadata,
                    metadata_source=instrument_metadata_source,
                )
                collected_reports.append(
                    {
                        "instrument_code": instrument_code,
                        "report_lines": report_lines,
                        "adjusted_prices": adjusted_prices,
                        "sampled_contracts": sampled_contracts,
                    }
                )
                print("\n" + "\n".join(report_lines))
            except Exception as e:
                print(f"\nCould not retrieve contracts for {instrument_code}: {e}")

        # Write summary first, then figures
        if pdf_writer is not None:
            _add_sampled_index_pages(collected_reports, pdf_writer)
            _add_summary_pages(
                [r["report_lines"] for r in collected_reports], pdf_writer
            )

        # Combined coverage plot across instruments, scaled by own max
        if pdf_writer is not None or plot_output_dir or should_show_plots:
            combined_plot_path = (
                plot_output_dir / "all_instruments_scaled.png"
                if plot_output_dir
                else None
            )
            _plot_all_instruments_scaled(
                adjusted_by_instrument,
                combined_plot_path,
                pdf_writer=pdf_writer,
                show=should_show_plots,
            )

        for record in collected_reports:
            if should_show_plots or plot_output_dir or pdf_writer is not None:
                _plot_adjusted_prices(
                    record["instrument_code"],
                    record["adjusted_prices"],
                    plot_output_dir,
                    show=should_show_plots,
                    pdf_writer=pdf_writer,
                    info_lines=record["report_lines"],
                )
    if pdf_writer is not None:
        pdf_writer.close()
        print("Dashboard PDF complete.")


def interactive_list_sampled_contracts():
    """
    Run sampled-contract diagnostics with guided interactive prompts.

    Prompt flow:
    1) instrument selection (`ENTER` for all instruments)
    2) date-window selection (`ENTER` for full available range)
    3) output mode (`report` PDF or terminal with figures)
    """
    print(
        "Interactive mode: press ENTER to keep defaults at each step, "
        "or Ctrl+C to exit."
    )
    list_sampled_contracts()


def parse_args():
    parser = argparse.ArgumentParser(
        description="List sampled contracts and data quality for each instrument."
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Display an adjusted price plot for each instrument.",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default=None,
        help="Directory where adjusted price plots should be saved. "
        "Enables plotting even if --plot is not set.",
    )
    parser.add_argument(
        "--dashboard-pdf",
        type=str,
        default=None,
        help="Path to a single PDF dashboard containing one page per instrument. "
        "If omitted, falls back to config key sampled_contracts_dashboard_pdf.",
    )
    parser.add_argument(
        "--instrument",
        type=str,
        default=None,
        help="Optional instrument code to inspect without interactive prompt. "
        "Case-insensitive.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Optional start date (YYYY-MM-DD) for adjusted-price diagnostics.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Optional end date (YYYY-MM-DD) for adjusted-price diagnostics.",
    )
    parser.add_argument(
        "--output-mode",
        type=str,
        default=None,
        help="Output mode: report or terminal. "
        "If omitted, interactive sessions prompt for a choice.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    list_sampled_contracts(
        plot=args.plot or bool(args.plot_dir),
        plot_dir=args.plot_dir,
        dashboard_pdf=args.dashboard_pdf,
        instrument=args.instrument,
        start_date=args.start_date,
        end_date=args.end_date,
        output_mode=args.output_mode,
    )
