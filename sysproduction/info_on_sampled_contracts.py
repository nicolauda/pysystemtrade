import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from syscore.exceptions import missingData
from syscore.fileutils import (
    get_resolved_pathname,
    resolve_path_and_filename_for_package,
)
from sysdata.data_blob import dataBlob
from sysproduction.data.contracts import dataContracts
from sysproduction.data.prices import diagPrices

DEFAULT_DASHBOARD_FILENAME = "sampled_contracts_dashboard.pdf"

def _calculate_data_quality(adjusted_prices: pd.Series) -> Optional[dict]:
    """
    Return basic data quality stats for an adjusted price series.
    """
    if adjusted_prices is None or adjusted_prices.empty:
        return None

    series_without_na = adjusted_prices.dropna()
    if series_without_na.empty:
        return None

    index_as_dt = pd.DatetimeIndex(series_without_na.index)
    if index_as_dt.tz is not None:
        index_as_dt = index_as_dt.tz_convert(None)
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


def _resolve_dashboard_pdf_path(config, dashboard_pdf_arg: Optional[str]) -> Optional[Path]:
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
    fig.suptitle(f"{instrument_code} contracts", y=0.995, fontsize=12, fontweight="bold")
    pdf_writer.savefig(fig)
    plt.close(fig)


def _add_summary_pages(all_reports: list[list[str]], pdf_writer, lines_per_page: int = 45) -> None:
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


def _format_contract_report(
    instrument_code: str, sampled_contracts: list, data_quality: Optional[dict]
) -> list[str]:
    lines = [f"--- {instrument_code} ---"]
    if sampled_contracts:
        lines.extend([str(contract) for contract in sampled_contracts])
    else:
        lines.append("No sampled contracts found.")

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

    return lines


def list_sampled_contracts(
    plot: bool = False, plot_dir: Optional[str] = None, dashboard_pdf: Optional[str] = None
):
    """
    Lists all the sampled contracts for each instrument and reports basic data quality.
    """
    _silence_pil_debug_output()
    plot_output_dir = Path(plot_dir) if plot_dir else None
    pdf_writer = None
    dashboard_path = None

    with dataBlob(log_name="info_on_sampled_contracts") as data:
        diag_prices = diagPrices(data)
        data_contracts = dataContracts(data)
        dashboard_path = _resolve_dashboard_pdf_path(data.config, dashboard_pdf)
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

        instrument_list = diag_prices.get_list_of_instruments_in_multiple_prices()

        print("--- Sampled Contracts ---")
        collected_reports = []

        for instrument_code in instrument_list:
            try:
                sampled_contracts = data_contracts.get_all_sampled_contracts(
                    instrument_code
                )
                adjusted_prices = diag_prices.get_adjusted_prices(instrument_code)
                data_quality = _calculate_data_quality(adjusted_prices)
                report_lines = _format_contract_report(
                    instrument_code, sampled_contracts, data_quality
                )
                collected_reports.append(
                    {
                        "instrument_code": instrument_code,
                        "report_lines": report_lines,
                        "adjusted_prices": adjusted_prices,
                    }
                )
                print("\n" + "\n".join(report_lines))
            except Exception as e:
                print(f"\nCould not retrieve contracts for {instrument_code}: {e}")

        # Write summary first, then figures
        if pdf_writer is not None:
            _add_summary_pages([r["report_lines"] for r in collected_reports], pdf_writer)

        for record in collected_reports:
            if plot or plot_output_dir or pdf_writer is not None:
                _plot_adjusted_prices(
                    record["instrument_code"],
                    record["adjusted_prices"],
                    plot_output_dir,
                    show=plot,
                    pdf_writer=pdf_writer,
                    info_lines=record["report_lines"],
                )
    if pdf_writer is not None:
        pdf_writer.close()
        print("Dashboard PDF complete.")


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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    list_sampled_contracts(
        plot=args.plot or bool(args.plot_dir),
        plot_dir=args.plot_dir,
        dashboard_pdf=args.dashboard_pdf,
    )
