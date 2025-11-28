import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from syscore.exceptions import missingData
from syscore.fileutils import get_resolved_pathname
from sysdata.data_blob import dataBlob
from sysproduction.data.contracts import dataContracts
from sysproduction.data.prices import diagPrices


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
    adjusted_prices.dropna().plot(title=f"{instrument_code} adjusted prices")
    plt.xlabel("Date")
    plt.ylabel("Adjusted price")
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
        return Path(get_resolved_pathname(dashboard_pdf_arg))

    try:
        configured_path = config.get_element("sampled_contracts_dashboard_pdf")
    except missingData:
        return None

    if not configured_path:
        return None

    return Path(get_resolved_pathname(configured_path))


def list_sampled_contracts(
    plot: bool = False, plot_dir: Optional[str] = None, dashboard_pdf: Optional[str] = None
):
    """
    Lists all the sampled contracts for each instrument and reports basic data quality.
    """
    plot_output_dir = Path(plot_dir) if plot_dir else None
    pdf_writer = None
    dashboard_path = None

    with dataBlob(log_name="list_sampled_contracts") as data:
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

        for instrument_code in instrument_list:
            try:
                sampled_contracts = data_contracts.get_all_sampled_contracts(
                    instrument_code
                )
                adjusted_prices = diag_prices.get_adjusted_prices(instrument_code)
                data_quality = _calculate_data_quality(adjusted_prices)

                if sampled_contracts:
                    print(f"\n--- {instrument_code} ---")
                    for contract in sampled_contracts:
                        print(contract)
                else:
                    print(f"\n--- {instrument_code} ---")
                    print("No sampled contracts found.")

                if data_quality:
                    print(
                        f"Data window: {data_quality['start']} -> {data_quality['end']} "
                        f"({data_quality['business_days']} business days)"
                    )
                    print(
                        f"Available observations: {data_quality['available_points']} rows "
                        f"across {data_quality['available_days']} days "
                        f"({data_quality['coverage_pct']:.1f}% coverage of window)"
                    )
                else:
                    print("No adjusted price data available to assess coverage.")

                if plot or plot_output_dir or pdf_writer is not None:
                    _plot_adjusted_prices(
                        instrument_code,
                        adjusted_prices,
                        plot_output_dir,
                        show=plot,
                        pdf_writer=pdf_writer,
                    )
            except Exception as e:
                print(f"\nCould not retrieve contracts for {instrument_code}: {e}")
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
