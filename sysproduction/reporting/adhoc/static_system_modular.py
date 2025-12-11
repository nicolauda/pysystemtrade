"""Static instrument selection report and CLI helpers.

Logic (matches the blog post and reference report):
- Pull a futures system (`futures_system`) built on your sampled-contracts DB, then drop any instrument that lacks price data (adds to `ignore_instruments` and removes its weight) to avoid missingData failures.
- Build the instrument correlation matrix once (capital-independent), then for each capital level:
  - Estimate instrument count (log-space interpolation of the reference ladder) unless you override it.
  - Compute a notional IDM = instruments**0.25 and max single-instrument weight = 1 / instruments.
  - Call `find_best_ordered_set_of_instruments` from `optimise_small_system` to greedily pick the next market that maximises portfolio SR given handcrafted risk weights from `handcraftPortfolio`, mean/stdev estimates (unit stdev, SR via net_SR), and the correlation subset.
  - Return both the ordered list and the alphabetically sorted list in the report.

Usage:
- Full report (default capital ladder): `python -m sysproduction.reporting.adhoc.static_system_modular --report`
- Report with custom capital levels: `python -m sysproduction.reporting.adhoc.static_system_modular --report --capital 100000 250000`
- Use capital from DB if present: `python -m sysproduction.reporting.adhoc.static_system_modular --use-db-capital --report`
- Get only the instrument list for a capital: `python -m sysproduction.reporting.adhoc.static_system_modular --capital 500000`
- Override estimated instrument counts (single or per capital): `python -m sysproduction.reporting.adhoc.static_system_modular --capital 500000 --estimated-count 45`
- Pick a config file: `python -m sysproduction.reporting.adhoc.static_system_modular --config systems.provided.rob_system.config.yaml`
- Force using all sampled instruments instead of production list: add `--use-all-sampled-instruments`

Notes:
- The script auto-prunes instruments that are missing from your sampled contracts DB before computing correlations.
- Defaults mirror https://github.com/robcarver17/reports/blob/master/Static_selection_of_instruments.
- Blog post https://qoppac.blogspot.com/2021/06/static-optimisation-of-best-set-of.html
"""

import argparse
import math
import os
from typing import Iterable, List, Sequence, Tuple

from systems.provided.rob_system.run_system import futures_system, System
from systems.provided.static_small_system_optimise.optimise_small_system import (
    find_best_ordered_set_of_instruments,
    get_correlation_matrix,
)

from sysquant.estimators.correlation_estimator import correlationEstimate
from sysproduction.reporting.reporting_functions import (
    parse_report_results,
    output_file_report,
    header,
    body_text,
)

from sysdata.data_blob import dataBlob

from sysproduction.data.capital import dataCapital
from sysproduction.reporting.report_configs import reportConfig

DEFAULT_CONFIG_FILENAME = "systems.provided.rob_system.config.yaml"


# Default pairs taken from the reference report:
# https://github.com/robcarver17/reports/blob/master/Static_selection_of_instruments
DEFAULT_CAPITAL_AND_INSTRUMENT_COUNT_PAIRS: List[Tuple[float, int]] = [
    (10_000, 7),
    (25_000, 13),
    (50_000, 22),
    (100_000, 25),
    (250_000, 39),
    (500_000, 45),
    (1_000_000, 54),
    (2_500_000, 65),
    (5_000_000, 72),
    (10_000_000, 59),
    (25_000_000, 59),
]


def static_system_adhoc_report(
    system_function=futures_system,
    list_of_capital_and_estimate_instrument_count_tuples: Sequence[
        Tuple[float, int]
    ] = DEFAULT_CAPITAL_AND_INSTRUMENT_COUNT_PAIRS,
    title_suffix: str = "",
):
    """Build the full static selection report."""
    data = dataBlob()
    base_title = "Static selection of instruments"
    title_with_suffix = (
        f"{base_title} {title_suffix}" if title_suffix else base_title
    )
    report_config = reportConfig(
        title=title_with_suffix, function="not_used", output="file"
    )

    report_results = build_static_selection_report(
        system_function=system_function,
        capital_and_estimate_pairs=list_of_capital_and_estimate_instrument_count_tuples,
    )

    parsed_report_results = parse_report_results(data, report_results=report_results)

    output_file_report(
        parsed_report=parsed_report_results, data=data, report_config=report_config
    )


def build_static_selection_report(
    system_function,
    capital_and_estimate_pairs: Sequence[Tuple[float, int]],
) -> list:
    """Return the report payload for the static selection run."""
    system = system_function()
    prune_system_instruments_without_data(system)
    corr_matrix = get_correlation_matrix(system)  # capital irrelevant for correlation

    all_results = []
    all_results.append(
        header(
            "Selected instruments using static selection for different levels of capital"
        )
    )
    for (
        capital,
        est_number_of_instruments,
    ) in capital_and_estimate_pairs:
        instrument_list = select_instruments_for_capital(
            capital=capital,
            estimated_instrument_count=est_number_of_instruments,
            system_function=system_function,
            corr_matrix=corr_matrix,
        )

        ordered_text = body_text(
            "For capital of %d, %d instruments, Selected order: %s"
            % (capital, len(instrument_list), str(instrument_list))
        )
        all_results.append(ordered_text)

        sorted_instruments = sorted(instrument_list)
        sorted_text = body_text("Sorted: %s \n" % (str(sorted_instruments)))
        all_results.append(sorted_text)

    return all_results


def select_instruments_for_capital(
    capital: float,
    estimated_instrument_count: int | None = None,
    system_function=futures_system,
    corr_matrix: correlationEstimate | None = None,
) -> list:
    """Return the ordered list of instruments for a given capital level."""
    if capital <= 0:
        raise ValueError("Capital must be positive.")

    system = system_function()
    prune_system_instruments_without_data(system)
    corr_matrix = corr_matrix or get_correlation_matrix(system)

    est_number_of_instruments = estimated_instrument_count or (
        estimate_instrument_count_from_capital(capital)
    )

    return static_system_results_for_capital(
        system=system,
        corr_matrix=corr_matrix,
        est_number_of_instruments=est_number_of_instruments,
        capital=capital,
    )


def static_system_results_for_capital(
    system: System,
    corr_matrix: correlationEstimate,
    est_number_of_instruments: int,
    capital: float,
):
    notional_starting_IDM = est_number_of_instruments**0.25
    max_instrument_weight = 1.0 / est_number_of_instruments

    return find_best_ordered_set_of_instruments(
        system=system,
        corr_matrix=corr_matrix,
        capital=capital,
        max_instrument_weight=max_instrument_weight,
        notional_starting_IDM=notional_starting_IDM,
    )


def estimate_instrument_count_from_capital(
    capital: float,
    capital_instrument_table: Sequence[Tuple[float, int]] = DEFAULT_CAPITAL_AND_INSTRUMENT_COUNT_PAIRS,
) -> int:
    """Estimate instrument count from capital using log-space interpolation."""
    if capital <= 0:
        raise ValueError("Capital must be positive.")

    sorted_table = sorted(capital_instrument_table, key=lambda pair: pair[0])
    capitals = [pair[0] for pair in sorted_table]
    counts = [pair[1] for pair in sorted_table]

    if len(capitals) == 1:
        return counts[0]

    log_capital = math.log(capital)
    log_capitals = [math.log(cap) for cap in capitals]

    if capital <= capitals[0]:
        return counts[0]
    if capital >= capitals[-1]:
        return counts[-1]

    for idx in range(1, len(capitals)):
        lower_capital = capitals[idx - 1]
        upper_capital = capitals[idx]
        if lower_capital <= capital <= upper_capital:
            lower_count = counts[idx - 1]
            upper_count = counts[idx]
            lower_log_capital = log_capitals[idx - 1]
            upper_log_capital = log_capitals[idx]
            span = upper_log_capital - lower_log_capital
            if span == 0:
                return lower_count
            interpolation_weight = (log_capital - lower_log_capital) / span
            estimated_count = lower_count + interpolation_weight * (
                upper_count - lower_count
            )
            return max(1, int(round(estimated_count)))

    return counts[-1]


def get_current_capital_from_db(data: dataBlob | None = None) -> float:
    """Fetch current total capital from the production DB."""
    data = data or dataBlob()
    return dataCapital(data).get_current_total_capital()


def build_system_function(
    config_filename: str = DEFAULT_CONFIG_FILENAME,
    use_all_sampled_instruments: bool = False,
):
    """
    Return a factory that builds a futures system using the chosen config.
    Optionally override the production instrument list by using every instrument
    found in the sampled-contracts DB.
    """

    def _system():
        system = futures_system(config_filename=config_filename)
        if use_all_sampled_instruments:
            try:
                available = (
                    system.data.db_futures_multiple_prices_data.get_list_of_instruments()
                )
                forecast_weights_cfg = getattr(system.config, "forecast_weights", {})

                # Only keep instruments that also have forecast weights defined; otherwise
                # the forecasting stage will try to use rule names equal to instrument codes.
                filtered_available = [
                    code for code in available if code in forecast_weights_cfg
                ]
                if not filtered_available:
                    filtered_available = list(available)

                system.config.instrument_weights = {
                    code: 1.0 for code in filtered_available
                }
            except Exception:
                # If anything goes wrong, fall back to config-defined instruments.
                pass
        return system

    return _system


def build_capital_and_estimate_pairs(
    capitals: Iterable[float] | None,
    estimated_counts: Iterable[int] | None,
    default_pairs: Sequence[Tuple[float, int]] = DEFAULT_CAPITAL_AND_INSTRUMENT_COUNT_PAIRS,
) -> List[Tuple[float, int]]:
    """Pair capitals with estimated instrument counts, applying validation and defaults."""
    if capitals is None:
        if estimated_counts:
            estimated_counts = list(estimated_counts)
            default_capitals = [capital for capital, _ in default_pairs]
            if len(estimated_counts) == 1:
                estimated_counts = estimated_counts * len(default_capitals)
            if len(estimated_counts) != len(default_capitals):
                raise ValueError(
                    "When overriding default capitals, provide one estimated count "
                    "or the same number as default capital levels."
                )
            return list(zip(default_capitals, estimated_counts))

        return list(default_pairs)

    capitals = list(capitals)
    if not capitals:
        return list(default_pairs)

    for capital in capitals:
        if capital <= 0:
            raise ValueError("Capital must be positive.")

    if estimated_counts:
        estimated_counts = list(estimated_counts)
        if len(estimated_counts) == 1 and len(capitals) > 1:
            estimated_counts = estimated_counts * len(capitals)
        if len(estimated_counts) != len(capitals):
            raise ValueError(
                "Provide a single estimated count or one per capital value."
            )
        counts = estimated_counts
    else:
        counts = [
            estimate_instrument_count_from_capital(capital) for capital in capitals
        ]

    return list(zip(capitals, counts))


def prune_system_instruments_without_data(system: System):
    """
    Ensure the system ignores instruments that are not present in the sampled contracts DB.
    We do this by expanding the ignore list in config before any heavy calculations start.
    """
    try:
        available_multiple_price_instruments = set(
            system.data.db_futures_multiple_prices_data.get_list_of_instruments()
        )
    except Exception:
        # If we cannot introspect the data source, fall back to existing behaviour.
        return

    config = system.config
    instrument_weights = getattr(config, "instrument_weights", {})
    missing_instruments = [
        instrument
        for instrument in instrument_weights.keys()
        if instrument not in available_multiple_price_instruments
    ]
    if not missing_instruments:
        return

    exclude_config = getattr(config, "exclude_instrument_lists", {}) or {}
    ignore_list = exclude_config.get("ignore_instruments", [])
    ignore_set = set(ignore_list) | set(missing_instruments)
    exclude_config["ignore_instruments"] = sorted(ignore_set)
    config.exclude_instrument_lists = exclude_config

    # Optional: drop weights for missing instruments so downstream code is lighter.
    config.instrument_weights = {
        k: v for k, v in instrument_weights.items() if k in available_multiple_price_instruments
    }


def parse_cli_args():
    parser = argparse.ArgumentParser(
        description="Generate the static instrument selection report or fetch a list for a given capital."
    )
    parser.add_argument(
        "--capital",
        type=float,
        nargs="+",
        help="Capital level(s) to evaluate. Omit to use the reference report levels.",
    )
    parser.add_argument(
        "--estimated-count",
        type=int,
        nargs="+",
        help="Optional estimated instrument count(s) to pair with the supplied capital levels.",
    )
    parser.add_argument(
        "--use-db-capital",
        action="store_true",
        help="Pull capital from the production database if no capital is supplied.",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Write the report file instead of only printing instrument lists.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_FILENAME,
        help="Config file to load for the system (default: systems.provided.rob_system.config.yaml).",
    )
    parser.add_argument(
        "--use-all-sampled-instruments",
        action="store_true",
        help="Ignore production instrument list and use every instrument found in the sampled-contracts DB.",
    )
    return parser.parse_args()


def main():
    args = parse_cli_args()

    system_function = build_system_function(
        config_filename=args.config,
        use_all_sampled_instruments=args.use_all_sampled_instruments,
    )

    suffix_tokens = []
    if args.use_db_capital:
        suffix_tokens.append("use_db_capital")
    if args.use_all_sampled_instruments:
        suffix_tokens.append("all_sampled_instruments")
    if args.config and args.config != DEFAULT_CONFIG_FILENAME:
        suffix_tokens.append(f"config_{os.path.basename(args.config)}")
    title_suffix = ""
    if suffix_tokens:
        title_suffix = "_".join(suffix_tokens) + "_report"

    capitals = args.capital
    if args.use_db_capital and not capitals:
        capitals = [get_current_capital_from_db()]

    capital_and_estimate_pairs = build_capital_and_estimate_pairs(
        capitals=capitals, estimated_counts=args.estimated_count
    )

    # Default to report mode if the user does not request a specific list
    report_mode = args.report or capitals is None
    if report_mode:
        static_system_adhoc_report(
            system_function=system_function,
            list_of_capital_and_estimate_instrument_count_tuples=capital_and_estimate_pairs,
            title_suffix=title_suffix,
        )
        return

    corr_matrix = get_correlation_matrix(system_function())
    for capital, est_count in capital_and_estimate_pairs:
        instruments = select_instruments_for_capital(
            capital=capital,
            estimated_instrument_count=est_count,
            system_function=system_function,
            corr_matrix=corr_matrix,
        )
        print(
            "Capital %.0f (estimated %d instruments) -> %s"
            % (capital, est_count, instruments)
        )


if __name__ == "__main__":
    main()
