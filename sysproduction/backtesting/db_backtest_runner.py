"""
Core routines for running a futures backtest using DB data (Mongo + Parquet)
via `dbFuturesSimData`, so you match what production will read.

Naming convention for zero-config runs:
- If your entrypoint is named <name>_backtest.py in some folder, and you place
  a <name>_config.yaml in the same folder, the runner will auto-discover it.
- Otherwise, you must pass --config (or set BacktestConfig.config_path).

Prerequisites:
- A private config (`private/private_config.yaml` or `PYSYS_PRIVATE_CONFIG_DIR`) with
  `parquet_store` and optional Mongo credentials configured.
- Data already seeded into Mongo/Parquet and enviroment variables already set 
  (see docs/production.md for data loading).
"""

from pathlib import Path
from datetime import datetime
import argparse
from dataclasses import dataclass
import logging
import warnings
import sys
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from sysdata.config.configdata import Config
from sysdata.sim.db_futures_sim_data import dbFuturesSimData
from systems.provided.futures_chapter15.basesystem import futures_system

DEFAULT_FALLBACK_SPREAD = 1.0  # price units if spread missing


def _infer_default_config_path() -> Optional[Path]:
    """
    If called from a script named <name>_backtest.py, try to use
    <name>_config.yaml in the same folder. Otherwise return None.
    """
    try:
        main_path = Path(sys.argv[0]).resolve()
    except Exception:
        return None

    if not main_path.name.endswith("_backtest.py"):
        return None

    base = main_path.stem[: -len("_backtest")]
    candidate = main_path.with_name(f"{base}_config.yaml")
    if candidate.exists():
        return candidate
    return None


@dataclass
class BacktestConfig:
    """
    Configuration for reusing the backtest without rewriting the script when
    inputs/outputs change.
    """

    config_path: Optional[Path] = None
    results_dir: Optional[Path] = None
    timestamp: Optional[str] = None
    fallback_spread: float = DEFAULT_FALLBACK_SPREAD
    instrument_filter: Optional[Sequence[str]] = None
    include_plots: bool = True
    include_quantstats: bool = True
    include_pdf: bool = True
    include_debug_txt: bool = True
    keep_intermediate_figs: bool = False

    def with_defaults(self) -> "BacktestConfig":
        inferred = _infer_default_config_path()
        if self.config_path:
            resolved_config = Path(self.config_path).resolve()
        elif inferred:
            resolved_config = inferred
        else:
            raise ValueError(
                "config_path is required (no <name>_config.yaml inferred). "
                "Pass BacktestConfig.config_path or --config."
            )
        resolved_results = (
            Path(self.results_dir)
            if self.results_dir is not None
            else resolved_config.parent / "backtest_results"
        )
        resolved_timestamp = self.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        resolved_instruments = (
            tuple(self.instrument_filter) if self.instrument_filter else None
        )
        return BacktestConfig(
            config_path=resolved_config,
            results_dir=resolved_results,
            timestamp=resolved_timestamp,
            fallback_spread=float(self.fallback_spread),
            instrument_filter=resolved_instruments,
            include_plots=self.include_plots,
            include_quantstats=self.include_quantstats,
            include_pdf=self.include_pdf,
            include_debug_txt=self.include_debug_txt,
            keep_intermediate_figs=self.keep_intermediate_figs,
        )


@dataclass
class BacktestOutputs:
    results_dir: Path
    figures: dict
    quantstats_report: Optional[Path]
    pdf_report: Optional[Path]
    debug_report: Optional[Path]


@dataclass
class BacktestResult:
    system: Any
    summary_rows: list
    per_inst_rows: list
    per_rule_rows: list
    cost_rows: list
    notional_rows: list
    trades_rows: list
    outputs: BacktestOutputs


def _set_matplotlib_font_defaults():
    """
    Set a safe default font to avoid 'Arial not found' warnings.
    """
    try:
        import matplotlib

        matplotlib.rcParams["font.family"] = "DejaVu Sans"
        matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "sans-serif"]
        matplotlib.rcParams["font.serif"] = ["DejaVu Serif", "serif"]

        logging.getLogger("matplotlib").setLevel(logging.ERROR)
        logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
        warnings.filterwarnings(
            "ignore",
            message=".*[Ff]ont family 'Arial' not found.*",
            category=UserWarning,
        )
    except Exception:
        pass


def _clean_series(series: Any) -> pd.Series:
    """
    Drop NA and remove leading zeros (keep data from first non-zero onward).
    """
    if series is None:
        return pd.Series(dtype=float)

    if isinstance(series, pd.DataFrame):
        if series.shape[1] == 0:
            return pd.Series(dtype=float)
        s = series.iloc[:, 0]
    else:
        try:
            s = pd.Series(series)
        except Exception:
            return pd.Series(dtype=float)

    s = s.dropna()
    if s.empty:
        return s

    mask = s.ne(0).cummax().astype(bool)
    if mask.any():
        return s.loc[mask]
    # all zeros: keep the original (no trimming) to avoid empty plots
    return s


def _filter_instruments(
    instruments: Iterable[str], desired: Optional[Sequence[str]]
) -> list:
    """
    Limit the instrument list to the requested subset; if the filter is empty
    return the full list.
    """
    full_list = list(instruments or [])
    if not desired:
        return full_list

    desired_set = set(desired)
    filtered = [inst for inst in full_list if inst in desired_set]
    missing = sorted(desired_set - set(full_list))
    if missing:
        print(f"Instruments not found and ignored: {', '.join(missing)}")
    return filtered


def run_backtest(
    backtest_config: BacktestConfig,
    data_factory=dbFuturesSimData,
    system_factory=futures_system,
) -> BacktestResult:
    """
    Build the system on DB data and print basic metrics.
    Pass BacktestConfig to reuse the flow with other settings without rewriting
    the script.
    """
    cfg = backtest_config.with_defaults()
    _set_matplotlib_font_defaults()
    data = data_factory()

    config_obj = Config(str(cfg.config_path))
    config_instruments = list(getattr(config_obj, "instruments", []) or [])
    weights_dict = getattr(config_obj, "instrument_weights", {}) or {}
    weights_instruments = list(weights_dict.keys())

    all_instruments = list(data.get_instrument_list())
    # Universe precedence: explicit config instruments > weight keys > all from DB
    base_universe = (
        config_instruments
        if config_instruments
        else weights_instruments or all_instruments
    )

    instrument_list = _filter_instruments(base_universe, cfg.instrument_filter)
    if cfg.instrument_filter and not instrument_list:
        print("Instrument filter empty: using base instrument universe.")
        instrument_list = base_universe

    spread_costs_used, spread_costs_missing = _ensure_spread_costs(
        data, instrument_list, fallback_default=cfg.fallback_spread
    )
    system = system_factory(data=data, config=config_obj)

    results_dir = cfg.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = cfg.timestamp

    portfolio = system.accounts.portfolio()
    instruments_for_output = instrument_list or system.get_instrument_list()

    print(f"Instruments: {', '.join(instruments_for_output)}")
    print("\nLatest portfolio stats:")
    stats = portfolio.stats()
    _print_stats(stats)

    print(f"\nSharpe: {portfolio.sharpe():.2f}")

    print("\nEquity curve (last 5):")
    curve = portfolio.curve()
    _safe_tail_print(curve, 5)

    print("\nDrawdown (last 5):")
    drawdown = portfolio.drawdown()
    _safe_tail_print(drawdown, 5)

    print("\nRolling annualised std (last 5):")
    rolling_std = portfolio.rolling_ann_std()
    _safe_tail_print(rolling_std, 5)

    print("\nPerformance per instrument (net % returns):")
    per_inst_rows = _print_per_instrument_perf(
        portfolio,
        instrument_filter=instrument_list,
    )

    print(
        "\nPerformance per strategy/rule (net % returns, scaled to portfolio capital):"
    )
    per_rule_rows = _print_per_strategy_perf(system, portfolio)

    print("\nRecent notional positions per instrument (last 3):")
    for inst in instruments_for_output:
        print(f"- {inst}:")
        _safe_tail_print(system.portfolio.get_notional_position(inst), 3, indent="  ")

    figures = {}
    if cfg.include_plots:
        figures["equity"] = results_dir / f"equity_curve_{timestamp}.png"
        figures["drawdown"] = results_dir / f"drawdown_{timestamp}.png"
        figures["rolling_std"] = results_dir / f"rolling_ann_std_{timestamp}.png"
        figures["notional_uncapped"] = (
            results_dir / f"notional_positions_uncapped_{timestamp}.png"
        )
        figures["notional_capped"] = (
            results_dir / f"notional_positions_capped_{timestamp}.png"
        )
        figures["buffered_positions"] = (
            results_dir / f"buffered_positions_{timestamp}.png"
        )

        _plot_series(curve, figures["equity"], title="Equity curve")
        _plot_series(drawdown, figures["drawdown"], title="Drawdown")
        _plot_series(
            rolling_std, figures["rolling_std"], title="Rolling annualised std"
        )
        _plot_notional_positions(
            system,
            figures["notional_uncapped"],
            instruments_for_output,
            clip_bounds=None,
        )
        _plot_notional_positions(
            system,
            figures["notional_capped"],
            instruments_for_output,
            clip_bounds=(-20, 20),
        )
        _plot_buffered_positions(
            system,
            figures["buffered_positions"],
            instruments_for_output,
            clip_bounds=None,
        )

    qs_html = (
        results_dir / f"backtest_report_qs_{timestamp}.html"
        if cfg.include_quantstats
        else None
    )
    if cfg.include_quantstats:
        _quantstats_report(portfolio.percent, qs_html)

    per_inst_rows = sorted(per_inst_rows, key=lambda r: r[0]) if per_inst_rows else []
    per_rule_rows = sorted(per_rule_rows, key=lambda r: r[0]) if per_rule_rows else []
    cost_rows = _build_spread_cost_rows(spread_costs_used, spread_costs_missing)

    notional_rows = _collect_notional_positions_by_year(system, instruments_for_output)
    trades_rows = _collect_trades(
        system,
        instrument_filter=instruments_for_output,
        max_rows=200,
        base_currency=getattr(config_obj, "base_currency", ""),
    )

    summary_rows = _build_summary_rows(
        stats,
        portfolio,
        base_currency=getattr(config_obj, "base_currency", ""),
    )

    pdf_path = (
        results_dir / f"backtest_report_{timestamp}.pdf" if cfg.include_pdf else None
    )
    debug_path = (
        results_dir / f"backtest_debug_{timestamp}.txt"
        if cfg.include_debug_txt
        else None
    )

    if cfg.include_pdf:
        _build_unified_pdf(
            output_path=pdf_path,
            summary_rows=summary_rows,
            per_inst_rows=per_inst_rows,
            per_rule_rows=per_rule_rows,
            cost_rows=cost_rows,
            notional_rows=notional_rows,
            trades_rows=trades_rows,
            figure_paths=list(figures.values()),
        )

    if cfg.include_debug_txt:
        _write_debug_txt(
            output_path=debug_path,
            summary_rows=summary_rows,
            per_inst_rows=per_inst_rows,
            per_rule_rows=per_rule_rows,
            cost_rows=cost_rows,
            notional_rows=notional_rows,
            trades_rows=trades_rows,
        )

    if cfg.include_plots and not cfg.keep_intermediate_figs:
        for tmp_fig in figures.values():
            try:
                tmp_fig.unlink(missing_ok=True)
            except Exception:
                pass

    figures_for_output = (
        {name: path for name, path in figures.items() if path.exists()}
        if cfg.include_plots
        else {}
    )

    outputs = BacktestOutputs(
        results_dir=results_dir,
        figures=figures_for_output,
        quantstats_report=qs_html,
        pdf_report=pdf_path,
        debug_report=debug_path,
    )

    return BacktestResult(
        system=system,
        summary_rows=summary_rows,
        per_inst_rows=per_inst_rows,
        per_rule_rows=per_rule_rows,
        cost_rows=cost_rows,
        notional_rows=notional_rows,
        trades_rows=trades_rows,
        outputs=outputs,
    )


def _print_stats(stats: Any, indent: str = ""):
    """
    Print stats in a readable way whether they are a pandas object or a list of tuples.
    """
    if hasattr(stats, "to_dict"):
        stats_dict = stats.to_dict()
    elif isinstance(stats, dict):
        stats_dict = stats
    elif isinstance(stats, Iterable):
        try:
            # Handle list of tuples like [(key, val), ...]
            stats_dict = dict(stats)
        except Exception:
            stats_dict = None
    else:
        stats_dict = None

    if stats_dict is not None:
        for k, v in stats_dict.items():
            print(f"{indent}{k}: {v}")
    else:
        # Fallback: print raw object
        print(f"{indent}{stats}")


def _safe_tail_print(obj: Any, n: int = 5, indent: str = ""):
    """
    Print tail of a pandas object if available; otherwise print the object.
    """
    if hasattr(obj, "tail"):
        try:
            print(obj.tail(n))
            return
        except Exception:
            pass
    print(f"{indent}{obj}")


def _plot_series(series: Any, path: Optional[Path], title: str = ""):
    """
    Save a simple line plot for a pandas-like series if matplotlib is available.
    """
    if path is None:
        return
    try:
        import matplotlib.pyplot as plt

        _set_matplotlib_font_defaults()

        s = _clean_series(series)
        if s.empty:
            return

        plt.figure(figsize=(10, 4))
        s.plot()
        plt.title(title or path.name)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print(f"Saved plot: {path}")
    except Exception as err:
        print(f"Plot skipped ({err})")


def _quantstats_report(returns: Any, output_path: Optional[Path]):
    """
    Generate a QuantStats HTML report if quantstats is available.

    Expects periodic percentage returns (e.g. portfolio.percent), not a cumulative curve.
    """
    if output_path is None:
        return
    try:
        _set_matplotlib_font_defaults()
        import quantstats as qs

        qs.extend_pandas()

        if not hasattr(returns, "__len__"):
            print("QuantStats skipped (returns is not iterable).")
            return

        returns = _clean_series(pd.Series(returns))
        if returns.empty:
            print("QuantStats skipped (no returns).")
            return

        # portfolio.percent is expressed in percent units; convert to decimal for QuantStats
        returns = returns.astype(float) / 100.0

        # Normalise index safely for quantstats:
        # 1) tz-naive datetime
        # 2) resample to daily with ffill
        # 3) rebuild index as a clean daily DateRange (freq='D') to avoid 'ME' bugs
        if isinstance(returns.index, pd.PeriodIndex):
            returns.index = returns.index.to_timestamp()
        else:
            returns.index = pd.DatetimeIndex(returns.index).tz_localize(None)
        returns = returns.resample("D").ffill()
        returns = pd.Series(
            returns.values,
            index=pd.date_range(
                start=returns.index.min().normalize(),
                periods=len(returns),
                freq="D",
            ),
        )

        qs.reports.html(
            returns,
            output=output_path,
            title="pysystemtrade backtest",
            periods_per_year=252,
        )
        print(f"Saved QuantStats report: {output_path}")
    except Exception as err:
        print(f"QuantStats skipped ({err})")


def _ensure_spread_costs(data, instruments, fallback_default: float):
    """
    Ensure spread costs exist; if zero/missing use fallback and patch get_spread_cost to return it.
    Returns (used_costs_dict, missing_list).
    """
    scd = data.db_spread_cost_data
    original_get = scd.get_spread_cost
    missing = set()
    used = {}

    for inst in instruments:
        try:
            cost = original_get(inst)
        except Exception:
            cost = None
        if cost is None or cost == 0.0:
            missing.add(inst)
            cost = fallback_default
        used[inst] = cost

    # Patch getter so the rest of the system uses the chosen values
    def patched_get_spread_cost(instrument_code: str) -> float:
        return used.get(instrument_code, fallback_default)

    scd.get_spread_cost = patched_get_spread_cost  # type: ignore[attr-defined]

    if missing:
        print(
            f"Spread costs missing or zero for: {', '.join(sorted(missing))}. "
            f"Using fallback={fallback_default}."
        )

    return used, sorted(missing)


def _plot_notional_positions(
    system, path: Optional[Path], instruments: Optional[Sequence[str]], clip_bounds=None
):
    """
    Plot notional positions for each instrument on a single chart (or the provided subset).
    If clip_bounds is provided, positions are clipped to that range.
    """
    if path is None:
        return
    try:
        import matplotlib.pyplot as plt

        _set_matplotlib_font_defaults()

        instrs = instruments or system.get_instrument_list()
        plt.figure(figsize=(12, 6))
        for inst in instrs:
            pos = system.portfolio.get_notional_position(inst)
            pos = _clean_series(pd.Series(pos).astype(float)).sort_index()
            if clip_bounds:
                pos = pos.clip(lower=clip_bounds[0], upper=clip_bounds[1])
            plt.plot(pos.index, pos.values, label=inst)

        if clip_bounds:
            plt.axhline(
                clip_bounds[0], color="red", linestyle="--", linewidth=0.8, alpha=0.5
            )
            plt.axhline(
                clip_bounds[1], color="red", linestyle="--", linewidth=0.8, alpha=0.5
            )
            plt.title(
                f"Notional positions (capped to [{clip_bounds[0]}, {clip_bounds[1]}])"
            )
        else:
            plt.title("Notional positions (uncapped)")
        plt.ylabel("Notional position")
        plt.xlabel("Date")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print(f"Saved notional positions plot: {path}")
    except Exception as err:
        print(f"Notional position plot skipped ({err})")


def _plot_buffered_positions(
    system, path: Optional[Path], instruments: Optional[Sequence[str]], clip_bounds=None
):
    """
    Plot buffered (executed) positions for each instrument on a single chart (or subset).
    """
    if path is None:
        return
    try:
        import matplotlib.pyplot as plt

        _set_matplotlib_font_defaults()

        instrs = instruments or system.get_instrument_list()
        plt.figure(figsize=(12, 6))
        for inst in instrs:
            pos = system.accounts.get_buffered_position(inst, roundpositions=True)
            pos = _clean_series(pd.Series(pos).astype(float)).sort_index()
            if clip_bounds:
                pos = pos.clip(lower=clip_bounds[0], upper=clip_bounds[1])
            plt.step(
                pos.index, pos.values, where="post", label=inst
            )  # step to show fills

        if clip_bounds:
            plt.axhline(
                clip_bounds[0], color="red", linestyle="--", linewidth=0.8, alpha=0.5
            )
            plt.axhline(
                clip_bounds[1], color="red", linestyle="--", linewidth=0.8, alpha=0.5
            )
            plt.title(
                f"Buffered positions (capped to [{clip_bounds[0]}, {clip_bounds[1]}])"
            )
        else:
            plt.title("Buffered positions (executed, rounded contracts)")
        plt.ylabel("Contracts")
        plt.xlabel("Date")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print(f"Saved buffered positions plot: {path}")
    except Exception as err:
        print(f"Buffered position plot skipped ({err})")


def _compute_return_stats(returns_pct: pd.Series, periods_per_year: int = 252) -> dict:
    """Return basic perf stats from a series of percent returns."""
    if returns_pct is None:
        return {}

    returns = pd.Series(returns_pct).astype(float) / 100.0
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if returns.empty:
        return {}

    log_returns = np.log1p(returns)
    cumulative = np.exp(log_returns.cumsum())
    total_return = cumulative.iloc[-1] - 1.0
    n = len(returns)
    ann_return = np.expm1(log_returns.mean() * periods_per_year)
    vol = returns.std(ddof=0) * np.sqrt(periods_per_year)
    sharpe = ann_return / vol if vol > 0 else np.nan

    peak = cumulative.cummax()
    dd = cumulative / peak - 1.0
    max_dd = dd.min()

    return dict(
        total_return=total_return,
        ann_return=ann_return,
        vol=vol,
        sharpe=sharpe,
        max_dd=max_dd,
    )


def _collect_notional_positions_by_year(
    system, instrument_filter: Optional[Sequence[str]] = None
) -> list:
    """
    Return annual average notional positions per instrument (optionally filtered)
    for summary tables.
    """
    rows = []
    try:
        instruments = instrument_filter or system.get_instrument_list()
    except Exception:
        return rows

    for inst in instruments:
        try:
            pos = system.portfolio.get_notional_position(inst)
            pos = _clean_series(pd.Series(pos))
            if pos.empty:
                continue
            df = pos.to_frame("pos")
            df.index = pd.to_datetime(df.index)
            grouped = df.resample("YE").mean().dropna()
            for ts, row in grouped.iterrows():
                rows.append(
                    [
                        inst,
                        str(ts.year),
                        f"{row['pos']:.2f}",
                    ]
                )
        except Exception:
            continue
    return rows


def _collect_trades(
    system,
    instrument_filter: Optional[Sequence[str]] = None,
    max_rows: int = 200,
    base_currency: str = "",
) -> list:
    """
    Collect trade events (position changes) from buffered positions.
    Optionally filter by instrument. Rows: [Date, Instrument, Trade (contracts),
    New position, Position (base ccy), Spread cost, Commission cost, Total cost]
    where costs/position are in base ccy.
    Limited to the most recent max_rows to keep the report readable.
    """
    rows = []
    try:
        instruments = instrument_filter or system.get_instrument_list()
    except Exception:
        return rows

    for inst in instruments:
        try:
            pos = system.accounts.get_buffered_position(inst, roundpositions=True)
            pos = _clean_series(pd.Series(pos))
            if pos.empty:
                continue
            diff = pos.diff().fillna(0)
            trades = diff[diff != 0]

            # Data needed for cost/valuation
            price = system.portfolio.get_contract_prices(inst)
            fx = system.accounts.get_fx_rate(inst)
            block_mult = system.accounts.get_value_of_block_price_move(inst)
            costs = system.accounts.get_raw_cost_data(inst)

            for ts, delta in trades.items():
                ts_dt = pd.to_datetime(ts)
                clean_px = _clean_series(price)
                clean_fx = _clean_series(fx)
                try:
                    px = clean_px.reindex([ts_dt], method="ffill").iloc[0]
                    fx_rate = clean_fx.reindex([ts_dt], method="ffill").iloc[0]
                except Exception:
                    continue
                if pd.isna(px) or pd.isna(fx_rate):
                    continue

                trade_blocks = float(delta)
                position_blocks = float(pos.loc[ts])
                value_per_block = px * block_mult

                # Costs in instrument currency
                spread_ccy = abs(trade_blocks) * costs.price_slippage * block_mult
                per_block_comm = abs(trade_blocks) * costs.value_of_block_commission
                per_trade_comm = costs.value_of_pertrade_commission
                pct_comm = costs.percentage_cost * value_per_block * abs(trade_blocks)
                commission_ccy = max(per_block_comm, per_trade_comm, pct_comm)
                total_ccy = spread_ccy + commission_ccy

                # Convert to base
                spread_base = spread_ccy * fx_rate
                commission_base = commission_ccy * fx_rate
                total_base = total_ccy * fx_rate
                position_base = position_blocks * value_per_block * fx_rate

                rows.append(
                    [
                        str(ts_dt.date()),
                        inst,
                        f"{int(trade_blocks):+d}",
                        f"{int(position_blocks)}",
                        f"{position_base:,.2f} {base_currency}".strip(),
                        f"{spread_base:,.4f} {base_currency}".strip(),
                        f"{commission_base:,.4f} {base_currency}".strip(),
                        f"{total_base:,.4f} {base_currency}".strip(),
                    ]
                )
        except Exception:
            continue

    # Sort by date ascending then instrument
    rows.sort(key=lambda r: (r[0], r[1]))

    if len(rows) > max_rows:
        rows = rows[-max_rows:]

    return rows


def _format_pct(x: float) -> str:
    if x is None or np.isnan(x):
        return "-"
    return f"{100 * x:,.2f}%"


def _print_perf_table(title: str, rows: list, verbose: bool = True):
    if not rows:
        if verbose:
            print(f"{title}: (no data)")
        return []
    col_names = ["Name", "Total", "CAGR", "Vol", "Sharpe", "MaxDD"]
    if verbose:
        print(title)
        print(
            f"{col_names[0]:<25} {col_names[1]:>12} {col_names[2]:>12} {col_names[3]:>12} {col_names[4]:>8} {col_names[5]:>12}"
        )
    printable = []
    for name, stats in rows:
        row = [
            name,
            _format_pct(stats.get("total_return", np.nan)),
            _format_pct(stats.get("ann_return", np.nan)),
            _format_pct(stats.get("vol", np.nan)),
            f"{stats.get('sharpe', np.nan):.2f}"
            if not np.isnan(stats.get("sharpe", np.nan))
            else "-",
            _format_pct(stats.get("max_dd", np.nan)),
        ]
        printable.append(row)
        if verbose:
            print(
                f"{row[0]:<25} {row[1]:>12} {row[2]:>12} {row[3]:>12} {row[4]:>8} {row[5]:>12}"
            )

    return printable


def _print_per_instrument_perf(
    portfolio, instrument_filter: Optional[Sequence[str]] = None, verbose: bool = True
):
    try:
        instrument_group = portfolio.percent  # per-instrument % P&L
        rows = []
        for inst in instrument_group.asset_columns:
            if instrument_filter and inst not in instrument_filter:
                continue
            inst_curve = instrument_group[inst]
            stats = _compute_return_stats(inst_curve.as_ts)
            rows.append((inst, stats))
        return _print_perf_table("Per instrument", rows, verbose=verbose)
    except Exception as err:
        if verbose:
            print(f"Per-instrument stats skipped ({err})")
        return []


def _print_per_strategy_perf(system, portfolio, verbose: bool = True):
    try:
        rules_group = system.accounts.pandl_for_all_trading_rules()
        rules_group = rules_group.value_terms  # P&L in currency per rule
        capital = portfolio.capital
        rows = []
        for rule in rules_group.asset_columns:
            rule_curve = rules_group[rule]
            pnl = pd.Series(rule_curve.as_ts).astype(float)
            cap = pd.Series(capital).astype(float).reindex(pnl.index).ffill()
            returns_pct = (pnl / cap) * 100.0
            stats = _compute_return_stats(returns_pct)
            rows.append((rule, stats))
        return _print_perf_table("Per strategy/rule", rows, verbose=verbose)
    except Exception as err:
        if verbose:
            print(f"Per-strategy stats skipped ({err})")
        return []


def _build_spread_cost_rows(used_costs: dict, missing: list) -> list:
    rows = []
    missing_set = set(missing)
    for inst, cost in used_costs.items():
        source = "fallback" if inst in missing_set else "db"
        rows.append([inst, source, f"{cost:.4f}"])
    return rows


def _build_summary_rows(stats: Any, portfolio, base_currency: str = "") -> list:
    """
    Build summary rows with metric name, value, UoM, and description.
    Ensures Sharpe is included even if stats is malformed. Values are formatted
    in plain decimal (non-scientific) where possible.
    """
    rows = []
    descriptions = _metric_descriptions()

    # account_curve.stats returns [list_of_pairs, comment]; handle that shape first
    if isinstance(stats, (list, tuple)) and stats:
        first = stats[0]
        if isinstance(first, (list, tuple)):
            try:
                rows.extend(
                    [
                        _format_summary_row(k, v, descriptions, base_currency)
                        for k, v in first
                    ]
                )
            except Exception:
                pass

    # dict-like stats (e.g. statsDict)
    if not rows and hasattr(stats, "items"):
        try:
            rows = [
                _format_summary_row(k, v, descriptions, base_currency)
                for k, v in stats.items()
            ]
        except Exception:
            rows = []

    # final fallback: try dict(...) coercion
    if not rows:
        try:
            rows = [
                _format_summary_row(k, v, descriptions, base_currency)
                for k, v in dict(stats).items()
            ]
        except Exception:
            rows = []

    # Always append Sharpe (avoids losing it if stats parsing failed)
    existing = {str(r[0]).lower() for r in rows}
    if "sharpe" not in existing:
        try:
            rows.append(
                _format_summary_row(
                    "Sharpe",
                    portfolio.sharpe(),
                    descriptions,
                    base_currency,
                )
            )
        except Exception:
            pass

    return rows


def _metric_descriptions() -> dict:
    """
    Short English descriptions for common metrics shown in the summary.
    """
    return {
        "min": "Smallest daily return",
        "max": "Largest daily return",
        "median": "Median daily return",
        "mean": "Average daily return",
        "std": "Std dev of daily returns",
        "skew": "Skew of daily returns",
        "ann_mean": "Annualised mean return",
        "ann_std": "Annualised volatility",
        "sharpe": "Annualised Sharpe ratio",
        "Sharpe": "Annualised Sharpe ratio",
        "sortino": "Sortino ratio (downside risk)",
        "avg_drawdown": "Average drawdown depth",
        "time_in_drawdown": "Fraction of time in drawdown",
        "calmar": "Calmar ratio (CAGR / max DD)",
        "avg_return_to_drawdown": "Avg gain per drawdown",
        "avg_loss": "Average losing return",
        "avg_gain": "Average winning return",
        "gaintolossratio": "Average gain / average loss",
        "profitfactor": "Sum gains / sum losses",
        "hitrate": "Fraction of positive periods",
        "t_stat": "t-stat of mean return",
        "p_value": "p-value of mean return",
        "max_drawdown": "Maximum drawdown",
    }


def _metric_uom(metric: str, base_currency: str) -> str:
    """
    Unit of measure per metric: currency for P&L/returns, ratio/fraction otherwise.
    """
    currency_metrics = {
        "min",
        "max",
        "median",
        "mean",
        "std",
        "ann_mean",
        "ann_std",
        "avg_drawdown",
        "avg_return_to_drawdown",
        "avg_loss",
        "avg_gain",
        "max_drawdown",
    }
    ratio_metrics = {
        "sharpe",
        "Sharpe",
        "sortino",
        "calmar",
        "gaintolossratio",
        "profitfactor",
        "t_stat",
        "skew",
    }
    fraction_metrics = {"time_in_drawdown", "hitrate", "p_value"}
    if metric in currency_metrics:
        return base_currency or ""
    if metric in ratio_metrics:
        return "/"
    if metric in fraction_metrics:
        return "/"
    return ""


def _format_summary_row(metric, value, descriptions: dict, base_currency: str) -> list:
    return [
        str(metric),
        _format_decimal(value),
        _metric_uom(str(metric), base_currency),
        descriptions.get(str(metric), ""),
    ]


def _format_decimal(value: Any) -> str:
    """
    Format numbers with fixed-point to avoid scientific notation.
    """
    try:
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        num = float(value)
        if np.isnan(num) or np.isinf(num):
            return str(value)
        return f"{num:.6f}"
    except Exception:
        return str(value)


def _add_table_pages(pdf, rows, col_labels, title, rows_per_page=25, col_widths=None):
    import matplotlib.pyplot as plt

    _set_matplotlib_font_defaults()

    total = len(rows)
    pages = (total + rows_per_page - 1) // rows_per_page
    for i in range(pages):
        chunk = rows[i * rows_per_page : (i + 1) * rows_per_page]
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        page_title = title if pages == 1 else f"{title} (page {i+1}/{pages})"
        ax.set_title(page_title, fontsize=14, pad=10)
        table = ax.table(
            cellText=chunk,
            colLabels=col_labels,
            loc="upper left",
            colWidths=col_widths,
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.2)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def _build_unified_pdf(
    output_path: Path,
    summary_rows: list,
    per_inst_rows: list,
    per_rule_rows: list,
    cost_rows: list,
    notional_rows: list,
    trades_rows: list,
    figure_paths: list,
):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.image as mpimg

        with PdfPages(output_path) as pdf:
            if summary_rows:
                _add_table_pages(
                    pdf,
                    summary_rows,
                    ["Metric", "Value", "UoM", "Description"],
                    "Portfolio summary",
                    rows_per_page=30,
                    col_widths=[0.28, 0.16, 0.10, 0.46],
                )

            if per_inst_rows:
                _add_table_pages(
                    pdf,
                    per_inst_rows,
                    ["Name", "Total", "CAGR", "Vol", "Sharpe", "MaxDD"],
                    "Performance per instrument (net % returns)",
                    rows_per_page=25,
                )

            if per_rule_rows:
                _add_table_pages(
                    pdf,
                    per_rule_rows,
                    ["Name", "Total", "CAGR", "Vol", "Sharpe", "MaxDD"],
                    "Performance per strategy/rule (net % returns)",
                    rows_per_page=20,
                )

            if cost_rows:
                _add_table_pages(
                    pdf,
                    cost_rows,
                    ["Instrument", "Source", "Spread"],
                    "Spread costs used",
                    rows_per_page=35,
                )

            if notional_rows:
                _add_table_pages(
                    pdf,
                    notional_rows,
                    ["Instrument", "Year", "Avg notional"],
                    "Notional positions (annual average)",
                    rows_per_page=35,
                )

            if trades_rows:
                # Note: only the most recent max_rows trades are shown to keep the report concise.
                _add_table_pages(
                    pdf,
                    trades_rows,
                    [
                        "Date",
                        "Instrument",
                        "Trade",
                        "New pos",
                        "Pos (base)",
                        "Spread",
                        "Commission",
                        "Total cost",
                    ],
                    "Executed trades (from buffered positions)",
                    rows_per_page=30,
                    col_widths=[0.12, 0.14, 0.08, 0.08, 0.18, 0.12, 0.14, 0.14],
                )

            for fig_path in figure_paths:
                if fig_path is None:
                    continue
                if not Path(fig_path).exists():
                    continue
                img = mpimg.imread(fig_path)
                fig, ax = plt.subplots(figsize=(11.69, 8.27))
                ax.axis("off")
                ax.imshow(img)
                ax.set_title(Path(fig_path).name, fontsize=12, pad=10)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

        print(f"Saved unified report: {output_path}")
    except Exception as err:
        print(f"Unified PDF report skipped ({err})")


def _write_debug_txt(
    output_path: Path,
    summary_rows: list,
    per_inst_rows: list,
    per_rule_rows: list,
    cost_rows: list,
    notional_rows: list,
    trades_rows: list,
):
    """
    Write a plain-text summary for easier debugging.
    """
    sections = []

    def add_table(title: str, rows: list, headers: list):
        if not rows:
            return
        sections.append(title)
        sections.append(" | ".join(headers))
        sections.append("-" * 80)
        for row in rows:
            sections.append(" | ".join(row))
        sections.append("")  # blank line between tables

    add_table(
        "Summary",
        summary_rows,
        ["Metric", "Value", "UoM", "Description"],
    )
    add_table(
        "Performance per instrument",
        per_inst_rows,
        ["Name", "Total", "CAGR", "Vol", "Sharpe", "MaxDD"],
    )
    add_table(
        "Performance per strategy/rule",
        per_rule_rows,
        ["Name", "Total", "CAGR", "Vol", "Sharpe", "MaxDD"],
    )
    add_table("Spread costs used", cost_rows, ["Instrument", "Source", "Spread"])
    add_table(
        "Notional positions (annual average)",
        notional_rows,
        ["Instrument", "Year", "Avg notional"],
    )
    add_table(
        "Executed trades (from buffered positions)",
        trades_rows,
        [
            "Date",
            "Instrument",
            "Trade",
            "New pos",
            "Pos (base)",
            "Spread",
            "Commission",
            "Total cost",
        ],
    )
    if trades_rows:
        sections.append(
            "Note: only the most recent trades are shown to keep the report concise."
        )
        sections.append("")

    if not sections:
        return

    try:
        output_path.write_text("\n".join(sections))
        print(f"Saved debug summary: {output_path}")
    except Exception as err:
        print(f"Debug text write skipped ({err})")


def parse_args(argv=None) -> BacktestConfig:
    parser = argparse.ArgumentParser(
        description="Run a futures backtest using DB data (Mongo + Parquet).",
    )
    parser.add_argument(
        "--config",
        dest="config_path",
        default=None,
        help="Config file path (required unless <script>_config.yaml sits next to the entrypoint).",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Output directory (default: backtest_results next to the config).",
    )
    parser.add_argument(
        "--timestamp",
        default=None,
        help="Custom timestamp for output files (default: now).",
    )
    parser.add_argument(
        "--fallback-spread",
        type=float,
        default=DEFAULT_FALLBACK_SPREAD,
        help="Fallback spread if missing or zero in the DB.",
    )
    parser.add_argument(
        "--instruments",
        default=None,
        help="Comma-separated list of instruments to include (default: all).",
    )
    parser.add_argument(
        "--no-plots", action="store_true", help="Disable PNG generation."
    )
    parser.add_argument(
        "--no-quantstats",
        action="store_true",
        help="Disable the QuantStats HTML report.",
    )
    parser.add_argument(
        "--no-pdf", action="store_true", help="Disable the unified PDF."
    )
    parser.add_argument(
        "--no-debug", action="store_true", help="Disable the debug txt output."
    )
    parser.add_argument(
        "--keep-intermediate-figs",
        action="store_true",
        help="Keep PNG figures instead of deleting them after building the PDF.",
    )

    args = parser.parse_args(argv)
    instruments = None
    if args.instruments:
        instruments = [x.strip() for x in args.instruments.split(",") if x.strip()]

    return BacktestConfig(
        config_path=Path(args.config_path) if args.config_path else None,
        results_dir=Path(args.results_dir) if args.results_dir else None,
        timestamp=args.timestamp,
        fallback_spread=args.fallback_spread,
        instrument_filter=instruments,
        include_plots=not args.no_plots,
        include_quantstats=not args.no_quantstats,
        include_pdf=not args.no_pdf,
        include_debug_txt=not args.no_debug,
        keep_intermediate_figs=args.keep_intermediate_figs,
    )


if __name__ == "__main__":
    cli_config = parse_args()
    run_backtest(cli_config)
