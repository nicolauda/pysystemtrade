"""backtest_runner_reportlab.py

Core routines for running a futures backtest using DB data (Mongo + Parquet)
via `dbFuturesSimData`, so you match what production will read.

This version fixes PDF table layout issues by rendering the unified PDF with
**ReportLab (Platypus)** (real layout engine), instead of matplotlib's table
artist (which is great for plots but weak for pagination/wrapping).

Key improvements
- Robust wrapping with Paragraphs (no text spilling out of cells)
- Auto row height based on wrapped content
- Repeat table headers across pages
- Policy-driven tables (wrap / alignment / landscape / wide-table chunking)
- Correlation matrices split into manageable column blocks

You can drop this file in place of your current backtest runner (or merge the
PDF-related parts into your existing script).

"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import io
import logging
import math
import sys
import warnings
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from sysdata.config.configdata import Config
from sysdata.sim.db_futures_sim_data import dbFuturesSimData
from sysdata.data_blob import dataBlob
from sysproduction.data.capital import dataCapital
from sysproduction.data.currency_data import dataCurrency
from systems.custom_system.run_system import futures_system
from systems.diagoutput import systemDiag
from syscore.capital import fixed_capital, full_compounding, half_compounding
from syscore.exceptions import missingData
from syscore.fileutils import (
    get_resolved_pathname,
    resolve_path_and_filename_for_package,
)

# --- ReportLab PDF rendering ---
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    BaseDocTemplate,
    PageTemplate,
    Frame,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    PageBreak,
    NextPageTemplate,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


DEFAULT_FALLBACK_SPREAD = 1.0  # price units if spread missing


# =========================
# Path helpers
# =========================


def _infer_default_config_path() -> Optional[Path]:
    """If entrypoint is <name>_backtest.py, try <name>_config.yaml next to it."""
    try:
        main_path = Path(sys.argv[0]).resolve()
    except Exception:
        return None

    if not main_path.name.endswith("_backtest.py"):
        return None

    base = main_path.stem[: -len("_backtest")]
    candidate = main_path.with_name(f"{base}_config.yaml")
    return candidate if candidate.exists() else None


def _resolve_file_path(pathlike: Optional[Path]) -> Optional[Path]:
    if pathlike is None:
        return None
    try:
        resolved = resolve_path_and_filename_for_package(str(pathlike))
        return Path(resolved)
    except Exception:
        return Path(pathlike).expanduser()


def _resolve_dir_path(pathlike: Optional[Path]) -> Optional[Path]:
    if pathlike is None:
        return None
    try:
        resolved = get_resolved_pathname(str(pathlike))
        return Path(resolved)
    except Exception:
        return Path(pathlike).expanduser()


@contextlib.contextmanager
def _tee_output(log_path: Optional[Path]):
    """Tee stdout/stderr to log_path while still echoing to terminal."""
    if log_path is None:
        yield
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)

    class _Tee(io.TextIOBase):
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for stream in self.streams:
                try:
                    stream.write(data)
                except Exception:
                    pass
            try:
                self.flush()
            except Exception:
                pass
            return len(data)

        def flush(self):
            for stream in self.streams:
                try:
                    stream.flush()
                except Exception:
                    pass

    with log_path.open("w", encoding="utf-8") as log_file:
        tee_out = _Tee(sys.stdout, log_file)
        tee_err = _Tee(sys.stderr, log_file)
        with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
            yield


# =========================
# Config + result models
# =========================


@dataclass
class BacktestConfig:
    config_path: Optional[Path] = None
    results_dir: Optional[Path] = None
    timestamp: Optional[str] = None
    fallback_spread: float = DEFAULT_FALLBACK_SPREAD
    instrument_filter: Optional[Sequence[str]] = None

    include_plots: bool = True
    include_quantstats: bool = True
    include_pdf: bool = True
    include_report_txt: bool = True
    include_debug_log: bool = True

    keep_intermediate_figs: bool = False

    use_cache: bool = False
    cache_file: Optional[Path] = None
    cache_compress: bool = True

    export_estimates: bool = True
    use_db_capital: bool = False
    strategy_name: Optional[str] = None
    capital_multiplier: Optional[str] = None
    ic_only: bool = False

    def with_defaults(self) -> "BacktestConfig":
        inferred = _infer_default_config_path()
        if self.config_path:
            resolved_config = _resolve_file_path(self.config_path)
        elif inferred:
            resolved_config = _resolve_file_path(inferred)
        else:
            raise ValueError(
                "config_path is required (no <name>_config.yaml inferred). "
                "Pass BacktestConfig.config_path or --config."
            )

        resolved_results = (
            _resolve_dir_path(self.results_dir)
            if self.results_dir is not None
            else _resolve_dir_path(resolved_config.parent / "backtest_results")
        )
        resolved_timestamp = self.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        resolved_instruments = (
            tuple(self.instrument_filter) if self.instrument_filter else None
        )

        cache_ext = "pckz" if self.cache_compress else "pck"
        resolved_cache = (
            _resolve_file_path(self.cache_file)
            if self.cache_file is not None
            else resolved_results / f"{resolved_config.stem}_system_cache.{cache_ext}"
        )

        return BacktestConfig(
            config_path=resolved_config,
            results_dir=resolved_results,
            timestamp=resolved_timestamp,
            fallback_spread=float(self.fallback_spread),
            instrument_filter=resolved_instruments,
            include_plots=bool(self.include_plots),
            include_quantstats=bool(self.include_quantstats),
            include_pdf=bool(self.include_pdf),
            include_report_txt=bool(self.include_report_txt),
            include_debug_log=bool(self.include_debug_log),
            keep_intermediate_figs=bool(self.keep_intermediate_figs),
            use_cache=bool(self.use_cache),
            cache_file=resolved_cache,
            cache_compress=bool(self.cache_compress),
            export_estimates=bool(self.export_estimates),
            use_db_capital=bool(self.use_db_capital),
            strategy_name=self.strategy_name or resolved_config.stem,
            capital_multiplier=self.capital_multiplier,
            ic_only=bool(self.ic_only),
        )


@dataclass
class BacktestOutputs:
    results_dir: Path
    figures: dict
    quantstats_report: Optional[Path]
    pdf_report: Optional[Path]
    report_txt: Optional[Path]
    debug_log: Optional[Path]
    estimates_yaml: Optional[Path]


@dataclass
class BacktestResult:
    system: Any
    summary_rows: list
    per_inst_rows: list
    per_rule_rows: list
    per_rule_headers: list
    rule_variation_rows: list
    rule_variation_headers: list
    rule_correlation_rows: list
    rule_correlation_headers: list
    per_inst_headers: list
    cost_rows: list
    notional_rows: list
    trades_rows: list
    outputs: BacktestOutputs


# =========================
# Matplotlib plotting helpers
# =========================


def _set_matplotlib_font_defaults():
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
    return s.loc[mask] if mask.any() else s


def _plot_series(series: Any, path: Optional[Path], title: str = ""):
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


def _plot_multiple_series(
    series_dict: dict, path: Optional[Path], title: str = "", ylabel: str = ""
):
    if path is None or not series_dict:
        return
    try:
        import matplotlib.pyplot as plt

        _set_matplotlib_font_defaults()
        plt.figure(figsize=(12, 6))
        plotted = 0
        for name, series in series_dict.items():
            s = _clean_series(series)
            if s.empty:
                continue
            plt.plot(s.index, s.values, label=name)
            plotted += 1
        if plotted == 0:
            plt.close()
            return
        plt.title(title or path.name)
        plt.xlabel("Date")
        if ylabel:
            plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print(f"Saved plot: {path}")
    except Exception as err:
        print(f"Aggregated rule plot skipped ({err})")


def _plot_notional_positions(
    system, path: Optional[Path], instruments: Optional[Sequence[str]], clip_bounds=None
):
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
            plt.axhline(clip_bounds[0], linestyle="--", linewidth=0.8, alpha=0.5)
            plt.axhline(clip_bounds[1], linestyle="--", linewidth=0.8, alpha=0.5)
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
            plt.step(pos.index, pos.values, where="post", label=inst)
        if clip_bounds:
            plt.axhline(clip_bounds[0], linestyle="--", linewidth=0.8, alpha=0.5)
            plt.axhline(clip_bounds[1], linestyle="--", linewidth=0.8, alpha=0.5)
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


# =========================
# Quantstats
# =========================


def _quantstats_report(returns: Any, output_path: Optional[Path]):
    if output_path is None:
        return
    try:
        _set_matplotlib_font_defaults()
        import quantstats as qs

        qs.extend_pandas()

        returns = _clean_series(pd.Series(returns))
        if returns.empty:
            print("QuantStats skipped (no returns).")
            return
        returns = returns.astype(float) / 100.0

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


# =========================
# Core calculations + formatting
# =========================


def _compute_return_stats(returns_pct: pd.Series, periods_per_year: int = 252) -> dict:
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

    t_stat = np.nan
    p_value = np.nan
    if n > 1:
        mean_return = returns.mean()
        sample_std = returns.std(ddof=1)
        if sample_std > 0:
            t_stat = mean_return / (sample_std / math.sqrt(n))
            try:
                from scipy import stats

                p_value = float(stats.t.sf(abs(t_stat), df=n - 1) * 2.0)
            except Exception:
                p_value = float(math.erfc(abs(t_stat) / math.sqrt(2)))

    peak = cumulative.cummax()
    dd = cumulative / peak - 1.0
    max_dd = dd.min()

    return dict(
        total_return=total_return,
        ann_return=ann_return,
        vol=vol,
        sharpe=sharpe,
        max_dd=max_dd,
        t_stat=t_stat,
        p_value=p_value,
    )


def _format_pct(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{100 * float(x):,.2f}%"


def _format_ccy(x: float, base_currency: str = "") -> str:
    try:
        if x is None or np.isnan(x):
            return "-"
        return f"{float(x):,.2f} {base_currency}".strip()
    except Exception:
        return "-"


def _format_number(x: float, decimals: int = 2) -> str:
    try:
        if x is None or np.isnan(x):
            return "-"
        return f"{float(x):.{decimals}f}"
    except Exception:
        return "-"


def _format_p_value(x: float) -> str:
    try:
        if x is None or np.isnan(x):
            return "-"
        if x < 0.001:
            return "<0.001"
        return f"{float(x):.3f}"
    except Exception:
        return "-"


def _print_perf_table(
    title: str,
    rows: list,
    verbose: bool = True,
    include_significance: bool = False,
    include_pnl: bool = False,
    include_ic: bool = False,
    base_currency: str = "",
):
    """Render a performance table while optionally including IC and significance columns."""
    if not rows:
        if verbose:
            print(f"{title}: (no data)")
        return [], []

    col_names = ["Name"]
    if include_pnl:
        col_names.append("P&L")
    col_names += ["Total", "CAGR", "Vol", "Sharpe"]
    if include_ic:
        col_names.append("IC")
    col_names.append("MaxDD")
    if include_significance:
        col_names += ["t-stat", "p-value"]

    if verbose:
        print(title)
        print(" | ".join(col_names))

    printable = []
    for name, stats in rows:
        row = [name]
        if include_pnl:
            row.append(_format_ccy(stats.get("pnl", np.nan), base_currency))
        row.extend(
            [
                _format_pct(stats.get("total_return", np.nan)),
                _format_pct(stats.get("ann_return", np.nan)),
                _format_pct(stats.get("vol", np.nan)),
                f"{stats.get('sharpe', np.nan):.2f}"
                if not np.isnan(stats.get("sharpe", np.nan))
                else "-",
            ]
        )
        if include_ic:
            row.append(
                _format_number(stats.get("info_coefficient", np.nan), decimals=3)
            )
        row.append(_format_pct(stats.get("max_dd", np.nan)))
        if include_significance:
            row.append(_format_number(stats.get("t_stat", np.nan)))
            row.append(_format_p_value(stats.get("p_value", np.nan)))
        printable.append(row)
        if verbose:
            print(" | ".join(row))
    return printable, col_names


def _print_per_instrument_perf(
    portfolio, instrument_filter: Optional[Sequence[str]] = None, verbose: bool = True
):
    try:
        instrument_group = portfolio.percent
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
        return [], []


def _extract_forecast_series(forecast: Any) -> Optional[pd.Series]:
    """Extract a single numeric pandas Series from various forecast representations."""
    if forecast is None:
        return None
    if isinstance(forecast, pd.DataFrame):
        if forecast.shape[1] == 0:
            return None
        series = forecast.iloc[:, 0]
    elif isinstance(forecast, pd.Series):
        series = forecast
    else:
        try:
            series = pd.Series(forecast).squeeze()
        except Exception:
            return None
    series = pd.to_numeric(series, errors="coerce")
    series = series.dropna()
    return series if not series.empty else None


def _sum_series_list(series_list: list[pd.Series]) -> Optional[pd.Series]:
    """Sum a list of Series along their shared index, returning None if empty."""
    if not series_list:
        return None
    if len(series_list) == 1:
        return series_list[0].astype(float)
    combined = pd.concat(series_list, axis=1)
    combined = combined.fillna(0.0)
    summed = combined.sum(axis=1)
    return summed.astype(float)


def _build_instrument_rule_map(
    system, instrument_list: Sequence[str]
) -> dict[str, set[str]]:
    """Map each instrument to the trading rules that reference it."""
    mapping: dict[str, set[str]] = {}
    for inst in instrument_list:
        try:
            rules = system.accounts.list_of_rules_for_code(inst)
        except Exception:
            rules = []
        mapping[inst] = set(rules or [])
    return mapping


def _aggregate_forecast_for_rule(
    system,
    rule: str,
    instrument_list: Sequence[str],
    instrument_rule_map: dict[str, set[str]],
) -> Optional[pd.Series]:
    """Aggregate the capped forecast series across all instruments supporting a rule."""
    series_list: list[pd.Series] = []
    for inst in instrument_list:
        if rule not in instrument_rule_map.get(inst, set()):
            continue
        try:
            forecast = system.accounts.get_capped_forecast(inst, rule)
        except Exception:
            continue
        series = _extract_forecast_series(forecast)
        if series is None or series.empty:
            continue
        series_list.append(series)
    return _sum_series_list(series_list)


def _compute_info_coefficient(
    forecast_series: Optional[pd.Series], returns_series: Optional[pd.Series]
) -> float:
    """Compute the Pearson correlation between aligned forecast and return series."""
    if (
        forecast_series is None
        or forecast_series.empty
        or returns_series is None
        or returns_series.empty
    ):
        return float("nan")
    aligned_forecast = forecast_series.reindex(returns_series.index)
    aligned_forecast = pd.to_numeric(aligned_forecast, errors="coerce")
    aligned_returns = pd.to_numeric(returns_series, errors="coerce")
    mask = (~aligned_forecast.isna()) & (~aligned_returns.isna())
    if mask.sum() < 2:
        return float("nan")
    try:
        return float(aligned_forecast[mask].corr(aligned_returns[mask]))
    except Exception:
        return float("nan")


def _print_per_strategy_perf(
    system, portfolio, base_currency: str = "", verbose: bool = True
):
    """Gather per-rule variation stats plus information coefficient data."""
    try:
        rules_group = system.accounts.pandl_for_all_trading_rules().value_terms
        capital = pd.Series(portfolio.capital).astype(float)

        rows = []
        returns_df = {}
        pnl_by_rule = {}
        forecast_by_rule: dict[str, Optional[pd.Series]] = {}

        try:
            instrument_list = system.get_instrument_list()
        except Exception:
            instrument_list = []
        instrument_rule_map = _build_instrument_rule_map(system, instrument_list)

        for rule in rules_group.asset_columns:
            rule_curve = rules_group[rule]
            pnl = pd.Series(rule_curve.as_ts).astype(float)
            pnl_by_rule[rule] = pnl
            cap = capital.reindex(pnl.index).ffill()
            returns_pct = (pnl / cap) * 100.0

            forecast_series = _aggregate_forecast_for_rule(
                system, rule, instrument_list, instrument_rule_map
            )
            ic = _compute_info_coefficient(forecast_series, returns_pct)

            stats = _compute_return_stats(returns_pct)
            stats["pnl"] = pnl.sum()
            stats["info_coefficient"] = ic

            returns_df[rule] = returns_pct
            rows.append((rule, stats))
            forecast_by_rule[rule] = forecast_series

        table_rows, headers = _print_perf_table(
            "Performance per rule variation (net % returns, scaled to portfolio capital):",
            rows,
            verbose=verbose,
            include_significance=True,
            include_pnl=True,
            include_ic=True,
            base_currency=base_currency,
        )
        returns_df = pd.DataFrame(returns_df)
        return table_rows, headers, returns_df, pnl_by_rule, capital, forecast_by_rule
    except Exception as err:
        if verbose:
            print(f"Per-strategy stats skipped ({err})")
        return [], [], pd.DataFrame(), {}, pd.Series(dtype=float), {}


def _group_rules_by_function(system) -> dict[str, list]:
    groups: dict[str, list] = defaultdict(list)
    try:
        trading_rules = system.rules.trading_rules()
    except Exception:
        trading_rules = {}
    for rule_name, rule_obj in trading_rules.items():
        func = getattr(rule_obj, "function", None)
        func_name = getattr(func, "__name__", "") or str(func)
        module_name = getattr(func, "__module__", "")
        base_module = module_name.split(".")[-1] if module_name else ""
        group_key = f"{base_module}.{func_name}" if base_module else func_name
        group_key = group_key or rule_name
        groups[group_key].append(rule_name)

    if not groups and hasattr(system.accounts, "list_of_trading_rules"):
        try:
            for rule_name in system.accounts.list_of_trading_rules():
                groups[rule_name].append(rule_name)
        except Exception:
            pass

    return {k: sorted(v) for k, v in sorted(groups.items())}


def _build_rule_group_perf(
    system,
    pnl_by_rule: dict,
    capital_series: pd.Series,
    forecast_by_rule: Optional[dict[str, Optional[pd.Series]]] = None,
    base_currency: str = "",
    verbose: bool = True,
):
    """Group rule variations by function and compute aggregated performance plus IC."""
    groups = _group_rules_by_function(system)
    if not groups:
        groups = {rule_name: [rule_name] for rule_name in pnl_by_rule.keys()}

    rows = []
    group_curves = {}
    for group_name, rule_names in groups.items():
        group_pnl = None
        for rule in rule_names:
            pnl = pnl_by_rule.get(rule)
            if pnl is None or pnl.empty:
                continue
            group_pnl = pnl if group_pnl is None else group_pnl.add(pnl, fill_value=0.0)
        if group_pnl is None or group_pnl.empty:
            continue

        group_pnl = group_pnl.sort_index()
        group_curves[group_name] = group_pnl.cumsum().ffill()

        cap = capital_series.reindex(group_pnl.index).ffill()
        returns_pct = (group_pnl / cap) * 100.0
        stats = _compute_return_stats(returns_pct)
        stats["pnl"] = group_pnl.sum()
        forecast_map = forecast_by_rule or {}
        group_forecasts = []
        for rule in rule_names:
            series = forecast_map.get(rule)
            if series is None or series.empty:
                continue
            group_forecasts.append(series)
        group_forecast_series = _sum_series_list(group_forecasts)
        stats["info_coefficient"] = _compute_info_coefficient(
            group_forecast_series, returns_pct
        )
        rows.append((group_name, stats))

    table_rows, headers = _print_perf_table(
        "Performance aggregata per rule (somma delle variations):",
        rows,
        verbose=verbose,
        include_significance=True,
        include_pnl=True,
        include_ic=True,
        base_currency=base_currency,
    )
    return table_rows, headers, group_curves


def _build_correlation_table(returns_df: pd.DataFrame):
    if returns_df is None or returns_df.empty or returns_df.shape[1] < 2:
        return [], []
    clean = returns_df.dropna(how="all")
    if clean.empty or clean.shape[1] < 2:
        return [], []
    corr = clean.corr()
    headers = ["Strategy"] + list(corr.columns)
    rows = []
    for idx, row in corr.iterrows():
        formatted = [
            "-" if pd.isna(val) else _format_number(val, decimals=3)
            for val in row.values
        ]
        rows.append([idx] + formatted)
    return rows, headers


def _collect_rule_performance_tables(
    system, portfolio, base_currency: str = "", verbose: bool = True
):
    (
        var_rows,
        var_headers,
        returns_df,
        pnl_by_rule,
        capital_series,
        forecast_by_rule,
    ) = _print_per_strategy_perf(
        system, portfolio, base_currency=base_currency, verbose=verbose
    )
    group_rows, group_headers, group_curves = _build_rule_group_perf(
        system,
        pnl_by_rule,
        capital_series,
        forecast_by_rule,
        base_currency=base_currency,
        verbose=verbose,
    )
    corr_rows, corr_headers = _build_correlation_table(returns_df)
    return dict(
        group_rows=group_rows,
        group_headers=group_headers,
        variation_rows=var_rows,
        variation_headers=var_headers,
        correlation_rows=corr_rows,
        correlation_headers=corr_headers,
        group_curves=group_curves,
    )


def _collect_notional_positions_by_year(
    system, instrument_filter: Optional[Sequence[str]] = None
) -> list:
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
                rows.append([inst, str(ts.year), f"{row['pos']:.2f}"])
        except Exception:
            continue
    return rows


def _collect_trades(
    system,
    instrument_filter: Optional[Sequence[str]] = None,
    max_rows: int = 200,
    base_currency: str = "",
) -> list:
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

                spread_ccy = abs(trade_blocks) * costs.price_slippage * block_mult
                per_block_comm = abs(trade_blocks) * costs.value_of_block_commission
                per_trade_comm = costs.value_of_pertrade_commission
                pct_comm = costs.percentage_cost * value_per_block * abs(trade_blocks)
                commission_ccy = max(per_block_comm, per_trade_comm, pct_comm)
                total_ccy = spread_ccy + commission_ccy

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

    rows.sort(key=lambda r: (r[0], r[1]))
    if len(rows) > max_rows:
        rows = rows[-max_rows:]
    return rows


def _build_spread_cost_rows(
    used_costs: dict, missing: list, instruments: Optional[Sequence[str]] = None
) -> list:
    rows = []
    missing_set = set(missing)
    ordered = list(instruments) if instruments is not None else list(used_costs.keys())
    for inst in ordered:
        if inst not in used_costs:
            continue
        source = "fallback" if inst in missing_set else "db"
        rows.append([inst, source, f"{used_costs[inst]:.4f}"])
    return rows


def _metric_descriptions() -> dict:
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
    if metric in ratio_metrics or metric in fraction_metrics:
        return "/"
    return ""


def _format_decimal(value: Any) -> str:
    try:
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        num = float(value)
        if np.isnan(num) or np.isinf(num):
            return str(value)
        return f"{num:.6f}"
    except Exception:
        return str(value)


def _format_summary_row(metric, value, descriptions: dict, base_currency: str) -> list:
    return [
        str(metric),
        _format_decimal(value),
        _metric_uom(str(metric), base_currency),
        descriptions.get(str(metric), ""),
    ]


def _build_summary_rows(stats: Any, portfolio, base_currency: str = "") -> list:
    rows = []
    descriptions = _metric_descriptions()

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

    if not rows and hasattr(stats, "items"):
        try:
            rows = [
                _format_summary_row(k, v, descriptions, base_currency)
                for k, v in stats.items()
            ]
        except Exception:
            rows = []

    if not rows:
        try:
            rows = [
                _format_summary_row(k, v, descriptions, base_currency)
                for k, v in dict(stats).items()
            ]
        except Exception:
            rows = []

    existing = {str(r[0]).lower() for r in rows}
    if "sharpe" not in existing:
        try:
            rows.append(
                _format_summary_row(
                    "Sharpe", portfolio.sharpe(), descriptions, base_currency
                )
            )
        except Exception:
            pass
    return rows


def _print_stats(stats: Any, indent: str = ""):
    if hasattr(stats, "to_dict"):
        stats_dict = stats.to_dict()
    elif isinstance(stats, dict):
        stats_dict = stats
    elif isinstance(stats, Iterable):
        try:
            stats_dict = dict(stats)
        except Exception:
            stats_dict = None
    else:
        stats_dict = None

    if stats_dict is not None:
        for k, v in stats_dict.items():
            print(f"{indent}{k}: {v}")
    else:
        print(f"{indent}{stats}")


def _safe_tail_print(obj: Any, n: int = 5, indent: str = ""):
    if hasattr(obj, "tail"):
        try:
            print(obj.tail(n))
            return
        except Exception:
            pass
    print(f"{indent}{obj}")


# =========================
# Spread cost patching
# =========================


def _ensure_spread_costs(data, instruments, fallback_default: float):
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

    def patched_get_spread_cost(instrument_code: str) -> float:
        return used.get(instrument_code, fallback_default)

    scd.get_spread_cost = patched_get_spread_cost  # type: ignore[attr-defined]

    if missing:
        print(
            f"Spread costs missing or zero for: {', '.join(sorted(missing))}. "
            f"Using fallback={fallback_default}."
        )
    return used, sorted(missing)


# =========================
# Report object + Policies (ReportLab)
# =========================


@dataclass
class TablePolicy:
    # wrap column idx -> approx chars
    wrap_cols: dict[int, int] | None = None
    # alignment idx -> LEFT/RIGHT/CENTER
    align_cols: dict[int, str] | None = None
    font_size: int = 8
    # for correlation-like tables: max number of data columns per page chunk
    max_data_cols: int = 6
    # landscape threshold
    landscape_if_cols_gte: int = 9
    # first col width fraction bounds
    first_col_min_frac: float = 0.28
    first_col_max_frac: float = 0.55


@dataclass
class ReportBlock:
    kind: str  # table | image | pagebreak | heading
    payload: dict


@dataclass
class ReportSection:
    title: str
    blocks: list[ReportBlock]


@dataclass
class ReportDocument:
    title: str
    sections: list[ReportSection]


def _safe_register_dejavu() -> str:
    """Best-effort font registration. Falls back to Helvetica."""
    try:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf",
        ]
        for p in candidates:
            if Path(p).exists():
                pdfmetrics.registerFont(TTFont("DejaVuSans", p))
                return "DejaVuSans"
    except Exception:
        pass
    return "Helvetica"


def _wrap_paragraph(text: Any, style: ParagraphStyle) -> Paragraph:
    s = "" if text is None else str(text)
    s = s.replace("\n", "<br/>")
    return Paragraph(s, style)


def _estimate_first_col_fraction(
    headers: list, rows: list, min_frac: float, max_frac: float
) -> float:
    try:
        first_len = max([len(str(headers[0]))] + [len(str(r[0])) for r in rows if r])
        other_lens = []
        for j in range(1, len(headers)):
            col_len = max(
                [len(str(headers[j]))] + [len(str(r[j])) for r in rows if len(r) > j]
            )
            other_lens.append(col_len)
        if not other_lens:
            return min(max_frac, max(min_frac, 0.40))
        avg_other = max(6, int(sum(other_lens) / len(other_lens)))
        denom = first_len + (len(headers) - 1) * avg_other
        frac = (first_len / denom) if denom else 0.35
        frac = frac + 0.06
        return min(max_frac, max(min_frac, frac))
    except Exception:
        return min(max_frac, max(min_frac, 0.35))


def _build_col_widths_points(
    page_width_pts: float, margin_pts: float, ncols: int, first_frac: float
) -> list:
    usable = max(100, page_width_pts - 2 * margin_pts)
    if ncols <= 1:
        return [usable]
    first = usable * first_frac
    rest = usable - first
    per = rest / (ncols - 1)
    return [first] + [per] * (ncols - 1)


def _estimate_max_data_cols_from_width(
    page_width_pts: float,
    margin_pts: float,
    first_col_frac: float,
    *,
    min_data_col_width_pts: float = 48.0,
) -> int:
    usable = max(100.0, page_width_pts - 2 * margin_pts)
    first_w = usable * first_col_frac
    remaining = max(0.0, usable - first_w)
    return max(1, int(remaining // max(min_data_col_width_pts, 1.0)))


def _split_wide_table(headers: list, rows: list, max_data_cols: int) -> list[dict]:
    if not headers or not rows or len(headers) <= 2:
        return [{"headers": headers, "rows": rows, "page": 1, "total_pages": 1}]

    fixed_header = headers[0]
    data_headers = headers[1:]
    max_data_cols = max(1, int(max_data_cols))
    total_chunks = (len(data_headers) + max_data_cols - 1) // max_data_cols
    out = []
    for idx in range(total_chunks):
        start = idx * max_data_cols
        end = start + max_data_cols
        h = [fixed_header] + data_headers[start:end]
        r = [[row[0]] + row[1 + start : 1 + end] for row in rows]
        out.append(
            {"headers": h, "rows": r, "page": idx + 1, "total_pages": total_chunks}
        )
    return out


def _default_table_style(font_name: str, font_size: int) -> TableStyle:
    return TableStyle(
        [
            ("FONTNAME", (0, 0), (-1, -1), font_name),
            ("FONTSIZE", (0, 0), (-1, -1), font_size),
            ("LEADING", (0, 0), (-1, -1), font_size + 2),
            ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]
    )


def _apply_alignment(
    ts: TableStyle, align_cols: dict[int, str] | None, nrows: int, ncols: int
):
    if not align_cols:
        return
    for col, align in align_cols.items():
        if col < 0 or col >= ncols:
            continue
        a = str(align).upper().strip()
        if a not in ("LEFT", "RIGHT", "CENTER"):
            a = "LEFT"
        ts.add("ALIGN", (col, 0), (col, nrows - 1), a)


def _make_table_flowable(
    title: str,
    headers: list,
    rows: list,
    policy: TablePolicy,
    font_name: str,
    page_width_pts: float,
    margin_pts: float,
    first_col_frac: float | None = None,
) -> list:
    styles = getSampleStyleSheet()
    hstyle = styles["Heading3"]
    hstyle.fontName = font_name

    body = ParagraphStyle(
        name="Cell",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=policy.font_size,
        leading=policy.font_size + 2,
        spaceAfter=0,
        spaceBefore=0,
    )

    import textwrap

    wrap_cols = policy.wrap_cols or {}
    align_cols = policy.align_cols or {}

    first_frac = first_col_frac
    if first_frac is None:
        first_frac = _estimate_first_col_fraction(
            headers, rows, policy.first_col_min_frac, policy.first_col_max_frac
        )

    col_widths = _build_col_widths_points(
        page_width_pts, margin_pts, len(headers), first_frac
    )

    def _effective_wrap_chars(col_idx: int, base_chars: int) -> int:
        if col_idx >= len(col_widths):
            return base_chars
        try:
            approx_chars = int(col_widths[col_idx] / max(1.0, policy.font_size * 0.55))
            return max(base_chars, approx_chars)
        except Exception:
            return base_chars

    def cell_to_para(col_idx: int, value: Any) -> Paragraph:
        s = "" if value is None else str(value)
        if col_idx in wrap_cols:
            base_width_chars = int(wrap_cols[col_idx])
            width_chars = _effective_wrap_chars(col_idx, base_width_chars)
            if width_chars > 0 and len(s) > width_chars:
                lines = []
                for part in s.split("\n"):
                    # IMPORTANT: allow long tokens like accel.accel_calc_v fast_wrap to break
                    wrapped = textwrap.wrap(
                        part,
                        width=width_chars,
                        break_long_words=True,
                        break_on_hyphens=True,
                    )
                    lines.extend(wrapped or [""])
                s = "\n".join(lines)
        return _wrap_paragraph(s, body)

    # header row
    data = [[cell_to_para(j, h) for j, h in enumerate(headers)]]
    for r in rows:
        data.append(
            [cell_to_para(j, r[j] if j < len(r) else "") for j in range(len(headers))]
        )

    ncols = len(headers)
    nrows = len(data)

    t = Table(data, colWidths=col_widths, repeatRows=1)
    ts = _default_table_style(font_name=font_name, font_size=policy.font_size)
    _apply_alignment(ts, align_cols, nrows, ncols)
    t.setStyle(ts)

    return [Paragraph(f"<b>{title}</b>", hstyle), Spacer(1, 6), t, Spacer(1, 12)]


def _render_reportlab_pdf(output_path: Path, report: ReportDocument):
    font_name = _safe_register_dejavu()

    page_portrait = A4
    page_landscape = landscape(A4)
    margin = 1.2 * cm

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(font_name, 8)
        canvas.drawRightString(doc.pagesize[0] - margin, 0.8 * cm, f"Page {doc.page}")
        canvas.restoreState()

    doc = BaseDocTemplate(
        str(output_path),
        pagesize=page_portrait,
        leftMargin=margin,
        rightMargin=margin,
        topMargin=margin,
        bottomMargin=margin,
    )

    frame_p = Frame(
        doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="portrait_frame"
    )
    frame_l = Frame(
        margin,
        margin,
        page_landscape[0] - 2 * margin,
        page_landscape[1] - 2 * margin,
        id="landscape_frame",
    )

    tpl_p = PageTemplate(
        id="PORTRAIT", frames=[frame_p], onPage=footer, pagesize=page_portrait
    )
    tpl_l = PageTemplate(
        id="LANDSCAPE", frames=[frame_l], onPage=footer, pagesize=page_landscape
    )
    doc.addPageTemplates([tpl_p, tpl_l])

    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    title_style.fontName = font_name

    story = [Paragraph(report.title, title_style), Spacer(1, 12)]
    current_template = "PORTRAIT"

    def ensure_template(template_id: str):
        nonlocal current_template
        template_id = template_id.upper()
        if template_id == current_template:
            return
        story.append(NextPageTemplate(template_id))
        story.append(PageBreak())
        current_template = template_id

    for si, section in enumerate(report.sections):
        section_heading_rendered = False

        def render_section_heading():
            nonlocal section_heading_rendered
            if section_heading_rendered:
                return
            story.append(Paragraph(section.title, styles["Heading2"]))
            story.append(Spacer(1, 10))
            section_heading_rendered = True

        for blk in section.blocks:
            kind = blk.kind
            payload = blk.payload

            if kind == "pagebreak":
                render_section_heading()
                story.append(PageBreak())
                continue

            if kind == "heading":
                render_section_heading()
                story.append(Paragraph(payload["text"], styles["Heading3"]))
                story.append(Spacer(1, 6))
                continue

            if kind == "table":
                headers = payload["headers"]
                rows = payload["rows"]
                policy: TablePolicy = payload["policy"]
                tb_title = payload.get("title", "")

                ncols = len(headers) if headers else 0
                use_landscape = True
                target_template = "LANDSCAPE" if use_landscape else "PORTRAIT"
                ensure_template(target_template)
                render_section_heading()

                # correlation-like splitting
                chunks = payload.get("chunks")
                if chunks is None:
                    first_frac = _estimate_first_col_fraction(
                        headers,
                        rows,
                        policy.first_col_min_frac,
                        policy.first_col_max_frac,
                    )
                    pagesize = page_landscape if use_landscape else page_portrait
                    page_width_pts = pagesize[0]

                    width_based_cols = _estimate_max_data_cols_from_width(
                        page_width_pts, margin, first_frac, min_data_col_width_pts=46.0
                    )
                    effective_max_data_cols = max(
                        policy.max_data_cols, width_based_cols
                    )

                    if payload.get("split_wide", False) and ncols > (
                        effective_max_data_cols + 1
                    ):
                        chunks = _split_wide_table(
                            headers, rows, effective_max_data_cols
                        )
                    else:
                        chunks = [
                            {
                                "headers": headers,
                                "rows": rows,
                                "page": 1,
                                "total_pages": 1,
                            }
                        ]

                pagesize = page_landscape if use_landscape else page_portrait
                page_width_pts = pagesize[0]

                for ch in chunks:
                    ch_title = tb_title
                    if ch.get("total_pages", 1) > 1:
                        ch_title = (
                            f"{tb_title} (block {ch['page']}/{ch['total_pages']})"
                        )
                    ch_headers = ch["headers"]
                    ch_rows = ch["rows"]
                    ch_first_frac = _estimate_first_col_fraction(
                        ch_headers,
                        ch_rows,
                        policy.first_col_min_frac,
                        policy.first_col_max_frac,
                    )
                    story.extend(
                        _make_table_flowable(
                            title=ch_title,
                            headers=ch_headers,
                            rows=ch_rows,
                            policy=policy,
                            font_name=font_name,
                            page_width_pts=page_width_pts,
                            margin_pts=margin,
                            first_col_frac=ch_first_frac,
                        )
                    )
                continue

            if kind == "image":
                img_path = Path(payload["path"])
                if not img_path.exists():
                    continue
                use_landscape = bool(payload.get("landscape", True))
                ensure_template("LANDSCAPE" if use_landscape else "PORTRAIT")
                render_section_heading()

                pagesize = page_landscape if use_landscape else page_portrait
                max_w = pagesize[0] - 2 * margin
                max_h = pagesize[1] - 2 * margin - 1.0 * cm

                story.append(
                    RLImage(
                        str(img_path), width=max_w, height=max_h, kind="proportional"
                    )
                )
                story.append(Spacer(1, 12))
                continue

        # page break after section (avoid last one)
        if si < len(report.sections) - 1:
            story.append(PageBreak())

    doc.build(story)


def _build_report_document(
    *,
    summary_rows: list,
    per_inst_rows: list,
    per_rule_rows: list,
    cost_rows: list,
    notional_rows: list,
    trades_rows: list,
    figure_paths: list,
    per_inst_headers: Optional[list] = None,
    per_rule_headers: Optional[list] = None,
    rule_variation_rows: Optional[list] = None,
    rule_variation_headers: Optional[list] = None,
    rule_correlation_rows: Optional[list] = None,
    rule_correlation_headers: Optional[list] = None,
) -> ReportDocument:
    per_inst_labels = per_inst_headers or [
        "Name",
        "Total",
        "CAGR",
        "Vol",
        "Sharpe",
        "MaxDD",
    ]
    per_rule_labels = per_rule_headers or [
        "Name",
        "Total",
        "CAGR",
        "Vol",
        "Sharpe",
        "IC",
        "MaxDD",
    ]
    rule_variation_rows = rule_variation_rows or []
    rule_variation_labels = rule_variation_headers or per_rule_labels
    rule_correlation_rows = rule_correlation_rows or []
    rule_correlation_labels = rule_correlation_headers or []

    def numeric_right_align(headers: list) -> dict[int, str]:
        return {0: "LEFT", **{i: "RIGHT" for i in range(1, len(headers))}}

    sections: list[ReportSection] = []

    if summary_rows:
        sections.append(
            ReportSection(
                title="Portfolio summary",
                blocks=[
                    ReportBlock(
                        kind="table",
                        payload=dict(
                            title="Portfolio summary",
                            headers=["Metric", "Value", "UoM", "Description"],
                            rows=summary_rows,
                            policy=TablePolicy(
                                wrap_cols={0: 24, 3: 52},
                                align_cols={
                                    0: "LEFT",
                                    1: "RIGHT",
                                    2: "CENTER",
                                    3: "LEFT",
                                },
                                font_size=8,
                                landscape_if_cols_gte=99,
                                first_col_min_frac=0.26,
                                first_col_max_frac=0.40,
                            ),
                            landscape=False,
                        ),
                    )
                ],
            )
        )

    if per_inst_rows:
        sections.append(
            ReportSection(
                title="Performance",
                blocks=[
                    ReportBlock(
                        kind="table",
                        payload=dict(
                            title="Performance per instrument (net % returns)",
                            headers=per_inst_labels,
                            rows=per_inst_rows,
                            policy=TablePolicy(
                                wrap_cols={0: 18},
                                align_cols=numeric_right_align(per_inst_labels),
                                font_size=8,
                                landscape_if_cols_gte=10,
                                first_col_min_frac=0.30,
                                first_col_max_frac=0.55,
                            ),
                            landscape=False,
                        ),
                    )
                ],
            )
        )

    if per_rule_rows:
        sections.append(
            ReportSection(
                title="Rule performance",
                blocks=[
                    ReportBlock(
                        kind="table",
                        payload=dict(
                            title="Performance per rule (aggregated variations)",
                            headers=per_rule_labels,
                            rows=per_rule_rows,
                            policy=TablePolicy(
                                # this is one of the problematic tables in your sample:
                                # long names like 'cs_mm.cross_sectional_mean_reversion' need real wrap + row height
                                wrap_cols={0: 20, 1: 80},
                                align_cols=numeric_right_align(per_rule_labels),
                                font_size=8,
                                landscape_if_cols_gte=10,
                                first_col_min_frac=0.20,
                                first_col_max_frac=0.28,
                            ),
                            landscape=False,
                        ),
                    )
                ],
            )
        )

    if rule_variation_rows:
        sections.append(
            ReportSection(
                title="Rule variations",
                blocks=[
                    ReportBlock(
                        kind="table",
                        payload=dict(
                            title="Performance per rule variation",
                            headers=rule_variation_labels,
                            rows=rule_variation_rows,
                            policy=TablePolicy(
                                wrap_cols={0: 20},
                                align_cols=numeric_right_align(rule_variation_labels),
                                font_size=8,
                                landscape_if_cols_gte=10,
                                first_col_min_frac=0.20,
                                first_col_max_frac=0.66,
                            ),
                            landscape=False,
                        ),
                    )
                ],
            )
        )

    if cost_rows:
        sections.append(
            ReportSection(
                title="Costs",
                blocks=[
                    ReportBlock(
                        kind="table",
                        payload=dict(
                            title="Spread costs used",
                            headers=["Instrument", "Source", "Spread"],
                            rows=cost_rows,
                            policy=TablePolicy(
                                wrap_cols={0: 24},
                                align_cols={0: "LEFT", 1: "CENTER", 2: "RIGHT"},
                                font_size=8,
                                landscape_if_cols_gte=99,
                                first_col_min_frac=0.38,
                                first_col_max_frac=0.58,
                            ),
                            landscape=False,
                        ),
                    )
                ],
            )
        )

    if notional_rows:
        sections.append(
            ReportSection(
                title="Positions",
                blocks=[
                    ReportBlock(
                        kind="table",
                        payload=dict(
                            title="Notional positions (annual average)",
                            headers=["Instrument", "Year", "Avg notional"],
                            rows=notional_rows,
                            policy=TablePolicy(
                                wrap_cols={0: 24},
                                align_cols={0: "LEFT", 1: "CENTER", 2: "RIGHT"},
                                font_size=8,
                                landscape_if_cols_gte=99,
                                first_col_min_frac=0.40,
                                first_col_max_frac=0.60,
                            ),
                            landscape=False,
                        ),
                    )
                ],
            )
        )

    if trades_rows:
        headers = [
            "Date",
            "Instrument",
            "Trade",
            "New pos",
            "Pos (base)",
            "Spread",
            "Commission",
            "Total cost",
        ]
        sections.append(
            ReportSection(
                title="Trades",
                blocks=[
                    ReportBlock(
                        kind="table",
                        payload=dict(
                            title="Executed trades (from buffered positions)",
                            headers=headers,
                            rows=trades_rows,
                            policy=TablePolicy(
                                wrap_cols={1: 14, 4: 18},
                                align_cols={
                                    0: "LEFT",
                                    1: "LEFT",
                                    2: "RIGHT",
                                    3: "RIGHT",
                                    4: "RIGHT",
                                    5: "RIGHT",
                                    6: "RIGHT",
                                    7: "RIGHT",
                                },
                                font_size=7,
                                landscape_if_cols_gte=7,
                                first_col_min_frac=0.14,
                                first_col_max_frac=0.20,
                            ),
                            landscape=True,
                        ),
                    )
                ],
            )
        )

    if rule_correlation_rows and rule_correlation_labels:
        sections.append(
            ReportSection(
                title="Correlations",
                blocks=[
                    ReportBlock(
                        kind="table",
                        payload=dict(
                            title="Correlation between strategies/rules",
                            headers=rule_correlation_labels,
                            rows=rule_correlation_rows,
                            split_wide=True,
                            policy=TablePolicy(
                                wrap_cols={0: 24},
                                align_cols={
                                    0: "LEFT",
                                    **{
                                        i: "RIGHT"
                                        for i in range(1, len(rule_correlation_labels))
                                    },
                                },
                                font_size=7,
                                max_data_cols=12,
                                landscape_if_cols_gte=4,
                                first_col_min_frac=0.34,
                                first_col_max_frac=0.62,
                            ),
                            landscape=True,
                        ),
                    )
                ],
            )
        )

    fig_blocks: list[ReportBlock] = []
    for p in figure_paths or []:
        pp = Path(p)
        if pp.exists():
            fig_blocks.append(
                ReportBlock(
                    kind="image",
                    payload={"path": str(pp), "caption": pp.name, "landscape": True},
                )
            )
    if fig_blocks:
        sections.append(ReportSection(title="Figures", blocks=fig_blocks))

    return ReportDocument(title="pysystemtrade backtest report", sections=sections)


def _build_unified_pdf(
    output_path: Path,
    summary_rows: list,
    per_inst_rows: list,
    per_rule_rows: list,
    cost_rows: list,
    notional_rows: list,
    trades_rows: list,
    figure_paths: list,
    per_inst_headers: Optional[list] = None,
    per_rule_headers: Optional[list] = None,
    rule_variation_rows: Optional[list] = None,
    rule_variation_headers: Optional[list] = None,
    rule_correlation_rows: Optional[list] = None,
    rule_correlation_headers: Optional[list] = None,
):
    report = _build_report_document(
        summary_rows=summary_rows,
        per_inst_rows=per_inst_rows,
        per_rule_rows=per_rule_rows,
        cost_rows=cost_rows,
        notional_rows=notional_rows,
        trades_rows=trades_rows,
        figure_paths=figure_paths,
        per_inst_headers=per_inst_headers,
        per_rule_headers=per_rule_headers,
        rule_variation_rows=rule_variation_rows,
        rule_variation_headers=rule_variation_headers,
        rule_correlation_rows=rule_correlation_rows,
        rule_correlation_headers=rule_correlation_headers,
    )
    _render_reportlab_pdf(output_path, report)


# =========================
# Cache helpers
# =========================


def _alternate_cache_path(path: Path) -> Optional[Path]:
    suffix = path.suffix.lower()
    if suffix == ".pckz":
        alt = path.with_suffix(".pck")
    elif suffix == ".pck":
        alt = path.with_suffix(".pckz")
    else:
        return None
    return alt if alt.exists() else None


def _infer_cache_compress(path: Path, default: bool) -> bool:
    suffix = path.suffix.lower()
    if suffix == ".pckz":
        return True
    if suffix == ".pck":
        return False
    return default


def _estimated_attr_names_from_config(config_obj: Config) -> list:
    names = []
    if getattr(config_obj, "use_instrument_weight_estimates", False):
        names.append("instrument_weights")
    if getattr(config_obj, "use_instrument_div_mult_estimates", False):
        names.append("instrument_div_multiplier")
    if getattr(config_obj, "use_forecast_weight_estimates", False):
        names.append("forecast_weights")
    if getattr(config_obj, "use_forecast_div_mult_estimates", False):
        names.append("forecast_div_multiplier")
    if getattr(config_obj, "use_forecast_scale_estimates", False):
        names.append("forecast_scalars")
    return names


def _resolve_capital_multiplier(arg: str) -> str:
    aliases = {
        "fixed": f"{fixed_capital.__module__}.{fixed_capital.__name__}",
        "full": f"{full_compounding.__module__}.{full_compounding.__name__}",
        "half": f"{half_compounding.__module__}.{half_compounding.__name__}",
    }
    return aliases.get(arg.strip().lower(), arg)


# =========================
# Runner
# =========================


def _filter_instruments(
    instruments: Iterable[str], desired: Optional[Sequence[str]]
) -> list:
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
    """Drive the DB backtest or IC-only flow and build the requested reports."""
    cfg = backtest_config.with_defaults()
    logger = logging.getLogger(__name__)
    results_dir = cfg.results_dir
    timestamp = cfg.timestamp

    if cfg.ic_only:
        cfg.include_plots = False
        cfg.include_quantstats = False
        if not cfg.use_cache:
            msg = "IC-only run requested; enabling cache loading to reuse a previous execution."
            print(msg)
            logger.info(msg)
            cfg.use_cache = True

    log_path = (
        (results_dir / f"backtest_output_{timestamp}.log")
        if cfg.include_debug_log
        else None
    )

    with _tee_output(log_path):
        results_dir.mkdir(parents=True, exist_ok=True)
        _set_matplotlib_font_defaults()
        data = data_factory()

        config_obj = Config(str(cfg.config_path))

        if cfg.use_db_capital:
            strategy_name = cfg.strategy_name or cfg.config_path.stem
            capital_data_source = (
                data if hasattr(data, "add_class_object") else dataBlob()
            )
            try:
                capital_data = dataCapital(capital_data_source)
                notional_trading_capital = (
                    capital_data.get_current_capital_for_strategy(strategy_name)
                )
            except missingData as err:
                raise Exception(
                    f"Capital data is missing for strategy '{strategy_name}': can't run backtest"
                ) from err
            base_currency = dataCurrency(capital_data_source).get_base_currency()
            config_obj.notional_trading_capital = notional_trading_capital
            config_obj.base_currency = base_currency
            msg = f"Using DB capital for {strategy_name}: {base_currency} {notional_trading_capital:,.2f}"
            print(msg)
            logger.info(msg)

        if cfg.capital_multiplier:
            cm_func = _resolve_capital_multiplier(cfg.capital_multiplier)
            config_obj.capital_multiplier = {"func": cm_func}
            logger.info(f"Applied capital multiplier override: {cm_func}")

        config_instruments = list(getattr(config_obj, "instruments", []) or [])
        weights_dict = getattr(config_obj, "instrument_weights", {}) or {}
        weights_instruments = list(weights_dict.keys())
        all_instruments = list(data.get_instrument_list())

        base_universe = (
            config_instruments
            if config_instruments
            else (weights_instruments or all_instruments)
        )
        instrument_list = _filter_instruments(base_universe, cfg.instrument_filter)
        if cfg.instrument_filter and not instrument_list:
            print("Instrument filter empty: using base instrument universe.")
            instrument_list = base_universe

        if cfg.ic_only:
            spread_costs_used = {}
            spread_costs_missing = []
        else:
            spread_costs_used, spread_costs_missing = _ensure_spread_costs(
                data, instrument_list, fallback_default=cfg.fallback_spread
            )

        system = system_factory(data=data, config=config_obj)

        cache_loaded = False
        cache_items_loaded = 0
        cache_path = cfg.cache_file if cfg.use_cache else None
        if cfg.use_cache and cache_path is not None:
            effective_path = (
                cache_path if cache_path.exists() else _alternate_cache_path(cache_path)
            )
            if effective_path is None:
                msg = f"No cache found at {cache_path}; building results from scratch."
                print(msg)
                logger.info(msg)
            else:
                try:
                    system.config.backtest_compress = _infer_cache_compress(
                        effective_path, cfg.cache_compress
                    )
                except Exception:
                    pass
                try:
                    system.cache.unpickle(str(effective_path))
                    cache_loaded = True
                    cache_path = effective_path
                    try:
                        cache_items_loaded = len(system.cache.get_items_with_data())
                    except Exception:
                        cache_items_loaded = 0
                    msg = f"Loaded cached system state from {effective_path} ({cache_items_loaded} cached items)."
                    print(msg)
                    logger.info(msg)
                except Exception as err:
                    msg = f"Cache load failed from {effective_path} ({err}); continuing without cache."
                    print(msg)
                    logger.warning(msg)

        if cfg.use_cache and cache_loaded and cache_items_loaded == 0:
            msg = "Cache loaded but contained 0 items; computations will run from scratch."
            print(msg)
            logger.info(msg)

        portfolio = system.accounts.portfolio()
        instruments_for_output = instrument_list or system.get_instrument_list()

        if cfg.ic_only:
            print(f"Instruments: {', '.join(instruments_for_output)}")
            print("\nIC-only run; collecting information coefficient metrics.")
            stats = None
            curve = drawdown = rolling_std = None
            base_currency = getattr(config_obj, "base_currency", "") or ""
            per_inst_rows = []
            per_inst_headers = None
        else:
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

            base_currency = getattr(config_obj, "base_currency", "") or ""

            print("\nPerformance per instrument (net % returns):")
            per_inst_rows, per_inst_headers = _print_per_instrument_perf(
                portfolio, instrument_filter=instrument_list
            )

        try:
            rule_perf_tables = _collect_rule_performance_tables(
                system, portfolio, base_currency=base_currency, verbose=True
            )
        except Exception as err:
            print(f"Rule-level performance skipped ({err})")
            rule_perf_tables = {
                "group_rows": [],
                "group_headers": [],
                "variation_rows": [],
                "variation_headers": [],
                "correlation_rows": [],
                "correlation_headers": [],
                "group_curves": {},
            }

        per_rule_rows = rule_perf_tables["group_rows"]
        per_rule_headers = rule_perf_tables["group_headers"]
        rule_variation_rows = rule_perf_tables["variation_rows"]
        rule_variation_headers = rule_perf_tables["variation_headers"]
        rule_correlation_rows = rule_perf_tables["correlation_rows"]
        rule_correlation_headers = rule_perf_tables["correlation_headers"]
        rule_group_curves = rule_perf_tables["group_curves"]

        figures: dict[str, Path] = {}
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
            if rule_group_curves:
                figures["rule_groups"] = results_dir / f"rule_groups_{timestamp}.png"

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
            if rule_group_curves:
                _plot_multiple_series(
                    rule_group_curves,
                    figures["rule_groups"],
                    title="P&L for trading rule (aggregated variations)",
                    ylabel=f"P&L {base_currency}".strip(),
                )

        qs_html = (
            (results_dir / f"backtest_report_qs_{timestamp}.html")
            if cfg.include_quantstats
            else None
        )
        if cfg.include_quantstats:
            _quantstats_report(portfolio.percent, qs_html)

        per_inst_rows = (
            sorted(per_inst_rows, key=lambda r: r[0]) if per_inst_rows else []
        )
        per_rule_rows = (
            sorted(per_rule_rows, key=lambda r: r[0]) if per_rule_rows else []
        )
        rule_variation_rows = (
            sorted(rule_variation_rows, key=lambda r: r[0])
            if rule_variation_rows
            else []
        )

        per_inst_headers = per_inst_headers or [
            "Name",
            "Total",
            "CAGR",
            "Vol",
            "Sharpe",
            "MaxDD",
        ]
        per_rule_headers = per_rule_headers or [
            "Name",
            "P&L",
            "Total",
            "CAGR",
            "Vol",
            "Sharpe",
            "IC",
            "MaxDD",
            "t-stat",
            "p-value",
        ]
        rule_variation_headers = rule_variation_headers or per_rule_headers

        if cfg.ic_only:
            summary_rows = []
            cost_rows = []
            notional_rows = []
            trades_rows = []
        else:
            cost_rows = _build_spread_cost_rows(
                spread_costs_used, spread_costs_missing, instruments_for_output
            )
            notional_rows = _collect_notional_positions_by_year(
                system, instruments_for_output
            )
            trades_rows = _collect_trades(
                system,
                instrument_filter=instruments_for_output,
                max_rows=200,
                base_currency=base_currency,
            )
            summary_rows = _build_summary_rows(
                stats, portfolio, base_currency=base_currency
            )

        report_prefix = "ic_report" if cfg.ic_only else "backtest_report"
        pdf_path = (
            (results_dir / f"{report_prefix}_{timestamp}.pdf")
            if cfg.include_pdf
            else None
        )
        report_txt_path = (
            (results_dir / f"{report_prefix}_{timestamp}.txt")
            if cfg.include_report_txt
            else None
        )

        if cfg.include_pdf and pdf_path is not None:
            _build_unified_pdf(
                output_path=pdf_path,
                summary_rows=summary_rows,
                per_inst_rows=per_inst_rows,
                per_rule_rows=per_rule_rows,
                cost_rows=cost_rows,
                notional_rows=notional_rows,
                trades_rows=trades_rows,
                figure_paths=list(figures.values()),
                per_inst_headers=per_inst_headers,
                per_rule_headers=per_rule_headers,
                rule_variation_rows=rule_variation_rows,
                rule_variation_headers=rule_variation_headers,
                rule_correlation_rows=rule_correlation_rows or [],
                rule_correlation_headers=rule_correlation_headers or [],
            )
            print(f"Saved unified report: {pdf_path}")

        if cfg.include_report_txt and report_txt_path is not None:
            _write_report_txt(
                output_path=report_txt_path,
                summary_rows=summary_rows,
                per_inst_rows=per_inst_rows,
                per_rule_rows=per_rule_rows,
                cost_rows=cost_rows,
                notional_rows=notional_rows,
                trades_rows=trades_rows,
                per_inst_headers=per_inst_headers,
                per_rule_headers=per_rule_headers,
                rule_variation_rows=rule_variation_rows,
                rule_variation_headers=rule_variation_headers,
                rule_correlation_rows=rule_correlation_rows or [],
                rule_correlation_headers=rule_correlation_headers or [],
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

        estimates_yaml = None
        if not cfg.ic_only:
            estimated_names = _estimated_attr_names_from_config(config_obj)
            if cfg.export_estimates and estimated_names:
                estimates_yaml = (
                    results_dir
                    / f"{cfg.config_path.stem}_estimated_params_{timestamp}.yaml"
                )
                try:
                    systemDiag(system).yaml_config_with_estimated_parameters(
                        str(estimates_yaml),
                        attr_names=estimated_names,
                    )
                    print(
                        f"Saved estimated parameters ({', '.join(estimated_names)}) to {estimates_yaml}"
                    )
                except Exception as err:
                    print(f"Estimated parameters export skipped ({err})")
                    estimates_yaml = None

        outputs = BacktestOutputs(
            results_dir=results_dir,
            figures=figures_for_output,
            quantstats_report=qs_html,
            pdf_report=pdf_path,
            report_txt=report_txt_path,
            debug_log=log_path,
            estimates_yaml=estimates_yaml,
        )

        if cfg.use_cache and cache_path is not None:
            try:
                try:
                    system.config.backtest_compress = _infer_cache_compress(
                        cache_path, cfg.cache_compress
                    )
                except Exception:
                    pass
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                system.cache.pickle(str(cache_path))
                verb = "Updated" if cache_loaded else "Saved"
                print(f"{verb} system cache at {cache_path}")
            except Exception as err:
                print(f"Cache save skipped ({err})")

        return BacktestResult(
            system=system,
            summary_rows=summary_rows,
            per_inst_rows=per_inst_rows,
            per_rule_rows=per_rule_rows,
            per_rule_headers=per_rule_headers,
            rule_variation_rows=rule_variation_rows,
            rule_variation_headers=rule_variation_headers,
            rule_correlation_rows=rule_correlation_rows or [],
            rule_correlation_headers=rule_correlation_headers or [],
            per_inst_headers=per_inst_headers,
            cost_rows=cost_rows,
            notional_rows=notional_rows,
            trades_rows=trades_rows,
            outputs=outputs,
        )


# =========================
# Text report
# =========================


def _write_report_txt(
    output_path: Path,
    summary_rows: list,
    per_inst_rows: list,
    per_rule_rows: list,
    cost_rows: list,
    notional_rows: list,
    trades_rows: list,
    per_inst_headers: Optional[list] = None,
    per_rule_headers: Optional[list] = None,
    rule_variation_rows: Optional[list] = None,
    rule_variation_headers: Optional[list] = None,
    rule_correlation_rows: Optional[list] = None,
    rule_correlation_headers: Optional[list] = None,
):
    sections = []

    def add_table(title: str, rows: list, headers: list):
        if not rows:
            return
        sections.append(title)
        sections.append(" | ".join(headers))
        sections.append("-" * 80)
        for row in rows:
            sections.append(" | ".join([str(x) for x in row]))
        sections.append("")

    per_inst_labels = per_inst_headers or [
        "Name",
        "Total",
        "CAGR",
        "Vol",
        "Sharpe",
        "MaxDD",
    ]
    per_rule_labels = per_rule_headers or [
        "Name",
        "Total",
        "CAGR",
        "Vol",
        "Sharpe",
        "IC",
        "MaxDD",
    ]
    rule_variation_rows = rule_variation_rows or []
    rule_variation_labels = rule_variation_headers or per_rule_labels
    rule_correlation_rows = rule_correlation_rows or []
    rule_correlation_labels = rule_correlation_headers or []

    add_table("Summary", summary_rows, ["Metric", "Value", "UoM", "Description"])
    add_table("Performance per instrument", per_inst_rows, per_inst_labels)
    add_table(
        "Performance per rule (aggregated variations)", per_rule_rows, per_rule_labels
    )
    add_table(
        "Performance per rule variation", rule_variation_rows, rule_variation_labels
    )
    add_table(
        "Correlation between strategies/rules",
        rule_correlation_rows,
        rule_correlation_labels,
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

    if not sections:
        return
    try:
        output_path.write_text("\n".join(sections), encoding="utf-8")
        print(f"Saved text summary: {output_path}")
    except Exception as err:
        print(f"Text summary write skipped ({err})")


# =========================
# CLI
# =========================


def parse_args(argv=None) -> BacktestConfig:
    parser = argparse.ArgumentParser(
        description="Run a futures backtest using DB data (Mongo + Parquet)."
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
        "--use-db-capital",
        action="store_true",
        help="Pull capital and base currency from DB instead of YAML.",
    )

    parser.add_argument(
        "--capital-multiplier",
        default=None,
        help="Override capital compounding: fixed / full / half, or dotted function path.",
    )

    parser.add_argument(
        "--strategy-name",
        default=None,
        help="Strategy name for DB capital lookup (default: config filename stem).",
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
        "--no-report",
        dest="no_report",
        action="store_true",
        help="Disable the txt summary output.",
    )

    parser.add_argument(
        "--no-debug-log",
        action="store_true",
        help="Disable the terminal output .log file.",
    )
    parser.add_argument(
        "--keep-intermediate-figs", action="store_true", help="Keep PNG figures."
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Load/save the system cache to speed up reruns.",
    )

    parser.add_argument(
        "--cache-file", default=None, help="Custom path for the system cache pickle."
    )
    parser.add_argument(
        "--no-cache-compress",
        action="store_true",
        help="Disable backtest_compress before pickling.",
    )
    parser.add_argument(
        "--ic-only",
        action="store_true",
        help="Only compute information coefficient metrics and emit a reduced report (prefers an existing cache).",
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
        include_report_txt=not args.no_report,
        include_debug_log=not args.no_debug_log,
        keep_intermediate_figs=args.keep_intermediate_figs,
        use_cache=args.cache,
        cache_file=Path(args.cache_file) if args.cache_file else None,
        cache_compress=not args.no_cache_compress,
        use_db_capital=args.use_db_capital,
        strategy_name=args.strategy_name,
        capital_multiplier=args.capital_multiplier,
        ic_only=args.ic_only,
    )


if __name__ == "__main__":
    cli_config = parse_args()
    run_backtest(cli_config)
