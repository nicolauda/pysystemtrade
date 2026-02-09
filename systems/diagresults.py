from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


def _as_tuple_str(values: Any) -> tuple[str, ...]:
    """Coerce an iterable of values into a tuple of strings."""
    if values is None:
        return ()
    return tuple(str(v) for v in values)


def _safe_float(value: Any) -> float:
    """Best-effort float conversion returning NaN for non-finite/unparseable."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return x if np.isfinite(x) else float("nan")


def _t_stat_1samp_zero(x: pd.Series) -> float:
    """Compute a simple 1-sample t-stat for mean(x) vs 0 with NaN handling."""
    vals = pd.to_numeric(x, errors="coerce").dropna().to_numpy(dtype=float)
    n = int(vals.size)
    if n < 2:
        return float("nan")
    std = float(vals.std(ddof=1))
    if std <= 0.0 or not np.isfinite(std):
        return float("nan")
    mean = float(vals.mean())
    return float(mean / (std / np.sqrt(n)))


def _t_stat_hac_zero(x: pd.Series, *, lags: int) -> float:
    """Compute a Newey–West (HAC) t-stat for mean(x) vs 0.

    This is a heteroskedasticity- and autocorrelation-consistent (HAC) variant
    of the naive 1-sample t-stat for time-series data. It replaces the IID
    standard error of the sample mean with a Newey–West long-run variance
    estimate using Bartlett weights.

    Args:
        x: Time series of observations.
        lags: Truncation lag L for the Newey–West estimator. Values <= 0 fall
            back to the naive t-stat.

    Returns:
        HAC-adjusted t-stat, or NaN if insufficient data.
    """
    if lags <= 0:
        return _t_stat_1samp_zero(x)

    vals = pd.to_numeric(x, errors="coerce").dropna().to_numpy(dtype=float)
    n = int(vals.size)
    if n < 2:
        return float("nan")

    L = int(min(lags, n - 1))
    u = vals - float(vals.mean())

    # gamma_k = (1/n) * sum_{t=k}^{n-1} u[t] * u[t-k]
    gamma0 = float(np.dot(u, u) / n)
    if not np.isfinite(gamma0) or gamma0 <= 0.0:
        return float("nan")

    lrv = gamma0
    for k in range(1, L + 1):
        w = 1.0 - (k / (L + 1.0))  # Bartlett kernel
        gamma_k = float(np.dot(u[k:], u[:-k]) / n)
        lrv += 2.0 * w * gamma_k

    if not np.isfinite(lrv) or lrv <= 0.0:
        return float("nan")

    var_mean = lrv / n
    if not np.isfinite(var_mean) or var_mean <= 0.0:
        return float("nan")

    return float(vals.mean() / np.sqrt(var_mean))


def _hitrate_positive(x: pd.Series) -> float:
    """Fraction of finite observations strictly greater than 0 (NaN -> ignored)."""
    vals = pd.to_numeric(x, errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size == 0:
        return float("nan")
    return float(np.mean(vals > 0.0))


@dataclass(frozen=True, slots=True)
class CrossSectionalICGridResult:
    """Cross-sectional IC results over a horizon x rule grid.

    This wraps the raw output of `systemDiag.get_cross_sectional_ic`:
    `ic[horizon][rule] -> pd.Series(date -> IC)`.
    """

    rules: tuple[str, ...]
    ic_by_horizon: dict[int, dict[str, pd.Series]]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_diagoutput(
        cls,
        rules_list: list[str],
        ic_by_horizon: dict[int, dict[str, pd.Series]],
        **metadata: Any,
    ) -> "CrossSectionalICGridResult":
        """Build from the raw `(rules_list, ic_by_horizon)` diagoutput return."""
        return cls(
            rules=_as_tuple_str(rules_list),
            ic_by_horizon=dict(ic_by_horizon),
            metadata=dict(metadata),
        )

    @property
    def horizons(self) -> tuple[int, ...]:
        """Sorted available horizons."""
        try:
            return tuple(sorted(int(h) for h in self.ic_by_horizon.keys()))
        except Exception:
            return tuple(self.ic_by_horizon.keys())

    def nearest_horizon(self, target: float) -> int | None:
        """Return the available horizon closest to `target`, or None if missing."""
        horizons = self.horizons
        if len(horizons) == 0:
            return None
        target_f = _safe_float(target)
        if not np.isfinite(target_f):
            return None
        return min(horizons, key=lambda h: abs(float(h) - target_f))

    def ic_series(self, *, horizon: int, rule: str) -> pd.Series:
        """Return the IC time series for a single (horizon, rule) pair."""
        series = self.ic_by_horizon.get(int(horizon), {}).get(str(rule))
        if series is None:
            return pd.Series(dtype=float)
        return series

    def t_stat_naive(self, *, horizon: int, rule: str) -> float:
        """Naive (IID) t-stat for the IC mean at a given (horizon, rule)."""
        s = self.ic_series(horizon=horizon, rule=rule)
        return _t_stat_1samp_zero(s)

    def t_stat_hac(self, *, horizon: int, rule: str, lags: int | None = None) -> float:
        """HAC (Newey–West) t-stat for the IC mean at a given (horizon, rule).

        Args:
            horizon: Forward-return horizon (days) used to compute IC.
            rule: Trading rule name.
            lags: Newey–West truncation lag. If None, uses the convention
                `max(0, horizon - 1)` and caps it to the available sample size.
        """
        s = self.ic_series(horizon=horizon, rule=rule)
        n_obs = int(pd.to_numeric(s, errors="coerce").dropna().shape[0])
        if lags is None:
            lags_use = max(0, int(horizon) - 1)
        else:
            lags_use = int(lags)
        lags_use = min(lags_use, max(0, n_obs - 1))
        return _t_stat_hac_zero(s, lags=lags_use)

    def ic_frame_for_horizon(
        self, horizon: int, *, rules: list[str] | None = None
    ) -> pd.DataFrame:
        """Return a DataFrame of IC series for one horizon (columns = rules)."""
        rules_use = _as_tuple_str(rules) if rules is not None else self.rules
        data = {rule: self.ic_series(horizon=horizon, rule=rule) for rule in rules_use}
        if len(data) == 0:
            return pd.DataFrame()
        return pd.DataFrame(data).sort_index()

    def mean_ic_by_horizon(self, *, rules: list[str] | None = None) -> pd.DataFrame:
        """Return mean IC per (horizon, rule). Index = horizon."""
        rules_use = _as_tuple_str(rules) if rules is not None else self.rules
        rows: list[dict[str, Any]] = []
        for h in self.horizons:
            row: dict[str, Any] = {"horizon": int(h)}
            for rule in rules_use:
                row[str(rule)] = _safe_float(
                    self.ic_series(horizon=h, rule=rule).mean()
                )
            rows.append(row)
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows).set_index("horizon").sort_index()
        return df

    def std_ic_by_horizon(self, *, rules: list[str] | None = None) -> pd.DataFrame:
        """Return IC std dev per (horizon, rule). Index = horizon."""
        rules_use = _as_tuple_str(rules) if rules is not None else self.rules
        rows: list[dict[str, Any]] = []
        for h in self.horizons:
            row: dict[str, Any] = {"horizon": int(h)}
            for rule in rules_use:
                row[str(rule)] = _safe_float(
                    self.ic_series(horizon=h, rule=rule).std(ddof=1)
                )
            rows.append(row)
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows).set_index("horizon").sort_index()
        return df

    def summary_for_horizon(self, horizon: int) -> pd.DataFrame:
        """Per-rule summary for a single horizon.

        Includes two t-stats for the IC mean:
        - `t_stat_naive`: assumes IID observations.
        - `t_stat_hac`: Newey–West (HAC) adjusted, using the convention
          `hac_lags = max(0, horizon - 1)` capped to the available sample size.

        The `horizon - 1` choice is a pragmatic default when the forward returns
        used to compute IC are overlapping h-day returns (common in daily data),
        which typically induces autocorrelation up to roughly h-1 lags.
        """
        rows: list[dict[str, Any]] = []
        for rule in self.rules:
            s = self.ic_series(horizon=horizon, rule=rule)
            s_clean = pd.to_numeric(s, errors="coerce")
            n_obs = int(s_clean.dropna().shape[0])
            hac_lags = max(0, int(horizon) - 1)
            hac_lags = min(hac_lags, max(0, n_obs - 1))
            rows.append(
                {
                    "rule": rule,
                    "mean": _safe_float(s_clean.mean()),
                    "std": _safe_float(s_clean.std(ddof=1)),
                    "t_stat_naive": _t_stat_1samp_zero(s_clean),
                    "t_stat_hac": _t_stat_hac_zero(s_clean, lags=hac_lags),
                    "hac_lags": int(hac_lags),
                    "n_obs": n_obs,
                    "hitrate_pos": _hitrate_positive(s_clean),
                }
            )
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).set_index("rule")

    def plot_mean_ic_by_horizon(
        self,
        *,
        rules: list[str] | None = None,
        show_std: bool = False,
        ax=None,
        title: str | None = None,
    ):
        """Plot mean IC by horizon for each rule.
        Mirrors the notebook helper used in `strategy_builder/workbench.ipynb`.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        rules_use = (
            list(_as_tuple_str(rules)) if rules is not None else list(self.rules)
        )
        mean_df = self.mean_ic_by_horizon(rules=rules_use)
        if mean_df.empty:
            ax.set_title(title or "Mean IC by horizon")
            return ax

        if show_std:
            std_df = self.std_ic_by_horizon(rules=rules_use)

        horizons = mean_df.index.to_numpy(dtype=float)
        for rule in rules_use:
            ic_mean = mean_df[rule].to_numpy(dtype=float)
            ax.plot(horizons, ic_mean, label=rule)
            if show_std and rule in std_df.columns:
                ic_std = std_df[rule].to_numpy(dtype=float)
                ax.fill_between(
                    horizons,
                    ic_mean - ic_std,
                    ic_mean + ic_std,
                    alpha=0.2,
                )

        ax.axhline(0.0, linestyle="--", linewidth=1, color="r")
        ax.set_xlabel("Horizon (days)")
        ax.set_ylabel("Mean IC")
        ax.grid(True, alpha=0.2)
        ax.legend()
        ax.set_title(title or "Mean IC by horizon")
        return ax


@dataclass(frozen=True, slots=True)
class ForecastPersistenceResult:
    """Forecast persistence results for multiple rules.

    Wraps the raw output of `systemDiag.get_persistence_function`:
    `persistence_by_rule[rule] -> pd.Series(lag -> rho)`,
    `summary_by_rule[rule] -> dict` (half_life_abs, half_life_rel, rho1, ...).
    """

    persistence_by_rule: dict[str, pd.Series]
    summary_by_rule: dict[str, dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_diagoutput(
        cls,
        persistence_by_rule: dict[str, pd.Series],
        summary_by_rule: dict[str, dict[str, Any]],
        **metadata: Any,
    ) -> "ForecastPersistenceResult":
        """Build from the raw `(persistence_by_rule, summary_by_rule)` return."""
        return cls(
            persistence_by_rule=dict(persistence_by_rule),
            summary_by_rule=dict(summary_by_rule),
            metadata=dict(metadata),
        )

    @property
    def rules(self) -> tuple[str, ...]:
        """Sorted rule names present in the result."""
        return tuple(sorted(self.persistence_by_rule.keys()))

    def persistence_series(self, rule: str) -> pd.Series:
        """Return the persistence curve for a single rule (index = lag)."""
        series = self.persistence_by_rule.get(str(rule))
        if series is None:
            return pd.Series(dtype=float)
        return series

    def half_life_abs(self, rule: str) -> float:
        """Absolute half-life (first crossing of `half_life_level`) for a rule."""
        return _safe_float(self.summary_by_rule.get(str(rule), {}).get("half_life_abs"))

    def summary_table(self) -> pd.DataFrame:
        """Return the per-rule persistence summary as a DataFrame."""
        if not self.summary_by_rule:
            return pd.DataFrame()
        df = pd.DataFrame(self.summary_by_rule).T
        df.index.name = "rule"
        return df.sort_index()

    def persistence_frame(self, *, rules: list[str] | None = None) -> pd.DataFrame:
        """Return a DataFrame of persistence curves (columns = rules)."""
        rules_use = _as_tuple_str(rules) if rules is not None else self.rules
        data = {rule: self.persistence_series(rule) for rule in rules_use}
        if len(data) == 0:
            return pd.DataFrame()
        return pd.DataFrame(data).sort_index()

    def plot_persistence_function_for_rules(
        self,
        *,
        rules: list[str] | None = None,
        ax=None,
        title: str | None = None,
    ):
        """Plot persistence curves for rules and annotate half-life on the x-axis.

        Mirrors the notebook helper used in `strategy_builder/workbench.ipynb`.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        rules_use = (
            list(_as_tuple_str(rules)) if rules is not None else list(self.rules)
        )
        if not rules_use:
            ax.set_title(title or "Persistence function")
            return ax

        ax.set_title(title or "Persistence function and HF")
        for rule in rules_use:
            p = self.persistence_series(rule)
            hl = self.half_life_abs(rule)
            ax.plot(p, label=f"{rule} - HF = {hl:.3g}" if np.isfinite(hl) else rule)
            if np.isfinite(hl) and not p.empty:
                ax.vlines(
                    float(hl),
                    ymin=float(np.nanmin(p.to_numpy(dtype=float))),
                    ymax=0.5,
                    linestyle=":",
                    linewidth=2,
                )
        ax.legend()
        ax.grid(True, alpha=0.2)
        return ax


@dataclass(frozen=True, slots=True)
class RuleDiagnostics:
    """Joined IC + persistence diagnostics for a single rule."""

    rule: str
    ic_by_horizon: dict[int, pd.Series]
    persistence: pd.Series
    persistence_summary: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def half_life_abs(self) -> float:
        """Absolute half-life for this rule's persistence curve."""
        return _safe_float(self.persistence_summary.get("half_life_abs"))


@dataclass(frozen=True, slots=True)
class RulesDiagnostics:
    """Convenience wrapper used by notebooks to plot IC + persistence separately."""

    ic: CrossSectionalICGridResult
    persistence: ForecastPersistenceResult
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(
        cls,
        *,
        rules_list: list[str],
        ic_by_horizon: dict[int, dict[str, pd.Series]],
        persistence_by_rule: dict[str, pd.Series],
        summary_by_rule: dict[str, dict[str, Any]],
        ic_metadata: dict[str, Any] | None = None,
        persistence_metadata: dict[str, Any] | None = None,
        **metadata: Any,
    ) -> "RulesDiagnostics":
        """Build a combined diagnostics object from raw IC + persistence outputs."""
        ic_res = CrossSectionalICGridResult.from_diagoutput(
            rules_list=rules_list,
            ic_by_horizon=ic_by_horizon,
            **(ic_metadata or {}),
        )
        p_res = ForecastPersistenceResult.from_diagoutput(
            persistence_by_rule=persistence_by_rule,
            summary_by_rule=summary_by_rule,
            **(persistence_metadata or {}),
        )
        return cls(ic=ic_res, persistence=p_res, metadata=dict(metadata))

    @property
    def rules(self) -> tuple[str, ...]:
        """Preferred rule order (IC order if present, else persistence order)."""
        if self.ic.rules:
            return self.ic.rules
        return self.persistence.rules

    def rule_diagnostics(self, rule: str) -> RuleDiagnostics:
        """Return a per-rule view containing IC-by-horizon and persistence info."""
        ic_by_horizon = {
            h: self.ic.ic_series(horizon=h, rule=rule) for h in self.ic.horizons
        }
        p = self.persistence.persistence_series(rule)
        s = self.persistence.summary_by_rule.get(str(rule), {})
        return RuleDiagnostics(
            rule=str(rule),
            ic_by_horizon=ic_by_horizon,
            persistence=p,
            persistence_summary=dict(s),
        )

    def plot_ic_at_half_life_for_rules(
        self,
        *,
        rules: list[str] | None = None,
        smoothing: int = 60,
        axes=None,
        title: str | None = None,
    ):
        """Plot IC series at each rule's half-life horizon (raw + rolling mean).

        For each rule, picks the available IC horizon closest to the estimated
        persistence half-life, then plots the IC time series at that horizon.
        If the half-life is NaN, it falls back to the maximum available horizon.
        """
        import matplotlib.pyplot as plt

        rules_use = (
            list(_as_tuple_str(rules)) if rules is not None else list(self.rules)
        )
        n = len(rules_use)
        if n == 0:
            return None, []

        if axes is None:
            fig, axes = plt.subplots(
                n, 1, figsize=(10, 2.2 * n), sharex=True, constrained_layout=True
            )
            if n == 1:
                axes = [axes]
        else:
            fig = None
            if n == 1:
                axes = [axes]

        horizons = self.ic.horizons
        max_horizon = max(horizons) if horizons else None
        for ax, rule in zip(axes, rules_use):
            hl_val = self.persistence.half_life_abs(rule)
            use_max_horizon = not np.isfinite(hl_val)
            if use_max_horizon:
                h = max_horizon
            else:
                h = self.ic.nearest_horizon(hl_val)
            if h is None:
                continue

            ic_series = self.ic.ic_series(horizon=h, rule=rule)
            if ic_series.empty:
                continue

            ax.plot(ic_series, lw=0.6, alpha=0.2, color="C0")
            if smoothing and smoothing > 1:
                ax.plot(ic_series.rolling(smoothing).mean(), lw=1.6, color="C0")
            ax.axhline(0.0, ls="--", lw=1, color="r")
            if use_max_horizon:
                title_text = f"{rule} (HF=NaN → h=h_max={h})"
            else:
                title_text = f"{rule} (HF≈{hl_val:.3g} → h={h})"
            ax.set_title(title_text, loc="left", fontsize=10)
            ax.grid(True, alpha=0.2)

        if axes is not None:
            first_ax = axes[0] if isinstance(axes, (list, tuple, np.ndarray)) else axes
            if first_ax is not None:
                first_ax.set_title(
                    title or "IC at half-life: raw + MA", loc="center", fontsize=12
                )

        return fig, axes

    def plot_rules_ic_and_persistence(
        self,
        *,
        rules: list[str] | None = None,
        plot_ic: bool = True,
        plot_persistence: bool = True,
        plot_ic_half_life: bool = True,
        smoothing_ic_half_life: int = 60,
        show_ic_std: bool = False,
    ) -> None:
        """Notebook-friendly wrapper for mean-IC, persistence, and IC-at-half-life plots.

        Mean IC plots include a red marker at each rule's persistence half-life
        when that value is available.
        """
        import matplotlib.pyplot as plt

        rules_use = (
            list(_as_tuple_str(rules)) if rules is not None else list(self.rules)
        )
        if plot_ic:
            _, ax = plt.subplots()
            self.ic.plot_mean_ic_by_horizon(
                rules=rules_use, show_std=show_ic_std, ax=ax
            )
            mean_df = self.ic.mean_ic_by_horizon(rules=rules_use)
            self._plot_mean_ic_half_life_markers(
                rules=rules_use,
                mean_df=mean_df,
                ax=ax,
            )
        if plot_persistence:
            _, ax = plt.subplots()
            self.persistence.plot_persistence_function_for_rules(rules=rules_use, ax=ax)
        if plot_ic_half_life:
            self.plot_ic_at_half_life_for_rules(
                rules=rules_use,
                smoothing=smoothing_ic_half_life,
            )

    def _plot_mean_ic_half_life_markers(
        self,
        *,
        rules: list[str],
        mean_df: pd.DataFrame,
        ax,
    ) -> None:
        """Plot red dots at each rule's persistence half-life on mean IC plots."""
        if mean_df.empty:
            return
        mean_index = mean_df.index
        for rule in (r for r in rules if r in mean_df.columns):
            half_life_val = self.persistence.half_life_abs(rule)
            if not np.isfinite(half_life_val):
                continue
            half_life = int(np.floor(half_life_val))
            if half_life not in mean_index:
                half_life = self.ic.nearest_horizon(half_life_val)
            if half_life is None or half_life not in mean_index:
                continue
            ic_at_hf = _safe_float(mean_df.at[half_life, rule])
            if not np.isfinite(ic_at_hf):
                continue
            ax.plot(
                [half_life],
                [ic_at_hf],
                marker="o",
                color="r",
                linestyle="",
            )
