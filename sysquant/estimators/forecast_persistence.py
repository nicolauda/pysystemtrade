import numpy as np
import pandas as pd
from sysquant.estimators.cross_sectional_ic import CrossSectionalIC


def _first_crossing_linear_interp(k: np.ndarray, v: np.ndarray, level: float) -> float:
    """
    Return the first lag at which a curve crosses a given threshold, using
    linear interpolation.

    Given a sequence of values v(k) defined at lags k, this helper identifies
    the first index i such that:

        v(k_i) <= level  and  v(k_{i-1}) > level

    and returns a continuously valued estimate of the crossing lag obtained
    by linear interpolation between the two adjacent points.

    Parameters
    ----------
    k : np.ndarray
        One-dimensional array of lag values (strictly increasing), aligned
        with `v`. Must have the same length as `v`.

    v : np.ndarray
        One-dimensional array of curve values evaluated at lags `k`.
        All values are assumed to be finite.

    level : float
        Threshold value defining the crossing condition.

    Returns
    -------
    float
        Interpolated lag at which the curve first crosses the threshold.
        Returns NaN if:
        - the curve never crosses the threshold,
        - the threshold is not finite,
        - or a unique crossing cannot be identified (e.g. flat segment
        above the threshold).

    Notes
    -----
    - Only the *first downward crossing* is considered.
    - The function is agnostic to the meaning of the curve (persistence,
    autocorrelation, decay profile, etc.) and is reused by higher-level
    half-life estimators.
    """
    if not np.isfinite(level):
        return np.nan

    for i in range(1, len(v)):
        if v[i] <= level:
            k0, k1 = float(k[i - 1]), float(k[i])
            v0, v1 = float(v[i - 1]), float(v[i])

            # Flat segment: avoid division by zero
            if v1 == v0:
                return k0 if v0 <= level else np.nan

            return k0 + (level - v0) * (k1 - k0) / (v1 - v0)

    return np.nan


def half_life_from_array(rho: np.ndarray, level: float = 0.5) -> float:
    """
    Estimate the half-life of a persistence curve stored as a NumPy array.

    The half-life is defined as the smallest lag k at which the persistence
    value falls below (or equals) a specified threshold. If the threshold
    is crossed between two integer lags, linear interpolation is used to
    obtain a continuous estimate.

    Formally, the half-life k* satisfies:

        rho(k*) = level

    where rho(k) is the persistence at lag k.

    Parameters
    ----------
    rho : np.ndarray
        One-dimensional array of persistence values, where rho[i] corresponds
        to lag k = i + 1 (i.e. rho[0] = rho(1)). The array may contain NaNs,
        which are ignored.

    level : float, default 0.5
        Threshold defining the half-life. A common choice is 0.5, corresponding
        to a 50% decay level.

    Returns
    -------
    float
        Estimated half-life expressed in lag units. Returns NaN if:
        - fewer than two finite persistence values are available, or
        - the persistence curve never crosses the specified threshold.

    Notes
    -----
    - The method is non-parametric and makes no assumptions about the functional
    form of the decay.
    - If the persistence curve is non-monotonic, the first downward crossing
    is used.
    - Lags are implicitly assumed to be consecutive integers starting at 1.
    """
    rho = np.asarray(rho, dtype=float)
    if rho.ndim != 1:
        raise ValueError("rho must be a 1D array")

    k_vals = np.arange(1, len(rho) + 1, dtype=float)

    mask = np.isfinite(rho)
    if mask.sum() < 2:
        return np.nan

    k = k_vals[mask]
    v = rho[mask]
    return float(_first_crossing_linear_interp(k, v, level))


def half_life_relative_to_rho1(rho: np.ndarray, frac: float = 0.5) -> float:
    """
    Estimate the half-life of a persistence curve relative to its initial value.

    This variant defines the half-life as the lag k at which persistence decays
    to a fixed fraction of its value at lag 1:

        rho(k) = frac * rho(1)

    If rho(1) is not finite, the first finite persistence value is used as
    the reference instead.

    Parameters
    ----------
    rho : np.ndarray
        One-dimensional array of persistence values, where rho[0] corresponds
        to lag k = 1.

    frac : float, default 0.5
        Fraction of the reference persistence value defining the half-life.
        The default corresponds to a 50% decay relative to rho(1).

    Returns
    -------
    float
        Estimated half-life expressed in lag units. Returns NaN if:
        - no finite reference value exists, or
        - the persistence curve never crosses the implied threshold.

    Notes
    -----
    - This definition is scale-adaptive and is useful when rho(1) is
    substantially below 1 due to noise, smoothing, or forecast clipping.
    - Internally, this function delegates to `half_life_from_array`.
    """
    rho = np.asarray(rho, dtype=float)
    if rho.ndim != 1:
        raise ValueError("rho must be a 1D array")
    if len(rho) == 0:
        return np.nan

    finite = np.isfinite(rho)
    if not finite.any():
        return np.nan

    ref = rho[0] if np.isfinite(rho[0]) else rho[finite][0]
    return half_life_from_array(rho, level=float(frac * ref))


def half_life_from_series(p: pd.Series, level: float = 0.5) -> float:
    """
    Estimate the half-life of a persistence curve stored as a pandas Series.

    The Series is expected to be indexed by lag values k (e.g. k = 1, 2, ..., H),
    with values representing persistence at each lag. Missing values are
    dropped prior to computation.

    Unlike `half_life_from_array`, this function respects the actual lag values
    in the index, which may be non-integer and/or irregularly spaced.

    Parameters
    ----------
    p : pd.Series
        Persistence curve indexed by lag k. The index must be numeric.
        NaN values are dropped before computation.

    level : float, default 0.5
        Threshold defining the half-life.

    Returns
    -------
    float
        Estimated half-life expressed in lag units. Returns NaN if:
        - fewer than two valid observations are available, or
        - the persistence curve never crosses the specified threshold.

    Notes
    -----
    - The Series is internally sorted by its index before analysis.
    - Linear interpolation is used to obtain a continuous estimate of the
    crossing lag.
    """

    if p is None or len(p) == 0:
        return np.nan

    p = p.dropna()
    if len(p) < 2:
        return np.nan

    k = p.index.to_numpy(dtype=float)
    v = p.to_numpy(dtype=float)

    order = np.argsort(k)
    k = k[order]
    v = v[order]

    return float(_first_crossing_linear_interp(k, v, level))


def persistence_function(
    df_forecast: pd.DataFrame,
    horizon: int = 60,
    agg: str = "median",
    min_obs: int = 30,
    spearman: bool = False,
) -> pd.Series:
    """
    Compute the forecast persistence function.

    The persistence function measures how stable a forecasting rule is over
    time by quantifying how strongly current forecasts resemble future
    forecasts of the same rule.

    For each lag k and instrument i, persistence is defined as the time-series
    correlation:

        rho_i(k) = corr_t( f_i(t), f_i(t+k) )

    where f_i(t) is the forecast for instrument i at time t.
    Instrument-level persistence values are then aggregated across instruments
    to obtain a single persistence measure per lag.

    Parameters
    ----------
    df_forecast : pd.DataFrame
        Forecast matrix with shape (T, N), where rows correspond to time
        (DateTimeIndex) and columns correspond to instruments. Entry
        df_forecast.loc[t, i] represents the forecast for instrument i at time t.

    horizon : int, default 60
        Maximum lag (in rows) for which persistence is computed.
        Lags are evaluated for k = 1, ..., min(horizon, T - 1).

    agg : {"median", "mean"}, default "median"
        Aggregation method used to combine instrument-level persistence values.
        The median is typically more robust to outliers and unstable instruments.

    min_obs : int, default 30
        Minimum number of finite forecast pairs (f_i(t), f_i(t+k)) required for
        an instrument to be included at lag k. Values less than 2 are coerced
        to 2.

    spearman : bool, default False
        If True, compute Spearman (rank) correlation instead of Pearson
        correlation. Spearman persistence is more robust to clipping,
        outliers, and non-linear transformations.

    Returns
    -------
    pd.Series
        Persistence curve indexed by lag k. The Series name encodes the
        correlation type and aggregation method (e.g. "persistence_pearson_median").

    Notes
    -----
    - Persistence is computed *per instrument* over time (column-wise).
    - Missing values are handled via pairwise deletion.
    - High persistence at lag k indicates slowly varying forecasts, while
    low or negative persistence indicates rapid changes or instability.

    Interpretation
    --------------
    The persistence curve characterizes the effective temporal memory of a
    forecasting rule and can be summarized further using half-life estimators.
    It is particularly useful for comparing strategies with different
    turnover profiles or signal construction methods.
    """

    method = "spearman" if spearman else "pearson"
    if df_forecast is None or df_forecast.empty:
        return pd.Series(dtype=float, name=f"persistence_{method}_{agg}")

    forecast = df_forecast.to_numpy(dtype=float)  # (T, N)
    T, _ = forecast.shape
    if T < 2:
        return pd.Series(dtype=float, name=f"persistence_{method}_{agg}")

    min_obs = max(int(min_obs), 2)
    max_k = min(int(horizon), T - 1)

    out = {}
    for k in range(1, max_k + 1):
        x = forecast[:-k, :]  # (T-k, N)
        y = forecast[k:, :]  # (T-k, N)

        if spearman:
            corr_per_instr = CrossSectionalIC.columnwise_spearman(x, y)
        else:
            corr_per_instr = CrossSectionalIC.columnwise_pearson(x, y)

        mask = np.isfinite(x) & np.isfinite(y)
        counts = mask.sum(axis=0)
        corr_per_instr = np.where(counts >= min_obs, corr_per_instr, np.nan)

        if agg == "median":
            out[k] = float(np.nanmedian(corr_per_instr))
        elif agg == "mean":
            out[k] = float(np.nanmean(corr_per_instr))
        else:
            raise ValueError("agg must be 'median' or 'mean'")

    return pd.Series(out, name=f"persistence_{method}_{agg}")


def get_persistence_function_for_rule(
    df_forecast_for_rule: pd.DataFrame,
    horizon: int = 60,
    agg: str = "median",
    min_obs: int = 30,
    spearman: bool = False,
    half_life_level: float = 0.5,
    half_life_frac_of_rho1: float = 0.5,
) -> tuple[pd.Series, float, float]:
    """
    Compute the persistence curve for a single forecasting rule and summarize it
    with two half-life estimates.

    Parameters
    ----------
    df_forecast_for_rule : pd.DataFrame
        Forecast matrix for a single rule with shape (T, N):
        index = time, columns = instruments.

    horizon : int, default 60
        Maximum lag k (in rows) to evaluate. Persistence is computed for
        k = 1..min(horizon, T-1).

    agg : {"median", "mean"}, default "median"
        Aggregation across instruments of instrument-level persistence values.

    min_obs : int, default 30
        Minimum number of finite pairs (f_i(t), f_i(t+k)) required for an
        instrument to be included at lag k. Values < 2 are coerced to 2.

    spearman : bool, default False
        If True, compute Spearman (rank) correlation; otherwise Pearson.

    half_life_level : float, default 0.5
        Absolute threshold used to extract half-life from the aggregated
        persistence curve (e.g. 0.5).

    half_life_frac_of_rho1 : float, default 0.5
        Relative half-life definition based on rho(1):
            rho(k) = frac * rho(1)
        This can be more stable when rho(1) is far from 1.

    Returns
    -------
    persistence : pd.Series
        Aggregated persistence curve indexed by lag k.

    half_life_abs : float
        Half-life computed at the absolute level `half_life_level`
        using the Series index as lag values.

    half_life_rel : float
        Half-life computed at the relative level `half_life_frac_of_rho1 * rho(1)`.
        Returns NaN if rho(1) is not finite or no crossing occurs.

    Notes
    -----
    - The persistence curve is computed column-wise over time (per instrument)
      and aggregated across instruments.
    - Missing values are handled via pairwise deletion.
    - Both half-life estimates use linear interpolation between adjacent lags.
    """
    persistence = persistence_function(
        df_forecast=df_forecast_for_rule,
        horizon=horizon,
        agg=agg,
        min_obs=min_obs,
        spearman=spearman,
    )

    hl_abs = half_life_from_series(persistence, level=half_life_level)
    hl_rel = half_life_relative_to_rho1(
        persistence.to_numpy(dtype=float),
        frac=half_life_frac_of_rho1,
    )

    # normalize to plain floats
    hl_abs = float(hl_abs) if np.isfinite(hl_abs) else np.nan
    hl_rel = float(hl_rel) if np.isfinite(hl_rel) else np.nan

    return persistence, hl_abs, hl_rel


class ForecastPersistence:
    """Namespace wrapper for forecast persistence helpers."""

    half_life_from_array = staticmethod(half_life_from_array)
    half_life_relative_to_rho1 = staticmethod(half_life_relative_to_rho1)
    half_life_from_series = staticmethod(half_life_from_series)
    persistence_function = staticmethod(persistence_function)
    get_persistence_function_for_rule = staticmethod(get_persistence_function_for_rule)
