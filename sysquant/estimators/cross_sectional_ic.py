import numpy as np
import pandas as pd
from scipy.stats import rankdata


def _rowwise_pearson(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute Pearson correlation per row with pairwise NaN handling.

    Args:
        x: 2D array of shape (n_rows, n_cols).
        y: 2D array of the same shape.

    Returns:
        1D array of length n_rows with Pearson correlations. Rows with
        fewer than two finite pairs or zero variance return NaN.

    Notes:
        The computation is vectorized and uses pairwise deletion via
        finite masks.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    count = mask.sum(axis=1)
    x_sum = np.where(mask, x, 0.0).sum(axis=1)
    y_sum = np.where(mask, y, 0.0).sum(axis=1)
    mean_x = np.divide(x_sum, count, out=np.full_like(x_sum, np.nan), where=count > 0)
    mean_y = np.divide(y_sum, count, out=np.full_like(y_sum, np.nan), where=count > 0)
    x0 = np.where(mask, x - mean_x[:, None], 0.0)
    y0 = np.where(mask, y - mean_y[:, None], 0.0)
    numer = (x0 * y0).sum(axis=1)
    denom = np.sqrt((x0**2).sum(axis=1) * (y0**2).sum(axis=1))
    corr = np.divide(numer, denom, out=np.full_like(numer, np.nan), where=denom > 0)
    corr = np.where(count >= 2, corr, np.nan)
    return corr


def _columnwise_pearson(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute Pearson correlation per column with pairwise NaN handling.

    Args:
        x: 2D array of shape (n_rows, n_cols).
        y: 2D array of the same shape.

    Returns:
        1D array of length n_cols with Pearson correlations. Rows with
        fewer than two finite pairs or zero variance return NaN.

    Notes:
        The computation is vectorized and uses pairwise deletion via
        finite masks.
    """
    return _rowwise_pearson(x.T, y.T)


def _rowwise_spearman(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute Spearman correlation per row using rank-transformed data.

    Args:
        x: 2D array of shape (n_rows, n_cols).
        y: 2D array of the same shape.

    Returns:
        1D array of length n_rows with Spearman correlations. NaN handling
        matches `_rowwise_pearson`.

    Notes:
        Ranks are computed per row with `nan_policy="omit"`, then Pearson
        correlation is applied to the ranked arrays.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    x_rank = rankdata(x, axis=1, method="average", nan_policy="omit")
    y_rank = rankdata(y, axis=1, method="average", nan_policy="omit")
    x_rank = np.where(mask, x_rank, np.nan)
    y_rank = np.where(mask, y_rank, np.nan)
    return _rowwise_pearson(x_rank, y_rank)


def _columnwise_spearman(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute Spearman correlation per column using rank-transformed data.

    Args:
        x: 2D array of shape (n_rows, n_cols).
        y: 2D array of the same shape.

    Returns:
        1D array of length n_cols with Spearman correlations. NaN handling
        matches `_rowwise_pearson`.

    Notes:
        Ranks are computed per column with `nan_policy="omit"`, then Pearson
        correlation is applied to the ranked arrays.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    x_rank = rankdata(x, axis=0, method="average", nan_policy="omit")
    y_rank = rankdata(y, axis=0, method="average", nan_policy="omit")
    x_rank = np.where(mask, x_rank, np.nan)
    y_rank = np.where(mask, y_rank, np.nan)
    return _columnwise_pearson(x_rank, y_rank)


def cross_sectional_ic(
    forecast_df: pd.DataFrame, returns_df: pd.DataFrame
) -> tuple[pd.Series, pd.Series]:
    """Compute cross-sectional IC time series for forecasts vs. returns.

    Args:
        forecast_df: Forecasts with dates as index and instruments as columns.
        returns_df: Forward returns with the same structure.

    Returns:
        Tuple of (pearson_ic, spearman_ic), each a Series indexed by the
        common dates. If no column or index overlap exists, empty Series
        are returned.

    Notes:
        Columns are intersected first, then indices aligned before row-wise
        correlations are computed.
    """
    common_cols = forecast_df.columns.intersection(returns_df.columns)
    if common_cols.empty:
        empty = pd.Series(dtype=float)
        return empty, empty

    forecast_aligned = forecast_df[common_cols]
    returns_aligned = returns_df[common_cols]
    common_idx = forecast_aligned.index.intersection(returns_aligned.index)
    if common_idx.empty:
        empty = pd.Series(dtype=float)
        return empty, empty

    forecast_aligned = forecast_aligned.loc[common_idx]
    returns_aligned = returns_aligned.loc[common_idx]
    x = forecast_aligned.to_numpy(dtype=float)
    y = returns_aligned.to_numpy(dtype=float)
    pearson = pd.Series(_rowwise_pearson(x, y), index=common_idx)
    spearman = pd.Series(_rowwise_spearman(x, y), index=common_idx)
    return pearson, spearman


class CrossSectionalIC:
    """Namespace wrapper for cross-sectional IC helpers."""

    _rowwise_pearson = staticmethod(_rowwise_pearson)
    _rowwise_spearman = staticmethod(_rowwise_spearman)
    _columnwise_pearson = staticmethod(_columnwise_pearson)
    _columnwise_spearman = staticmethod(_columnwise_spearman)
    cross_sectional_ic = staticmethod(cross_sectional_ic)
