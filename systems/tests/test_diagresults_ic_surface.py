"""Tests for IC date × horizon surface helpers in systems.diagresults."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from systems.diagresults import CrossSectionalICGridResult, RulesDiagnostics


def _minimal_ic_by_horizon() -> tuple[list[str], dict[int, dict[str, pd.Series]]]:
    rules_list = ["carry", "ewmac16_64"]
    idx5 = pd.date_range("2024-01-01", periods=10, freq="B")
    idx20 = pd.date_range("2024-01-05", periods=8, freq="B")
    ic_by_horizon: dict[int, dict[str, pd.Series]] = {
        5: {
            "carry": pd.Series(np.linspace(0.01, 0.05, len(idx5)), index=idx5),
            "ewmac16_64": pd.Series(np.linspace(-0.02, 0.03, len(idx5)), index=idx5),
        },
        20: {
            "carry": pd.Series(np.linspace(0.0, 0.04, len(idx20)), index=idx20),
            "ewmac16_64": pd.Series(np.linspace(0.02, -0.01, len(idx20)), index=idx20),
        },
    }
    return rules_list, ic_by_horizon


def test_ic_surface_frame_columns_and_index() -> None:
    rules_list, ic_by_horizon = _minimal_ic_by_horizon()
    res = CrossSectionalICGridResult.from_diagoutput(
        rules_list=rules_list,
        ic_by_horizon=ic_by_horizon,
    )
    df = res.ic_surface_frame("carry")
    assert list(df.columns) == [5, 20]
    assert df.index.is_monotonic_increasing
    assert len(df) == len(df.index.unique())
    first_date = pd.Timestamp("2024-01-01")
    assert np.isfinite(df.loc[first_date, 5])


def test_ic_surface_frame_empty_rule() -> None:
    rules_list, ic_by_horizon = _minimal_ic_by_horizon()
    res = CrossSectionalICGridResult.from_diagoutput(
        rules_list=rules_list,
        ic_by_horizon=ic_by_horizon,
    )
    df = res.ic_surface_frame("missing_rule")
    assert df.empty


def test_plot_ic_surface_for_rule_heatmap_and_3d_smoke() -> None:
    rules_list, ic_by_horizon = _minimal_ic_by_horizon()
    res = CrossSectionalICGridResult.from_diagoutput(
        rules_list=rules_list,
        ic_by_horizon=ic_by_horizon,
    )
    ax_h = res.plot_ic_surface_for_rule("carry", kind="heatmap")
    assert ax_h is not None
    plt.close(ax_h.figure)

    fig = plt.figure()
    ax3 = fig.add_subplot(111, projection="3d")
    ax3_out = res.plot_ic_surface_for_rule("ewmac16_64", kind="3d", ax=ax3, fig=fig)
    assert ax3_out is not None
    plt.close(fig)


def test_plot_ic_surface_for_rule_raises_when_empty() -> None:
    rules_list, ic_by_horizon = _minimal_ic_by_horizon()
    res = CrossSectionalICGridResult.from_diagoutput(
        rules_list=rules_list,
        ic_by_horizon=ic_by_horizon,
    )
    with pytest.raises(ValueError, match="No IC surface data"):
        res.plot_ic_surface_for_rule("missing_rule", kind="heatmap")


def test_rules_diagnostics_plot_rule_ic_surface_smoke() -> None:
    rules_list, ic_by_horizon = _minimal_ic_by_horizon()
    persistence_by_rule = {
        "carry": pd.Series([1.0, 0.5], index=[1, 2]),
        "ewmac16_64": pd.Series([1.0, 0.4], index=[1, 2]),
    }
    summary_by_rule = {
        "carry": {
            "half_life_abs": 5.0,
            "half_life_rel": 5.0,
            "rho1": 1.0,
            "n_lags": 2,
            "method": "pearson",
            "agg": "median",
        },
        "ewmac16_64": {
            "half_life_abs": 6.0,
            "half_life_rel": 6.0,
            "rho1": 1.0,
            "n_lags": 2,
            "method": "pearson",
            "agg": "median",
        },
    }
    diags = RulesDiagnostics.from_raw(
        rules_list=rules_list,
        ic_by_horizon=ic_by_horizon,
        persistence_by_rule=persistence_by_rule,
        summary_by_rule=summary_by_rule,
    )
    ax = diags.plot_rule_ic_surface("carry", kind="heatmap")
    plt.close(ax.figure)


def test_plot_rules_ic_surfaces_heatmap_smoke() -> None:
    rules_list, ic_by_horizon = _minimal_ic_by_horizon()
    persistence_by_rule = {
        "carry": pd.Series([1.0], index=[1]),
        "ewmac16_64": pd.Series([1.0], index=[1]),
    }
    summary_by_rule = {
        "carry": {
            "half_life_abs": 5.0,
            "half_life_rel": 5.0,
            "rho1": 1.0,
            "n_lags": 1,
            "method": "pearson",
            "agg": "median",
        },
        "ewmac16_64": {
            "half_life_abs": 6.0,
            "half_life_rel": 6.0,
            "rho1": 1.0,
            "n_lags": 1,
            "method": "pearson",
            "agg": "median",
        },
    }
    diags = RulesDiagnostics.from_raw(
        rules_list=rules_list,
        ic_by_horizon=ic_by_horizon,
        persistence_by_rule=persistence_by_rule,
        summary_by_rule=summary_by_rule,
    )
    diags.plot_rules_ic_surfaces(kind="heatmap", ncols=2)
    plt.close("all")


def test_plot_rules_ic_surfaces_unknown_kind() -> None:
    rules_list, ic_by_horizon = _minimal_ic_by_horizon()
    diags = RulesDiagnostics.from_raw(
        rules_list=rules_list,
        ic_by_horizon=ic_by_horizon,
        persistence_by_rule={"carry": pd.Series([1.0], index=[1])},
        summary_by_rule={
            "carry": {
                "half_life_abs": 5.0,
                "half_life_rel": 5.0,
                "rho1": 1.0,
                "n_lags": 1,
                "method": "pearson",
                "agg": "median",
            },
        },
    )
    with pytest.raises(ValueError, match="Unknown kind"):
        diags.plot_rules_ic_surfaces(kind="typo")  # type: ignore[arg-type]
