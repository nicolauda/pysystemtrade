import sys
import types

import pandas as pd


class _LazyIbAsyncModule(types.ModuleType):
    """Stand-in for `ib_async` and `ib_async.*` so reporting imports work without real IB."""

    def __getattr__(self, name: str):
        placeholder = type(name, (), {})
        setattr(self, name, placeholder)
        return placeholder


def _install_ib_async_stub() -> None:
    if "ib_async" in sys.modules:
        return
    sys.modules["ib_async"] = _LazyIbAsyncModule("ib_async")
    for sub_name in ("ib_async.ib", "ib_async.order"):
        sys.modules[sub_name] = _LazyIbAsyncModule(sub_name)


_install_ib_async_stub()

import sysproduction.reporting.api as api_module
from sysproduction.reporting.api import reportingApi


class _FakeLog:
    def __init__(self):
        self.warning_messages = []

    def warning(self, message):
        self.warning_messages.append(message)


class _FakeData:
    def __init__(self):
        self.log = _FakeLog()


def test_get_correlations_for_configured_duplicates_except_branch(monkeypatch):
    fake_data = _FakeData()
    api = reportingApi(fake_data, min_correlation=0.5)

    def _fixed_pairs(_data):
        return [("SYM_A", "SYM_B")]

    def _raise_correlation(*_args, **_kwargs):
        raise RuntimeError("messaggio deterministico")

    monkeypatch.setattr(api_module, "generate_duplicate_pairs", _fixed_pairs)
    monkeypatch.setattr(
        api_module, "get_correlation_matrix_for_instruments", _raise_correlation
    )

    result = api.get_correlations_for_configured_duplicates()

    assert len(fake_data.log.warning_messages) == 1
    warn = fake_data.log.warning_messages[0]
    assert "Could not calculate configured duplicate correlations" in warn
    assert "messaggio deterministico" in warn

    body = result.Body
    assert list(body.columns) == ["included", "excluded", "correlation", "note"]
    assert len(body) == 1
    row = body.iloc[0]
    assert row["included"] == "SYM_A"
    assert row["excluded"] == "SYM_B"
    assert pd.isna(row["correlation"])
    assert row["note"] == "Correlation unavailable: messaggio deterministico"
