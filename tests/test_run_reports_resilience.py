import sys
import types

import pandas as pd

if "ib_async" not in sys.modules:
    ib_async_stub = types.ModuleType("ib_async")
    ib_async_stub.IB = object
    sys.modules["ib_async"] = ib_async_stub

from sysproduction import run_reports as run_reports_module
from sysproduction.data.risk import _normalise_returns_index_for_correlation


class _FakeLog:
    def __init__(self):
        self.critical_messages = []

    def critical(self, message):
        self.critical_messages.append(message)


class _FakeData:
    def __init__(self):
        self.log = _FakeLog()


class _FakeConfig:
    title = "Duplicate markets report"


def test_run_report_logs_critical_and_continues_on_failure(monkeypatch):
    fake_data = _FakeData()
    report_runner = run_reports_module.runReport(
        fake_data, _FakeConfig(), "duplicate_market_report"
    )

    def _raise_runtime_error(*_args, **_kwargs):
        raise RuntimeError("simulated report crash")

    monkeypatch.setattr(run_reports_module, "run_report", _raise_runtime_error)

    report_runner.run_generic_report()

    assert len(fake_data.log.critical_messages) == 1
    critical_message = fake_data.log.critical_messages[0]
    assert "Duplicate markets report" in critical_message
    assert "RuntimeError: simulated report crash" in critical_message


def test_normalise_returns_drops_rangeindex_series():
    returns = pd.Series([0.1, 0.2, 0.3], index=pd.RangeIndex(0, 3))

    normalised = _normalise_returns_index_for_correlation(returns)

    assert normalised.empty
