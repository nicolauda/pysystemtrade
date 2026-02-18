import datetime
from types import SimpleNamespace

from syscore.constants import arg_not_supplied
from sysexecution.orders.list_of_orders import listOfOrders
from sysexecution.strategies import strategy_order_handling


def _make_order(instrument_code: str) -> SimpleNamespace:
    return SimpleNamespace(instrument_code=instrument_code)


class _DummyLog:
    def debug(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


class _DummyDiagInstruments:
    def __init__(self, region_by_instrument: dict[str, str]):
        self._region_by_instrument = region_by_instrument

    def get_region(self, instrument_code: str) -> str:
        return self._region_by_instrument[instrument_code]


class _DummyOrderGenerator(strategy_order_handling.orderGeneratorForStrategy):
    def __init__(self, order_list: listOfOrders):
        self._strategy_name = "test_strategy"
        self._data = object()
        self._log = _DummyLog()
        self._data_orders = SimpleNamespace(db_instrument_stack_data=None)
        self._order_generation_by_zone = arg_not_supplied
        self._order_generation_time_manager = None
        self._order_list = order_list
        self.submitted_order_lists: list[listOfOrders] = []

    def get_required_orders(self) -> listOfOrders:
        return self._order_list

    def apply_overrides_and_position_limits(
        self, order_list: listOfOrders
    ) -> listOfOrders:
        return order_list

    def submit_order_list(self, order_list: listOfOrders):
        self.submitted_order_lists.append(order_list)


def test_get_and_place_orders_legacy_without_zone_config():
    generator = _DummyOrderGenerator(
        listOfOrders([_make_order("ASIA_CODE"), _make_order("US_CODE")])
    )

    generator.get_and_place_orders()

    assert len(generator.submitted_order_lists) == 1
    assert [order.instrument_code for order in generator.submitted_order_lists[0]] == [
        "ASIA_CODE",
        "US_CODE",
    ]


def test_zone_runs_when_time_reached(monkeypatch):
    monkeypatch.setattr(
        strategy_order_handling.orderGenerationTimeManager,
        "_current_datetime",
        lambda self: datetime.datetime(2026, 2, 18, 5, 30),
    )
    monkeypatch.setattr(
        strategy_order_handling,
        "diagInstruments",
        lambda data: _DummyDiagInstruments(
            {"ASIA_CODE": "ASIA", "EMEA_CODE": "EMEA", "US_CODE": "US"}
        ),
    )

    generator = _DummyOrderGenerator(
        listOfOrders(
            [_make_order("ASIA_CODE"), _make_order("EMEA_CODE"), _make_order("US_CODE")]
        )
    )
    generator.get_and_place_orders(
        order_generation_by_zone={"ASIA": "05:00", "EMEA": "11:00", "US": "18:00"}
    )

    assert len(generator.submitted_order_lists) == 1
    assert [order.instrument_code for order in generator.submitted_order_lists[0]] == [
        "ASIA_CODE"
    ]


def test_zone_is_not_run_twice_in_same_day(monkeypatch):
    monkeypatch.setattr(
        strategy_order_handling.orderGenerationTimeManager,
        "_current_datetime",
        lambda self: datetime.datetime(2026, 2, 18, 6, 0),
    )
    monkeypatch.setattr(
        strategy_order_handling,
        "diagInstruments",
        lambda data: _DummyDiagInstruments({"ASIA_CODE": "ASIA"}),
    )

    generator = _DummyOrderGenerator(listOfOrders([_make_order("ASIA_CODE")]))
    config = {"ASIA": "05:00"}

    generator.get_and_place_orders(order_generation_by_zone=config)
    generator.get_and_place_orders(order_generation_by_zone=config)

    assert len(generator.submitted_order_lists) == 1
    assert [order.instrument_code for order in generator.submitted_order_lists[0]] == [
        "ASIA_CODE"
    ]


def test_late_start_catches_up_overdue_zones(monkeypatch):
    monkeypatch.setattr(
        strategy_order_handling.orderGenerationTimeManager,
        "_current_datetime",
        lambda self: datetime.datetime(2026, 2, 18, 17, 30),
    )
    monkeypatch.setattr(
        strategy_order_handling,
        "diagInstruments",
        lambda data: _DummyDiagInstruments(
            {"ASIA_CODE": "ASIA", "EMEA_CODE": "EMEA", "US_CODE": "US"}
        ),
    )

    generator = _DummyOrderGenerator(
        listOfOrders(
            [_make_order("ASIA_CODE"), _make_order("EMEA_CODE"), _make_order("US_CODE")]
        )
    )
    generator.get_and_place_orders(
        order_generation_by_zone={"ASIA": "05:00", "EMEA": "11:00", "US": "18:00"}
    )

    assert len(generator.submitted_order_lists) == 2
    assert [order.instrument_code for order in generator.submitted_order_lists[0]] == [
        "ASIA_CODE"
    ]
    assert [order.instrument_code for order in generator.submitted_order_lists[1]] == [
        "EMEA_CODE"
    ]
