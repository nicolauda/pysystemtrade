from types import SimpleNamespace

from sysobjects.production.roll_state import RollState
from sysproduction.interactive_update_roll_status import (
    ASK_FOR_STATE,
    RollDataWithStateReporting,
    autoRollParameters,
)
from sysproduction.update_roll_status import updateRollStatus


class _DummyLog:
    def __init__(self):
        self.warnings = []
        self.debug_messages = []

    def warning(self, msg, **kwargs):
        self.warnings.append(msg)

    def debug(self, msg, **kwargs):
        self.debug_messages.append(msg)


def _build_update_roll_status(config_dict: dict) -> updateRollStatus:
    dummy_log = _DummyLog()
    obj = updateRollStatus.__new__(updateRollStatus)
    obj.data = SimpleNamespace(
        config=SimpleNamespace(roll_status_auto_update=config_dict), log=dummy_log
    )
    obj.api = None
    return obj


def _build_auto_roll_parameters() -> autoRollParameters:
    return autoRollParameters(
        auto_roll_if_relative_volume_higher_than=1.0,
        min_relative_volume=0.01,
        min_absolute_volume=100,
        near_expiry_days=10,
        default_roll_state_if_undecided=ASK_FOR_STATE,
        auto_roll_expired=True,
    )


def _build_roll_data(
    instrument_code: str,
    position_priced_contract: int = 1,
    days_until_roll: int = 0,
    relative_volume: float = 2.0,
    absolute_forward_volume: int = 1000,
    days_until_expiry: int = 5,
) -> RollDataWithStateReporting:
    return RollDataWithStateReporting(
        instrument_code=instrument_code,
        original_roll_status=RollState.Passive,
        position_priced_contract=position_priced_contract,
        allowable_roll_states_as_list_of_str=[],
        days_until_roll=days_until_roll,
        relative_volume=relative_volume,
        absolute_forward_volume=absolute_forward_volume,
        days_until_expiry=days_until_expiry,
    )


def test_fallback_roll_state_uses_force_outright_when_expiry_imminent():
    obj = _build_update_roll_status({})
    params = _build_auto_roll_parameters()
    roll_data = _build_roll_data("RUBBER", days_until_expiry=2)

    state = obj._fallback_roll_state_for_undecided(
        roll_data=roll_data,
        auto_parameters=params,
    )

    assert state is RollState.Force_Outright


def test_fallback_roll_state_uses_force_when_expiry_not_imminent():
    obj = _build_update_roll_status({})
    params = _build_auto_roll_parameters()
    roll_data = _build_roll_data("IRON", days_until_expiry=11)

    state = obj._fallback_roll_state_for_undecided(
        roll_data=roll_data,
        auto_parameters=params,
    )

    assert state is RollState.Force


def test_fallback_roll_state_uses_force_outright_on_threshold_boundary():
    obj = _build_update_roll_status({})
    params = _build_auto_roll_parameters()
    roll_data = _build_roll_data("IRON", days_until_expiry=10)

    state = obj._fallback_roll_state_for_undecided(
        roll_data=roll_data,
        auto_parameters=params,
    )

    assert state is RollState.Force_Outright


def test_ensure_non_interactive_defaults_keeps_ask_sentinel():
    obj = _build_update_roll_status({})
    params = _build_auto_roll_parameters()

    normalised = obj._ensure_non_interactive_defaults(params)

    assert normalised.default_roll_state_if_undecided == ASK_FOR_STATE


def test_ensure_non_interactive_defaults_replaces_invalid_state_with_ask():
    obj = _build_update_roll_status({})
    params = _build_auto_roll_parameters()
    params.default_roll_state_if_undecided = object()

    normalised = obj._ensure_non_interactive_defaults(params)

    assert normalised.default_roll_state_if_undecided == ASK_FOR_STATE
