import pytest

from sysobjects.production.roll_state import RollState
from sysproduction.interactive_update_roll_status import (
    ASK_FOR_STATE,
    include_instrument_in_auto_cycle,
    normalise_default_roll_state_if_undecided,
    RollDataWithStateReporting,
    autoRollParameters,
    suggest_roll_state_for_instrument,
)


def test_normalise_default_roll_state_maps_valid_state_to_enum():
    result = normalise_default_roll_state_if_undecided("Force")
    assert result is RollState.Force


def test_normalise_default_roll_state_preserves_ask_sentinel():
    result = normalise_default_roll_state_if_undecided("ask")
    assert result == ASK_FOR_STATE


def test_normalise_default_roll_state_rejects_unknown_state():
    with pytest.raises(ValueError, match="Invalid default roll state"):
        normalise_default_roll_state_if_undecided("Not_A_Real_State")


def test_include_instrument_in_auto_cycle_uses_days_until_roll(monkeypatch):
    monkeypatch.setattr(
        "sysproduction.interactive_update_roll_status.days_until_desired_roll_date",
        lambda data, instrument_code: 5,
    )

    assert include_instrument_in_auto_cycle(object(), "EDOLLAR", days_ahead=5)
    assert not include_instrument_in_auto_cycle(object(), "EDOLLAR", days_ahead=4)


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
    *,
    position_priced_contract: int,
    days_until_roll: int,
    relative_volume: float,
    absolute_forward_volume: int,
    days_until_expiry: int = 5,
) -> RollDataWithStateReporting:
    return RollDataWithStateReporting(
        instrument_code="EDOLLAR",
        original_roll_status=RollState.No_Roll,
        position_priced_contract=position_priced_contract,
        allowable_roll_states_as_list_of_str=[],
        days_until_roll=days_until_roll,
        relative_volume=relative_volume,
        absolute_forward_volume=absolute_forward_volume,
        days_until_expiry=days_until_expiry,
    )


@pytest.mark.parametrize(
    ("roll_data", "expected_state"),
    [
        (
            _build_roll_data(
                position_priced_contract=1,
                days_until_roll=11,
                relative_volume=2.0,
                absolute_forward_volume=1000,
            ),
            RollState.Passive,
        ),
        (
            _build_roll_data(
                position_priced_contract=1,
                days_until_roll=11,
                relative_volume=0.0,
                absolute_forward_volume=0,
            ),
            RollState.No_Roll,
        ),
        (
            _build_roll_data(
                position_priced_contract=0,
                days_until_roll=11,
                relative_volume=2.0,
                absolute_forward_volume=1000,
            ),
            RollState.Roll_Adjusted,
        ),
        (
            _build_roll_data(
                position_priced_contract=1,
                days_until_roll=5,
                relative_volume=2.0,
                absolute_forward_volume=1000,
            ),
            RollState.Force,
        ),
        (
            _build_roll_data(
                position_priced_contract=1,
                days_until_roll=5,
                relative_volume=0.0,
                absolute_forward_volume=0,
            ),
            RollState.No_Open,
        ),
        (
            _build_roll_data(
                position_priced_contract=0,
                days_until_roll=5,
                relative_volume=0.0,
                absolute_forward_volume=0,
            ),
            RollState.No_Open,
        ),
        (
            _build_roll_data(
                position_priced_contract=1,
                days_until_roll=0,
                relative_volume=2.0,
                absolute_forward_volume=1000,
            ),
            RollState.Force_Outright,
        ),
        (
            _build_roll_data(
                position_priced_contract=1,
                days_until_roll=0,
                relative_volume=0.0,
                absolute_forward_volume=0,
            ),
            RollState.Close,
        ),
        (
            _build_roll_data(
                position_priced_contract=0,
                days_until_roll=0,
                relative_volume=0.0,
                absolute_forward_volume=0,
            ),
            RollState.Roll_Adjusted,
        ),
    ],
)
def test_suggest_roll_state_for_instrument_uses_three_window_logic(
    roll_data: RollDataWithStateReporting,
    expected_state: RollState,
):
    params = _build_auto_roll_parameters()

    state = suggest_roll_state_for_instrument(
        roll_data=roll_data,
        auto_parameters=params,
    )

    assert state is expected_state
