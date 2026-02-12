import pytest

from sysobjects.production.roll_state import RollState
from sysproduction.interactive_update_roll_status import (
    ASK_FOR_STATE,
    normalise_default_roll_state_if_undecided,
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
