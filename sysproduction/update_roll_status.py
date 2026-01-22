from syscore.constants import success

from sysdata.data_blob import dataBlob
from sysproduction.reporting.api import reportingApi

from sysobjects.production.roll_state import RollState

from sysproduction.interactive_update_roll_status import (
    ASK_FOR_STATE,
    auto_selected_roll_state_instrument,
    get_auto_roll_parameters_potentially_using_default,
    get_list_of_instruments_to_auto_cycle,
    modify_roll_state,
    no_change_required,
    setup_roll_data_with_state_reporting,
    warn_not_rolling,
)


class updateRollStatus:
    """
    Automated roll status updater for scheduled runs.
    Uses the same decision logic as the interactive workflow but runs with
    default parameters and without user prompts.
    """

    def __init__(self, data: dataBlob):
        self.data = data
        self.api = reportingApi(data)

    def update_roll_status(self):
        auto_parameters = get_auto_roll_parameters_potentially_using_default(
            data=self.data, use_default=True
        )
        auto_parameters = self._ensure_non_interactive_defaults(auto_parameters)
        fallback_roll_state = RollState.Force

        days_ahead = auto_parameters.near_expiry_days
        instrument_list = get_list_of_instruments_to_auto_cycle(
            self.data, days_ahead=days_ahead
        )

        if len(instrument_list) == 0:
            self.data.log.debug(
                "No instruments near expiry requiring roll status update"
            )
            return success

        for instrument_code in instrument_list:
            try:
                roll_data = setup_roll_data_with_state_reporting(
                    self.data, instrument_code
                )
                roll_state_required = auto_selected_roll_state_instrument(
                    api=self.api,
                    roll_data=roll_data,
                    auto_parameters=auto_parameters,
                    fallback_roll_state_if_ask=fallback_roll_state,
                )

                if roll_state_required is no_change_required:
                    warn_not_rolling(instrument_code, auto_parameters)
                    continue

                modify_roll_state(
                    data=self.data,
                    instrument_code=instrument_code,
                    original_roll_state=roll_data.original_roll_status,
                    roll_state_required=roll_state_required,
                    confirm_adjusted_price_change=False,
                    allow_forward_fill_without_prompt=True,
                )
            except Exception as e:
                self.data.log.exception(
                    "Automatic roll status update failed for %s: %s"
                    % (instrument_code, str(e))
                )

        return success

    def _ensure_non_interactive_defaults(self, auto_parameters):
        default_state = auto_parameters.default_roll_state_if_undecided

        if isinstance(default_state, str) and default_state != ASK_FOR_STATE:
            try:
                auto_parameters.default_roll_state_if_undecided = RollState[
                    default_state
                ]
            except KeyError:
                self.data.log.warning(
                    "Unknown default roll state %s; using Force instead" % default_state
                )
                auto_parameters.default_roll_state_if_undecided = RollState.Force
            else:
                default_state = auto_parameters.default_roll_state_if_undecided

        if auto_parameters.default_roll_state_if_undecided == ASK_FOR_STATE:
            self.data.log.debug(
                "Default roll state set to Ask; using Force to avoid interactive prompt during scheduled roll update"
            )
            auto_parameters.default_roll_state_if_undecided = RollState.Force

        if not isinstance(
            auto_parameters.default_roll_state_if_undecided, RollState
        ):
            self.data.log.warning(
                "Default roll state %s not recognised; using Force instead"
                % str(auto_parameters.default_roll_state_if_undecided)
            )
            auto_parameters.default_roll_state_if_undecided = RollState.Force

        return auto_parameters
