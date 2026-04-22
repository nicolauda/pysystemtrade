from syscore.constants import success
from syscore.interactive.display import print_with_landing_strips_around

from sysdata.data_blob import dataBlob
from sysproduction.reporting.api import reportingApi

from sysobjects.production.roll_state import RollState

from sysproduction.interactive_update_roll_status import (
    ASK_FOR_STATE,
    get_auto_roll_parameters_potentially_using_default,
    get_list_of_instruments_to_auto_cycle,
    modify_roll_state,
    no_change_required,
    run_roll_report,
    setup_roll_data_with_state_reporting,
    suggest_roll_state_for_instrument,
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

        days_ahead = auto_parameters.near_expiry_days
        instrument_list = get_list_of_instruments_to_auto_cycle(
            self.data, days_ahead=days_ahead
        )

        if len(instrument_list) == 0:
            self.data.log.debug(
                "No instruments near their desired roll date requiring roll status update"
            )
            return success

        for instrument_code in instrument_list:
            try:
                roll_data = setup_roll_data_with_state_reporting(
                    self.data, instrument_code
                )
                roll_state_required = self._auto_selected_roll_state_instrument(
                    roll_data=roll_data,
                    auto_parameters=auto_parameters,
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
                "Default roll state set to Ask; scheduled updater will pick automatic fallback (Force before the desired roll date, Force_Outright on or after it)"
            )

        default_state = auto_parameters.default_roll_state_if_undecided
        default_state_is_valid = isinstance(default_state, RollState) or (
            default_state == ASK_FOR_STATE
        )
        if not default_state_is_valid:
            self.data.log.warning(
                "Default roll state %s not recognised; using Ask so scheduled updater can pick fallback"
                % str(default_state)
            )
            auto_parameters.default_roll_state_if_undecided = ASK_FOR_STATE

        return auto_parameters

    def _auto_selected_roll_state_instrument(
        self,
        roll_data,
        auto_parameters,
    ):
        run_roll_report(self.api, roll_data.instrument_code)
        roll_state_required = suggest_roll_state_for_instrument(
            roll_data=roll_data, auto_parameters=auto_parameters
        )

        if roll_state_required == ASK_FOR_STATE:
            fallback_roll_state_if_ask = self._fallback_roll_state_for_undecided(
                roll_data=roll_data, auto_parameters=auto_parameters
            )
            print_with_landing_strips_around(
                "No automatic roll state available; defaulting to %s"
                % fallback_roll_state_if_ask
            )
            roll_state_required = fallback_roll_state_if_ask

        original_roll_status = roll_data.original_roll_status
        if original_roll_status == roll_state_required:
            print_with_landing_strips_around(
                "Roll status already set to %s for %s: not changing"
                % (original_roll_status, roll_data.instrument_code)
            )
            return no_change_required

        print_with_landing_strips_around(
            "Automatically changing state from %s to %s for %s"
            % (original_roll_status, roll_state_required, roll_data.instrument_code)
        )

        return roll_state_required

    def _fallback_roll_state_for_undecided(
        self,
        roll_data,
        auto_parameters,
    ) -> RollState:
        if self._is_at_or_past_desired_roll_date(
            roll_data=roll_data, auto_parameters=auto_parameters
        ):
            return RollState.Force_Outright

        return RollState.Force

    @staticmethod
    def _is_at_or_past_desired_roll_date(roll_data, auto_parameters) -> bool:
        return roll_data.days_until_roll <= 0
