"""
Called from sysproduction code in a while loop, each time it runs loops over strategies
For each strategy gets the required trades per instrument
It then passes these to the 'virtual' order queue
So called because it deals with instrument level trades, not contract implementation
"""

import datetime
from copy import copy

from syscore.constants import arg_not_supplied
from sysdata.data_blob import dataBlob

from sysexecution.orders.list_of_orders import listOfOrders
from sysexecution.orders.instrument_orders import instrumentOrder
from sysexecution.order_stacks.instrument_order_stack import zeroOrderException
from syslogging.logger import *
from sysproduction.data.instruments import diagInstruments
from sysproduction.data.positions import diagPositions
from sysproduction.data.orders import dataOrders
from sysproduction.data.controls import diagOverrides, dataLocks, dataPositionLimits

name_of_main_generator_method = "get_and_place_orders"


class orderGenerationTimeManager:
    """Track which configured zones are due and already processed for the day.

    The manager keeps an in-memory progress map (`zone -> done`) and resets it
    automatically when the calendar day changes.
    """

    def __init__(self, order_generation_by_zone: dict):
        """Build a zone scheduler from a `zone -> HH:MM` mapping.

        Args:
            order_generation_by_zone: Zone schedule used to trigger intraday
                order generation. Dict insertion order is used as execution
                priority when multiple zones are due.
        """
        self._order_generation_by_zone = copy(order_generation_by_zone)
        self._current_day = self._current_datetime().date()
        self._init_progress_dict()

    def list_of_zones_due_now(self) -> list[str]:
        """Return zones that are due now and not yet completed today.

        Returns:
            Ordered list of zones ready for execution.
        """
        self._reset_progress_if_new_day()
        return [
            zone for zone in self.list_of_zones if self.can_zone_be_generated_now(zone)
        ]

    def can_zone_be_generated_now(self, zone: str) -> bool:
        """Return `True` when a zone is due and not yet completed today.

        Args:
            zone: Zone key configured in `order_generation_by_zone`.
        """
        if self.progress_by_zone[zone]:
            return False

        now_time = self._current_datetime().time()
        return now_time >= self.time_to_start_zone(zone)

    def time_to_start_zone(self, zone: str) -> datetime.time:
        """Return the configured start time for a zone.

        Args:
            zone: Zone key configured in `order_generation_by_zone`.
        """
        time_as_string = self.order_generation_by_zone[zone]
        return datetime.datetime.strptime(time_as_string, "%H:%M").time()

    def mark_zone_as_completed(self, zone: str):
        """Mark a zone as completed for the current day.

        Args:
            zone: Zone key configured in `order_generation_by_zone`.
        """
        progress_by_zone = self._progress_by_zone
        progress_by_zone[zone] = True
        self._progress_by_zone = progress_by_zone

    def _reset_progress_if_new_day(self):
        """Reset daily progress when the date changes."""
        current_day = self._current_datetime().date()
        if current_day == self._current_day:
            return

        self._current_day = current_day
        self._init_progress_dict()

    def _init_progress_dict(self):
        """Initialise `zone -> completed` state for the active day."""
        self._progress_by_zone = {zone: False for zone in self.list_of_zones}

    def _current_datetime(self) -> datetime.datetime:
        """Return current local datetime.

        Isolated in a method so tests can monkeypatch deterministic timestamps.
        """
        return datetime.datetime.now()

    @property
    def list_of_zones(self) -> list[str]:
        return list(self.order_generation_by_zone.keys())

    @property
    def order_generation_by_zone(self) -> dict:
        return self._order_generation_by_zone

    @property
    def progress_by_zone(self) -> dict:
        return self._progress_by_zone


class orderGeneratorForStrategy(object):
    """

    Order generators are strategy specific but have common methods used by the order handler

    """

    def __init__(self, data: dataBlob, strategy_name: str):
        self._strategy_name = strategy_name
        self._data = data
        data_orders = dataOrders(data)
        self._log = data.log
        self._data_orders = data_orders
        self._order_generation_by_zone = arg_not_supplied
        self._order_generation_time_manager = None

    @property
    def data(self) -> dataBlob:
        return self._data

    @property
    def strategy_name(self) -> str:
        return self._strategy_name

    @property
    def log(self):
        return self._log

    @property
    def data_orders(self):
        return self._data_orders

    @property
    def order_stack(self):
        return self.data_orders.db_instrument_stack_data

    def get_and_place_orders(self, order_generation_by_zone: dict = arg_not_supplied):
        """Generate and place strategy orders, optionally by configured zones.

        Args:
            order_generation_by_zone: Optional mapping `zone -> HH:MM`. If
                omitted, all orders are generated and submitted with legacy
                behaviour. If provided, only zones due at current time are run,
                and each zone is processed at most once per day.
        """
        # THIS IS THE MAIN FUNCTION THAT IS RUN
        if order_generation_by_zone is arg_not_supplied:
            self._get_and_submit_order_list(self.get_required_orders())
            return

        self._get_and_place_orders_by_zone(order_generation_by_zone)

    def _get_and_place_orders_by_zone(self, order_generation_by_zone: dict):
        """Execute order generation for all zones currently due."""
        order_generation_time_manager = (
            self._get_or_create_order_generation_time_manager(order_generation_by_zone)
        )
        zones_due_now = order_generation_time_manager.list_of_zones_due_now()
        if len(zones_due_now) == 0:
            self.log.debug(
                "No order-generation zones due now for strategy %s"
                % self.strategy_name,
                method="temp",
            )
            return

        order_list = self.get_required_orders()
        instrument_data = diagInstruments(self.data)
        for zone in zones_due_now:
            zone_order_list = self._get_orders_for_zone(
                order_list=order_list, zone=zone, instrument_data=instrument_data
            )
            if len(zone_order_list) == 0:
                self.log.debug(
                    "No orders available for zone %s in strategy %s; marking complete"
                    % (zone, self.strategy_name),
                    method="temp",
                )
                order_generation_time_manager.mark_zone_as_completed(zone)
                continue

            self.log.debug(
                "Generating %d orders for zone %s in strategy %s"
                % (len(zone_order_list), zone, self.strategy_name),
                method="temp",
            )
            self._get_and_submit_order_list(zone_order_list)
            order_generation_time_manager.mark_zone_as_completed(zone)
            self.log.debug(
                "Finished zone %s for strategy %s" % (zone, self.strategy_name),
                method="temp",
            )

    def _get_or_create_order_generation_time_manager(
        self, order_generation_by_zone: dict
    ) -> orderGenerationTimeManager:
        """Return a reusable daily zone manager, recreating on config changes."""
        if (
            self._order_generation_time_manager is None
            or self._order_generation_by_zone != order_generation_by_zone
        ):
            self._order_generation_by_zone = copy(order_generation_by_zone)
            self._order_generation_time_manager = orderGenerationTimeManager(
                order_generation_by_zone=order_generation_by_zone
            )

        return self._order_generation_time_manager

    def _get_orders_for_zone(
        self, order_list: listOfOrders, zone: str, instrument_data: diagInstruments
    ) -> listOfOrders:
        """Filter orders for a single configured zone using instrument region."""
        zone_order_list = [
            order
            for order in order_list
            if instrument_data.get_region(order.instrument_code) == zone
        ]
        return listOfOrders(zone_order_list)

    def _get_and_submit_order_list(self, order_list: listOfOrders):
        """Apply controls and submit a ready list of instrument orders."""
        order_list_with_overrides = self.apply_overrides_and_position_limits(order_list)
        self.submit_order_list(order_list_with_overrides)

    def get_required_orders(self) -> listOfOrders:
        raise Exception(
            "Need to inherit with a specific method for your type of strategy"
        )

    def get_actual_positions_for_strategy(self) -> dict:
        """
        Actual positions held by a strategy

        Useful to know, usually

        :return: dict, keys are instrument codes, values are positions
        """
        data = self.data
        strategy_name = self.strategy_name

        diag_positions = diagPositions(data)
        actual_positions = diag_positions.get_dict_of_actual_positions_for_strategy(
            strategy_name
        )

        return actual_positions

    def apply_overrides_and_position_limits(
        self, order_list: listOfOrders
    ) -> listOfOrders:
        new_order_list = [
            self.apply_overrides_and_position_limits_for_instrument_and_strategy(
                proposed_order
            )
            for proposed_order in order_list
        ]
        new_order_list = listOfOrders(new_order_list)

        return new_order_list

    def apply_overrides_and_position_limits_for_instrument_and_strategy(
        self, proposed_order: instrumentOrder
    ) -> instrumentOrder:
        revised_order = self.apply_overrides_for_instrument_and_strategy(proposed_order)
        new_order = self.adjust_order_for_position_limits(revised_order)

        return new_order

    def apply_overrides_for_instrument_and_strategy(
        self, proposed_order: instrumentOrder
    ) -> instrumentOrder:
        """
        Apply an override to a trade

        :param strategy_name: str
        :param instrument_code: str
        :return: int, updated position
        """

        diag_overrides = diagOverrides(self.data)
        diag_positions = diagPositions(self.data)

        instrument_strategy = proposed_order.instrument_strategy

        original_position = diag_positions.get_current_position_for_instrument_strategy(
            instrument_strategy
        )

        override = diag_overrides.get_cumulative_override_for_instrument_strategy(
            instrument_strategy
        )

        revised_order = override.apply_override(original_position, proposed_order)

        if revised_order.trade != proposed_order.trade:
            self.log.debug(
                "%s trade change from %s to %s because of override %s"
                % (
                    instrument_strategy.key,
                    str(proposed_order.trade),
                    str(revised_order.trade),
                    str(override),
                ),
                **proposed_order.log_attributes(),
                method="temp",
            )

        return revised_order

    def adjust_order_for_position_limits(
        self, order: instrumentOrder
    ) -> instrumentOrder:
        log_attrs = {**order.log_attributes(), "method": "temp"}

        data_position_limits = dataPositionLimits(self.data)
        new_order = data_position_limits.apply_position_limit_to_order(order)

        if new_order.trade != order.trade:
            if new_order.is_zero_trade():
                ## at position limit, can't do anything
                self.log.warning(
                    "Can't trade at all because of position limits %s" % str(order),
                    **log_attrs,
                )
            else:
                self.log.warning(
                    "Can't do trade of %s because of position limits,instead will do %s"
                    % (str(order), str(new_order.trade)),
                    **log_attrs,
                )

        return new_order

    def submit_order_list(self, order_list: listOfOrders):
        data_lock = dataLocks(self.data)
        for order in order_list:
            # try:
            # we allow existing orders to be modified
            log_attrs = {**order.log_attributes(), "method": "temp"}
            self.log.debug("Required order %s" % str(order), **log_attrs)

            instrument_locked = data_lock.is_instrument_locked(order.instrument_code)
            if instrument_locked:
                self.log.debug("Instrument locked, not submitting", **log_attrs)
                continue
            self.submit_order(order)

    def submit_order(self, order: instrumentOrder):
        log_attrs = {**order.log_attributes(), "method": "temp"}

        try:
            order_id = self.order_stack.put_order_on_stack(order)
            log_attrs[INSTRUMENT_ORDER_ID_LABEL] = order_id
        except zeroOrderException:
            # we checked for zero already, which means that there is an existing order
            # on the stack
            # An existing order of the same size
            self.log.warning(
                "Ignoring new order as either zero size or it replicates an existing "
                "order on the stack",
                **log_attrs,
            )

        else:
            self.log.debug(
                "Added order %s to instrument order stack with order id %d"
                % (str(order), order_id),
                **log_attrs,
            )
