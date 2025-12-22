"""
Custom futures system mirroring rob_system.
Accepts either a Config object or a path/str for the config and will build
dbFuturesSimData automatically when data is omitted.
"""

from pathlib import Path

from syscore.constants import arg_not_supplied

from sysdata.sim.db_futures_sim_data import dbFuturesSimData
from sysdata.config.configdata import Config

from systems.forecasting import Rules
from systems.basesystem import System
from systems.forecast_combine import ForecastCombine
from systems.provided.attenuate_vol.vol_attenuation_forecast_scale_cap import (
    volAttenForecastScaleCap,
)
from systems.provided.rob_system.rawdata import myFuturesRawData
from systems.positionsizing import PositionSizing
from systems.portfolio import Portfolios
from systems.provided.dynamic_small_system_optimise.optimised_positions_stage import (
    optimisedPositions,
)
from systems.risk import Risk
from systems.provided.dynamic_small_system_optimise.accounts_stage import (
    accountForOptimisedStage,
)


def futures_system(
    data=arg_not_supplied,
    config=arg_not_supplied,
    rules=arg_not_supplied,
):
    """
    Build the futures system using the rob_system pipeline but with flexible inputs.
    - data: dbFuturesSimData auto-created if omitted.
    - config: Config object or path/str (must be provided).
    - rules: custom Rules instance; defaults to Rules().
    - config_filename/config_file: alternate names for config (kept for compatibility).
    """

    if data is arg_not_supplied:
        data = dbFuturesSimData()

    if config is arg_not_supplied:
        raise ValueError("config must be provided to build the futures system")

    if rules is arg_not_supplied:
        rules = Rules()

    system = System(
        [
            Risk(),
            accountForOptimisedStage(),
            optimisedPositions(),
            Portfolios(),
            PositionSizing(),
            myFuturesRawData(),
            ForecastCombine(),
            volAttenForecastScaleCap(),
            rules,
        ],
        data,
        config,
    )

    return system
