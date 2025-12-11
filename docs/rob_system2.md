# Rob system: stage wiring and rawdata naming

This note clarifies how the production `rob_system` is wired and why the config keeps referring to `rawdata.*` even though the stage class is `myFuturesRawData`.

## Stage list (from `systems/provided/rob_system/run_system.py`)

```python
from systems.provided.rob_system.rawdata import myFuturesRawData
from systems.provided.attenuate_vol.vol_attenuation_forecast_scale_cap import volAttenForecastScaleCap
from systems.provided.dynamic_small_system_optimise.optimised_positions_stage import optimisedPositions
from systems.provided.dynamic_small_system_optimise.accounts_stage import accountForOptimisedStage

def futures_system(...):
    return System(
        [
            Risk(),
            accountForOptimisedStage(),
            optimisedPositions(),
            Portfolios(),
            PositionSizing(),
            myFuturesRawData(),          # <-- rawdata stage instance
            ForecastCombine(),
            volAttenForecastScaleCap(),
            rules,
        ],
        sim_data,
        config,
    )
```

The order above is the execution order: `rawdata → rules → scaling → combination → portfolio → size → optimisation → accounts/risk`.

## Why config strings say `rawdata.*`

The system uses the **stage name** (not the class name) when wiring stages. `myFuturesRawData` inherits from `systems/rawdata.py`, which defines:

```python
class RawData(SystemStage):
    @property
    def name(self):
        return "rawdata"
```

When the `System` is constructed, it sets `system.rawdata = myFuturesRawData()` because the `name` property returns `"rawdata"`. Therefore any config entry such as:

```yaml
data:
  - "rawdata.get_daily_prices"
  - "rawdata.get_cumulative_daily_vol_normalised_returns"
```

is resolved against the `rawdata` attribute on the system, which is the `myFuturesRawData` instance. You can extend the class (e.g., skew, demeaned factors) without changing config paths: they always refer to the stage name `rawdata`.

## Practical takeaway

- Keep using `rawdata.*` in `config.yaml` and rule definitions; the stage name is stable even if the class changes.
- To add new raw series, implement methods in `myFuturesRawData` (or a subclass) and reference them via `rawdata.your_method` in config.
- The same naming rule applies to other stages (e.g., `forecastScaleCap`, `portfolio`, `accounts`): config strings use the stage name exposed on the system instance.
