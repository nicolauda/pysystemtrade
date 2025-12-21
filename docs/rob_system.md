# Base system vs Rob system — an in-depth tour

This chapter is written as a step-by-step walkthrough of two reference pipelines shipped with `pysystemtrade`:

- **Base system** — the minimal, teaching-oriented stack from *Systematic Trading* (Chapter 15): `systems.provided.futures_chapter15.basesystem.futures_system` with `futuresconfig.yaml`.
- **Rob system** — the production-grade stack: `systems.provided.rob_system.run_system.futures_system` with `config.yaml`.

We explain why each stage exists, how data flows, and how the production pipeline extends the teaching version. Blog references are included when they give extra context (mainly from Rob Carver’s “Qoppac” / “This Blog Is Systematic” site).

## 1) System architecture at 10,000 ft

Both systems use the stage framework in `systems/basesystem.py`. A stage has:
- a stable `name` (used to wire the DAG),
- methods annotated for caching/diagnostics,
- access to the parent system, config, and data.

Stages compose into a directed acyclic graph; caching avoids recomputation when a series is reused downstream. This mirrors the modular “factory” approach in *Systematic Trading*.

- **Base stack (teaching path):** `accounts`, `portfolio`, `positionSize`, `rawdata`, `combForecast`, `forecastScaleCap`, `rules`.
- **Rob stack (production path):** `risk`, `accounts` (custom), `optimisedPositions`, `portfolio`, `positionSize`, `rawdata` (custom), `combForecast`, `forecastScaleCap` (vol-attenuated), `rules`.

The extra stages in the rob system deliver production needs: richer signals, volatility-aware scaling, cost-aware optimisation, and explicit risk checks.

### Quick code map (rob system pipeline)

```python
# systems/provided/rob_system/run_system.py
from systems.provided.rob_system.rawdata import myFuturesRawData
from systems.provided.attenuate_vol.vol_attenuation_forecast_scale_cap import volAttenForecastScaleCap
from systems.provided.dynamic_small_system_optimise.optimised_positions_stage import optimisedPositions
from systems.provided.dynamic_small_system_optimise.accounts_stage import accountForOptimisedStage

def futures_system(sim_data=arg_not_supplied, config_filename="systems.provided.rob_system.config.yaml", rules=arg_not_supplied):
    if sim_data is arg_not_supplied:
        sim_data = dbFuturesSimData()
    config = Config(config_filename)
    rules = Rules() if rules is arg_not_supplied else rules
    return System(
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
        sim_data,
        config,
    )
```

Reading top-to-bottom mirrors the execution: rawdata → rules → vol attenuation/scale → combine → portfolio → size → optimise → accounts/risk.

## 1bis) How rob stages talk to each other (data flow and calculations)

Think of the rob system as a left-to-right conveyor belt with feedback from cached results:

1. **rawdata → rules**  
   - Raw series: prices, returns, vol, carry, factors (skew, demeaned factors).  
   - Rules pull these series to compute raw forecasts in SR units (e.g., EWMAC normalises by vol; carry is scaled to SR-like units via the scalar in config).

2. **rules → forecastScaleCap (volAttenForecastScaleCap)**  
   - For rules in `use_attenuation`, compute vol quantile vs 10y mean; attenuation = `2 - 1.5 * quantile` (smoothed).  
   - Scaled forecast = `clip(raw_forecast * attenuation * forecast_scalar, -cap, +cap)` with `cap=20`.

3. **forecastScaleCap → combForecast**  
   - Per-instrument weighted blend: combined_forecast = sum(weight_i * scaled_forecast_i).  
   - Apply `forecast_div_multiplier` (discounts intra-style correlation).  
   - Output is a single combined forecast per instrument, still in SR units.

4. **combForecast → portfolio**  
   - Portfolio stage uses combined forecast, instrument weights, covariance estimates (`instrument_returns_correlation`) and diversification multiplier to form desired notional risk per instrument.  
   - Risk overlay checks (`max_risk_fraction_*`, leverage cap) can downscale or block portfolios that exceed limits.

5. **portfolio → positionSize**  
   - Converts desired risk to contracts: contracts = (target_weight * capital) / (per-contract risk).  
   - Uses vol target (25%) and notional capital (500k) plus buffering to avoid churn.

6. **positionSize → optimisedPositions (rob only)**  
   - Takes raw optimal contracts, the covariance matrix, per-contract values, and costs.  
   - Objective balances tracking error vs transaction cost using `shadow_cost` and `tracking_error_buffer`; constraints enforce reduce-only or long-only where configured.  
   - Output: smoothed contract paths and equivalent weights.

7. **optimisedPositions → accounts / risk**  
   - Accounts compute P&L on optimised positions (costs, roll costs, FX).  
   - Risk stage recomputes portfolio risk on both optimised and original positions to validate overlays and optimiser impact.

Caching ensures repeated queries (e.g., risk and accounts both asking for per-contract value) are served from the same stored output, keeping stages consistent.

## 2) Data sources and configuration

- **Base system**
  - Default data: `csvFuturesSimData` (toy CSV bundle).
  - Config: `systems/provided/futures_chapter15/futuresconfig.yaml`.
  - Universe: six futures (SOFR, US10, EUROSTX, V2X, MXP, CORN).
  - Notional: 250,000 USD; vol target: 20%.
  - Diversification multipliers: `forecast_div_multiplier 1.31`, `instrument_div_multiplier 1.89`.

- **Rob system**
  - Default data: `dbFuturesSimData` (production Parquet/Mongo); requires sampled-contracts DB, FX, roll calendars, and costs.
  - Config: `systems/provided/rob_system/config.yaml`.
  - Universe: broad, curated list (rates, equity indices, commodities, FX, crypto, sectors, IRS, etc.).
  - Notional: 500,000 USD; vol target: 25%.
  - Diversification multipliers: `instrument_div_multiplier 2.75` with `use_instrument_div_mult_estimates: True`; forecast weights are per instrument.
  - Risk overlay: hard limits on risk and leverage (see section 8).

**Key concepts (used throughout):**
- **Forecast**: directional score in annualised SR units.
- **Scaling & cap**: apply `forecast_scalar`, then cap to ±20 to keep SR units bounded.
- **Diversification multipliers**: discount correlated bets intra-rule and across instruments (see blog [“Correlations, weights and multipliers”](https://qoppac.blogspot.com/2016/01/correlations-weights-multipliers.html), 2016).
- **Vol target**: annualised risk budget for sizing.
- **Buffering**: hysteresis to cut turnover (related to blog [“Diversification and small account size”](https://qoppac.blogspot.com/2016/03/diversification-and-small-account-size.html), 2016).
- **Risk overlay**: post-portfolio constraints on risk and leverage (see [“Capital correction”](https://qoppac.blogspot.com/2016/06/capital-correction-pysystemtrade.html), 2016).
- **Optimisation**: cost-aware smoothing of positions (see [“Optimising weights with costs”](https://qoppac.blogspot.com/2016/05/optimising-weights-with-costs.html), 2016).

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


## 3) Signal layer — trading rules

### Base system (`futuresconfig.yaml`)
- Trend: six EWMAC rules on prices (2/8 to 64/256).
- Carry: one rule, 90-day smoothing.
- Fixed forecast scalars (e.g., 10.6 on `ewmac2_8`); no attenuation, no cross-sectional or factor inputs.

### Rob system (`config.yaml`)
- **Breakout trend**: lookbacks 10–320.
- **Price+vol trend** (`momentum4`…`momentum64`): EWMAC using daily prices and daily vol.
- **Asset-class trend** (`assettrend2`…`assettrend64`): EWMAC on class-normalised prices.
- **Vol-normalised trend** (`normmom2`…`normmom64`): EWMAC on vol-normalised returns.
- **Carry family**: 10/30/60/125 plus `relcarry` cross-section.
- **Cross-sectional momentum / MR**: `relmomentum10`…`relmomentum80`, `mrinasset1000`.
- **Acceleration**: `accel16/32/64`.
- **Skew factors**: `skewabs*` (demeaned vs universe) and `skewrv*` (demeaned vs asset class).
- **Attenuation list**: most directional/factor rules are flagged for volatility attenuation.
- All forecast scalars are explicit; estimation flags are off (`use_forecast_*_estimates: False`).

**Why the breadth?** The production stack follows the “many small, weakly correlated edges” principle in the book and blog: mix trend horizons, cross-sectional effects, carry, and simple factors to smooth returns.

### Example: defining a rule in config

```yaml
# systems/provided/rob_system/config.yaml
trading_rules:
  breakout80:
     function: systems.provided.rules.breakout.breakout
     data:
         - "rawdata.get_daily_prices"
     other_args:
       lookback: 80
```

This asks the rules stage to call `breakout` on daily prices with an 80-day lookback; the forecast scalar for `breakout80` is applied downstream in scaling.

## 4) Forecast scaling, capping, and volatility attenuation

- **Base**: `ForecastScaleCap` multiplies by `forecast_scalar`, then caps to ±20. No conditioning on volatility.
- **Rob**: `volAttenForecastScaleCap` adds volatility-aware attenuation (related to the buffering/forecast mapping ideas in [“Diversification and small account size”](https://qoppac.blogspot.com/2016/03/diversification-and-small-account-size.html)):
  1) compute daily vol; 2) compute a 10-year rolling mean; 3) normalise vol and convert to a quantile; 4) apply `2 - 1.5 * quantile` (smoothed) to rules in `use_attenuation`; 5) scale and cap.

Effect: when an instrument is unusually volatile, forecasts shrink before combination, keeping SR units comparable across regimes (a theme in Rob’s scaling posts on keeping SR units meaningful when volatility shifts).

### Code path (attenuation)

```python
# systems/provided/attenuate_vol/vol_attenuation_forecast_scale_cap.py
@diagnostic()
def get_vol_attenuation(self, instrument_code):
    normalised_vol_q = self.get_vol_quantile_points(instrument_code)
    vol_attenuation = normalised_vol_q.apply(multiplier_function)  # 2 - 1.5 * quantile
    return vol_attenuation.ewm(span=10).mean()
```

Only rules listed in `use_attenuation` are multiplied by this factor before capping to ±20.

## 5) Combining forecasts

- **Base**: global weights for all instruments (carry 50%; trend buckets 50%). `forecast_div_multiplier 1.31` discounts correlated trend rules (see [“Correlations, weights and multipliers”](https://qoppac.blogspot.com/2016/01/correlations-weights-multipliers.html), 2016).
- **Rob**: per-instrument weights (`forecast_weights` is a large dict by instrument). `instrument_div_multiplier 2.75` with estimation turned on to reflect realised inter-instrument correlation. Forecast cap stays at 20; forecast/div estimation flags are off, so weights are hand-crafted.

Effect: different markets lean on different styles (e.g., more carry in rates, more breakout in equities), while correlated instruments are penalised more aggressively.

### Example: per-instrument weights

```yaml
# systems/provided/rob_system/config.yaml
forecast_weights:
  AEX:
    breakout10: 0.01
    breakout160: 0.05
    carry10: 0.025
    momentum32: 0.03
    normmom8: 0.02
    relcarry: 0.05
    skewabs180: 0.025
    # ... more rules per instrument
```

The combine stage multiplies each scaled forecast by its weight and sums to one combined forecast per instrument, then applies diversification multipliers.

## 6) Raw data layer

- **What lives here (base `systems/rawdata.py`)**: stitched prices, returns, two flavours of volatility (price-diff for forecasts; percentage returns for sizing), carry inputs, FX/asset-class normalisation, and diagnostics. The stage caches series so rules/portfolio don’t recompute them.
- **How volatility is calculated**: by default a robust EWMA (35-day span, 10-day warm-up) with an additional floor using a 500-day 5% quantile. Price-diff vol uses stitched prices (Panama-style splicing); percentage-return vol uses `daily_denominator_price` to avoid using the stitched price as the denominator on positive-carry assets.
- **Why stitched vs denominator prices**: stitched prices give smooth inputs for rules, but they would explode historical percentage returns if used as denominators. The `daily_denominator_price` method switches to the *current* contract price for the denominator so percentage returns and vol stay sensible.
- **Customising raw data**: add new methods to a subclass (e.g., `myFuturesRawData`) whenever you need reusable diagnostics (carry components, moving averages, factor transforms) or asset-class-specific denominators. Config still calls them via `rawdata.*` because the stage name is stable.
- **Rob extensions (`systems/provided/rob_system/rawdata.py`)**: rolling `skew`, `neg_skew`, `kurtosis`; factor de-meaning vs universe/asset class; cross-sectional factor matrices and averages; carry detail. These feed skew/factor and cross-sectional rules.

### Example: factor de-meaning

```python
# systems/provided/rob_system/rawdata.py
@output()
def get_demeanded_factor_value(
    self,
    instrument_code,
    factor_name="skew",
    demean_method="average_factor_value_in_asset_class_for_instrument",
    **kwargs,
):
    demean_value = getattr(self, demean_method)(
        instrument_code, factor_name=factor_name, **kwargs
    )
    factor_value = self.get_factor_value_for_instrument(
        instrument_code, factor_name=factor_name, **kwargs
    )
    return factor_value - demean_value
```

Skew-based rules consume this demeaned factor; the demean method controls whether the benchmark is the asset class, the entire universe, or instrument history.

## 7) Position sizing and buffering

- **Base**: `PositionSizing` with `percentage_vol_target 20%`, `notional_trading_capital 250k`, default buffering (`buffer_method: forecast`, `buffer_size: 0.10`, `buffer_trade_to_edge: True`). Output goes straight to accounts.
- **Rob**: same stage, but `percentage_vol_target 25%`, `notional_trading_capital 500k`. Output feeds the optimisation stage.

Mechanics: position size is proportional to forecast, inverse vol, and instrument weight; buffering delays trades until the forecast moves beyond the buffer, reducing churn (related to the [“Diversification and small account size”](https://qoppac.blogspot.com/2016/03/diversification-and-small-account-size.html) blog, 2016).

### Pseudocode: from forecast to contracts

```python
desired_risk = combined_forecast * instrument_weight / instrument_vol
contract_value = per_contract_risk(instrument_code)  # vol-adjusted notional
contracts_raw = (desired_risk * capital) / contract_value
contracts_buffered = apply_buffer(contracts_raw, buffer_size=0.10)  # hysteresis
```

The optimiser (next stage) receives `contracts_buffered` as its starting point.

## 8) Portfolio construction and risk overlay

- **Base**: six instruments, fixed `instrument_weights`, `instrument_div_multiplier 1.89`, no explicit `risk_overlay`.
- **Rob**: broad universe with specific weights; `instrument_div_multiplier 2.75` (estimable); `risk_overlay` enforces:
  - `max_risk_fraction_normal_risk: 1.75`
  - `max_risk_fraction_stdev_risk: 4.0`
  - `max_risk_limit_sum_abs_risk: 4.0`
  - `max_risk_leverage: 20.0`
  Correlations come from `instrument_returns_correlation` (EW 75 weeks, clipped at 0.99).

Effect: the rob system can block portfolios that exceed risk or leverage limits before execution, echoing the [“capital correction”](https://qoppac.blogspot.com/2016/06/capital-correction-pysystemtrade.html) and overlay guidance on the blog.

### Overlay parameters (config excerpt)

```yaml
risk_overlay:
  max_risk_fraction_normal_risk: 1.75
  max_risk_fraction_stdev_risk: 4.0
  max_risk_limit_sum_abs_risk: 4.0
  max_risk_leverage: 20.0
```

If any limit is breached, risk is clipped/downscaled before orders are sent.

## 9) Optimisation stage (rob only)

Stage: `optimisedPositions` (`systems/provided/dynamic_small_system_optimise/optimised_positions_stage.py`).

Purpose: make raw optimal positions tradable for small/medium capital by accounting for costs, constraints, and speed limits. This is the practical implementation of the [“Optimising weights with costs”](https://qoppac.blogspot.com/2016/05/optimising-weights-with-costs.html) ideas (2016).

Inputs: covariance matrix, per-contract value, raw optimal contracts, cost per contract (deflated), constraints (reduce-only / long-only), speed control (`shadow_cost`, `tracking_error_buffer`).

Process: build `objectiveFunctionForGreedy`, run the greedy optimiser over dates, carry positions forward to avoid churn, and emit optimised positions/weights.

Key parameters (`small_system` in `sysdata/config/defaults.yaml`):
- `shadow_cost: 50`
- `cost_multiplier: 1.0`
- `tracking_error_buffer: 0.0125`
- `shrink_instrument_returns_correlation: 0.5`

Outcome: lower turnover, fewer trades in expensive/illiquid markets, and adherence to operational constraints.

### Optimiser sketch

```python
# systems/provided/dynamic_small_system_optimise/optimised_positions_stage.py
def get_optimal_positions_with_fixed_contract_values(
    self, relevant_date=arg_not_supplied, previous_positions=arg_not_supplied, maximum_positions=arg_not_supplied
):
    obj = self._get_optimal_positions_objective_instance(
        relevant_date=relevant_date,
        previous_positions=previous_positions,
        maximum_positions=maximum_positions,
    )
    return obj.optimise_positions()  # greedy search balances tracking error vs cost
```

The objective includes covariance, per-contract value, cost deflators, constraints (reduce-only/long-only), and speed control (`shadow_cost`, `tracking_error_buffer`).

## 10) Accounts and P&L

- **Base**: `Account` computes P&L on buffered/rounded positions; costs come from data; roll costs halved by default (`multiply_roll_costs_by: 0.5`); SR cost off (`use_SR_costs: False`).
- **Rob**: `accountForOptimisedStage` computes P&L on optimised positions and exposes turnover diagnostics (`total_optimised_portfolio_level_turnover`) for live-style monitoring.

## 11) Risk reporting (rob only)

Stage: `Risk` (`systems/risk.py`) reports portfolio risk for optimised positions and for original positions (with/without buffering/rounding). Use it to sanity-check leverage and overlay constraints before orders go out.

## 12) Instrument selection and universe hygiene

- Config lists drive the universe in both systems.
- Production hygiene: `exclude_instrument_lists.ignore_instruments` and `trading_restrictions` in `sysdata/config/defaults.yaml`.
- `prune_system_instruments_without_data` (`sysproduction/reporting/adhoc/static_system_modular.py`) drops instruments missing in sampled-contracts DB when building static reports.
- Static selection report: `python -m sysproduction.reporting.adhoc.static_system_modular --report` ranks markets per capital using rob-system correlations. This aligns with the diversification and small-account discussions on the blog.

## 13) Capital, vol target, and scaling philosophy

- Base: vol target 20%, notional 250k — simple, small universe.
- Rob: vol target 25%, notional 500k — broader universe and risk budget.
- Shared rules: forecast cap 20; scalars/weights are hand-picked; instrument diversification is estimated in rob, fixed in base.

These choices match the capital-sizing guidance in *Systematic Trading* and the “capital correction” blog post: pick a vol target consistent with drawdown tolerance, then diversify so the target is reachable without concentration.

## 14) Putting it together — execution paths

- **Base backtest**: call `futures_system()` from `futures_chapter15/basesystem.py` and backtest. No optimisation stage.
- **Rob backtest/production**: call `futures_system()` from `rob_system/run_system.py`; for production, ensure sampled-contracts DB, FX, costs, and (optionally) capital from DB (`sysproduction/data/capital.py`). Optimisation and risk overlay are active.
- **Static selection report**: `python -m sysproduction.reporting.adhoc.static_system_modular --report` to propose instrument sets per capital. For intuition on ladders and correlation-aware selection, see the blog pieces on diversification and small accounts.

## 15) Key differences at a glance

- **Signals**: Base = trend + carry; Rob = multi-trend, carry, cross-sectional momentum/MR, skew factors, acceleration.
- **Scaling**: Base = static; Rob = static + vol attenuation.
- **Combination**: Base = global weights; Rob = per-instrument weights + estimated instrument diversification.
- **Universe**: Base = 6 instruments; Rob = broad multi-asset.
- **Risk controls**: Base = implicit via weights; Rob = explicit risk overlay + leverage caps.
- **Optimisation**: Base = none; Rob = cost-aware greedy optimiser.
- **Data**: Base = CSV toy; Rob = DB futures with factors and costs.
- **Vol target**: Base 20% @ 250k; Rob 25% @ 500k.

## 16) Practical guidance for readers

- Use the **base system** to learn stage APIs and the classic trend+carry flow.
- Use the **rob system** for anything production-like: volatility attenuation, cost-aware optimisation, and overlays matter.
- When extending from the book: add vol attenuation, cross-sectional signals, per-instrument weights, then optimisation. Cross-check the blog series on correlations, small accounts, and cost-aware optimisation for the rationale.
- Keep data hygiene tight (prices, costs, FX). Missing data triggers pruning in static reports and changes correlations/diversification multipliers. Rob’s blog posts on adding instruments (e.g., [2021 “adding new instruments”](https://qoppac.blogspot.com/2021/05/adding-new-instruments-or-how-i-learned.html)) are a useful operational companion.
