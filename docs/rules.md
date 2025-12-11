## Provided trading rules

These functions live under `systems.provided.rules` and are referenced in YAML under `trading_rules:` as `function: systems.provided.rules.<module>.<callable>`. Arguments go in `other_args`, and data series are pulled by name from `rawdata` in `data:`. Outputs are raw forecasts (unscaled, uncapped) that your system later clips and rescales.

Example pattern (taken from `pysystemtrade/systems/provided/rob_system/config.yaml`):
```yaml
trading_rules:
  breakout80:
    function: systems.provided.rules.breakout.breakout
    data:
      - "rawdata.get_daily_prices"
    other_args:
      lookback: 80
```

### Breakout (`breakout.breakout`)
Config names: `breakout10`, `breakout20`, `breakout40`, `breakout80`, `breakout160`, `breakout320` (different `lookback`).
- Step 1: rolling high/low over `lookback` (min periods is half of available data or half of lookback).  
- Step 2: compute mid = (high + low) / 2.  
- Step 3: normalise distance: `40 * (price - mid) / (high - low)` to get a naturally scaled score.  
- Step 4: EWMA smooth with span `smooth` (default `lookback/4`).  
- Output: smoothed breakout score.  
Code:
```python
roll_max = price.rolling(lookback, min_periods=int(min(len(price), np.ceil(lookback/2.0)))).max()
roll_min = price.rolling(lookback, min_periods=int(min(len(price), np.ceil(lookback/2.0)))).min()
roll_mean = (roll_max + roll_min) / 2.0
output = 40.0 * ((price - roll_mean) / (roll_max - roll_min))
smoothed_output = output.ewm(span=smooth, min_periods=np.ceil(smooth/2.0)).mean()
```
Python:
```python
from systems.provided.rules.breakout import breakout
signal = breakout(price_series, lookback=80)  # optional smooth overrides default
```
YAML example:
```yaml
trading_rules:
  breakout80:
    function: systems.provided.rules.breakout.breakout
    data:
      - "rawdata.get_daily_prices"
    other_args:
      lookback: 80
```

### EWMAC trend (`ewmac.ewmac` and `ewmac.ewmac_calc_vol`)
Config names:  
- `momentum4/8/16/32/64` call `ewmac.ewmac` with price + precomputed vol (`rawdata.get_daily_prices`, `rawdata.daily_returns_volatility`).  
- `normmom2/4/8/16/32/64` and `assettrend2/4/8/16/32/64` call `ewmac.ewmac_calc_vol`, which computes volatility internally on the supplied price series (normalised returns or asset-class price).
Algorithm (ewmac core):
- Step 1: fast EWMA of price with span `Lfast`; slow EWMA with span `Lslow`.  
- Step 2: subtract to get raw crossover.  
- Step 3: divide by price volatility (passed in or recomputed with `robust_vol_calc` on price diffs for `ewmac_calc_vol`).  
- Output: volatility-adjusted trend forecast.  
Code:
```python
fast_ewma = price.ewm(span=Lfast, min_periods=1).mean()
slow_ewma = price.ewm(span=Lslow, min_periods=1).mean()
raw_ewmac = fast_ewma - slow_ewma
return raw_ewmac / vol.ffill()              # ewmac

vol = robust_vol_calc(price.diff(), vol_days)
forecast = ewmac(price, vol, Lfast, Lslow)  # ewmac_calc_vol
```
Python:
```python
from systems.provided.rules.ewmac import ewmac, ewmac_calc_vol
signal_direct = ewmac(price_series, vol_series, Lfast=32, Lslow=128)
signal_recalc_vol = ewmac_calc_vol(price_series, Lfast=32, Lslow=128, vol_days=35)
```
YAML examples:
```yaml
  momentum32:
    function: systems.provided.rules.ewmac.ewmac
    data:
      - "rawdata.get_daily_prices"
      - "rawdata.daily_returns_volatility"
    other_args:
      Lfast: 32
      Lslow: 128

  normmom32:
    function: systems.provided.rules.ewmac.ewmac_calc_vol
    data:
      - "rawdata.get_cumulative_daily_vol_normalised_returns"
    other_args:
      Lfast: 32
      Lslow: 128
```

### Acceleration (`accel.accel`)
Config names: `accel16`, `accel32`, `accel64`.
- Step 1: build an EWMAC with `Lslow = 4 * Lfast` on price and vol.  
- Step 2: compare today’s EWMAC to the value `Lfast` days ago (`signal - signal.shift(Lfast)`).  
- Output: positive when momentum is speeding up, negative when it is fading.  
Code:
```python
Lslow = Lfast * 4
ewmac_signal = ewmac(price, vol, Lfast, Lslow)
accel = ewmac_signal - ewmac_signal.shift(Lfast)
```
Python:
```python
from systems.provided.rules.accel import accel
signal = accel(price_series, vol_series, Lfast=32)
```
YAML example:
```yaml
  accel32:
    function: systems.provided.rules.accel.accel
    data:
      - "rawdata.get_daily_prices"
      - "rawdata.daily_returns_volatility"
    other_args:
      Lfast: 32
```

### Cross-asset trend (`assettrend*` variants)
EWMACs (`ewmac_calc_vol`) run on the asset-class composite price (`rawdata.normalised_price_for_asset_class`) instead of the instrument’s own price.
- Step 1: compute fast/slow EWMAs of the asset-class normalised price (`Lfast`, `Lslow`).  
- Step 2: subtract fast − slow to get raw trend.  
- Step 3: recompute vol internally on price diffs with `robust_vol_calc`; divide raw trend by this vol.  
- Output: asset-class-level trend applied to the instrument.  
Code:
```python
vol = robust_vol_calc(price.diff(), vol_days)
forecast = ewmac(price, vol, Lfast, Lslow)
```
Python:
```python
from systems.provided.rules.ewmac import ewmac_calc_vol
signal = ewmac_calc_vol(asset_class_price, Lfast=8, Lslow=32, vol_days=35)
```
YAML example:
```yaml
  assettrend8:
    function: systems.provided.rules.ewmac.ewmac_calc_vol
    data:
      - "rawdata.normalised_price_for_asset_class"
    other_args:
      Lfast: 8
      Lslow: 32
```

### Normalised momentum (`normmom*` variants)
EWMACs (`ewmac_calc_vol`) applied to instrument prices that are already normalised by their own volatility (`rawdata.get_cumulative_daily_vol_normalised_returns`).
- Step 1: fast/slow EWMAs on the vol-normalised price.  
- Step 2: subtract to get raw trend.  
- Step 3: recompute vol on that series’ diffs and divide, giving a doubly vol-aware momentum.  
- Output: trend on an already volatility-adjusted price path.  
Code:
```python
vol = robust_vol_calc(price.diff(), vol_days)
forecast = ewmac(price, vol, Lfast, Lslow)
```
Python:
```python
from systems.provided.rules.ewmac import ewmac_calc_vol
signal = ewmac_calc_vol(norm_price, Lfast=16, Lslow=64, vol_days=35)
```
YAML example:
```yaml
  normmom16:
    function: systems.provided.rules.ewmac.ewmac_calc_vol
    data:
      - "rawdata.get_cumulative_daily_vol_normalised_returns"
    other_args:
      Lfast: 16
      Lslow: 64
```

### Relative momentum (`rel_mom.relative_momentum`)
Config names: `relmomentum10/20/40/80`.
- Inputs: instrument normalised price (`rawdata.get_cumulative_daily_vol_normalised_returns`) and asset-class normalised price (`rawdata.normalised_price_for_asset_class`).  
- Step 1: outperformance = instrument - asset-class, forward-filled.  
- Step 2: compute average outperformance over `horizon`: `(outperf - outperf.shift(horizon)) / horizon`.  
- Step 3: EWMA smooth over `ewma_span` (default `horizon/4`, min 2).  
- Output: positive when the instrument has recently outperformed peers.  
Code:
```python
outperformance = normalised_price_this_instrument.ffill() - normalised_price_for_asset_class.ffill()
average_outperformance_over_horizon = (outperformance - outperformance.shift(horizon)) / horizon
forecast = average_outperformance_over_horizon.ewm(span=ewma_span).mean()
```
Python:
```python
from systems.provided.rules.rel_mom import relative_momentum
signal = relative_momentum(instr_norm_price, asset_class_norm_price, horizon=40)
```
YAML example:
```yaml
  relmomentum40:
    function: systems.provided.rules.rel_mom.relative_momentum
    data:
      - "rawdata.get_cumulative_daily_vol_normalised_returns"
      - "rawdata.normalised_price_for_asset_class"
    other_args:
      horizon: 40
```

### Cross-sectional mean reversion (`cs_mr.cross_sectional_mean_reversion`)
Config name: `mrinasset1000` (long horizon).  
- Inputs: same pair as relative momentum.  
- Step 1: outperformance = instrument - asset-class, forward-filled.  
- Step 2: relative returns = `outperformance.diff()`.  
- Step 3: rolling mean of relative returns over `horizon`.  
- Step 4: EWMA smooth (default `horizon/4`, min 2) and negate to bet on reversion toward peers.  
- Output: negative when recent relative returns were positive, encouraging a mean-reversion stance.  
Code:
```python
outperformance = normalised_price_this_instrument.ffill() - normalised_price_for_asset_class.ffill()
relative_return = outperformance.diff()
outperformance_over_horizon = relative_return.rolling(horizon).mean()
forecast = -outperformance_over_horizon.ewm(span=ewma_span).mean()
```
Python:
```python
from systems.provided.rules.cs_mr import cross_sectional_mean_reversion
signal = cross_sectional_mean_reversion(instr_norm_price, asset_class_norm_price, horizon=250)
```
YAML example:
```yaml
  mrinasset1000:
    function: systems.provided.rules.cs_mr.cross_sectional_mean_reversion
    data:
      - "rawdata.get_cumulative_daily_vol_normalised_returns"
      - "rawdata.normalised_price_for_asset_class"
    other_args:
      horizon: 1000
```

### Carry (`carry.carry`)
Config names: `carry10/30/60/125`.
- Input: `rawdata.raw_carry` (annualised rolldown Sharpe).  
- Step 1: EWMA smooth over `smooth_days`.  
- Output: smoother carry forecast.  
Code:
```python
smooth_carry = raw_carry.ewm(smooth_days).mean()
```
Python:
```python
from systems.provided.rules.carry import carry
signal = carry(raw_carry_series, smooth_days=60)
```
YAML example:
```yaml
  carry60:
    function: systems.provided.rules.carry.carry
    data:
      - "rawdata.raw_carry"
    other_args:
      smooth_days: 60
```

### Relative carry (`carry.relative_carry`)
Config name: `relcarry`.
- Inputs: `rawdata.smoothed_carry` for the instrument and `rawdata.median_carry_for_asset_class`.  
- Step: subtract asset-class median from instrument carry to express relative attractiveness.  
- Output: positive when the instrument’s carry exceeds its peers.  
Code:
```python
relative_carry_forecast = smoothed_carry_this_instrument - median_carry_for_asset_class
```
Python:
```python
from systems.provided.rules.carry import relative_carry
signal = relative_carry(smoothed_carry, asset_class_median_carry)
```
YAML example:
```yaml
  relcarry:
    function: systems.provided.rules.carry.relative_carry
    data:
      - "rawdata.smoothed_carry"
      - "rawdata.median_carry_for_asset_class"
```

### Factor rules (`factors.factor_trading_rule`)
Config names: `skewabs180/365`, `skewrv180/365` (different lookbacks and demeaning methods).
- Inputs: `rawdata.get_demeanded_factor_value` retrieves a demeaned factor based on `_factor_name`, `_demean_method`, `_lookback_days`.  
- Step 1: estimate factor vol with `robust_vol_calc`.  
- Step 2: normalise factor by its vol.  
- Step 3: EWMA smooth over `smooth` days.  
- Output: a volatility-normalised factor signal; demeaning settings control whether you compare to global history or asset-class peers.  
Code:
```python
vol = robust_vol_calc(demean_factor_value)
normalised_factor_value = demean_factor_value / vol
smoothed_normalised_factor_value = normalised_factor_value.ewm(span=smooth).mean()
```
Python:
```python
from systems.provided.rules.factors import factor_trading_rule
signal = factor_trading_rule(demeaned_factor_series, smooth=90)
```
YAML example:
```yaml
  skewabs365:
    function: systems.provided.rules.factors.factor_trading_rule
    data:
      - "rawdata.get_demeanded_factor_value"
    other_args:
      smooth: 90
      _factor_name: "neg_skew"
      _demean_method: "historic_average_factor_value_all_assets"
      _lookback_days: 365
```

### Mean-reversion wings (`mr_wings.mr_wings`)
Not used in the Rob config but available.
- Step 1: short-term EWMAC (`Lfast`, `Lslow=4*Lfast`).  
- Step 2: rolling std of that EWMAC over a very long window (5,000 days).  
- Step 3: zero out small signals inside ±3 standard deviations.  
- Step 4: flip the sign to trade reversion only in the tails.  
- Output: contrarian signal active only when trend signals are extreme.  
Code:
```python
ewmac_signal = ewmac(price, vol, Lfast, Lslow)
ewmac_std = ewmac_signal.rolling(5000, min_periods=3).std()
ewmac_signal[ewmac_signal.abs() < ewmac_std * 3] = 0.0
mr_signal = -ewmac_signal
```
Python:
```python
from systems.provided.rules.mr_wings import mr_wings
signal = mr_wings(price_series, vol_series, Lfast=4)
```
YAML example:
```yaml
  mrwings4:
    function: systems.provided.rules.mr_wings.mr_wings
    data:
      - "rawdata.get_daily_prices"
      - "rawdata.daily_returns_volatility"
    other_args:
      Lfast: 4
```

> Legacy: `_TO_DELETE_OLD.py` holds older variants (e.g., breakout plus fixed long/short biases) and is not referenced in the config.
