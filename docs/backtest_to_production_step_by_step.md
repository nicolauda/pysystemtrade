# Step-by-step: Backtest to Production (Mongo + Parquet)

This recipe assumes a Linux host (matching the provided scripts and crontab), but the flow is portable to other OSes with equivalent scheduling and path tweaks. It shows how to move from a clean checkout to a live-ready stack using MongoDB for state and Parquet for time-series. It bridges the [backtesting guide](/docs/backtesting.md) and the [production manual](/docs/production.md) with a single path that relies on the built-in futures system. Each step links to the main docs so you can dive deeper and includes validation checks so you know if you’re ready to proceed.

Table of Contents
=================

- [Prerequisites](#prerequisites)
- [Step 0: Environment variables and directories](#step-0-environment-variables-and-directories)
- [Step 1: Install and set private config](#step-1-install-and-set-private-config)
- [Step 2: Bring up Mongo and Parquet stores](#step-2-bring-up-mongo-and-parquet-stores)
- [Step 3: Seed data into Mongo/Parquet](#step-3-seed-data-into-mongoparquet)
- [Step 4: Run a backtest on DB data](#step-4-run-a-backtest-on-db-data)
- [Classic vs dynamic systems](#classic-vs-dynamic-systems)
- [Step 5: System stages and continuous positioning](#step-5-system-stages-and-continuous-positioning)
- [Step 6: Shape a base strategy (weights, multipliers, optimisation)](#step-6-shape-a-base-strategy-weights-multipliers-optimisation)
- [Step 7: Promote to production](#step-7-promote-to-production)
- [Step 8: Operate and iterate](#step-8-operate-and-iterate)

## Prerequisites

- Python environment set up with `pip install --editable .` (or normal install) from the repo root.
- A running MongoDB instance and a writable Parquet directory (fast local disk is ideal).
- Interactive Brokers paper/live access (for price polling and execution) and market data subscriptions for traded instruments.
- Disk space for Parquet history plus regular backups (CSV and Mongo dumps).
- Optional but recommended: set `PYSYS_PRIVATE_CONFIG_DIR` if you want private config outside the repo tree.
- If you need fresher shipped data, use the community-maintained CSV set at https://github.com/bug-or-feature/pst-csv-data (see Step 3).
- See: [Installation](/docs/installation.md), [Production: code and configuration management](/docs/production.md#code-and-configuration-management), [Backtesting: project defaults and private configuration](/docs/backtesting.md#project-defaults-and-private-configuration).

## Step 0: Environment variables and directories

Define these early (add to your shell profile), adjusting paths to taste:

- `MONGO_DATA=/home/you/data/mongodb` (Mongo dbpath)
- `PYSYS_CODE=/home/you/pysystemtrade` (repo root)
- `SCRIPT_PATH=/home/you/pysystemtrade/sysproduction/linux/scripts` (for cron wrappers)
- `ECHO_PATH=/home/you/echos` (echo/log output)
- `MONGO_BACKUP_PATH=/home/you/data/mongo_dump` (Mongo dumps)
- `PYSYS_PRIVATE_CONFIG_DIR=/home/you/private` (optional, if private config is outside the repo)

Create the corresponding directories, plus a Parquet root (e.g. `/home/you/data/parquet`) and CSV backup/report/backtest folders if you follow the production defaults.  
See: [Production quick start env vars](/docs/production.md#quick-start-guide) and [Production: data storage](/docs/production.md#data-storage).

On Linux, plan to use `crontab` with the bundled shell wrappers in `sysproduction/linux/scripts` (see [Production: scheduling](/docs/production.md#scheduling)). On other OSes, use an equivalent scheduler and point it at the same commands.
Validation:
- `echo $PYSYS_CODE $MONGO_DATA $PYSYS_PRIVATE_CONFIG_DIR` shows sensible paths.
- `ls $SCRIPT_PATH` lists the cron wrappers; `crontab -l` contains your schedule (Linux).

## Step 1: Install and set private config

1. Create or activate your virtualenv, then install pysystemtrade.
2. Copy the sample private config and edit it:
   ```
   cp examples/production/private_config_example.yaml private/private_config.yaml
   ```
3. In `private/private_config.yaml` fill at least (override defaults where needed):
   - Broker/IB: `broker_account`, `ib_ipaddress`, `ib_port`, `ib_idoffset`
   - Storage: `mongo_host`, `mongo_db`, `parquet_store`, `backtest_store_directory`, `csv_backup_directory`, `mongo_dump_directory`, `echo_directory`
   - Capital/portfolio: `percentage_vol_target`, `notional_trading_capital`, `base_currency`
   - Strategy routing: `strategy_list` (or include strategy config YAMLs you will call from production scripts), `strategy_weights` (used by `strategy_capital_allocation`)
   - Notifications/logging: `email_*`, optional `PYSYS_LOGGING_CONFIG`
4. If your private files live outside the repo, export `PYSYS_PRIVATE_CONFIG_DIR` (Step 0) so the code can find `private_config.yaml`.
5. Keep CSV configuration as shipped (instrument config, roll parameters) unless you have updated copies.
- See: [System defaults & private config](/docs/production.md#system-defaults--private-config), [Backtesting: project defaults and private configuration](/docs/backtesting.md#project-defaults-and-private-configuration).
Validation:
- `python - <<'PY'\nfrom sysdata.config.private_config import get_private_config_as_dict\nprint(get_private_config_as_dict())\nPY` prints the merged private config without missing keys.
- Directories in `parquet_store`, `backtest_store_directory`, `mongo_dump_directory`, `csv_backup_directory`, `echo_directory` exist and are writable.

## Step 2: Bring up Mongo and Parquet stores

1. Start MongoDB against your chosen data directory, e.g.:
   ```
   mongod --dbpath /path/to/mongo_data --port 27017
   ```
2. Create and note the Parquet root you set in `parquet_store`, e.g.:
   ```
   mkdir -p /path/to/pst_parquet
   ```
3. If you keep private config elsewhere, export `PYSYS_PRIVATE_CONFIG_DIR=/path/to/private`.
4. At this point the code will be able to resolve Mongo/Parquet locations, but the stores are still empty.
- See: [Production: data storage](/docs/production.md#data-storage), [Data: storing and representing futures data](/docs/data.md#part-3-storing-and-representing-futures-data).
Validation:
- `mongo --host $mongo_host --port $mongo_port --eval "db.stats()"` succeeds.
- Parquet root exists and is writable: `touch /path/to/pst_parquet/_test && rm /path/to/pst_parquet/_test`.

## Step 3: Seed data into Mongo/Parquet

Populate Mongo (for spread costs and state) and Parquet (for prices) so backtests and production share the same storage.

1. Load spread costs into Mongo (used by both backtests and live):
   ```
   python -m sysinit.futures.repocsv_spread_costs
   ```
2. Load spot FX prices into Parquet:
   ```
   python -m sysinit.futures.repocsv_spotfx_prices
   ```
3. Load shipped multiple and adjusted futures prices from CSV into Parquet (safe starting point before you have fresher data):
   ```
   python -m sysinit.futures.multiple_and_adjusted_from_csv_to_db
   ```
   This writes to the Parquet store defined in `parquet_store`.
   - If the built-in CSVs are too old, clone the updated dataset:
     ```
     git clone https://github.com/bug-or-feature/pst-csv-data.git /path/to/pst-csv-data
     ```
     Then load from the fresher CSVs:
     ```
     python - <<'PY'
     from sysinit.futures.multiple_and_adjusted_from_csv_to_db import init_db_with_csv_futures_contract_prices
     init_db_with_csv_futures_contract_prices(
         multiple_price_datapath="/path/to/pst-csv-data/multiple_prices_csv",
         adj_price_datapath="/path/to/pst-csv-data/adjusted_prices_csv",
     )
     PY
     ```
4. Backfill contract-level history for instruments you plan to trade:
   - From IB: `python -m sysinit.futures.seed_price_data_from_IB` (interactive; run per instrument).
   - Or from CSV/API downloads: see `sysinit/futures/contract_prices_from_csv_to_db.py` and the Barchart flow in [data.md](/docs/data.md#getting-historical-data-for-individual-futures-contracts).
5. Quick sanity check that Parquet and Mongo are reachable:
   ```python
   from sysdata.sim.db_futures_sim_data import dbFuturesSimData
   data = dbFuturesSimData()
   print(data.get_instrument_list()[:5])
   print(data.data.config.get_element("parquet_store"))
   ```
- See: [Data: dbFuturesSimData](/docs/data.md#using-dbfuturessimdata), [Data: MongoDB](/docs/data.md#mongodb), [Data: Parquet](/docs/data.md#parquet).
Validation:
- Spot FX codes and futures instruments load: `dbFuturesSimData().get_instrument_list()` returns non-empty; pick an instrument and ensure `get_multiple_prices` returns data with recent dates.
- Spread costs present in Mongo (if applicable): check via `mongo` shell on the relevant collection or use `sysproduction.data.prices.diagPrices().db_spread_cost_data.get_spread_costs()`.

## Step 4: Run a backtest on DB data

Use the built-in Chapter 15 system but point it at Mongo/Parquet via `dbFuturesSimData` so the results match what production will see.

```python
from sysdata.sim.db_futures_sim_data import dbFuturesSimData
from systems.provided.futures_chapter15.estimatedsystem import futures_system

data = dbFuturesSimData()
system = futures_system(data=data)

# Basic perf check
print(system.accounts.portfolio().stats())
print(system.accounts.portfolio().sharpe())
```

Backtests will cache into the directories defined by `backtest_store_directory` in defaults/private config.
- See: [Backtesting: create a futures backtest](/docs/backtesting.md#how-do-icreate-a-standard-futures-backtest), [Backtesting: using pre-baked systems](/docs/backtesting.md#pre-baked-systems), [Data: using dbFuturesSimData](/docs/data.md#using-dbfuturessimdata).
Validation:
- `system.accounts.portfolio().stats()` returns sensible values (no NaNs/Infs), and the instrument count matches the data you loaded.
- Re-run and confirm caches are used (second run faster) and identical outputs.

## Classic vs dynamic systems

- Classic system (book): Mirrors the "Staunch Systems Trader" in Robert Carver, *Systematic Trading* (2015) stored under `pysystemtrade-private/private/docs`. The shipped config `systems/provided/futures_chapter15/futuresconfig.yaml` uses the same six instruments (Eurodollar, US 5yr, Euro Stoxx, V2X, MXP, Corn), the EWMAC stack plus carry, a 20% vol target, and a forecast cap of 20. Production hooks: `sysproduction/strategy_code/run_system_classic.py`, `sysexecution/strategies/classic_buffered_positions.py`, and `sysproduction/strategy_code/report_system_classic.py`. Use this when you want one-to-one parity with the book before layering extras.
- Dynamic system (cost-aware): Adds the `optimisedPositions` stage from `systems.provided.dynamic_small_system_optimise` so raw optimal positions (often from the same Chapter 15-style config) are optimised against cost, speed, and constraints. `run_dynamic_optimised_system.py` writes raw optimal positions; `sysexecution/strategies/dynamic_optimised_positions.py` then runs a greedy optimiser using shadow costs, reduce-only/don't-trade overrides, position limits, and speed control to smooth turnover.
- Researching it: swap in the dynamic pipeline on DB data to see the optimisation effect:
  ```python
  from sysdata.sim.db_futures_sim_data import dbFuturesSimData
  from sysproduction.strategy_code.run_dynamic_optimised_system import dynamic_system

  data = dbFuturesSimData()
  dyn = dynamic_system(data=data, config_filename="private/my_futuresconfig.yaml")
  print(dyn.optimisedPositions.get_optimised_weights_df().tail())
  ```
  Reuse the same config you validated in Step 4; the optimiser works on the raw optimal positions your backtest produces.
- Choosing: start with the classic flow to match the book and debug data; move to dynamic when you need tighter cost/turnover control or when you rely on production controls (shadow cost in `private_config.yaml`, reduce-only/don't-trade flags, position limits) to steer orders. Validate by checking that raw and optimised positions land in Mongo/Parquet and that optimisation logs show the expected constraints.

## Step 5: System stages and continuous positioning

- Rob’s standard pipeline (see [Introduction](/docs/introduction.md)): preprocess raw data → run trading rules → scale/cap forecasts → combine forecasts → size positions → build the portfolio → compute P&L. Pre-baked systems follow this structure; you typically configure stages via YAML (which rules, caps, scaling, buffers, weights) rather than editing Python.
- PST uses *continuous* forecasts and positions. Capping (`forecast_cap`) and scaling (`forecast_scalar`, `average_absolute_forecast`) happen before combining forecasts and sizing positions. Positions remain continuous; rounding to tradable lots happens in the execution stack, not in portfolio construction.
- Buffering and speed limits: use the buffering/inertia settings in position sizing to avoid over-trading; turnover and cost controls work with the caps/scalars to keep forecasts and positions stable.
- Validate that adjusted prices and roll calendars are correct, since continuous positioning relies on clean stitched series; see [Data: roll calendars](/docs/data.md#roll-calendars) and [Data: creating and storing back adjusted prices](/docs/data.md#creating-and-storing-back-adjusted-prices).
- Run an order-level backtest (`systems.provided.example.daily_with_order_simulation.futures_system` with `dbFuturesSimData`) to see how continuous desired positions become discrete orders with broker constraints (lot size, min tick), and to confirm capping/buffering behave as expected.
- See: [Backtesting: Stage – Forecast scale and cap](/docs/backtesting.md#stage-forecast-scale-and-cap), [Backtesting: Stage – Position scaling](/docs/backtesting.md#stage-position-scaling), [Backtesting: buffering and position inertia](/docs/backtesting.md#buffering-and-position-inertia), [Production: strategy order handling](/docs/production.md#strategy-order-handling).

## Step 6: Shape a base strategy (weights, multipliers, optimisation)

1. Start from `systems/provided/futures_chapter15/futuresconfig.yaml` (or `futuresestimateconfig.yaml`). Copy to your private area so edits are not overwritten:
   ```
   cp systems/provided/futures_chapter15/futuresconfig.yaml private/my_futuresconfig.yaml
   ```
2. Edit the private config:
   - Instrument universe and weights; set fixed `instrument_weights` or enable `use_instrument_weight_estimates/use_instrument_div_mult_estimates` to optimise weights and instrument diversification multipliers (IDM) from returns.
   - Forecast-level settings: set fixed `forecast_scalar/forecast_cap`, or enable `use_forecast_weight_estimates/use_forecast_div_mult_estimates` to optimise forecast weights and diversification multipliers (FDM); adjust `forecast_weight_estimate` and `instrument_weight_estimate` blocks if you want different optimisers from `sysquant.optimisation`.
   - Vol target, capital, buffering, speed/cost parameters, long-only flags, and any rule list changes.
3. For order-level realism in backtests, use the example order-simulation system (`systems.provided.example.daily_with_order_simulation.futures_system`) with `dbFuturesSimData`.
4. Iterate: run the snippet from Step 4 pointing `Config` to your private YAML, inspect portfolio stats, and stabilise weights/div multipliers before freezing the config for production.
- See: [Backtesting: Stage – Forecast combine](/docs/backtesting.md#stage-forecast-combine) (forecast weights/div multiplier), [Backtesting: Stage – Creating portfolios](/docs/backtesting.md#stage-creating-portfolios) (instrument weights/IDM), [Backtesting: Stage – Position scaling](/docs/backtesting.md#stage-position-scaling) (vol target), [Production: finalise your backtest configuration](/docs/production.md#finalise-your-backtest-configuration).
Validation:
- Check forecasts are capped and scaled: inspect `system.forecastScaleCap.get_forecast_scaled("EDOLLAR", "ewmac64")` (replace with a rule/instrument you have).
- Confirm IDM/FDM are as expected: `system.portfolio.get_instrument_diversification_multiplier()` and `system.portfolio.get_forecast_diversification_multiplier()`.
- Inspect turnover/cost: `system.accounts.portfolio().turnover_summary()` and `system.accounts.portfolio().costs_breakdown()` (or equivalent metrics).

## Step 7: Promote to production

Run the daily stack with Mongo + Parquet (these scripts already use the Parquet/Mongo classes defined in `sysproduction.data.production_data_objects`):

1. Data updates
   - `python -m sysproduction.run_daily_fx_and_contract_updates`
   - `python -m sysproduction.run_daily_price_updates`
   - `python -m sysproduction.run_daily_update_multiple_adjusted_prices`
2. Strategy refresh and orders
   - `python -m sysproduction.run_systems` (refresh backtests on latest data)
   - `python -m sysproduction.run_strategy_order_generator` (desired positions and strategy orders)
   - `python -m sysproduction.run_stack_handler` (creates broker orders, manages fills)
3. Reporting and hygiene
   - `python -m sysproduction.run_reports`
   - `python -m sysproduction.run_backups` (Mongo dumps, Parquet/CSV backups)

Keep the same `mongo_db`/`parquet_store` values as in your backtests so live runs and research stay in sync.
- Before going live, ensure the production `strategy_list` and configs point to the same YAML you validated in Step 6; rerun `run_systems` + `run_strategy_order_generator` after any config change to push the new desired positions.
- On Linux you can call the ready-made bash wrappers in `sysproduction/linux/scripts` from `cron` (see [Production: scheduling](/docs/production.md#scheduling)) instead of invoking the Python modules directly; the Python commands above remain valid on any OS or when running ad-hoc.
- See: [Production system data flow](/docs/production.md#production-system-data-flow) and [Core production system components](/docs/production.md#core-production-system-components) for how these scripts fit together; [Production: linking to a broker](/docs/production.md#linking-to-a-broker) and [IB](/docs/IB.md) for connectivity specifics.
Validation:
- Run each script once manually and confirm no critical errors in logs/echos.
- Confirm orders are not sent when in test mode (paper IB or controls set to block live trading).
- Crontab includes the desired schedule; logs/echos rotate as expected.

## Step 8: Operate and iterate

- Use the interactive tools for checks: `sysproduction.interactive_diagnostics`, `interactive_controls`, and `interactive_order_stack`.
- Monitor echoes/logs in the paths from `echo_directory` and log settings in your config.
- When you change strategy configuration, re-run `run_systems` then `run_strategy_order_generator` to push the new desired positions; the stack handler will take it from there.
- For deeper topics (dashboard, roll calendars, broker specifics) see [production.md](/docs/production.md), [data.md](/docs/data.md), and [IB.md](/docs/IB.md).
- See: [Production: interactive scripts](/docs/production.md#interactive-scripts), [Production: dashboard and monitor](/docs/dashboard_and_monitor.md), [Data: roll calendars](/docs/data.md#roll-calendars).
- Validation:
  - Monitoring shows fresh prices, positions, and no stale processes; dashboards or `interactive_diagnostics` reflect expected state.
  - Backups complete (Mongo dump and Parquet/CSV backups) and are restorable (spot-check a dump or Parquet file).
