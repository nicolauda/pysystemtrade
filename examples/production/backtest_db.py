"""
Example entrypoint for running a DB-backed futures backtest.

Prereqs:
- Configure your `private_config.yaml` (or `PYSYS_PRIVATE_CONFIG_DIR`) with
  `parquet_store` and optional Mongo credentials.
- Seed Mongo/Parquet with the data you want to backtest.

Usage (from repo root):
    python examples/production/backtest_db.py

Naming convention for zero-config runs:
- If your entrypoint is named <name>_backtest.py in some folder, and you place
  a <name>_config.yaml in the same folder, the runner will auto-discover it.
- Otherwise, you must pass --config (or set BacktestConfig.config_path).
"""

from pathlib import Path

from sysproduction.backtesting.db_backtest_runner import BacktestConfig, run_backtest

CONFIG_FILE = Path(__file__).resolve().with_name("futuresconfig.yaml")


def main() -> None:
    cfg = BacktestConfig(config_path=CONFIG_FILE)
    run_backtest(cfg)


if __name__ == "__main__":
    main()
