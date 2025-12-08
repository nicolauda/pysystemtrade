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

Outputs: by default you get QuantStats HTML, a PDF, a txt summary, and a
backtest_output_<timestamp>.log capturing stdout/stderr under backtest_results/.
Use BacktestConfig.include_report_txt / include_debug_log (or --no-report /
--no-debug-log) to disable them.
"""

from pathlib import Path

from sysproduction.backtesting.db_backtest_runner import BacktestConfig, run_backtest

# from systems.diagoutput import systemDiag

# the configuration file can be passed both as usual in pysystemtrade with dot notation
# or with a traditional path, Path or str with the absolute path
# CONFIG_FILE = Path(__file__).resolve().with_name("futuresconfig.yaml")
# CONFIG_FILE = "~/pst/pysystemtrade/examples/production/futuresconfig.yaml"
CONFIG_FILE = "examples.production.futures_config_estimated.yaml"
# YAML_EST_PARAM = Path(__file__).resolve().parent / "backtest_results" / "estimated_weights.yaml"


def main() -> None:
    cfg = BacktestConfig(
        config_path=CONFIG_FILE, use_cache=True, include_debug_log=False
    )
    run_backtest(cfg)


if __name__ == "__main__":
    main()
