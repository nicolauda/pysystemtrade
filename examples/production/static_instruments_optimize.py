"""Minimal wrapper to run static instrument optimisation against any config file.

Run via `python -m pysystemtrade.examples.production.static_instruments_optimize`
(inherits the CLI of `static_system_modular`: report by default; add `--no-report`
to print instrument lists).

To skip CLI arguments, set `USE_GLOBAL_SETTINGS = True` and adjust the globals:
- USE_GLOBAL_SETTINGS: toggle programmatic run vs CLI passthrough.
- GLOBAL_CONFIG_FILENAME: config source. None => auto-discover next to this
  script (`<script_stem>_config.yaml`, or `<folder>_config.yaml`); "rob_system"
  => use the provided rob_system config; anything else => treat as a path or
  dot-path.
- GLOBAL_CAPITALS: capital levels (list or single float).
- GLOBAL_ESTIMATED_COUNTS: estimated instrument counts (list or single int).
- GLOBAL_USE_DB_CAPITAL: pull capital from DB when capitals are not provided.
- GLOBAL_USE_ALL_SAMPLED_INSTRUMENTS: ignore config instrument list and use all
  sampled instruments.
- GLOBAL_REPORT: True writes the report; False prints instrument lists only.
- GLOBAL_TITLE_SUFFIX: optional custom suffix added to the report title/filename.
  By default the suffix is auto-built from flags (`use_db_capital`,
  `all_sampled_instruments`, `config_<basename>`). Example with defaults:
  `--use-db-capital --config my.yaml` => title "Static selection of instruments
  use_db_capital_config_my.yaml_report"; with `GLOBAL_TITLE_SUFFIX="20240501"`
  => "Static selection of instruments 20240501".
- GLOBAL_REPORT_IN_CONFIG_DIR: True writes the report next to the resolved
  config file; False uses the default reporting directory.
"""

from sysproduction.reporting.adhoc import static_system_modular as static_mod

USE_GLOBAL_SETTINGS = False
GLOBAL_CONFIG_FILENAME = None
GLOBAL_CAPITALS = None
GLOBAL_ESTIMATED_COUNTS = None
GLOBAL_USE_DB_CAPITAL = False
GLOBAL_USE_ALL_SAMPLED_INSTRUMENTS = False
GLOBAL_REPORT = True
GLOBAL_TITLE_SUFFIX = None
GLOBAL_REPORT_IN_CONFIG_DIR = False


def main():
    if USE_GLOBAL_SETTINGS:
        static_mod.static_system_modular(
            capital=GLOBAL_CAPITALS,
            estimated_count=GLOBAL_ESTIMATED_COUNTS,
            use_db_capital=GLOBAL_USE_DB_CAPITAL,
            config_filename=GLOBAL_CONFIG_FILENAME,
            use_all_sampled_instruments=GLOBAL_USE_ALL_SAMPLED_INSTRUMENTS,
            report=GLOBAL_REPORT,
            title_suffix=GLOBAL_TITLE_SUFFIX,
            report_in_config_dir=GLOBAL_REPORT_IN_CONFIG_DIR,
        )
        return

    static_mod.main()


if __name__ == "__main__":
    main()
