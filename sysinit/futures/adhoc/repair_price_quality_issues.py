from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import pandas as pd

from syscore.constants import arg_not_supplied
from sysdata.data_blob import dataBlob
from sysproduction.data.prices import diagPrices
from sysproduction.reporting.data.price_quality import (
    ALL_DB_PRICE_QUALITY_INSTRUMENTS,
    DEFAULT_DB_PRICE_QUALITY_CALENDAR_DAYS,
    DEFAULT_DB_PRICE_QUALITY_MIN_BAD_DAYS,
    QUALITY_STATUS_CLEAN,
    resolve_db_price_quality_dates,
)
from sysinit.futures.adhoc.price_quality_issues import (
    PriceQualityRepairPlan,
    build_price_quality_repair_plan,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Conservatively repair DB multiple/adjusted price-quality issues "
            "with dry-run or apply modes."
        )
    )
    parser.add_argument(
        "--instrument",
        dest="instrument_code",
        default=ALL_DB_PRICE_QUALITY_INSTRUMENTS,
        help="Instrument code to repair. Default: all instruments.",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Inclusive lower bound in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Inclusive upper bound in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--calendar-days-back",
        type=int,
        default=DEFAULT_DB_PRICE_QUALITY_CALENDAR_DAYS,
        help="Rolling lookback window used when explicit dates are not supplied.",
    )
    parser.add_argument(
        "--full-history",
        action="store_true",
        help="Repair against full history instead of a bounded date window.",
    )
    parser.add_argument(
        "--min-bad-days",
        type=int,
        default=DEFAULT_DB_PRICE_QUALITY_MIN_BAD_DAYS,
        help="Action threshold used when scoring candidates.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply repairs. Default mode is dry-run.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path for JSON output.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional path for CSV output.",
    )
    parser.add_argument(
        "--backup-dir",
        default=None,
        help=(
            "Optional directory for pre-repair backups. In apply mode a timestamped "
            "directory under tmp/ is used when not provided."
        ),
    )
    return parser.parse_args()


def _resolve_window(
    args: argparse.Namespace,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None, str]:
    if args.full_history:
        return resolve_db_price_quality_dates(
            start_date=arg_not_supplied,
            end_date=arg_not_supplied,
            calendar_days_back=arg_not_supplied,
        )

    start_date = pd.Timestamp(args.start_date) if args.start_date else arg_not_supplied
    end_date = pd.Timestamp(args.end_date) if args.end_date else arg_not_supplied
    calendar_days_back = args.calendar_days_back
    if args.start_date or args.end_date:
        calendar_days_back = arg_not_supplied

    return resolve_db_price_quality_dates(
        start_date=start_date,
        end_date=end_date,
        calendar_days_back=calendar_days_back,
    )


def _instrument_list(data: dataBlob, instrument_code: str) -> list[str]:
    if instrument_code != ALL_DB_PRICE_QUALITY_INSTRUMENTS:
        return [code.strip() for code in instrument_code.split(",") if code.strip()]

    return sorted(diagPrices(data).get_list_of_instruments_in_multiple_prices())


def _default_backup_dir() -> Path:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("tmp") / f"repair_price_quality_issues_{timestamp}"


def _backup_plan_state(plan: PriceQualityRepairPlan, backup_dir: Path):
    backup_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(plan.current_multiple_prices).to_csv(
        backup_dir / f"{plan.instrument_code}_multiple_before.csv"
    )
    pd.Series(plan.current_adjusted_prices).to_csv(
        backup_dir / f"{plan.instrument_code}_adjusted_before.csv",
        header=["price"],
    )


def _plan_to_record(plan: PriceQualityRepairPlan, applied: bool) -> dict[str, object]:
    current_quality = plan.current_candidate.quality
    selected_quality = plan.selected_candidate.quality

    return {
        "instrument": plan.instrument_code,
        "current_status": current_quality["status"],
        "selected_status": selected_quality["status"],
        "selected_strategy": plan.selected_candidate.label,
        "current_bad_days": int(current_quality["bad_days"]),
        "selected_bad_days": int(selected_quality["bad_days"]),
        "current_bad_rows": int(current_quality["bad_rows"]),
        "selected_bad_rows": int(selected_quality["bad_rows"]),
        "selected_full_gap_days": int(selected_quality["full_gap_days"]),
        "selected_full_gap_rows": int(selected_quality["full_gap_rows"]),
        "current_adjusted_missing_rows": int(
            plan.current_adjusted_alignment["missing_adjusted_rows"]
        ),
        "current_adjusted_mismatch_rows": int(
            plan.current_adjusted_alignment["adjusted_mismatch_rows"]
        ),
        "write_multiple": bool(plan.write_multiple),
        "write_adjusted": bool(plan.write_adjusted),
        "applied": bool(applied),
    }


def _write_optional_outputs(records: list[dict[str, object]], args: argparse.Namespace):
    if len(records) == 0:
        return

    output_df = pd.DataFrame(records)
    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(output_path, index=False)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(records, indent=2, default=str),
            encoding="utf-8",
        )


def main():
    args = _parse_args()
    start_date, end_date, window_label = _resolve_window(args)

    backup_dir = None
    if args.apply:
        backup_dir = Path(args.backup_dir) if args.backup_dir else _default_backup_dir()
        backup_dir.mkdir(parents=True, exist_ok=True)

    with dataBlob(log_name="Repair-Price-Quality-Issues") as data:
        instrument_codes = _instrument_list(data, args.instrument_code)
        records = []
        prices = diagPrices(data)
        for instrument_code in instrument_codes:
            plan = build_price_quality_repair_plan(
                data=data,
                instrument_code=instrument_code,
                start_date=start_date,
                end_date=end_date,
                min_bad_days=args.min_bad_days,
            )

            should_apply = args.apply and (plan.write_multiple or plan.write_adjusted)
            if should_apply:
                _backup_plan_state(plan, backup_dir)
                if plan.write_multiple:
                    prices.db_futures_multiple_prices_data.add_multiple_prices(
                        instrument_code,
                        plan.selected_multiple_prices,
                        ignore_duplication=True,
                    )
                if plan.write_adjusted:
                    prices.db_futures_adjusted_prices_data.add_adjusted_prices(
                        instrument_code,
                        plan.rebuilt_adjusted_prices,
                        ignore_duplication=True,
                    )

            record = _plan_to_record(plan, applied=should_apply)
            if (
                (not should_apply)
                and record["current_status"] == QUALITY_STATUS_CLEAN
                and record["selected_status"] == QUALITY_STATUS_CLEAN
                and not plan.write_adjusted
            ):
                continue
            records.append(record)

    if len(records) == 0:
        print(f"No repairs proposed for window {window_label}.")
        return

    output_df = pd.DataFrame(records).sort_values(
        by=["current_bad_days", "current_bad_rows"],
        ascending=[False, False],
    )
    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"Mode: {mode}")
    print(f"Window: {window_label}")
    if backup_dir is not None:
        print(f"Backup directory: {backup_dir}")
    print(output_df.to_string(index=False))
    _write_optional_outputs(records, args)


if __name__ == "__main__":
    main()
