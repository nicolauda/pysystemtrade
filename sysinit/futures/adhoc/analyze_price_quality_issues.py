from __future__ import annotations

import argparse
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
            "Analyse DB multiple/adjusted price-quality issues and classify "
            "safe repair candidates."
        )
    )
    parser.add_argument(
        "--instrument",
        dest="instrument_code",
        default=ALL_DB_PRICE_QUALITY_INSTRUMENTS,
        help="Instrument code to analyse. Default: all instruments.",
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
        help=(
            "Rolling lookback window used when explicit dates are not supplied. "
            "Use 0 with explicit dates or --full-history."
        ),
    )
    parser.add_argument(
        "--full-history",
        action="store_true",
        help="Analyse full history instead of a bounded date window.",
    )
    parser.add_argument(
        "--min-bad-days",
        type=int,
        default=DEFAULT_DB_PRICE_QUALITY_MIN_BAD_DAYS,
        help="Action threshold used for the quality status.",
    )
    parser.add_argument(
        "--include-clean",
        action="store_true",
        help="Include instruments that are already clean after analysis.",
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


def _classify_root_causes(plan: PriceQualityRepairPlan) -> list[str]:
    causes = []
    current_quality = plan.current_candidate.quality
    filtered_quality = plan.filtered_candidate.quality
    selected_quality = plan.selected_candidate.quality

    if current_quality["partial_gap_rows"] > 0:
        causes.append("partial_gap_alignment")
    if (
        plan.tail_candidate is not None
        and plan.selected_candidate.label == plan.tail_candidate.label
    ):
        causes.append("tail_rebuild_candidate_better")
    if selected_quality["full_gap_days"] > 0:
        causes.append("full_gap_source_gap")
    if plan.current_adjusted_alignment["missing_adjusted_rows"] > 0:
        causes.append("adjusted_missing_rows")
    if plan.current_adjusted_alignment["adjusted_mismatch_rows"] > 0:
        causes.append("adjusted_value_mismatch")
    if (
        filtered_quality["bad_days"] == current_quality["bad_days"]
        and selected_quality["bad_days"] == current_quality["bad_days"]
        and current_quality["bad_days"] > 0
    ):
        causes.append("no_safe_derived_improvement_found")

    if len(causes) == 0:
        causes.append("clean")

    return causes


def _plan_to_record(plan: PriceQualityRepairPlan) -> dict[str, object]:
    current_quality = plan.current_candidate.quality
    filtered_quality = plan.filtered_candidate.quality
    selected_quality = plan.selected_candidate.quality

    return {
        "instrument": plan.instrument_code,
        "current_status": current_quality["status"],
        "selected_status": selected_quality["status"],
        "selected_strategy": plan.selected_candidate.label,
        "current_bad_days": int(current_quality["bad_days"]),
        "filtered_bad_days": int(filtered_quality["bad_days"]),
        "selected_bad_days": int(selected_quality["bad_days"]),
        "current_bad_rows": int(current_quality["bad_rows"]),
        "selected_bad_rows": int(selected_quality["bad_rows"]),
        "current_partial_gap_days": int(current_quality["partial_gap_days"]),
        "current_partial_gap_rows": int(current_quality["partial_gap_rows"]),
        "selected_full_gap_days": int(selected_quality["full_gap_days"]),
        "selected_full_gap_rows": int(selected_quality["full_gap_rows"]),
        "current_adjusted_missing_rows": int(
            plan.current_adjusted_alignment["missing_adjusted_rows"]
        ),
        "current_adjusted_mismatch_rows": int(
            plan.current_adjusted_alignment["adjusted_mismatch_rows"]
        ),
        "multiple_repair_needed": bool(plan.write_multiple),
        "adjusted_repair_needed": bool(plan.write_adjusted),
        "root_causes": ",".join(_classify_root_causes(plan)),
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

    with dataBlob(log_name="Analyze-Price-Quality-Issues") as data:
        instrument_codes = _instrument_list(data, args.instrument_code)
        records = []
        for instrument_code in instrument_codes:
            plan = build_price_quality_repair_plan(
                data=data,
                instrument_code=instrument_code,
                start_date=start_date,
                end_date=end_date,
                min_bad_days=args.min_bad_days,
            )
            record = _plan_to_record(plan)
            if (not args.include_clean) and (
                record["current_status"] == QUALITY_STATUS_CLEAN
                and record["selected_status"] == QUALITY_STATUS_CLEAN
            ):
                continue
            records.append(record)

    if len(records) == 0:
        print(f"No price-quality issues found in window {window_label}.")
        return

    output_df = pd.DataFrame(records).sort_values(
        by=[
            "current_bad_days",
            "current_bad_rows",
            "current_adjusted_missing_rows",
        ],
        ascending=[False, False, False],
    )
    print(f"Window: {window_label}")
    print(output_df.to_string(index=False))
    _write_optional_outputs(records, args)


if __name__ == "__main__":
    main()
