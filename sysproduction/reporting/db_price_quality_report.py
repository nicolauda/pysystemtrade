from __future__ import annotations

import pandas as pd

from syscore.constants import arg_not_supplied
from sysdata.data_blob import dataBlob
from sysproduction.reporting.api import reportingApi
from sysproduction.reporting.data.price_quality import (
    ALL_DB_PRICE_QUALITY_INSTRUMENTS,
    DEFAULT_DB_PRICE_QUALITY_CALENDAR_DAYS,
    DEFAULT_DB_PRICE_QUALITY_MIN_BAD_DAYS,
    QUALITY_STATUS_ACTION,
    QUALITY_STATUS_CLEAN,
    QUALITY_STATUS_CRITICAL,
    QUALITY_STATUS_MINOR,
    get_db_price_quality_df,
    summarise_db_price_quality,
)
from sysproduction.reporting.reporting_functions import body_text, table


def db_price_quality_report(
    data: dataBlob = arg_not_supplied,
    instrument_code: str | object = arg_not_supplied,
    start_date=arg_not_supplied,
    end_date=arg_not_supplied,
    calendar_days_back: int | object = DEFAULT_DB_PRICE_QUALITY_CALENDAR_DAYS,
    min_bad_days: int = DEFAULT_DB_PRICE_QUALITY_MIN_BAD_DAYS,
):
    """Report DB multiple and adjusted price-quality issues.

    Args:
        data: Production data blob.
        instrument_code: Optional instrument filter. Use
            `ALL_DB_PRICE_QUALITY_INSTRUMENTS` to scan all instruments.
        start_date: Inclusive lower bound for the analysis window.
        end_date: Inclusive upper bound for the analysis window.
        calendar_days_back: Default rolling lookback window used when explicit
            dates are not supplied.
        min_bad_days: Minimum number of bad business days required before an
            instrument is classified as actionable.

    Returns:
        A list of report items consumable by the reporting framework.
    """

    if data is arg_not_supplied:
        data = dataBlob()

    reporting_api = reportingApi(data)
    quality_df, window_label = get_db_price_quality_df(
        data=data,
        instrument_code=instrument_code,
        start_date=start_date,
        end_date=end_date,
        calendar_days_back=calendar_days_back,
        min_bad_days=min_bad_days,
    )

    formatted_output = [reporting_api.terse_header("DB price quality report")]
    formatted_output.append(
        body_text(
            "Checks DB multiple and adjusted prices for missing current prices "
            "(`PRICE` is NaN while `FORWARD` is present) and backward "
            "`PRICE_CONTRACT` transitions."
        )
    )
    formatted_output.append(
        body_text(
            _build_scope_text(
                instrument_code=instrument_code,
                window_label=window_label,
                min_bad_days=min_bad_days,
                instrument_count=len(quality_df),
            )
        )
    )

    if quality_df.empty:
        formatted_output.append(
            body_text("No instruments found in DB multiple prices.")
        )
        formatted_output.append(reporting_api.footer())
        return formatted_output

    formatted_output.append(
        table("Status summary", summarise_db_price_quality(quality_df))
    )

    actionable_df = quality_df.loc[
        quality_df["status"].isin([QUALITY_STATUS_CRITICAL, QUALITY_STATUS_ACTION])
    ]
    if actionable_df.empty:
        formatted_output.append(
            body_text("No actionable issues found in the selected window.")
        )
    else:
        formatted_output.append(
            table("Actionable issues", _format_quality_table(actionable_df))
        )

    minor_df = quality_df.loc[quality_df["status"] == QUALITY_STATUS_MINOR]
    if not minor_df.empty:
        formatted_output.append(table("Minor issues", _format_quality_table(minor_df)))

    clean_count = int((quality_df["status"] == QUALITY_STATUS_CLEAN).sum())
    formatted_output.append(
        body_text(f"Clean instruments in selected window: {clean_count}")
    )
    formatted_output.append(reporting_api.footer())

    return formatted_output


def _build_scope_text(
    instrument_code: str | object,
    window_label: str,
    min_bad_days: int,
    instrument_count: int,
) -> str:
    selected_scope = (
        "all instruments"
        if instrument_code in [arg_not_supplied, ALL_DB_PRICE_QUALITY_INSTRUMENTS]
        else str(instrument_code)
    )

    return (
        f"Selected instrument scope: {selected_scope}\n"
        f"Selected data window: {window_label}\n"
        f"Action threshold: at least {min_bad_days} bad business days, "
        "or any contract regression\n"
        f"Instruments analysed: {instrument_count}"
    )


def _format_quality_table(quality_df: pd.DataFrame) -> pd.DataFrame:
    if quality_df.empty:
        return pd.DataFrame()

    table_df = quality_df[
        [
            "status",
            "bad_days",
            "bad_rows",
            "longest_bad_day_streak",
            "streak_start",
            "streak_end",
            "last_bad_day",
            "contract_regressions",
            "last_regression",
        ]
    ].copy()
    table_df = table_df.rename(columns={"longest_bad_day_streak": "longest_streak"})
    for column in ["streak_start", "streak_end", "last_bad_day", "last_regression"]:
        table_df[column] = table_df[column].apply(_format_timestamp_for_report)

    return table_df


def _format_timestamp_for_report(timestamp_value) -> str:
    if pd.isna(timestamp_value):
        return ""

    timestamp = pd.Timestamp(timestamp_value)
    if timestamp == timestamp.normalize():
        return str(timestamp.date())

    return timestamp.strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    db_price_quality_report()
