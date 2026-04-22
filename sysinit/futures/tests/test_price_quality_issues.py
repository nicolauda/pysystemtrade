from sysinit.futures.adhoc.price_quality_issues import (
    PriceQualityCandidate,
    choose_best_price_quality_candidate,
)
from sysobjects.multiple_prices import futuresMultiplePrices


def _candidate(
    label: str,
    *,
    bad_days: int,
    bad_rows: int,
    full_gap_days: int,
    regressions: int = 0,
    streak: int = 0,
) -> PriceQualityCandidate:
    return PriceQualityCandidate(
        label=label,
        multiple_prices=futuresMultiplePrices.create_empty(),
        quality=dict(
            contract_regressions=regressions,
            full_gap_days=full_gap_days,
            bad_days=bad_days,
            bad_rows=bad_rows,
            longest_bad_day_streak=streak,
        ),
        gap_summary={},
    )


def test_choose_best_price_quality_candidate_prefers_lower_issue_score():
    current = _candidate(
        "current",
        bad_days=10,
        bad_rows=50,
        full_gap_days=3,
    )
    filtered = _candidate(
        "drop_partial_gap_rows",
        bad_days=2,
        bad_rows=3,
        full_gap_days=2,
    )

    best = choose_best_price_quality_candidate([current, filtered])

    assert best.label == "drop_partial_gap_rows"


def test_choose_best_price_quality_candidate_keeps_earlier_candidate_on_tie():
    current = _candidate(
        "current",
        bad_days=2,
        bad_rows=3,
        full_gap_days=1,
    )
    filtered = _candidate(
        "drop_partial_gap_rows",
        bad_days=2,
        bad_rows=3,
        full_gap_days=1,
    )

    best = choose_best_price_quality_candidate([current, filtered])

    assert best.label == "current"
