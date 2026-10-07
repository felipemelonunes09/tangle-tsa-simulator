#!/usr/bin/env python3
"""Assess the paper's precision requirement from a metrics summary CSV.

The bootstrap percentile interval can be asymmetric around the sample mean.
For that reason, the operational margin is the larger distance from the mean
to either endpoint, rather than one half of the interval width.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


PERCENTAGE_POINT_METRICS = {
    "confirmation_rate_percent",
    "abandonment_percent",
}
RELATIVE_METRICS = {"fraud_rmst_seconds"}
REQUIRED_METRICS = PERCENTAGE_POINT_METRICS | RELATIVE_METRICS


def assess(
    row: dict[str, str],
    *,
    observed_events: int | None = None,
) -> dict[str, str | int | float | bool | None]:
    metric = row["metric"]
    estimate = float(row["estimate"])
    low = float(row["ci_low"])
    high = float(row["ci_high"])
    replicates = int(row["replicates"])
    margin = max(abs(estimate - low), abs(high - estimate))

    if metric in PERCENTAGE_POINT_METRICS:
        target = 5.0
        criterion = "5 percentage points"
    elif metric in RELATIVE_METRICS:
        target = 0.05 * abs(estimate)
        criterion = "5% of the mean"
    else:
        raise ValueError(f"unsupported metric: {metric}")

    all_censored = metric in RELATIVE_METRICS and observed_events == 0
    requirement_met = None if all_censored else margin <= target
    relative_margin = 100.0 * margin / abs(estimate) if estimate else math.inf
    if all_censored:
        planning_replicates: int | str = ""
    elif target > 0.0:
        planning_replicates = max(
            replicates,
            math.ceil(replicates * (margin / target) ** 2),
        )
    else:
        planning_replicates = replicates

    return {
        "scenario": row["scenario"],
        "algorithm": row["algorithm"],
        "metric": metric,
        "replicates": replicates,
        "estimate": estimate,
        "ci_low": low,
        "ci_high": high,
        "conservative_margin": margin,
        "relative_margin_percent": relative_margin,
        "target_margin": target,
        "criterion": criterion,
        "requirement_met": requirement_met,
        "assessment_status": (
            "not_assessable_all_censored"
            if all_censored
            else "met" if requirement_met else "not_met"
        ),
        "observed_events": "" if observed_events is None else observed_events,
        # Planning approximation only: CI width is assumed to scale as 1/sqrt(n).
        "planning_replicates_scaled": planning_replicates,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--attack-runs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    with args.input.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["metric"] in REQUIRED_METRICS]
    with args.attack_runs.open(encoding="utf-8", newline="") as handle:
        attack_rows = list(csv.DictReader(handle))
    observed_by_group: dict[tuple[str, str], int] = {}
    for row in attack_rows:
        key = (row["scenario"], row["algorithm"])
        observed = row["event_observed"].lower() in {"true", "1"}
        observed_by_group[key] = observed_by_group.get(key, 0) + int(observed)
    assessed = [
        assess(
            row,
            observed_events=(
                observed_by_group[(row["scenario"], row["algorithm"])]
                if row["metric"] in RELATIVE_METRICS
                else None
            ),
        )
        for row in rows
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(assessed[0]) if assessed else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(assessed)

    failed = [row for row in assessed if row["requirement_met"] is False]
    inconclusive = [row for row in assessed if row["requirement_met"] is None]
    print(json.dumps({
        "assessed": len(assessed),
        "passed": len(assessed) - len(failed) - len(inconclusive),
        "failed": len(failed),
        "inconclusive": len(inconclusive),
        "all_requirements_met": not failed and not inconclusive,
        "largest_planning_replicate_count": max(
            (
                int(row["planning_replicates_scaled"])
                for row in assessed
                if row["planning_replicates_scaled"] != ""
            ),
            default=0,
        ),
        "output": str(args.output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
