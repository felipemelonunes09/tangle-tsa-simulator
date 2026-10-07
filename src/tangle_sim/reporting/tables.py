from __future__ import annotations

from collections import defaultdict
import csv
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from tangle_sim.metrics.bootstrap import (
    Interval,
    bootstrap_mean,
    holm_adjust,
    paired_bootstrap_difference,
    paired_effect_size,
    paired_normal_pvalue,
)
from tangle_sim.metrics.survival import rmst


METRICS = (
    ("confirmation_rate_percent", "Taxa de aprovação", "percent"),
    ("abandonment_percent", "Pontas honestas deixadas para trás", "percent"),
    ("fraud_rmst_seconds", "Tempo restrito até consolidação da fraude", "seconds"),
    ("fraud_success_percent", "Taxa de sucesso da fraude", "percent"),
    ("fraud_confidence_final_percent", "Confiança fraudulenta final", "percent"),
)


def descriptive_statistics(values: np.ndarray, interval: Interval) -> dict[str, float]:
    """Return replicate-level dispersion and estimation-error diagnostics.

    The simulation replicate remains the experimental unit.  ``ddof=1`` is
    used for the sample variance/standard deviation, while the standard error
    describes uncertainty of the sample mean. Bootstrap intervals can be
    asymmetric. ``ci_half_width`` records half their total width, while
    ``margin_of_error`` records the larger distance to either endpoint.
    """
    observations = np.asarray(values, dtype=float)
    if observations.ndim != 1 or observations.size == 0 or np.any(~np.isfinite(observations)):
        raise ValueError("descriptive statistics require finite observations")
    variance = float(observations.var(ddof=1)) if observations.size > 1 else 0.0
    standard_deviation = float(np.sqrt(variance))
    standard_error = standard_deviation / float(np.sqrt(observations.size))
    estimate = float(interval.estimate)
    ci_half_width = 0.5 * float(interval.high - interval.low)
    margin_of_error = max(
        abs(estimate - float(interval.low)),
        abs(float(interval.high) - estimate),
    )
    return {
        "mean": estimate,
        "sample_variance": variance,
        "sample_standard_deviation": standard_deviation,
        "standard_error": standard_error,
        "relative_standard_error_percent": (
            100.0 * standard_error / abs(estimate) if estimate != 0.0 else 0.0
        ),
        "ci_half_width": ci_half_width,
        "margin_of_error": margin_of_error,
    }


def _with_descriptive_statistics(
    row: dict[str, Any],
    values: np.ndarray,
    interval: Interval,
    *,
    confirmation_error_rate: bool = False,
) -> dict[str, Any]:
    row.update(descriptive_statistics(values, interval))
    row["error_rate_percent"] = (
        100.0 - float(interval.estimate) if confirmation_error_rate else ""
    )
    return row


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Iterable[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: set[str] = set()
        for row in rows:
            keys.update(row)
        preferred = ["scenario", "algorithm", "replicate", "battery"]
        fieldnames = [key for key in preferred if key in keys] + sorted(keys - set(preferred))
    names = list(fieldnames)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> str | None:
    try:
        import pandas as pd
    except ImportError:
        return "pandas/pyarrow unavailable; parquet not emitted"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, engine="pyarrow", index=False)
    return None


def summarize(
    benign_rows: list[dict[str, Any]],
    attack_rows: list[dict[str, Any]],
    rng: np.random.Generator,
    bootstrap_samples: int,
    confidence: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary: list[dict[str, Any]] = []
    benign_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    attack_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in benign_rows:
        benign_groups[(str(row["scenario"]), str(row["algorithm"]))].append(row)
    for row in attack_rows:
        attack_groups[(str(row["scenario"]), str(row["algorithm"]))].append(row)

    for (scenario, algorithm), rows in benign_groups.items():
        for key, label, unit in METRICS[:2]:
            values = np.asarray([float(row[key]) for row in rows], dtype=float)
            interval = bootstrap_mean(values, rng, bootstrap_samples, confidence)
            summary.append(_with_descriptive_statistics({
                "scenario": scenario,
                "algorithm": algorithm,
                "metric": key,
                "metric_label": label,
                "unit": unit,
                "estimate": interval.estimate,
                "ci_low": interval.low,
                "ci_high": interval.high,
                "replicates": len(values),
                "definition_version": "2.0" if key == "confirmation_rate_percent" else "1.0",
            }, values, interval, confirmation_error_rate=key == "confirmation_rate_percent"))
    for (scenario, algorithm), rows in attack_groups.items():
        times = np.asarray([float(row["time_to_event_or_censoring"]) for row in rows])
        observed = np.asarray([str(row["event_observed"]).lower() in {"true", "1"} for row in rows])
        tau = min(float(row["censoring_horizon"]) for row in rows)
        restricted = np.minimum(times, tau)
        # With a common administrative horizon, mean restricted time equals KM RMST.
        estimate = rmst(times, observed, tau)
        boot = bootstrap_mean(restricted, rng, bootstrap_samples, confidence)
        summary.append(_with_descriptive_statistics({
            "scenario": scenario,
            "algorithm": algorithm,
            "metric": "fraud_rmst_seconds",
            "metric_label": METRICS[2][1],
            "unit": "seconds",
            "estimate": estimate,
            "ci_low": boot.low,
            "ci_high": boot.high,
            "replicates": len(rows),
            "tau": tau,
            "definition_version": "1.1",
        }, restricted, boot))
        final_confidence = 100.0 * np.asarray(
            [float(row["fraud_confidence_final"]) for row in rows], dtype=float,
        )
        confidence_ci = bootstrap_mean(final_confidence, rng, bootstrap_samples, confidence)
        summary.append(_with_descriptive_statistics({
            "scenario": scenario,
            "algorithm": algorithm,
            "metric": "fraud_confidence_final_percent",
            "metric_label": METRICS[4][1],
            "unit": "percent",
            "estimate": confidence_ci.estimate,
            "ci_low": confidence_ci.low,
            "ci_high": confidence_ci.high,
            "replicates": len(rows),
            "tau": tau,
            "definition_version": "1.0",
        }, final_confidence, confidence_ci))
        success = 100.0 * observed.astype(float)
        success_ci = bootstrap_mean(success, rng, bootstrap_samples, confidence)
        summary.append(_with_descriptive_statistics({
            "scenario": scenario,
            "algorithm": algorithm,
            "metric": "fraud_success_percent",
            "metric_label": METRICS[3][1],
            "unit": "percent",
            "estimate": success_ci.estimate,
            "ci_low": success_ci.low,
            "ci_high": success_ci.high,
            "replicates": len(rows),
            "tau": tau,
            "definition_version": "1.1",
        }, success, success_ci))

    pairs = (("urts", "mcmc"), ("urts", "sb_wrw"), ("mcmc", "sb_wrw"))
    comparison_rows: list[dict[str, Any]] = []
    for scenario in ("PACTCare", "TLIP", "ElaadNL"):
        metric_values: dict[str, dict[str, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
        for row in benign_rows:
            if row["scenario"] == scenario:
                for metric in ("confirmation_rate_percent", "abandonment_percent"):
                    metric_values[metric][str(row["algorithm"])][int(row["replicate"])] = float(row[metric])
        for row in attack_rows:
            if row["scenario"] == scenario:
                metric_values["fraud_rmst_seconds"][str(row["algorithm"])][int(row["replicate"])] = float(row["time_to_event_or_censoring"])
                metric_values["fraud_confidence_final_percent"][str(row["algorithm"])][int(row["replicate"])] = 100.0 * float(row["fraud_confidence_final"])
        scenario_start = len(comparison_rows)
        raw_pvalues: list[float] = []
        for metric, by_algorithm in metric_values.items():
            for first, second in pairs:
                indices = sorted(set(by_algorithm[first]) & set(by_algorithm[second]))
                a = np.asarray([by_algorithm[first][i] for i in indices])
                b = np.asarray([by_algorithm[second][i] for i in indices])
                interval = paired_bootstrap_difference(a, b, rng, bootstrap_samples, confidence)
                pvalue = paired_normal_pvalue(a, b)
                raw_pvalues.append(pvalue)
                denominator = np.mean(b)
                comparison_rows.append({
                    "scenario": scenario,
                    "metric": metric,
                    "algorithm_a": first,
                    "algorithm_b": second,
                    "replicates": len(indices),
                    "absolute_difference_a_minus_b": interval.estimate,
                    "relative_difference_percent": 100.0 * interval.estimate / denominator if denominator else float("nan"),
                    "ci_low": interval.low,
                    "ci_high": interval.high,
                    "paired_effect_size_dz": paired_effect_size(a, b),
                    "p_value_normal_approx": pvalue,
                })
        adjusted = holm_adjust(raw_pvalues)
        for row, value in zip(comparison_rows[scenario_start:], adjusted, strict=True):
            row["p_value_holm"] = value
    return summary, comparison_rows
