from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


class AuditError(RuntimeError):
    pass


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def audit_run(run_dir: str | Path, *, write: bool = True) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    benign_path = root / "raw" / "benign_runs.csv"
    attack_path = root / "raw" / "attack_runs.csv"
    summary_path = root / "summary" / "metrics_summary.csv"
    benign = _rows(benign_path)
    attack = _rows(attack_path)
    summary = _rows(summary_path)
    failures: list[str] = []
    warnings: list[str] = []

    for row in benign:
        denominator = int(row["eligible_denominator"])
        confirmed = int(row["confirmation_numerator"])
        confidence_confirmed = int(row.get("confidence_confirmation_numerator", confirmed))
        abandoned = int(row["abandoned_numerator"])
        if (
            denominator <= 0
            or not (0 <= confirmed <= denominator)
            or not (0 <= confidence_confirmed <= denominator)
            or not (0 <= abandoned <= denominator)
        ):
            failures.append(f"invalid benign counts: {row['scenario']}/{row['replicate']}/{row['algorithm']}")
        for key in (
            "confirmation_rate_percent", "abandonment_percent", "duration_seconds",
            "confidence_confirmation_rate_percent",
            "confirmation_confidence_mean_percent", "confirmation_early_80_percent",
            "confirmation_final_20_percent", "endpoint_effective_count",
            "endpoint_max_probability", "endpoint_uniform_total_variation",
        ):
            if key not in row:
                continue
            if not math.isfinite(float(row[key])):
                failures.append(f"non-finite {key}: {row['scenario']}/{row['replicate']}/{row['algorithm']}")
        if "endpoint_count" in row:
            endpoint_count = int(row["endpoint_count"])
            effective = float(row["endpoint_effective_count"])
            max_probability = float(row["endpoint_max_probability"])
            total_variation = float(row["endpoint_uniform_total_variation"])
            if endpoint_count < 1 or not 1.0 <= effective <= endpoint_count + 1e-9:
                failures.append(f"invalid endpoint effective count: {row['scenario']}/{row['replicate']}/{row['algorithm']}")
            if not 0.0 < max_probability <= 1.0 or not 0.0 <= total_variation <= 1.0:
                failures.append(f"invalid endpoint distribution diagnostics: {row['scenario']}/{row['replicate']}/{row['algorithm']}")
        if row["seed_selection"] == row["seed_measurement"]:
            failures.append(f"selection/measurement seed reuse: {row['scenario']}/{row['replicate']}/{row['algorithm']}")
        if "drain_window_transactions" in row and int(row["drain_window_transactions"]) != 0:
            failures.append(f"unexpected drain emission: {row['scenario']}/{row['replicate']}/{row['algorithm']}")

    for battery, rows in (("benign", benign), ("attack", attack)):
        groups: dict[tuple[str, str], list[dict[str, str]]] = {}
        for row in rows:
            groups.setdefault((row["scenario"], row["replicate"]), []).append(row)
        for key, group in groups.items():
            common_fields = [
                "external_schedule_sha256",
                "seed_workload",
                "seed_topology",
                "seed_delay",
            ]
            if battery == "attack" and group and "attacker_schedule_sha256" in group[0]:
                common_fields.append("attacker_schedule_sha256")
            for field in common_fields:
                if len({row[field] for row in group}) != 1:
                    failures.append(f"CRN mismatch {battery}/{key[0]}/{key[1]}/{field}")

    for row in attack:
        value = float(row["time_to_event_or_censoring"])
        tau = float(row["censoring_horizon"])
        observed = row["event_observed"].lower() in {"true", "1"}
        if not 0 <= value <= tau:
            failures.append(f"attack time outside [0,tau]: {row['scenario']}/{row['replicate']}/{row['algorithm']}")
        if not observed and value != tau:
            failures.append(f"incorrect right censoring: {row['scenario']}/{row['replicate']}/{row['algorithm']}")
        detection_mode = row.get("fraud_detection_mode", "periodic")
        if detection_mode == "event_driven" and float(
            row.get("fraud_detection_resolution_seconds", 0.0)
        ) != 0.0:
            failures.append(
                f"event-driven detector has nonzero resolution: "
                f"{row['scenario']}/{row['replicate']}/{row['algorithm']}"
            )
        if observed and row.get("fraud_threshold_crossing_time", "") != "":
            crossing = float(row["fraud_threshold_crossing_time"])
            if not 0 <= crossing <= value:
                failures.append(
                    f"invalid fraud threshold crossing time: "
                    f"{row['scenario']}/{row['replicate']}/{row['algorithm']}"
                )
        if (
            row.get("attacker_issuer_policy") == "topology_proportional"
            and int(row.get("attacker_distinct_sectors", 0)) < 2
        ):
            failures.append(
                f"distributed attacker did not span sectors: "
                f"{row['scenario']}/{row['replicate']}/{row['algorithm']}"
            )
        if (
            row.get(
                "effective_parasite_chain_strategy",
                row.get("parasite_chain_strategy"),
            ) == "simple_multi_pinned"
            and int(row.get("parasite_root_count", 0)) < 2
        ):
            warnings.append(
                f"multi-pinned attack collapsed to one available root: "
                f"{row['scenario']}/{row['replicate']}/{row['algorithm']}"
            )

    attack_groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in attack:
        attack_groups.setdefault((row["scenario"], row["algorithm"]), []).append(row)
    for (scenario, algorithm), rows in sorted(attack_groups.items()):
        if not any(row["event_observed"].lower() in {"true", "1"} for row in rows):
            warnings.append(
                f"all fraud observations censored: {scenario}/{algorithm}; "
                "RMST is the administrative horizon, not an observed fraud time"
            )

    for row in benign + attack:
        if row["algorithm"] == "sb_wrw":
            counts = sum(int(row[key]) for key in ("ring_r0_count", "ring_r1_count", "ring_r2_count", "ring_r3plus_count"))
            percentages = sum(float(row[key]) for key in ("ring_r0_percent", "ring_r1_percent", "ring_r2_percent", "ring_r3plus_percent"))
            if counts > 0 and abs(percentages - 100.0) > 1e-8:
                failures.append(f"SB ring percentages do not sum to 100: {row['battery']}/{row['scenario']}/{row['replicate']}")

    for row in summary:
        estimate, low, high = (float(row[key]) for key in ("estimate", "ci_low", "ci_high"))
        if not all(math.isfinite(x) for x in (estimate, low, high)) or not low <= estimate <= high:
            failures.append(f"invalid summary interval: {row['scenario']}/{row['algorithm']}/{row['metric']}")
        if row["metric"] == "fraud_rmst_seconds" and not 0 <= estimate <= float(row["tau"]):
            failures.append(f"RMST outside [0,tau]: {row['scenario']}/{row['algorithm']}")
        if row["metric"] == "fraud_confidence_final_percent" and not 0 <= estimate <= 100:
            failures.append(f"fraud confidence outside [0,100]: {row['scenario']}/{row['algorithm']}")

    required = [
        "manifest.json", "config_resolved.yaml", "environment.txt",
        "raw/benign_runs.parquet", "raw/attack_runs.parquet",
        "summary/pairwise_comparisons.csv", "summary/sensitivity_summary.csv",
        "plots/taxa_confirmacao.png", "plots/taxa_confirmacao.pdf", "plots/taxa_confirmacao.svg",
        "plots/pontas_abandonadas.png", "plots/pontas_abandonadas.pdf", "plots/pontas_abandonadas.svg",
        "plots/tempo_fraude.png", "plots/tempo_fraude.pdf", "plots/tempo_fraude.svg",
        "plots/confianca_fraude_final.png", "plots/confianca_fraude_final.pdf", "plots/confianca_fraude_final.svg",
        "plots/metricas_combinadas.png", "plots/curvas_sobrevivencia.pdf",
    ]
    for relative in required:
        path = root / relative
        if not path.is_file() or path.stat().st_size == 0:
            failures.append(f"missing or empty artifact: {relative}")

    parquet_counts: dict[str, int] = {}
    try:
        import pandas as pd
        for name, csv_rows in (("benign", benign), ("attack", attack)):
            count = len(pd.read_parquet(root / "raw" / f"{name}_runs.parquet"))
            parquet_counts[name] = count
            if count != len(csv_rows):
                failures.append(f"CSV/Parquet row count mismatch: {name}")
    except (ImportError, FileNotFoundError) as exc:
        failures.append(f"cannot audit parquet: {exc}")

    report = {
        "valid": not failures,
        "failures": failures,
        "warnings": warnings,
        "row_counts": {"benign": len(benign), "attack": len(attack), "summary": len(summary)},
        "parquet_row_counts": parquet_counts,
        "hashes": {
            "benign_csv_sha256": _sha256(benign_path),
            "attack_csv_sha256": _sha256(attack_path),
            "metrics_summary_csv_sha256": _sha256(summary_path),
        },
    }
    if write:
        (root / "audit.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if failures:
        raise AuditError("; ".join(failures))
    return report
