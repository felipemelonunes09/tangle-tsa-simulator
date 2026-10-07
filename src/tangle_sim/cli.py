from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time
from typing import Any

import yaml

from tangle_sim import __version__
from tangle_sim.audit import AuditError, audit_run
from tangle_sim.config import (
    ConfigError,
    ExperimentConfig,
    load_config,
    scenario_alpha,
    scenario_private_build_seconds,
)
from tangle_sim.metrics.confirmation import benign_metrics
from tangle_sim.reporting.plots import generate_plots
from tangle_sim.reporting.tables import summarize, write_csv, write_parquet
from tangle_sim.simulation.attack import run_attack
from tangle_sim.simulation.engine import Simulator


METRIC_DEFINITIONS = {
    "confirmation_rate_percent": "100 * eligible honest measurement-window transactions approved by at least one direct approver by the final snapshot / eligible honest transactions; liveness v2.0",
    "confidence_confirmation_rate_percent": "100 * eligible honest measurement-window transactions with exact final TSA-conditioned absorbing-DAG CC >= 0.95 / eligible honest transactions; retained consensus diagnostic v1.3",
    "abandonment_percent": "100 * eligible honest global tips aged at least abandonment_age / eligible honest transactions; v1.0",
    "fraud_rmst_seconds": "Kaplan-Meier restricted mean time from F0 publication to completion of the stable-threshold window, administratively censored at tau; v1.1",
    "fraud_success_percent": "100 * observed stable fraud consolidations by tau / attack replicas; v1.1",
    "fraud_confidence_final_percent": "100 * exact final TSA probability of ending at a tip that approves F0; diagnostic v1.0",
}

STATISTICAL_FIELD_DEFINITIONS = {
    "mean": "arithmetic mean across independent simulation replicates; identical to estimate",
    "sample_variance": "replicate-level sample variance with Bessel correction (ddof=1)",
    "sample_standard_deviation": "square root of replicate-level sample variance",
    "standard_error": "sample standard deviation divided by square root of the replicate count",
    "relative_standard_error_percent": "100 * standard error / absolute mean; zero when mean is zero",
    "ci_half_width": "half of the total width of the percentile-bootstrap confidence interval",
    "margin_of_error": "larger absolute distance from the estimate to either percentile-bootstrap endpoint",
    "error_rate_percent": "100 - mean confirmation rate; populated only for confirmation_rate_percent",
}


def _algorithm_label(name: str, conformance: str) -> str:
    return {
        "urts": "URTS",
        "mcmc": "MCMC/WRW",
        "sb_wrw": "SB-WRW" if conformance == "paper_exact" else "SB-WRW (reprodução operacional)",
    }[name]


def _base_row(config: ExperimentConfig, scenario: str, algorithm: str, replicate: int, battery: str) -> dict[str, Any]:
    seeds = Simulator(config).seeds
    conformance = config.raw["algorithms"]["sb_wrw"]["conformance"]
    return {
        "scenario": scenario,
        "algorithm": algorithm,
        "algorithm_label": _algorithm_label(algorithm, conformance),
        "replicate": replicate,
        "battery": battery,
        "sb_wrw_conformance": conformance,
        "simulator_version": __version__,
        "root_seed": config.root_seed,
        "seed_workload": seeds.identifier(scenario, replicate, "workload"),
        "seed_topology": seeds.identifier(scenario, replicate, "topology"),
        "seed_delay": seeds.identifier(scenario, replicate, "delay"),
        "seed_selection": seeds.identifier(scenario, replicate, "selection", algorithm),
        "seed_attack": seeds.identifier(scenario, replicate, "attack"),
        "seed_measurement": seeds.identifier(scenario, replicate, "measurement", algorithm),
        "mcmc_alpha": scenario_alpha(config, scenario),
        "entry_point_policy": config.raw["algorithms"]["mcmc"]["entry_point_policy"],
        "entry_point_depth_seconds": config.raw["algorithms"]["mcmc"]["entry_point_depth_seconds"],
    }


def _run_replica(config_path: str, scenario_name: str, replicate: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    config = load_config(config_path)
    scenario = next(item for item in config.scenarios if item.name == scenario_name)
    simulator = Simulator(config)
    algorithms = tuple(config.raw["algorithms"]["names"])
    sim_cfg = config.raw["simulation"]
    # Generate only through warm-up and measurement. Drain completes attachment
    # events already in flight and introduces no new issues.
    benign_horizon = (
        float(sim_cfg["warmup_seconds"])
        + float(sim_cfg["measurement_seconds"])
    )
    topology, schedule = simulator.common_inputs(scenario, replicate, benign_horizon)
    benign_rows: list[dict[str, Any]] = []
    for algorithm in algorithms:
        started = time.perf_counter()
        result = simulator.run_benign(scenario, algorithm, replicate, topology, schedule)
        measurement_rng = simulator.seeds.generator(scenario.name, replicate, "measurement", algorithm)
        metrics = benign_metrics(result, config, scenario, algorithm, measurement_rng)
        row = _base_row(config, scenario.name, algorithm, replicate, "benign")
        row.update(result.diagnostics)
        row.update(metrics.as_dict())
        row["duration_seconds"] = time.perf_counter() - started
        benign_rows.append(row)

    attack_cfg = config.raw["attack"]
    attack_final = (
        float(sim_cfg["warmup_seconds"])
        + scenario_private_build_seconds(config, scenario.name)
        + float(attack_cfg["attack_observation_seconds"])
    )
    attack_topology, attack_schedule = simulator.common_inputs(scenario, replicate, attack_final)
    attack_rows: list[dict[str, Any]] = []
    if bool(attack_cfg.get("enabled", True)):
        for algorithm in algorithms:
            started = time.perf_counter()
            outcome = run_attack(simulator, scenario, algorithm, replicate, attack_topology, attack_schedule)
            row = _base_row(config, scenario.name, algorithm, replicate, "attack")
            row.update(outcome.simulation.diagnostics)
            row.update(outcome.metrics.as_dict())
            row["duration_seconds"] = time.perf_counter() - started
            attack_rows.append(row)
    return benign_rows, attack_rows


def _normative_hash(project_root: Path) -> str:
    prompt = (project_root / "prompt_codex_simulador_tangle.md").read_text(encoding="utf-8")
    start = prompt.index("### 5.3 SB-WRW")
    end = prompt.index("## 6. Cenários sintéticos")
    return hashlib.sha256(prompt[start:end].encode("utf-8")).hexdigest()


def _git_state(project_root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=project_root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=project_root, check=True,
            capture_output=True, text=True,
        ).stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": None, "dirty": None, "note": "worktree sem repositório Git inicializado"}


def _environment() -> tuple[str, dict[str, str]]:
    packages: dict[str, str] = {}
    for package in ("numpy", "PyYAML", "matplotlib", "pandas", "pyarrow", "scipy", "pytest"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "not-installed"
    lines = [
        f"python={platform.python_version()}",
        f"implementation={platform.python_implementation()}",
        f"platform={platform.platform()}",
        f"machine={platform.machine()}",
    ] + [f"{key}={value}" for key, value in packages.items()]
    return "\n".join(lines) + "\n", packages


def run_experiment(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.replicates is not None:
        if args.replicates < 1:
            raise ConfigError("--replicates must be >= 1")
        config.raw["replicates"] = args.replicates
    if args.workers is not None:
        if args.workers < 1:
            raise ConfigError("--workers must be >= 1")
        config.raw["workers"] = args.workers
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    previous_manifest_path = output / "manifest.json"
    previous_manifest: dict[str, Any] = {}
    if previous_manifest_path.is_file():
        previous_manifest = json.loads(previous_manifest_path.read_text(encoding="utf-8"))
    temporary = output / ".replicas"
    temporary.mkdir(exist_ok=True)
    checkpoint_hash = temporary / "config.sha256"
    current_hash = config.sha256()
    finalized = [path for path in output.iterdir() if path.name != ".replicas"]
    if finalized and not checkpoint_hash.is_file():
        raise ConfigError(f"output directory contains artifacts without resumable checkpoints: {output}")
    if checkpoint_hash.exists() and checkpoint_hash.read_text(encoding="ascii").strip() != current_hash:
        raise ConfigError(f"checkpoint configuration mismatch: {temporary}")
    checkpoint_hash.write_text(current_hash + "\n", encoding="ascii")

    def checkpoint_path(scenario: str, replicate: int) -> Path:
        return temporary / f"{scenario}-{replicate:05d}.json"

    def persist_replica(
        scenario: str,
        replicate: int,
        benign: list[dict[str, Any]],
        attack: list[dict[str, Any]],
    ) -> None:
        path = checkpoint_path(scenario, replicate)
        path.write_text(
            json.dumps({"benign": benign, "attack": attack}, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )

    tasks = [
        (scenario.name, replicate)
        for scenario in config.scenarios
        for replicate in range(config.replicates)
        if not checkpoint_path(scenario.name, replicate).is_file()
    ]
    workers = int(config.raw.get("workers", 1))
    effective_workers = workers
    execution_warnings: list[str] = []
    if not tasks and previous_manifest:
        # Regenerating reports from complete checkpoints must not rewrite the
        # execution provenance as if a new parallel simulation had occurred.
        effective_workers = int(previous_manifest.get("effective_workers", workers))
        execution_warnings = list(previous_manifest.get("execution_warnings", []))
    elif workers == 1:
        for scenario, replicate in tasks:
            benign, attack = _run_replica(str(config.path), scenario, replicate)
            persist_replica(scenario, replicate, benign, attack)
    else:
        try:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(_run_replica, str(config.path), scenario, replicate): (scenario, replicate)
                    for scenario, replicate in tasks
                }
                for future in as_completed(futures):
                    scenario, replicate = futures[future]
                    benign, attack = future.result()
                    persist_replica(scenario, replicate, benign, attack)
        except (PermissionError, OSError) as exc:
            effective_workers = 1
            execution_warnings.append(
                f"process parallelism unavailable; sequential fallback used: {type(exc).__name__}: {exc}"
            )
            remaining = [
                (scenario, replicate)
                for scenario, replicate in tasks
                if not checkpoint_path(scenario, replicate).is_file()
            ]
            for scenario, replicate in remaining:
                benign, attack = _run_replica(str(config.path), scenario, replicate)
                persist_replica(scenario, replicate, benign, attack)

    benign_rows: list[dict[str, Any]] = []
    attack_rows: list[dict[str, Any]] = []
    for path in sorted(temporary.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        benign_rows.extend(payload["benign"])
        attack_rows.extend(payload["attack"])
    scenario_order = {name: i for i, name in enumerate(("PACTCare", "TLIP", "ElaadNL"))}
    algorithm_order = {name: i for i, name in enumerate(("urts", "mcmc", "sb_wrw"))}
    key = lambda row: (scenario_order[row["scenario"]], int(row["replicate"]), algorithm_order[row["algorithm"]])
    benign_rows.sort(key=key)
    attack_rows.sort(key=key)

    raw_dir = output / "raw"
    write_csv(raw_dir / "benign_runs.csv", benign_rows)
    write_csv(raw_dir / "attack_runs.csv", attack_rows)
    warnings = [warning for warning in (
        write_parquet(raw_dir / "benign_runs.parquet", benign_rows),
        write_parquet(raw_dir / "attack_runs.parquet", attack_rows),
    ) if warning]

    bootstrap_rng = Simulator(config).seeds.generator("summary", 0, "bootstrap")
    stats = config.raw["statistics"]
    summary, comparisons = summarize(
        benign_rows,
        attack_rows,
        bootstrap_rng,
        int(stats["bootstrap_samples"]),
        float(stats["confidence_level"]),
    )
    write_csv(output / "summary" / "metrics_summary.csv", summary)
    write_csv(output / "summary" / "pairwise_comparisons.csv", comparisons)
    sensitivity_rows = []
    sensitivity = config.raw.get("sensitivity", {})
    for parameter in ("load_multipliers", "delay_multipliers", "alphas", "attacker_rate_fractions"):
        for value in sensitivity.get(parameter, []):
            sensitivity_rows.append({"parameter": parameter, "value": value, "status": "preregistered_not_part_of_primary_run"})
    write_csv(output / "summary" / "sensitivity_summary.csv", sensitivity_rows, ("parameter", "value", "status"))

    resolved = config.resolved_yaml()
    (output / "config_resolved.yaml").write_text(resolved, encoding="utf-8")
    environment_text, packages = _environment()
    (output / "environment.txt").write_text(environment_text, encoding="utf-8")
    project_root = Path(__file__).resolve().parents[2]
    manifest = {
        "manifest_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "simulator_version": __version__,
        "experiment_id": config.raw["experiment_id"],
        "run_class": config.raw.get("run_class", "primary"),
        "config_sha256": hashlib.sha256(resolved.encode("utf-8")).hexdigest(),
        "sb_wrw_normative_section_sha256": _normative_hash(project_root),
        "sb_wrw_paper_sha256": None,
        "sb_wrw_conformance": config.raw["algorithms"]["sb_wrw"]["conformance"],
        "sb_wrw_label": _algorithm_label("sb_wrw", config.raw["algorithms"]["sb_wrw"]["conformance"]),
        "full_text_status": "not_available; operational reproduction used",
        "git": _git_state(project_root),
        "python": platform.python_version(),
        "dependencies": packages,
        "operating_system": platform.platform(),
        "root_seed": config.root_seed,
        "replicates": config.replicates,
        "workers": workers,
        "effective_workers": effective_workers,
        # Do not leak or hash a checkout-specific path from sys.argv[0].
        "commands": ["python -m tangle_sim " + shlex.join(sys.argv[1:])],
        "metric_definitions": METRIC_DEFINITIONS,
        "statistical_field_definitions": STATISTICAL_FIELD_DEFINITIONS,
        "drain_traffic": "quiescent; no new issues, only pending attachments complete",
        "volatile_fields": ["timestamp_utc", "duration_seconds", "commands", "environment.txt"],
        "scenario_statement": "cenários sintéticos inspirados nas características operacionais dos casos de uso",
        "artifact_warnings": warnings,
        "execution_warnings": execution_warnings,
        "primary_run_includes_sensitivity": False,
        "fraud_consolidation_rule": (
            "F0 confidence >= threshold and H0 confidence <= 1-threshold "
            "continuously for confidence_stability_seconds"
            if bool(config.raw["attack"].get("require_honest_confidence_below_complement", True))
            else "F0 confidence >= threshold continuously for confidence_stability_seconds"
        ),
        "fraud_detection_mode": config.raw["attack"].get(
            "fraud_detection_mode", "periodic"
        ),
        "fraud_event_time_definition": "end_of_stability_window_from_F0_publication",
        "attacker_rate_fraction_schedule": config.raw["attack"].get(
            "attacker_rate_fraction_schedule", []
        ),
        "stop_after_fraud_consolidation": bool(
            config.raw["attack"].get("stop_after_fraud_consolidation", False)
        ),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    generate_plots(output, "pt-BR")
    audit = audit_run(output)
    print(json.dumps({"output": str(output), "config_sha256": manifest["config_sha256"], "warnings": warnings, "audit_valid": audit["valid"]}, ensure_ascii=False))
    return 0


def report(args: argparse.Namespace) -> int:
    paths = generate_plots(Path(args.input).resolve(), args.language)
    print(json.dumps({"plots": [str(path) for path in paths]}, ensure_ascii=False))
    return 0


def validate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    payload = {
        "valid": True,
        "experiment_id": config.raw["experiment_id"],
        "config_sha256": config.sha256(),
        "replicates": config.replicates,
        "scenarios": [scenario.name for scenario in config.scenarios],
        "algorithms": config.raw["algorithms"]["names"],
        "sb_wrw_conformance": config.raw["algorithms"]["sb_wrw"]["conformance"],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def audit_command(args: argparse.Namespace) -> int:
    report = audit_run(args.input)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m tangle_sim")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate", help="validate and resolve a preregistered YAML configuration")
    validate_parser.add_argument("--config", required=True)
    validate_parser.set_defaults(func=validate)
    run_parser = subparsers.add_parser("run", help="run benign and attack batteries")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--output", required=True)
    run_parser.add_argument("--replicates", type=int)
    run_parser.add_argument("--workers", type=int)
    run_parser.set_defaults(func=run_experiment)
    report_parser = subparsers.add_parser("report", help="regenerate plots from raw summaries")
    report_parser.add_argument("--input", required=True)
    report_parser.add_argument("--language", default="pt-BR")
    report_parser.set_defaults(func=report)
    audit_parser = subparsers.add_parser("audit", help="audit raw data, CRN, censoring and required artifacts")
    audit_parser.add_argument("--input", required=True)
    audit_parser.set_defaults(func=audit_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        return int(args.func(args))
    except (ConfigError, AuditError) as exc:
        print(f"validation error: {exc}", file=sys.stderr)
        return 2
