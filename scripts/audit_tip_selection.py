#!/usr/bin/env python3
"""Run a labelled, non-primary audit of TSA confirmation behaviour.

This script exists to investigate rankings without modifying a preregistered
primary configuration.  It preserves common exogenous inputs within every
(scenario, replicate, variant) group and exports every replicate.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tangle-sim-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tangle_sim import __version__
from tangle_sim.config import ExperimentConfig, load_config
from tangle_sim.metrics.bootstrap import bootstrap_mean
from tangle_sim.metrics.confirmation import benign_metrics
from tangle_sim.reporting.tables import write_csv
from tangle_sim.simulation.engine import Simulator


ALGORITHMS = ("urts", "mcmc", "sb_wrw")
COLORS = {"urts": "#1f77b4", "mcmc": "#ff7f0e", "sb_wrw": "#2ca02c"}
LABELS = {"urts": "URTS", "mcmc": "MCMC", "sb_wrw": "SB-WRW*"}


def _variant_config(path: str, measurement_seconds: float, alpha: float) -> ExperimentConfig:
    config = load_config(path)
    config.raw["simulation"]["measurement_seconds"] = float(measurement_seconds)
    config.raw["algorithms"]["mcmc"]["alpha"] = float(alpha)
    config.raw["algorithms"]["sb_wrw"]["alpha"] = float(alpha)
    return config


def _run_variant(
    config_path: str,
    scenario_name: str,
    replicates: int,
    measurement_seconds: float,
    alpha: float,
    analysis: str,
    algorithms: tuple[str, ...],
) -> list[dict[str, object]]:
    config = _variant_config(config_path, measurement_seconds, alpha)
    scenario = next(item for item in config.scenarios if item.name == scenario_name)
    simulator = Simulator(config)
    warmup = float(config.raw["simulation"]["warmup_seconds"])
    horizon = warmup + measurement_seconds
    rows: list[dict[str, object]] = []
    for replicate in range(replicates):
        topology, schedule = simulator.common_inputs(scenario, replicate, horizon)
        for algorithm in algorithms:
            result = simulator.run_benign(
                scenario, algorithm, replicate, topology, schedule,
            )
            metrics = benign_metrics(
                result, config, scenario, algorithm,
                simulator.seeds.generator(
                    scenario.name, replicate, "measurement", algorithm,
                ),
            )
            row: dict[str, object] = {
                "analysis": analysis,
                "scenario": scenario.name,
                "algorithm": algorithm,
                "replicate": replicate,
                "measurement_seconds": measurement_seconds,
                "alpha": alpha,
                "external_schedule_sha256": result.diagnostics["external_schedule_sha256"],
                "seed_selection": simulator.seeds.identifier(
                    scenario.name, replicate, "selection", algorithm,
                ),
            }
            row.update(metrics.as_dict())
            rows.append(row)
        print(
            f"{analysis}: measurement={measurement_seconds:g}, alpha={alpha:g}, "
            f"replicate={replicate + 1}/{replicates}",
            flush=True,
        )
    return rows


def _summarize(
    rows: list[dict[str, object]],
    config: ExperimentConfig,
) -> list[dict[str, object]]:
    groups: dict[tuple[str, float, float, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["analysis"]), float(row["measurement_seconds"]),
            float(row["alpha"]), str(row["algorithm"]),
        )
        groups[key].append(row)
    rng = Simulator(config).seeds.generator("algorithm-audit", 0, "bootstrap")
    samples = int(config.raw["statistics"]["bootstrap_samples"])
    confidence = float(config.raw["statistics"]["confidence_level"])
    summary: list[dict[str, object]] = []
    for (analysis, measurement, alpha, algorithm), group in sorted(groups.items()):
        confirmation = np.asarray(
            [float(row["confirmation_rate_percent"]) for row in group], dtype=float,
        )
        interval = bootstrap_mean(confirmation, rng, samples, confidence)
        summary.append({
            "analysis": analysis,
            "measurement_seconds": measurement,
            "alpha": alpha,
            "algorithm": algorithm,
            "replicates": len(group),
            "confirmation_rate_percent": interval.estimate,
            "confirmation_ci_low": interval.low,
            "confirmation_ci_high": interval.high,
            "confirmation_sd": float(np.std(confirmation, ddof=1)) if len(group) > 1 else 0.0,
            "confirmation_confidence_mean_percent": float(np.mean([
                float(row["confirmation_confidence_mean_percent"]) for row in group
            ])),
            "confirmation_early_80_percent": float(np.mean([
                float(row["confirmation_early_80_percent"]) for row in group
            ])),
            "confirmation_final_20_percent": float(np.mean([
                float(row["confirmation_final_20_percent"]) for row in group
            ])),
            "abandonment_percent": float(np.mean([
                float(row["abandonment_percent"]) for row in group
            ])),
            "endpoint_uniform_total_variation": float(np.mean([
                float(row["endpoint_uniform_total_variation"]) for row in group
            ])),
        })
    return summary


def _plot_horizon(summary: list[dict[str, object]], output: Path, scenario: str) -> None:
    rows = [row for row in summary if row["analysis"] == "horizon_sensitivity"]
    durations = sorted({float(row["measurement_seconds"]) for row in rows})
    x = np.arange(len(durations), dtype=float)
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    for offset, algorithm in enumerate(ALGORITHMS):
        group = [
            next(
                row for row in rows
                if row["algorithm"] == algorithm
                and float(row["measurement_seconds"]) == duration
            )
            for duration in durations
        ]
        values = np.asarray([float(row["confirmation_rate_percent"]) for row in group])
        lows = np.asarray([float(row["confirmation_ci_low"]) for row in group])
        highs = np.asarray([float(row["confirmation_ci_high"]) for row in group])
        positions = x + (offset - 1) * width
        bars = ax.bar(
            positions, values, width, label=LABELS[algorithm], color=COLORS[algorithm],
            yerr=np.vstack([values - lows, highs - values]), capsize=3, zorder=2,
            error_kw={"zorder": 1, "elinewidth": 1.1, "capthick": 1.1},
        )
        for bar, value, high in zip(bars, values, highs, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2, high + 1.2, f"{value:.1f}",
                ha="center", va="bottom", fontsize=8, zorder=5,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 0.8},
            )
    ax.set_xticks(x, [f"{duration:g} s" for duration in durations])
    ax.set_xlabel("Duração da janela de medição")
    ax.set_ylabel("Taxa de confirmação (%)")
    title = (
        f"{scenario} — taxa de confirmação"
        if len(durations) == 1
        else f"{scenario} — sensibilidade da taxa de confirmação"
    )
    ax.set_title(title, pad=34)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.09), ncol=3, frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=320, facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/article.yaml")
    parser.add_argument("--scenario", default="TLIP")
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--measurement-seconds", nargs="+", type=float, default=(25.0, 60.0))
    parser.add_argument("--alphas", nargs="*", type=float, default=(0.0, 0.001, 0.01, 0.1))
    parser.add_argument("--output", default="results/tlip-algorithm-audit-v1")
    args = parser.parse_args()
    if args.replicates < 1 or any(value <= 0 for value in args.measurement_seconds):
        parser.error("replicates and measurement durations must be positive")
    if any(alpha < 0 for alpha in args.alphas):
        parser.error("alphas must be non-negative")

    base = load_config(args.config)
    if args.scenario not in {scenario.name for scenario in base.scenarios}:
        parser.error(f"unknown scenario: {args.scenario}")
    primary_alpha = float(base.raw["algorithms"]["mcmc"]["alpha"])
    primary_measurement = float(base.raw["simulation"]["measurement_seconds"])
    rows: list[dict[str, object]] = []
    for measurement in dict.fromkeys(args.measurement_seconds):
        rows.extend(_run_variant(
            args.config, args.scenario, args.replicates, measurement,
            primary_alpha, "horizon_sensitivity", ALGORITHMS,
        ))
    for alpha in dict.fromkeys(args.alphas):
        rows.extend(_run_variant(
            args.config, args.scenario, args.replicates, primary_measurement,
            alpha, "alpha_sensitivity", ("mcmc", "sb_wrw"),
        ))

    summary = _summarize(rows, base)
    output = Path(args.output).resolve()
    write_csv(output / "raw.csv", rows)
    write_csv(output / "summary.csv", summary)
    plot_name = (
        "taxa_confirmacao.png"
        if len(set(args.measurement_seconds)) == 1
        else "taxa_confirmacao_horizonte.png"
    )
    _plot_horizon(summary, output / plot_name, args.scenario)
    metadata = {
        "status": "diagnostic_non_primary",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "simulator_version": __version__,
        "source_config": str(base.path),
        "source_config_sha256": base.sha256(),
        "scenario": args.scenario,
        "replicates": args.replicates,
        "measurement_seconds": list(dict.fromkeys(args.measurement_seconds)),
        "alphas": list(dict.fromkeys(args.alphas)),
        "primary_alpha": primary_alpha,
        "common_random_numbers": "common topology, arrivals, issuers, and delays within replicate/variant",
        "sb_wrw_conformance": base.raw["algorithms"]["sb_wrw"]["conformance"],
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(f"audit written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
