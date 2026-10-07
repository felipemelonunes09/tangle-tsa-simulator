from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tangle-sim-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tangle_sim.metrics.survival import kaplan_meier


SCENARIOS = ("PACTCare", "TLIP", "ElaadNL")
ALGORITHMS = ("urts", "mcmc", "sb_wrw")
COLORS = {"urts": "#1f77b4", "mcmc": "#ff7f0e", "sb_wrw": "#2ca02c"}
LABELS = {"urts": "URTS", "mcmc": "MCMC", "sb_wrw": "SB-WRW"}
PLOTS = {
    "confirmation_rate_percent": ("taxa_confirmacao", "Taxa de aprovação", "Taxa de aprovação (%)"),
    "abandonment_percent": ("pontas_abandonadas", "Percentual médio de pontas honestas deixadas para trás", "Pontas honestas deixadas para trás (%)"),
    "fraud_rmst_seconds": ("tempo_fraude", "RMST até a consolidação da fraude", "Tempo restrito até a fraude (s)"),
    "fraud_success_percent": ("taxa_sucesso_fraude", "Taxa de sucesso da fraude", "Fraudes consolidadas no horizonte (%)"),
}


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _lookup(rows: list[dict[str, str]], metric: str, scenario: str, algorithm: str) -> dict[str, str]:
    return next(row for row in rows if row["metric"] == metric and row["scenario"] == scenario and row["algorithm"] == algorithm)


def _bar_panel(
    ax: plt.Axes,
    rows: list[dict[str, str]],
    metric: str,
    conformance: str,
    title: str,
    ylabel: str,
    *,
    show_legend: bool = True,
    show_values: bool = True,
) -> None:
    x = np.arange(len(SCENARIOS), dtype=float)
    width = 0.24
    all_rows = [_lookup(rows, metric, scenario, algorithm) for scenario in SCENARIOS for algorithm in ALGORITHMS]
    highest_ci = max(float(row["ci_high"]) for row in all_rows)
    if metric == "confirmation_rate_percent":
        top = 105.0
        label_padding = 1.2
    else:
        top = max(1.0, highest_ci * 1.20)
        label_padding = max(0.12, top * 0.018)
    for offset, algorithm in enumerate(ALGORITHMS):
        data = [_lookup(rows, metric, scenario, algorithm) for scenario in SCENARIOS]
        values = np.asarray([float(row["estimate"]) for row in data])
        ci_high = np.asarray([float(row["ci_high"]) for row in data])
        lower = values - np.asarray([float(row["ci_low"]) for row in data])
        upper = ci_high - values
        positions = x + (offset - 1) * width
        label = LABELS[algorithm]
        if algorithm == "sb_wrw" and conformance != "paper_exact":
            label += "*"
        bars = ax.bar(
            positions, values, width, label=label, color=COLORS[algorithm],
            yerr=np.vstack([lower, upper]), capsize=3, zorder=2,
            error_kw={"zorder": 1, "elinewidth": 1.1, "capthick": 1.1},
        )
        for bar, value, upper_limit, row in zip(bars, values, ci_high, data, strict=True):
            # Two decimals keep close confirmation means distinguishable (for
            # example 97.25 versus 97.18) without adding another annotation.
            decimals = (
                2
                if metric in {
                    "confirmation_rate_percent",
                    "fraud_rmst_seconds",
                }
                else (1 if abs(value) >= 10 else 2)
            )
            value_label = f"{value:.{decimals}f}"
            if metric == "fraud_rmst_seconds":
                success = _lookup(rows, "fraud_success_percent", row["scenario"], algorithm)
                if float(success["estimate"]) == 0.0:
                    value_label = f"≥{value_label}"
                    bar.set_hatch("///")
            if show_values:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    upper_limit + label_padding,
                    value_label,
                    ha="center", va="bottom", fontsize=8, zorder=5,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 0.8},
                )
    ax.set_xticks(x, SCENARIOS)
    ax.tick_params(axis="both", labelsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=13 if show_legend else 10, pad=34 if show_legend else 10)
    ax.set_ylim(0, top)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.6)
    ax.set_axisbelow(True)
    if show_legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.09), ncol=3, frameon=False, fontsize=9)


def generate_plots(run_dir: Path, language: str = "pt-BR") -> list[Path]:
    if language != "pt-BR":
        raise ValueError("only pt-BR reporting is currently preregistered")
    summary = _read(run_dir / "summary" / "metrics_summary.csv")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    conformance = manifest["sb_wrw_conformance"]
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    emitted: list[Path] = []
    for metric, (stem, title, ylabel) in PLOTS.items():
        fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
        _bar_panel(ax, summary, metric, conformance, title, ylabel)
        for extension in ("png", "pdf", "svg"):
            path = plot_dir / f"{stem}.{extension}"
            fig.savefig(path, dpi=320 if extension == "png" else None, facecolor="white")
            emitted.append(path)
        plt.close(fig)

    # Supplementary zoom; the mandatory 0--100 confirmation chart remains primary.
    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    _bar_panel(ax, summary, "confirmation_rate_percent", conformance, "Taxa de aprovação — ampliação suplementar", "Taxa de aprovação (%)")
    lows = [float(_lookup(summary, "confirmation_rate_percent", scenario, algorithm)["ci_low"]) for scenario in SCENARIOS for algorithm in ALGORITHMS]
    ax.set_ylim(max(0.0, min(lows) - 2.0), 101.0)
    zoom = plot_dir / "taxa_confirmacao_ampliada.png"
    fig.savefig(zoom, dpi=320, facecolor="white")
    emitted.append(zoom)
    plt.close(fig)

    # RMST values concentrate near tau when most runs are right-censored. A
    # point/interval view makes the small identified differences visible
    # without using truncated bars, which would exaggerate them visually.
    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    x = np.arange(len(SCENARIOS), dtype=float)
    width = 0.24
    lows: list[float] = []
    highs: list[float] = []
    for offset, algorithm in enumerate(ALGORITHMS):
        data = [_lookup(summary, "fraud_rmst_seconds", scenario, algorithm) for scenario in SCENARIOS]
        values = np.asarray([float(row["estimate"]) for row in data])
        ci_low = np.asarray([float(row["ci_low"]) for row in data])
        ci_high = np.asarray([float(row["ci_high"]) for row in data])
        lows.extend(ci_low.tolist())
        highs.extend(ci_high.tolist())
        positions = x + (offset - 1) * width
        label = LABELS[algorithm] + ("*" if algorithm == "sb_wrw" and conformance != "paper_exact" else "")
        ax.errorbar(
            positions, values, yerr=np.vstack([values - ci_low, ci_high - values]),
            fmt="o", color=COLORS[algorithm], label=label, capsize=4,
            linewidth=1.4, markersize=5, zorder=2,
        )
        for position, value, upper, row in zip(positions, values, ci_high, data, strict=True):
            success = _lookup(summary, "fraud_success_percent", row["scenario"], algorithm)
            value_label = f"{value:.2f}"
            if float(success["estimate"]) == 0.0:
                value_label = f"≥{value_label}"
            ax.text(
                position, upper + 0.22, value_label, ha="center", va="bottom",
                fontsize=8, zorder=5,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 0.7},
            )
    tau = max(highs)
    ax.axhline(tau, color="#666666", linestyle="--", linewidth=0.8, label=f"horizonte τ={tau:g} s")
    ax.set_ylim(max(0.0, min(lows) - 1.2), tau + 1.6)
    ax.set_xticks(x, SCENARIOS)
    ax.set_ylabel("RMST até a fraude (s)", fontsize=10)
    ax.set_title("Tempo até a fraude — ampliação com censura", fontsize=13, pad=34)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=4, frameon=False, fontsize=8)
    path = plot_dir / "tempo_fraude_ampliado.png"
    fig.savefig(path, dpi=320, facecolor="white")
    emitted.append(path)
    plt.close(fig)

    # Continuous attack outcome remains informative when every threshold event
    # is right-censored and RMST consequently equals tau.
    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    _bar_panel(
        ax, summary, "fraud_confidence_final_percent", conformance,
        "Confiança fraudulenta ao fim do horizonte", "Confiança fraudulenta final (%)",
    )
    ax.set_ylim(0, 105)
    for extension in ("png", "pdf", "svg"):
        path = plot_dir / f"confianca_fraude_final.{extension}"
        fig.savefig(path, dpi=320 if extension == "png" else None, facecolor="white")
        emitted.append(path)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    compact_titles = {
        "confirmation_rate_percent": "Taxa de aprovação",
        "abandonment_percent": "Pontas deixadas para trás",
        "fraud_rmst_seconds": "Tempo até fraude (RMST)",
    }
    combined_metrics = (
        "confirmation_rate_percent", "abandonment_percent",
        "fraud_rmst_seconds",
    )
    for ax, metric in zip(axes, combined_metrics, strict=True):
        ylabel = PLOTS[metric][2]
        _bar_panel(
            ax, summary, metric, conformance, compact_titles[metric], ylabel,
            show_legend=False, show_values=False,
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=3, frameon=False, fontsize=9)
    fig.subplots_adjust(top=0.82, bottom=0.15, wspace=0.34)
    # ``graficos_combinados.png`` is the exact filename referenced by the
    # manuscript. Keep the historical name as a compatibility alias.
    for filename in ("graficos_combinados.png", "metricas_combinadas.png"):
        combined = plot_dir / filename
        fig.savefig(combined, dpi=320, facecolor="white")
        emitted.append(combined)
    plt.close(fig)

    captions = [
        "SB-WRW*: reprodução operacional local-first; correspondência com o pseudocódigo integral dos autores ainda não verificada.",
        "Taxa de aprovação benigna: percentual de transações elegíveis que recebeu ao menos um aprovador direto até o snapshot final (liveness v2).",
        "Confiança condicionada CC>=0,95 permanece no CSV como confidence_confirmation_rate_percent e não é a barra de liveness.",
        "Tempo de fraude: RMST até tau, com censura à direita; valores maiores indicam maior resistência.",
        "Rótulo ≥tau e hachura: nenhuma consolidação observada; a barra é limite inferior censurado, não tempo de evento.",
        "Confiança fraudulenta final: probabilidade do TSA terminar em ponta que aprova F0 no último snapshot.",
        "Barras de erro: intervalos de confiança de 95% por bootstrap de réplicas.",
        "Média, variância amostral, desvio-padrão, erro-padrão, margem de erro e erro relativo: summary/metrics_summary.csv.",
        "Taxa de não aprovação: 100% menos a média da taxa de aprovação.",
        "Quando a maioria das réplicas é censurada, RMST se aproxima de tau; use também taxa_sucesso_fraude.png e tempo_fraude_ampliado.png.",
    ]
    fraction_schedule = manifest.get("attacker_rate_fraction_schedule", [])
    if fraction_schedule:
        schedule_text = ", ".join(
            f"{100 * float(step['fraction']):g}% após {float(step['after_seconds']):g} s"
            for step in fraction_schedule
        )
        captions.insert(
            1,
            "Bateria adversarial de estresse progressivo: "
            f"{schedule_text}; os tempos medem desde a publicação de F0.",
        )
    caption_path = plot_dir / "legendas_metodologicas.txt"
    caption_path.write_text("\n".join(captions) + "\n", encoding="utf-8")
    emitted.append(caption_path)

    attack = _read(run_dir / "raw" / "attack_runs.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True, constrained_layout=True)
    for ax, scenario in zip(axes, SCENARIOS, strict=True):
        for algorithm in ALGORITHMS:
            group = [row for row in attack if row["scenario"] == scenario and row["algorithm"] == algorithm]
            times = [float(row["time_to_event_or_censoring"]) for row in group]
            observed = [row["event_observed"].lower() in {"true", "1"} for row in group]
            points = kaplan_meier(times, observed)
            xs = [point.time for point in points]
            ys = [point.survival for point in points]
            tau = min(float(row["censoring_horizon"]) for row in group)
            xs.append(tau)
            ys.append(ys[-1])
            label = LABELS[algorithm] + ("*" if algorithm == "sb_wrw" and conformance != "paper_exact" else "")
            ax.step(xs, ys, where="post", color=COLORS[algorithm], label=label)
        ax.set_title(scenario)
        ax.set_xlabel("Tempo desde a publicação de F0 (s)")
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.6)
    axes[0].set_ylabel("Probabilidade de não consolidação")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3, frameon=False)
    survival_path = plot_dir / "curvas_sobrevivencia.pdf"
    fig.savefig(survival_path, facecolor="white")
    emitted.append(survival_path)
    plt.close(fig)
    return emitted
