from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tangle_sim.config import ExperimentConfig, ScenarioConfig, scenario_alpha
from tangle_sim.simulation.engine import SimulationResult
from .exit_probabilities import ancestor_confidences, mixed_endpoint_distribution


@dataclass(frozen=True, slots=True)
class BenignMetrics:
    confirmation_numerator: int
    eligible_denominator: int
    confirmation_rate_percent: float
    confirmation_method: str
    confidence_confirmation_numerator: int
    confidence_confirmation_rate_percent: float
    abandoned_numerator: int
    abandonment_percent: float
    eligible_ids: tuple[int, ...]
    confidence_samples: int
    confidence_method: str
    monte_carlo_standard_error_max: float
    abandoned_tip_age_mean: float
    abandoned_tip_age_max: float
    confirmation_confidence_mean_percent: float
    confirmation_early_80_percent: float
    confirmation_final_20_percent: float
    confirmation_early_80_denominator: int
    confirmation_final_20_denominator: int
    endpoint_count: int
    endpoint_effective_count: float
    endpoint_max_probability: float
    endpoint_uniform_total_variation: float

    def as_dict(self) -> dict[str, int | float | str]:
        return {
            "confirmation_numerator": self.confirmation_numerator,
            "eligible_denominator": self.eligible_denominator,
            "confirmation_rate_percent": self.confirmation_rate_percent,
            "confirmation_method": self.confirmation_method,
            "confidence_confirmation_numerator": self.confidence_confirmation_numerator,
            "confidence_confirmation_rate_percent": self.confidence_confirmation_rate_percent,
            "abandoned_numerator": self.abandoned_numerator,
            "abandonment_percent": self.abandonment_percent,
            "confidence_samples": self.confidence_samples,
            "confidence_method": self.confidence_method,
            "monte_carlo_standard_error_max": self.monte_carlo_standard_error_max,
            "abandoned_tip_age_mean": self.abandoned_tip_age_mean,
            "abandoned_tip_age_max": self.abandoned_tip_age_max,
            "confirmation_confidence_mean_percent": self.confirmation_confidence_mean_percent,
            "confirmation_early_80_percent": self.confirmation_early_80_percent,
            "confirmation_final_20_percent": self.confirmation_final_20_percent,
            "confirmation_early_80_denominator": self.confirmation_early_80_denominator,
            "confirmation_final_20_denominator": self.confirmation_final_20_denominator,
            "endpoint_count": self.endpoint_count,
            "endpoint_effective_count": self.endpoint_effective_count,
            "endpoint_max_probability": self.endpoint_max_probability,
            "endpoint_uniform_total_variation": self.endpoint_uniform_total_variation,
        }


def eligible_transactions(result: SimulationResult, config: ExperimentConfig) -> tuple[int, ...]:
    sim = config.raw["simulation"]
    start = float(sim["warmup_seconds"])
    end = start + float(sim["measurement_seconds"])
    return tuple(
        tx.id for tx in result.tangle.transactions
        if tx.id != 0
        and tx.honest
        and start <= tx.issue_time < end
        and tx.visible_time <= result.final_time
    )


def estimate_confirmation_confidence(
    result: SimulationResult,
    config: ExperimentConfig,
    scenario: ScenarioConfig,
    algorithm: str,
    rng: np.random.Generator,
    samples: int | None = None,
) -> tuple[np.ndarray, int]:
    del rng, samples  # Evolution and measurement streams remain separated; exact evaluation needs no draws.
    confidence, _ = _exact_confirmation_state(result, config, scenario, algorithm)
    return confidence, 0


def _exact_confirmation_state(
    result: SimulationResult,
    config: ExperimentConfig,
    scenario: ScenarioConfig,
    algorithm: str,
) -> tuple[np.ndarray, np.ndarray]:
    view = result.tangle.view(result.final_time)
    alpha = 0.0 if algorithm == "urts" else scenario_alpha(config, scenario.name)
    endpoints = mixed_endpoint_distribution(
        view, algorithm, result.topology,
        alpha=alpha, grid=scenario.grid, entry_point=0,
    )
    return ancestor_confidences(view, endpoints), endpoints


def _percent_approved(result: SimulationResult, ids: tuple[int, ...]) -> float:
    approved = sum(
        result.tangle.transactions[tx_id].first_direct_approval_time is not None
        and result.tangle.transactions[tx_id].first_direct_approval_time <= result.final_time
        for tx_id in ids
    )
    return 100.0 * approved / len(ids) if ids else 0.0


def benign_metrics(
    result: SimulationResult,
    config: ExperimentConfig,
    scenario: ScenarioConfig,
    algorithm: str,
    rng: np.random.Generator,
) -> BenignMetrics:
    ids = eligible_transactions(result, config)
    del rng  # Exact evaluation has no measurement draws.
    confidence, endpoints = _exact_confirmation_state(result, config, scenario, algorithm)
    samples = 0
    threshold = float(config.raw["simulation"]["confirmation_threshold"])
    confidence_confirmed = sum(confidence[tx_id] >= threshold for tx_id in ids)
    confirmed = sum(
        result.tangle.transactions[tx_id].first_direct_approval_time is not None
        and result.tangle.transactions[tx_id].first_direct_approval_time <= result.final_time
        for tx_id in ids
    )
    for tx_id in ids:
        result.tangle.transactions[tx_id].confirmed = bool(
            result.tangle.transactions[tx_id].first_direct_approval_time is not None
            and result.tangle.transactions[tx_id].first_direct_approval_time <= result.final_time
        )
    age_threshold = float(config.raw["simulation"]["abandonment_age_seconds"])
    tips = set(result.tangle.view(result.final_time).tips())
    abandoned_ids = [
        tx_id for tx_id in ids
        if tx_id in tips and result.tangle.transactions[tx_id].age(result.final_time) >= age_threshold
    ]
    ages = [result.tangle.transactions[tx_id].age(result.final_time) for tx_id in abandoned_ids]
    denominator = len(ids)
    rate = 100.0 * confirmed / denominator if denominator else float("nan")
    confidence_rate = 100.0 * confidence_confirmed / denominator if denominator else float("nan")
    abandonment = 100.0 * len(abandoned_ids) / denominator if denominator else float("nan")
    start = float(config.raw["simulation"]["warmup_seconds"])
    measurement = float(config.raw["simulation"]["measurement_seconds"])
    final_quintile_start = start + 0.8 * measurement
    early_ids = tuple(
        tx_id for tx_id in ids
        if result.tangle.transactions[tx_id].issue_time < final_quintile_start
    )
    early_id_set = set(early_ids)
    final_ids = tuple(tx_id for tx_id in ids if tx_id not in early_id_set)
    view = result.tangle.view(result.final_time)
    tips = tuple(view.tips())
    tip_probabilities = endpoints[list(tips)] if tips else np.asarray([], dtype=float)
    positive_probabilities = tip_probabilities[tip_probabilities > 0.0]
    entropy = -float(np.sum(positive_probabilities * np.log(positive_probabilities))) if positive_probabilities.size else 0.0
    endpoint_count = len(tips)
    uniform_probability = 1.0 / endpoint_count if endpoint_count else 0.0
    endpoint_tv = 0.5 * float(np.abs(tip_probabilities - uniform_probability).sum()) if endpoint_count else 0.0
    # Worst Bernoulli MC standard error; zero for exact URTS evaluation.
    mc_error = 0.0
    return BenignMetrics(
        confirmation_numerator=int(confirmed),
        eligible_denominator=denominator,
        confirmation_rate_percent=float(rate),
        confirmation_method="approved_by_at_least_one_direct_approver_by_final_snapshot",
        confidence_confirmation_numerator=int(confidence_confirmed),
        confidence_confirmation_rate_percent=float(confidence_rate),
        abandoned_numerator=len(abandoned_ids),
        abandonment_percent=float(abandonment),
        eligible_ids=ids,
        confidence_samples=samples,
        confidence_method="exact_absorbing_dag",
        monte_carlo_standard_error_max=float(mc_error),
        abandoned_tip_age_mean=float(np.mean(ages)) if ages else 0.0,
        abandoned_tip_age_max=float(max(ages)) if ages else 0.0,
        confirmation_confidence_mean_percent=(
            100.0 * float(np.mean(confidence[list(ids)])) if ids else float("nan")
        ),
        confirmation_early_80_percent=_percent_approved(result, early_ids),
        confirmation_final_20_percent=_percent_approved(result, final_ids),
        confirmation_early_80_denominator=len(early_ids),
        confirmation_final_20_denominator=len(final_ids),
        endpoint_count=endpoint_count,
        endpoint_effective_count=float(np.exp(entropy)) if positive_probabilities.size else 0.0,
        endpoint_max_probability=float(tip_probabilities.max()) if endpoint_count else 0.0,
        endpoint_uniform_total_variation=endpoint_tv,
    )
