from __future__ import annotations

from dataclasses import replace

from tangle_sim.config import load_config
from tangle_sim.metrics.confirmation import benign_metrics, eligible_transactions
from tangle_sim.simulation.engine import Simulator
from tangle_sim.simulation.workloads import ExogenousTransaction


def test_quiescent_drain_completes_pending_attachment_without_new_issues() -> None:
    config = load_config("configs/article.yaml")
    scenario = replace(config.scenarios[0], delay_median=0.01, delay_sigma=0.01)
    simulator = Simulator(config)
    topology, _ = simulator.common_inputs(scenario, 0, 1.0)
    warmup = float(config.raw["simulation"]["warmup_seconds"])
    measurement_end = warmup + float(config.raw["simulation"]["measurement_seconds"])
    schedule = (
        ExogenousTransaction(warmup + 0.1, 0, 0.01),
        # Issued during measurement, attached only during drain.
        ExogenousTransaction(measurement_end - 0.1, 1, 0.2),
    )

    result = simulator.run_benign(scenario, "urts", 0, topology, schedule)
    eligible = eligible_transactions(result, config)

    assert len(result.tangle) == 3
    assert len(eligible) == 2
    assert result.diagnostics["drain_window_transactions"] == 0
    assert result.diagnostics["attachments_completed_during_drain"] == 1


def test_extending_quiescent_drain_cannot_desconfirm_existing_transactions() -> None:
    short = load_config("configs/article.yaml")
    long = load_config("configs/article.yaml")
    short.raw["simulation"]["drain_seconds"] = 0.0
    long.raw["simulation"]["drain_seconds"] = 1.0
    scenario = replace(short.scenarios[0], delay_median=0.01, delay_sigma=0.01)
    simulator = Simulator(long)
    topology, _ = simulator.common_inputs(scenario, 0, 1.0)
    measurement_end = (
        float(short.raw["simulation"]["warmup_seconds"])
        + float(short.raw["simulation"]["measurement_seconds"])
    )
    schedule = (
        # Two visible tips at measurement end.
        ExogenousTransaction(measurement_end - 0.2, 0, 0.01),
        ExogenousTransaction(measurement_end - 0.2, 1, 0.01),
        # Selects both tips before the cutoff, then attaches during long drain.
        ExogenousTransaction(measurement_end - 0.05, 2, 0.2),
    )

    results = []
    metrics = []
    for config in (short, long):
        run_simulator = Simulator(config)
        result = run_simulator.run_benign(scenario, "urts", 0, topology, schedule)
        results.append(result)
        metrics.append(benign_metrics(
            result, config, scenario, "urts",
            run_simulator.seeds.generator(scenario.name, 0, "measurement", "urts"),
        ))

    common_ids = metrics[0].eligible_ids
    assert len(common_ids) == 2
    assert all(not results[0].tangle.transactions[tx_id].confirmed for tx_id in common_ids)
    assert all(results[1].tangle.transactions[tx_id].confirmed for tx_id in common_ids)
    assert metrics[1].confirmation_numerator >= metrics[0].confirmation_numerator


def test_observed_approval_rate_is_not_self_conditioned_tsa_confidence() -> None:
    config = load_config("configs/article.yaml")
    scenario = replace(config.scenarios[0], delay_median=0.01, delay_sigma=0.01)
    simulator = Simulator(config)
    topology, _ = simulator.common_inputs(scenario, 0, 1.0)
    warmup = float(config.raw["simulation"]["warmup_seconds"])
    schedule = (ExogenousTransaction(warmup + 0.1, 0, 0.01),)

    result = simulator.run_benign(scenario, "urts", 0, topology, schedule)
    metric = benign_metrics(
        result, config, scenario, "urts",
        simulator.seeds.generator(scenario.name, 0, "measurement", "urts"),
    )

    # The sole transaction is the only endpoint, so its TSA-conditioned
    # confidence is 100%. It nevertheless received no approval from a later
    # transaction and must not count as confirmed in the liveness metric.
    assert metric.eligible_denominator == 1
    assert metric.confidence_confirmation_numerator == 1
    assert metric.confirmation_numerator == 0
