from __future__ import annotations

from tangle_sim.config import load_config
from tangle_sim.metrics.confirmation import benign_metrics
from tangle_sim.simulation.engine import Simulator


def test_benign_algorithms_share_external_digest_and_export_denominators() -> None:
    config = load_config("configs/article.yaml")
    scenario = config.scenarios[0]
    simulator = Simulator(config)
    issue_horizon = sum(
        float(config.raw["simulation"][key])
        for key in ("warmup_seconds", "measurement_seconds")
    )
    topology, schedule = simulator.common_inputs(scenario, 0, issue_horizon)
    digests = set()
    for algorithm in ("urts", "mcmc", "sb_wrw"):
        result = simulator.run_benign(scenario, algorithm, 0, topology, schedule)
        metrics = benign_metrics(
            result, config, scenario, algorithm,
            simulator.seeds.generator(scenario.name, 0, "measurement", algorithm),
        )
        digests.add(result.diagnostics["external_schedule_sha256"])
        assert result.diagnostics["drain_window_transactions"] == 0
        assert metrics.eligible_denominator > 0
        assert 0 <= metrics.confirmation_numerator <= metrics.eligible_denominator
        assert 0 <= metrics.confidence_confirmation_numerator <= metrics.eligible_denominator
        assert 0 <= metrics.abandoned_numerator <= metrics.eligible_denominator
        assert metrics.confidence_method == "exact_absorbing_dag"
        assert metrics.confidence_samples == 0
        assert metrics.confirmation_method == "approved_by_at_least_one_direct_approver_by_final_snapshot"
    assert len(digests) == 1
