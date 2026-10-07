from __future__ import annotations

import pytest

from tangle_sim.config import load_config
from tangle_sim.metrics.exit_probabilities import mixed_endpoint_distribution
import tangle_sim.simulation.attack as attack_module
from tangle_sim.simulation.attack import run_attack
from tangle_sim.simulation.engine import Simulator


def test_attack_battery_executes_each_configured_tip_selector() -> None:
    config = load_config("configs/article.yaml")
    config.raw["attack"].update({
        "private_build_seconds": 1.0,
        "attack_observation_seconds": 4.0,
        "confidence_stability_seconds": 1.0,
        "confidence_check_interval_seconds": 1.0,
    })
    scenario = config.scenarios[0]
    simulator = Simulator(config)
    final_time = (
        float(config.raw["simulation"]["warmup_seconds"])
        + float(config.raw["attack"]["private_build_seconds"])
        + float(config.raw["attack"]["attack_observation_seconds"])
    )
    topology, schedule = simulator.common_inputs(scenario, 0, final_time)
    outcomes = {
        algorithm: run_attack(simulator, scenario, algorithm, 0, topology, schedule)
        for algorithm in ("urts", "mcmc", "sb_wrw")
    }

    assert len({outcome.simulation.diagnostics["external_schedule_sha256"] for outcome in outcomes.values()}) == 1
    assert outcomes["urts"].simulation.diagnostics["walks"] == 0
    assert outcomes["mcmc"].simulation.diagnostics["walks"] > 0
    assert outcomes["sb_wrw"].simulation.diagnostics["walks"] > 0
    assert outcomes["sb_wrw"].simulation.diagnostics["ring_r0_count"] > 0
    for outcome in outcomes.values():
        assert outcome.confidence_history
        assert 0 <= outcome.metrics.fraud_confidence_final <= 1
        assert 0 <= outcome.metrics.honest_confidence_final <= 1
        diagnostics = outcome.simulation.diagnostics
        assert diagnostics["private_attacker_transactions"] >= 1
        assert diagnostics["post_release_attacker_transactions"] >= 0
        assert diagnostics["honest_cumulative_weight_at_release"] >= 1
        assert diagnostics["fraud_cumulative_weight_at_release"] >= 1
        assert diagnostics["fraud_to_honest_weight_ratio_at_release"] >= 0
        assert diagnostics["fraud_consolidation_requires_honest_rejection"] is True


def test_simple_one_pinned_pc_links_every_malicious_tx_to_root() -> None:
    config = load_config("configs/article.yaml")
    config.raw["attack"].update({
        "parasite_chain_strategy": "simple_1_pinned",
        "private_build_seconds": 2.0,
        "attack_observation_seconds": 2.0,
        "continue_attacker_after_release": False,
    })
    scenario = config.scenarios[0]
    simulator = Simulator(config)
    final_time = (
        float(config.raw["simulation"]["warmup_seconds"])
        + float(config.raw["attack"]["private_build_seconds"])
        + float(config.raw["attack"]["attack_observation_seconds"])
    )
    topology, schedule = simulator.common_inputs(scenario, 0, final_time)
    outcome = run_attack(simulator, scenario, "urts", 0, topology, schedule)
    tangle = outcome.simulation.tangle
    diagnostics = outcome.simulation.diagnostics
    root = int(diagnostics["parasite_root_tx_id"])
    fraud = outcome.metrics.fraud_tx_id
    malicious_descendants = [
        tx for tx in tangle.transactions
        if not tx.honest and tx.id != fraud
    ]

    assert diagnostics["parasite_chain_strategy"] == "simple_1_pinned"
    assert diagnostics["parasite_links_to_root"] == len(malicious_descendants)
    assert malicious_descendants
    assert all(root in tx.parents and len(tx.parents) == 2 for tx in malicious_descendants)


def test_distributed_attacker_schedule_is_common_and_spatially_diverse() -> None:
    config = load_config("configs/article.yaml")
    config.raw["attack"].update({
        "attacker_rate_fraction": 0.80,
        "attacker_rate_semantics": "total_issuance_share",
        "attacker_issuer_policy": "topology_proportional",
        "private_build_seconds": 2.0,
        "attack_observation_seconds": 2.0,
        "continue_attacker_after_release": True,
    })
    scenario = config.scenarios[0]
    simulator = Simulator(config)
    final_time = float(config.raw["simulation"]["warmup_seconds"]) + 4.0
    topology, schedule = simulator.common_inputs(scenario, 0, final_time)
    outcomes = [
        run_attack(simulator, scenario, algorithm, 0, topology, schedule)
        for algorithm in ("urts", "mcmc", "sb_wrw")
    ]

    diagnostics = [outcome.simulation.diagnostics for outcome in outcomes]
    assert {item["attacker_schedule_sha256"] for item in diagnostics} == {
        diagnostics[0]["attacker_schedule_sha256"]
    }
    assert all(
        item["attacker_issuer_policy"] == "topology_proportional"
        for item in diagnostics
    )
    assert diagnostics[0]["attacker_distinct_issuers"] > 1
    assert diagnostics[0]["attacker_distinct_sectors"] > 1


def test_double_spend_can_use_the_same_attacker_origin() -> None:
    config = load_config("configs/article.yaml")
    config.raw["attack"].update({
        "conflict_issuer_policy": "same_attacker",
        "private_build_seconds": 1.0,
        "attack_observation_seconds": 2.0,
    })
    scenario = config.scenarios[0]
    simulator = Simulator(config)
    final_time = float(config.raw["simulation"]["warmup_seconds"]) + 3.0
    topology, schedule = simulator.common_inputs(scenario, 0, final_time)

    outcome = run_attack(
        simulator, scenario, "sb_wrw", 0, topology, schedule,
    )
    honest = outcome.simulation.tangle.transactions[outcome.metrics.honest_tx_id]
    fraud = outcome.simulation.tangle.transactions[outcome.metrics.fraud_tx_id]

    assert honest.issuer_id == fraud.issuer_id
    assert honest.sector_id == fraud.sector_id
    assert honest.position == fraud.position
    assert outcome.simulation.diagnostics["conflict_same_issuer"] is True


def test_multi_pinned_pc_uses_multiple_spatial_roots() -> None:
    config = load_config("configs/article.yaml")
    config.raw["attack"].update({
        "attacker_rate_fraction": 0.80,
        "attacker_rate_semantics": "total_issuance_share",
        "attacker_issuer_policy": "topology_proportional",
        "parasite_chain_strategy": "simple_multi_pinned",
        "private_build_seconds": 2.0,
        "attack_observation_seconds": 2.0,
        "continue_attacker_after_release": True,
    })
    scenario = config.scenarios[0]
    simulator = Simulator(config)
    final_time = float(config.raw["simulation"]["warmup_seconds"]) + 4.0
    topology, schedule = simulator.common_inputs(scenario, 0, final_time)
    outcome = run_attack(
        simulator, scenario, "sb_wrw", 0, topology, schedule,
    )
    diagnostics = outcome.simulation.diagnostics

    assert diagnostics["parasite_chain_strategy"] == "simple_multi_pinned"
    assert diagnostics["parasite_root_count"] > 1
    assert diagnostics["parasite_links_to_roots"] > 1


def test_tsa_adaptive_attack_fans_out_only_against_urts() -> None:
    config = load_config("configs/article.yaml")
    config.raw["attack"].update({
        "attacker_rate_fraction": 0.80,
        "attacker_rate_semantics": "total_issuance_share",
        "attacker_issuer_policy": "topology_proportional",
        "parasite_chain_strategy": "tsa_adaptive",
        "parasite_root_policy": "recent_visible_transactions",
        "private_build_seconds": 2.0,
        "attack_observation_seconds": 2.0,
        "continue_attacker_after_release": False,
    })
    scenario = config.scenarios[0]
    simulator = Simulator(config)
    final_time = float(config.raw["simulation"]["warmup_seconds"]) + 4.0
    topology, schedule = simulator.common_inputs(scenario, 0, final_time)

    urts = run_attack(simulator, scenario, "urts", 0, topology, schedule)
    mcmc = run_attack(simulator, scenario, "mcmc", 0, topology, schedule)
    assert urts.simulation.diagnostics[
        "effective_parasite_chain_strategy"
    ] == "urts_tip_flood"
    assert mcmc.simulation.diagnostics[
        "effective_parasite_chain_strategy"
    ] == "simple_multi_pinned"

    fraud_id = urts.metrics.fraud_tx_id
    descendants = [
        tx for tx in urts.simulation.tangle.transactions
        if not tx.honest and tx.id != fraud_id
    ]
    assert len(descendants) > 1
    assert all(fraud_id in tx.parents for tx in descendants)


def test_tsa_adaptive_v2_uses_conventional_pc_against_mcmc() -> None:
    config = load_config("configs/article.yaml")
    config.raw["attack"].update({
        "attacker_rate_fraction": 0.80,
        "attacker_rate_semantics": "total_issuance_share",
        "attacker_issuer_policy": "topology_proportional",
        "parasite_chain_strategy": "tsa_adaptive_v2",
        "parasite_root_policy": "visible_tips",
        "private_build_seconds": 2.0,
        "attack_observation_seconds": 2.0,
        "continue_attacker_after_release": False,
    })
    scenario = config.scenarios[0]
    simulator = Simulator(config)
    final_time = float(config.raw["simulation"]["warmup_seconds"]) + 4.0
    topology, schedule = simulator.common_inputs(scenario, 0, final_time)

    outcomes = {
        algorithm: run_attack(
            simulator, scenario, algorithm, 0, topology, schedule,
        )
        for algorithm in ("urts", "mcmc", "sb_wrw")
    }

    assert outcomes["urts"].simulation.diagnostics[
        "effective_parasite_chain_strategy"
    ] == "urts_tip_flood"
    assert outcomes["mcmc"].simulation.diagnostics[
        "effective_parasite_chain_strategy"
    ] == "simple_1_pinned"
    assert outcomes["sb_wrw"].simulation.diagnostics[
        "effective_parasite_chain_strategy"
    ] == "sector_hybrid_multi_pinned"


@pytest.mark.parametrize(
    ("fraud_confidence", "honest_confidence", "observed"),
    ((1.0, 0.0, True), (0.0, 1.0, False)),
)
def test_event_driven_fraud_detector_has_exact_stability_deadline(
    monkeypatch: pytest.MonkeyPatch,
    fraud_confidence: float,
    honest_confidence: float,
    observed: bool,
) -> None:
    config = load_config("configs/article.yaml")
    config.raw["attack"].update({
        "attacker_rate_fraction": 0.0,
        "private_build_seconds": 1.0,
        "attack_observation_seconds": 4.0,
        "confidence_stability_seconds": 1.25,
        # Deliberately coarser than stability: event-driven mode must ignore it.
        "confidence_check_interval_seconds": 2.0,
        "fraud_detection_mode": "event_driven",
    })
    monkeypatch.setattr(
        attack_module,
        "_branch_confidence",
        lambda *args, **kwargs: (fraud_confidence, honest_confidence),
    )
    scenario = config.scenarios[0]
    simulator = Simulator(config)
    final_time = 1.0 + 1.0 + 4.0
    topology, schedule = simulator.common_inputs(scenario, 0, final_time)

    outcome = run_attack(
        simulator, scenario, "urts", 0, topology, schedule,
    )

    assert outcome.metrics.event_observed is observed
    assert outcome.simulation.diagnostics["fraud_detection_mode"] == "event_driven"
    assert outcome.simulation.diagnostics["fraud_detection_resolution_seconds"] == 0.0
    if observed:
        assert outcome.metrics.time_to_event_or_censoring == pytest.approx(1.25)
        assert outcome.simulation.diagnostics["fraud_threshold_crossing_time"] == pytest.approx(0.0)
    else:
        assert outcome.metrics.time_to_event_or_censoring == 4.0
        assert outcome.simulation.diagnostics["fraud_threshold_crossing_time"] is None


def test_event_driven_deadline_rechecks_confidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config("configs/article.yaml")
    config.raw["attack"].update({
        "attacker_rate_fraction": 0.0,
        "private_build_seconds": 1.0,
        "attack_observation_seconds": 3.0,
        "confidence_stability_seconds": 1.0,
        "fraud_detection_mode": "event_driven",
    })
    release_time = float(config.raw["simulation"]["warmup_seconds"]) + 1.0

    def confidence(*args, **kwargs):
        at_time = float(args[3])
        return (1.0, 0.0) if at_time < release_time + 0.5 else (0.0, 1.0)

    monkeypatch.setattr(attack_module, "_branch_confidence", confidence)
    scenario = config.scenarios[0]
    simulator = Simulator(config)
    final_time = release_time + 3.0
    topology, schedule = simulator.common_inputs(scenario, 0, final_time)
    outcome = run_attack(simulator, scenario, "urts", 0, topology, schedule)

    assert outcome.metrics.event_observed is False
    assert outcome.metrics.time_to_event_or_censoring == 3.0


def test_event_driven_sb_zero_probability_uses_absorbing_shortcut(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config("configs/article.yaml")
    config.raw["attack"].update({
        "attacker_rate_fraction": 0.0,
        "private_build_seconds": 1.0,
        "attack_observation_seconds": 2.0,
        "fraud_detection_mode": "event_driven",
        "continue_attacker_after_release": False,
    })
    monkeypatch.setattr(
        attack_module,
        "_branch_confidence",
        lambda *args, **kwargs: (0.0, 1.0),
    )
    scenario = config.scenarios[0]
    simulator = Simulator(config)
    final_time = float(config.raw["simulation"]["warmup_seconds"]) + 3.0
    topology, _ = simulator.common_inputs(scenario, 0, final_time)
    outcome = run_attack(
        simulator, scenario, "sb_wrw", 0, topology, (),
    )

    assert outcome.metrics.event_observed is False
    assert outcome.simulation.diagnostics["fraud_zero_absorbing_shortcut"] is True
    assert outcome.simulation.diagnostics["fraud_confidence_evaluations"] == 2


def test_optimized_branch_confidence_matches_full_absorbing_dag() -> None:
    config = load_config("configs/article.yaml")
    config.raw["attack"].update({
        "private_build_seconds": 1.0,
        "attack_observation_seconds": 3.0,
    })
    scenario = config.scenarios[0]
    simulator = Simulator(config)
    final_time = float(config.raw["simulation"]["warmup_seconds"]) + 4.0
    topology, schedule = simulator.common_inputs(scenario, 0, final_time)

    for algorithm in ("urts", "mcmc", "sb_wrw"):
        outcome = run_attack(
            simulator, scenario, algorithm, 0, topology, schedule,
        )
        view = outcome.simulation.tangle.view(final_time)
        alpha = 0.0 if algorithm == "urts" else 2.25
        endpoints = mixed_endpoint_distribution(
            view,
            algorithm,
            topology,
            alpha=alpha,
            grid=scenario.grid,
            entry_point=0,
        )
        expected_fraud = sum(
            float(probability)
            for endpoint, probability in enumerate(endpoints)
            if probability and view.approves(endpoint, outcome.metrics.fraud_tx_id)
        )
        expected_honest = sum(
            float(probability)
            for endpoint, probability in enumerate(endpoints)
            if probability and view.approves(endpoint, outcome.metrics.honest_tx_id)
        )
        actual = attack_module._branch_confidence(
            outcome.simulation.tangle,
            outcome.metrics.fraud_tx_id,
            outcome.metrics.honest_tx_id,
            final_time,
            algorithm,
            config,
            scenario,
            topology,
            simulator.seeds.generator(
                scenario.name, 0, "measurement", algorithm,
            ),
        )
        assert actual == pytest.approx((expected_fraud, expected_honest))
