from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tangle_sim.config import (
    ConfigError,
    load_config,
    scenario_alpha,
    scenario_private_build_seconds,
)


def test_primary_configuration_is_operational_and_genesis_started() -> None:
    config = load_config("configs/article.yaml")
    assert config.raw["algorithms"]["mcmc"]["entry_point_policy"] == "genesis"
    assert config.raw["algorithms"]["sb_wrw"]["conformance"] == "operational_reproduction"
    assert config.raw["algorithms"]["mcmc"]["alpha"] == config.raw["algorithms"]["sb_wrw"]["alpha"]
    assert config.raw["algorithms"]["mcmc"]["alpha"] == 0.1


def test_scenario_alpha_defaults_and_explicit_overrides() -> None:
    config = load_config("configs/article.yaml")
    assert scenario_alpha(config, "PACTCare") == 2.25
    assert scenario_alpha(config, "TLIP") == 0.1


def test_final_config_keeps_validated_liveness_bias_and_attack_model() -> None:
    config = load_config("configs/article.yaml")

    assert scenario_alpha(config, "PACTCare") == 2.25
    assert scenario_alpha(config, "TLIP") == 0.1
    assert scenario_alpha(config, "ElaadNL") == 0.1
    assert config.raw["attack"]["private_build_seconds"] == 5.0
    assert config.raw["attack"]["parasite_chain_strategy"] == "tsa_adaptive_v2"
    assert config.raw["attack"]["attacker_issuer_policy"] == "topology_proportional"


def test_final_config_resolves_private_build_by_scenario() -> None:
    config = load_config("configs/article.yaml")

    assert scenario_private_build_seconds(config, "PACTCare") == 5.0
    assert scenario_private_build_seconds(config, "TLIP") == 15.0
    assert scenario_private_build_seconds(config, "ElaadNL") == 15.0


def test_attack_phase_switches_must_be_boolean() -> None:
    config = load_config("configs/article.yaml")
    assert isinstance(config.raw["attack"].get("continue_attacker_after_release", True), bool)


def test_unknown_parasite_chain_strategy_is_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("configs/article.yaml").read_text())
    raw["attack"]["parasite_chain_strategy"] = "invented"
    path = tmp_path / "invalid-parasite.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError, match="parasite_chain_strategy"):
        load_config(path)


def test_total_adversarial_share_must_be_below_one(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("configs/article.yaml").read_text())
    raw["attack"]["attacker_rate_semantics"] = "total_issuance_share"
    raw["attack"]["attacker_rate_fraction"] = 1.0
    path = tmp_path / "invalid-adversarial-share.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError, match="adversarial issuance share"):
        load_config(path)


def test_unknown_fraud_detection_mode_is_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("configs/article.yaml").read_text())
    raw["attack"]["fraud_detection_mode"] = "approximate_magic"
    path = tmp_path / "invalid-detection-mode.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError, match="fraud_detection_mode"):
        load_config(path)


def test_unknown_attacker_rate_profile_is_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("configs/article.yaml").read_text())
    raw["attack"]["attacker_rate_profile"] = "retroactive_magic"
    path = tmp_path / "invalid-rate-profile.yaml"
    path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ConfigError, match="attacker_rate_profile"):
        load_config(path)


def test_unknown_attacker_attachment_delay_policy_is_rejected(
    tmp_path: Path,
) -> None:
    raw = yaml.safe_load(Path("configs/article.yaml").read_text())
    raw["attack"]["attacker_attachment_delay_policy"] = "teleport"
    path = tmp_path / "invalid-attacker-delay.yaml"
    path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ConfigError, match="attacker_attachment_delay_policy"):
        load_config(path)


def test_progressive_attacker_schedule_is_validated(tmp_path: Path) -> None:
    config = load_config("configs/article.yaml")
    assert config.raw["attack"]["attacker_rate_fraction_schedule"] == [
        {"after_seconds": 60.0, "fraction": 0.70},
        {"after_seconds": 75.0, "fraction": 0.85},
    ]

    raw = yaml.safe_load(Path("configs/article.yaml").read_text())
    raw["attack"]["attacker_rate_fraction_schedule"] = [
        {"after_seconds": 60.0, "fraction": 0.70},
        {"after_seconds": 30.0, "fraction": 0.80},
    ]
    path = tmp_path / "invalid-progressive-schedule.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError, match="strictly increasing"):
        load_config(path)


def test_article_configuration_matches_declared_protocol() -> None:
    config = load_config("configs/article.yaml")

    assert config.replicates == 10
    assert config.root_seed == 20250701
    assert config.raw["simulation"] == {
        "warmup_seconds": 5.0,
        "measurement_seconds": 25.0,
        "drain_seconds": 5.0,
        "abandonment_age_seconds": 5.0,
        "confirmation_threshold": 0.95,
        "confidence_samples": 200,
    }
    assert scenario_alpha(config, "PACTCare") == 2.25
    assert scenario_alpha(config, "TLIP") == 0.1
    assert scenario_alpha(config, "ElaadNL") == 0.1
    assert config.raw["algorithms"]["sb_wrw"]["conformance"] == "operational_reproduction"
    assert config.raw["attack"]["attacker_rate_fraction"] == 0.36
    assert config.raw["attack"]["attacker_rate_fraction_schedule"] == [
        {"after_seconds": 60.0, "fraction": 0.70},
        {"after_seconds": 75.0, "fraction": 0.85},
    ]
    assert config.raw["attack"]["attack_observation_seconds"] == 110.0
    assert config.raw["attack"]["fraud_confidence_threshold"] == 0.95
    assert config.raw["attack"]["confidence_stability_seconds"] == 2.0
    assert config.raw["attack"]["confidence_check_interval_seconds"] == 0.5
    assert scenario_private_build_seconds(config, "PACTCare") == 5.0
    assert scenario_private_build_seconds(config, "TLIP") == 15.0
    assert scenario_private_build_seconds(config, "ElaadNL") == 15.0
    assert config.raw["statistics"]["bootstrap_samples"] == 10_000


def test_resolved_configuration_hash_does_not_embed_checkout_path() -> None:
    config = load_config(Path("configs/article.yaml").resolve())

    assert "scenarios_file" not in config.raw
    assert tuple(config.raw["scenarios"]) == ("PACTCare", "TLIP", "ElaadNL")
    assert str(Path.cwd().resolve()) not in config.resolved_yaml()


def test_unknown_attacker_issuer_policy_is_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("configs/article.yaml").read_text())
    raw["attack"]["attacker_issuer_policy"] = "teleporting_magic"
    path = tmp_path / "invalid-attacker-location.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError, match="attacker_issuer_policy"):
        load_config(path)


def test_unknown_conflict_issuer_policy_is_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("configs/article.yaml").read_text())
    raw["attack"]["conflict_issuer_policy"] = "different_planets"
    path = tmp_path / "invalid-conflict-origin.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError, match="conflict_issuer_policy"):
        load_config(path)


def test_paper_exact_is_rejected_without_completed_traceability(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("configs/article.yaml").read_text())
    raw["algorithms"]["sb_wrw"]["conformance"] = "paper_exact"
    path = tmp_path / "paper-exact.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError, match="paper_exact"):
        load_config(path)
