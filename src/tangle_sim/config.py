from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import copy
import hashlib

import yaml


class ConfigError(ValueError):
    pass


ALGORITHMS = ("urts", "mcmc", "sb_wrw")
SB_ALLOWED = {
    "conformance": {"operational_reproduction", "paper_exact"},
    "sector_assignment": {"regular_grid"},
    "sector_distance": {"manhattan"},
    "locality_policy": {"nearest_nonempty_ring"},
    "endpoint_fallback": {"nearest_ring_weighted_without_replacement"},
    "age_bias": {"disabled"},
    "continuous_distance_bias": {"disabled"},
}


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    issuers: int
    grid: tuple[int, int]
    workload: str
    delay_median: float
    delay_sigma: float
    values: dict[str, Any]

    @property
    def sectors(self) -> int:
        return self.grid[0] * self.grid[1]


@dataclass(frozen=True)
class ExperimentConfig:
    path: Path
    raw: dict[str, Any]
    scenarios: tuple[ScenarioConfig, ...]

    @property
    def root_seed(self) -> int:
        return int(self.raw["root_seed"])

    @property
    def replicates(self) -> int:
        return int(self.raw["replicates"])

    def algorithm_config(self, name: str) -> dict[str, Any]:
        if name == "urts":
            return {}
        return copy.deepcopy(self.raw["algorithms"][name])

    def resolved_yaml(self) -> str:
        return yaml.safe_dump(self.raw, sort_keys=True, allow_unicode=True)

    def sha256(self) -> str:
        return hashlib.sha256(self.resolved_yaml().encode("utf-8")).hexdigest()


def _positive(data: dict[str, Any], key: str, *, allow_zero: bool = False) -> float:
    if key not in data:
        raise ConfigError(f"missing required key: {key}")
    value = float(data[key])
    if value < 0 or (value == 0 and not allow_zero):
        raise ConfigError(f"{key} must be {'non-negative' if allow_zero else 'positive'}")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"configuration not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"configuration root must be a mapping: {path}")
    return value


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path).resolve()
    raw = _load_yaml(path)
    for key in ("schema_version", "experiment_id", "root_seed", "replicates", "algorithms", "simulation", "attack", "statistics"):
        if key not in raw:
            raise ConfigError(f"missing required key: {key}")
    if "scenarios" not in raw and "scenarios_file" not in raw:
        raise ConfigError("missing required key: scenarios")
    if int(raw["schema_version"]) != 1:
        raise ConfigError("unsupported schema_version")
    if int(raw["replicates"]) < 1:
        raise ConfigError("replicates must be >= 1")

    algorithms = raw["algorithms"]
    names = tuple(algorithms.get("names", ()))
    if not names or any(name not in ALGORITHMS for name in names) or len(set(names)) != len(names):
        raise ConfigError(f"algorithms.names must contain unique values from {ALGORITHMS}")
    mcmc = algorithms.get("mcmc", {})
    unknown_mcmc = set(mcmc) - {
        "alpha", "alpha_by_scenario", "entry_point_policy",
        "entry_point_depth_seconds", "max_walk_steps", "max_tip_retries",
    }
    if unknown_mcmc:
        raise ConfigError(f"unknown mcmc keys: {sorted(unknown_mcmc)}")
    if mcmc.get("entry_point_policy") not in {"genesis", "time_depth"}:
        raise ConfigError("mcmc.entry_point_policy must be genesis or time_depth")
    for key in ("alpha", "entry_point_depth_seconds", "max_walk_steps", "max_tip_retries"):
        _positive(mcmc, key, allow_zero=key in {"alpha", "entry_point_depth_seconds"})
    alpha_by_scenario = mcmc.get("alpha_by_scenario", {})
    if not isinstance(alpha_by_scenario, dict):
        raise ConfigError("mcmc.alpha_by_scenario must be a mapping")
    unknown_alpha_scenarios = set(alpha_by_scenario) - {"PACTCare", "TLIP", "ElaadNL"}
    if unknown_alpha_scenarios:
        raise ConfigError(
            f"unknown alpha_by_scenario keys: {sorted(unknown_alpha_scenarios)}"
        )
    for scenario_name, value in alpha_by_scenario.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"alpha_by_scenario.{scenario_name} must be numeric") from exc
        if numeric < 0:
            raise ConfigError(f"alpha_by_scenario.{scenario_name} must be non-negative")
        alpha_by_scenario[scenario_name] = numeric
    sb = algorithms.get("sb_wrw", {})
    unknown = set(sb) - (set(SB_ALLOWED) | {"alpha", "max_walk_steps", "max_tip_retries"})
    if unknown:
        raise ConfigError(f"unknown sb_wrw keys: {sorted(unknown)}")
    for key, choices in SB_ALLOWED.items():
        if sb.get(key) not in choices:
            raise ConfigError(f"sb_wrw.{key} must be one of {sorted(choices)}")
    if sb["conformance"] == "paper_exact":
        trace = path.parent.parent / "docs" / "sb_wrw_traceability.md"
        text = trace.read_text(encoding="utf-8") if trace.exists() else ""
        if "paper_exact_verified: true" not in text:
            raise ConfigError("paper_exact requires a completed traceability document")
    if sb.get("alpha") == "same_as_mcmc":
        sb["alpha"] = float(mcmc["alpha"])
    _positive(sb, "alpha", allow_zero=True)
    if float(sb["alpha"]) != float(mcmc["alpha"]):
        raise ConfigError("primary SB-WRW alpha must equal MCMC alpha")

    sim = raw["simulation"]
    for key in ("warmup_seconds", "measurement_seconds", "drain_seconds", "abandonment_age_seconds", "confirmation_threshold", "confidence_samples"):
        _positive(sim, key, allow_zero=key in {"warmup_seconds", "drain_seconds"})
    if not 0 < float(sim["confirmation_threshold"]) <= 1:
        raise ConfigError("confirmation_threshold must be in (0, 1]")
    attack = raw["attack"]
    for key in ("attacker_rate_fraction", "private_build_seconds", "attack_observation_seconds", "fraud_confidence_threshold", "confidence_stability_seconds", "confidence_check_interval_seconds", "confidence_samples"):
        _positive(attack, key, allow_zero=key == "attacker_rate_fraction")
    for key in (
        "private_build_seconds_by_scenario",
        "sector_merge_interval_by_scenario",
    ):
        values = attack.get(key, {})
        if not isinstance(values, dict):
            raise ConfigError(f"attack.{key} must be a mapping")
        unknown_scenarios = set(values) - {"PACTCare", "TLIP", "ElaadNL"}
        if unknown_scenarios:
            raise ConfigError(
                f"unknown {key} scenarios: {sorted(unknown_scenarios)}"
            )
        for scenario_name, value in values.items():
            numeric = float(value)
            if numeric <= 0 or (
                key == "sector_merge_interval_by_scenario"
                and not numeric.is_integer()
            ):
                raise ConfigError(
                    f"attack.{key}.{scenario_name} must be positive"
                    + (
                        " integer"
                        if key == "sector_merge_interval_by_scenario"
                        else ""
                    )
                )
    if not 0 < float(attack["fraud_confidence_threshold"]) <= 1:
        raise ConfigError("fraud_confidence_threshold must be in (0, 1]")
    detection_mode = attack.get("fraud_detection_mode", "periodic")
    if detection_mode not in {"periodic", "event_driven"}:
        raise ConfigError(
            "attack.fraud_detection_mode must be periodic or event_driven"
        )
    parasite_strategy = attack.get("parasite_chain_strategy", "linear_legacy")
    if parasite_strategy not in {
        "linear_legacy",
        "simple_1_pinned",
        "simple_multi_pinned",
        "sector_parallel_multi_pinned",
        "sector_hybrid_multi_pinned",
        "urts_tip_flood",
        "tsa_adaptive",
        "tsa_adaptive_v2",
    }:
        raise ConfigError(
            "attack.parasite_chain_strategy must be linear_legacy, "
            "simple_1_pinned, simple_multi_pinned, "
            "sector_parallel_multi_pinned, sector_hybrid_multi_pinned, "
            "urts_tip_flood, tsa_adaptive, or tsa_adaptive_v2"
        )
    if int(attack.get("sector_merge_interval", 2)) < 1:
        raise ConfigError("attack.sector_merge_interval must be >= 1")
    parasite_root_policy = attack.get("parasite_root_policy", "visible_tips")
    if parasite_root_policy not in {
        "visible_tips",
        "recent_visible_transactions",
    }:
        raise ConfigError(
            "attack.parasite_root_policy must be visible_tips or "
            "recent_visible_transactions"
        )
    rate_semantics = attack.get("attacker_rate_semantics", "honest_rate_ratio")
    if rate_semantics not in {"honest_rate_ratio", "total_issuance_share"}:
        raise ConfigError(
            "attack.attacker_rate_semantics must be honest_rate_ratio or total_issuance_share"
        )
    if rate_semantics == "total_issuance_share" and not 0 <= float(attack["attacker_rate_fraction"]) < 1:
        raise ConfigError("total adversarial issuance share must be in [0, 1)")
    rate_profile = attack.get("attacker_rate_profile", "nominal_average")
    if rate_profile not in {"nominal_average", "workload_matched"}:
        raise ConfigError(
            "attack.attacker_rate_profile must be nominal_average or "
            "workload_matched"
        )
    fraction_schedule = attack.get("attacker_rate_fraction_schedule", [])
    if not isinstance(fraction_schedule, list):
        raise ConfigError(
            "attack.attacker_rate_fraction_schedule must be a list"
        )
    previous_after = -1.0
    for index, step in enumerate(fraction_schedule):
        if not isinstance(step, dict) or set(step) != {
            "after_seconds", "fraction",
        }:
            raise ConfigError(
                "each attacker_rate_fraction_schedule step must contain "
                "after_seconds and fraction"
            )
        after_seconds = float(step["after_seconds"])
        fraction = float(step["fraction"])
        if after_seconds < 0 or after_seconds <= previous_after:
            raise ConfigError(
                "attacker_rate_fraction_schedule after_seconds must be "
                "non-negative and strictly increasing"
            )
        if rate_semantics == "total_issuance_share":
            if not 0 <= fraction < 1:
                raise ConfigError(
                    "scheduled total adversarial issuance share must be "
                    "in [0, 1)"
                )
        elif fraction < 0:
            raise ConfigError(
                "scheduled honest-rate ratio must be non-negative"
            )
        fraction_schedule[index] = {
            "after_seconds": after_seconds,
            "fraction": fraction,
        }
        previous_after = after_seconds
    attacker_issuer_policy = attack.get("attacker_issuer_policy", "single_fixed")
    if attacker_issuer_policy not in {"single_fixed", "topology_proportional"}:
        raise ConfigError(
            "attack.attacker_issuer_policy must be single_fixed or "
            "topology_proportional"
        )
    attachment_delay_policy = attack.get(
        "attacker_attachment_delay_policy", "instant"
    )
    if attachment_delay_policy not in {"instant", "scenario_lognormal"}:
        raise ConfigError(
            "attack.attacker_attachment_delay_policy must be instant or "
            "scenario_lognormal"
        )
    conflict_issuer_policy = attack.get(
        "conflict_issuer_policy", "legacy_split"
    )
    if conflict_issuer_policy not in {"legacy_split", "same_attacker"}:
        raise ConfigError(
            "attack.conflict_issuer_policy must be legacy_split or "
            "same_attacker"
        )
    for key in (
        "continue_attacker_after_release",
        "require_honest_confidence_below_complement",
        "stop_after_fraud_consolidation",
    ):
        if key in attack and not isinstance(attack[key], bool):
            raise ConfigError(f"attack.{key} must be boolean")

    if "scenarios" in raw:
        scenario_doc = {"scenarios": raw["scenarios"]}
    else:
        # Backward-compatible loader for external user configurations. The
        # repository's final experiment is intentionally self-contained.
        scenario_path = Path(str(raw["scenarios_file"]))
        if not scenario_path.is_absolute():
            candidates = [Path.cwd() / scenario_path, path.parent / scenario_path.name]
            scenario_path = next((p for p in candidates if p.exists()), candidates[0])
        scenario_doc = _load_yaml(scenario_path.resolve())
    scenarios: list[ScenarioConfig] = []
    for name, values in scenario_doc.get("scenarios", {}).items():
        if values.get("workload") not in {"poisson", "mmpp", "periodic_burst"}:
            raise ConfigError(f"unsupported workload for {name}")
        grid = tuple(int(x) for x in values.get("grid", ()))
        if len(grid) != 2 or min(grid) <= 0:
            raise ConfigError(f"{name}.grid must have two positive integers")
        scenarios.append(ScenarioConfig(
            name=name,
            issuers=int(values["issuers"]),
            grid=(grid[0], grid[1]),
            workload=str(values["workload"]),
            delay_median=float(values["delay_median"]),
            delay_sigma=float(values["delay_sigma"]),
            values=copy.deepcopy(values),
        ))
    if tuple(s.name for s in scenarios) != ("PACTCare", "TLIP", "ElaadNL"):
        raise ConfigError("scenarios must be ordered PACTCare, TLIP, ElaadNL")
    raw["scenarios"] = {s.name: s.values for s in scenarios}
    return ExperimentConfig(path=path, raw=raw, scenarios=tuple(scenarios))


def scenario_alpha(config: ExperimentConfig, scenario: str) -> float:
    """Resolved MCMC/SB-WRW alpha for a scenario.

    MCMC and SB-WRW always share this value within a scenario.  The optional
    map makes load-scale calibration explicit instead of hiding it in code.
    """

    mcmc = config.raw["algorithms"]["mcmc"]
    return float(mcmc.get("alpha_by_scenario", {}).get(scenario, mcmc["alpha"]))


def scenario_private_build_seconds(
    config: ExperimentConfig,
    scenario: str,
) -> float:
    attack = config.raw["attack"]
    return float(
        attack.get("private_build_seconds_by_scenario", {}).get(
            scenario, attack["private_build_seconds"],
        )
    )
