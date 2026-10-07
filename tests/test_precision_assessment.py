from scripts.assess_precision import assess


def test_percentage_point_precision_requirement() -> None:
    result = assess({
        "scenario": "PACTCare",
        "algorithm": "urts",
        "metric": "confirmation_rate_percent",
        "replicates": "10",
        "estimate": "94",
        "ci_low": "91",
        "ci_high": "95",
    })
    assert result["conservative_margin"] == 3.0
    assert result["target_margin"] == 5.0
    assert result["requirement_met"] is True


def test_relative_precision_uses_larger_asymmetric_distance() -> None:
    result = assess({
        "scenario": "TLIP",
        "algorithm": "urts",
        "metric": "fraud_rmst_seconds",
        "replicates": "10",
        "estimate": "20",
        "ci_low": "18.5",
        "ci_high": "20.5",
    })
    assert result["conservative_margin"] == 1.5
    assert result["target_margin"] == 1.0
    assert result["relative_margin_percent"] == 7.5
    assert result["requirement_met"] is False
    assert result["planning_replicates_scaled"] == 23


def test_all_censored_rmst_is_not_treated_as_perfect_precision() -> None:
    result = assess({
        "scenario": "PACTCare",
        "algorithm": "mcmc",
        "metric": "fraud_rmst_seconds",
        "replicates": "10",
        "estimate": "110",
        "ci_low": "110",
        "ci_high": "110",
    }, observed_events=0)
    assert result["conservative_margin"] == 0.0
    assert result["requirement_met"] is None
    assert result["assessment_status"] == "not_assessable_all_censored"
    assert result["planning_replicates_scaled"] == ""
