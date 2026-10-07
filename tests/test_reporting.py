from __future__ import annotations

import numpy as np
import pytest

from tangle_sim.metrics.bootstrap import Interval
from tangle_sim.reporting.tables import descriptive_statistics


def test_descriptive_statistics_use_replicates_as_experimental_units() -> None:
    values = np.asarray([10.0, 12.0, 14.0, 16.0])
    result = descriptive_statistics(values, Interval(13.0, 11.0, 15.0))

    assert result["mean"] == 13.0
    assert result["sample_variance"] == pytest.approx(20.0 / 3.0)
    assert result["sample_standard_deviation"] == pytest.approx(np.sqrt(20.0 / 3.0))
    assert result["standard_error"] == pytest.approx(np.sqrt(20.0 / 3.0) / 2.0)
    assert result["relative_standard_error_percent"] == pytest.approx(
        100.0 * np.sqrt(20.0 / 3.0) / 2.0 / 13.0
    )
    assert result["margin_of_error"] == 2.0


def test_descriptive_statistics_single_replicate_has_zero_sample_dispersion() -> None:
    result = descriptive_statistics(np.asarray([7.0]), Interval(7.0, 7.0, 7.0))
    assert result["sample_variance"] == 0.0
    assert result["sample_standard_deviation"] == 0.0
    assert result["standard_error"] == 0.0
    assert result["relative_standard_error_percent"] == 0.0
    assert result["margin_of_error"] == 0.0


def test_asymmetric_interval_separates_half_width_from_margin() -> None:
    result = descriptive_statistics(np.asarray([8.0, 12.0]), Interval(10.0, 9.0, 13.0))
    assert result["ci_half_width"] == 2.0
    assert result["margin_of_error"] == 3.0
