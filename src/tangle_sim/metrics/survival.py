from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class KaplanMeierPoint:
    time: float
    survival: float
    at_risk: int
    events: int


def kaplan_meier(times: np.ndarray | list[float], observed: np.ndarray | list[bool]) -> list[KaplanMeierPoint]:
    t = np.asarray(times, dtype=float)
    e = np.asarray(observed, dtype=bool)
    if t.ndim != 1 or e.ndim != 1 or len(t) != len(e) or np.any(t < 0) or np.any(~np.isfinite(t)):
        raise ValueError("invalid survival observations")
    points = [KaplanMeierPoint(0.0, 1.0, len(t), 0)]
    survival = 1.0
    for time in sorted(set(t[e])):
        at_risk = int(np.sum(t >= time))
        events = int(np.sum((t == time) & e))
        if at_risk:
            survival *= 1.0 - events / at_risk
        points.append(KaplanMeierPoint(float(time), float(survival), at_risk, events))
    return points


def rmst(times: np.ndarray | list[float], observed: np.ndarray | list[bool], tau: float) -> float:
    if tau <= 0:
        raise ValueError("tau must be positive")
    points = kaplan_meier(times, observed)
    area = 0.0
    previous_time = 0.0
    survival = 1.0
    for point in points[1:]:
        if point.time >= tau:
            break
        area += survival * (point.time - previous_time)
        previous_time = point.time
        survival = point.survival
    area += survival * (tau - previous_time)
    return float(area)


def survival_at(times: np.ndarray | list[float], observed: np.ndarray | list[bool], time: float) -> float:
    value = 1.0
    for point in kaplan_meier(times, observed)[1:]:
        if point.time > time:
            break
        value = point.survival
    return float(value)

