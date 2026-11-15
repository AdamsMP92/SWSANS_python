"""Derived ensemble observables and critical hysteresis states."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from swsans.models import (
    CriticalState,
    CriticalStates,
    HysteresisResult,
    InputValidationError,
)


def field_point_state(result: HysteresisResult, index: int) -> CriticalState:
    """Return one sampled hysteresis state with an unambiguous branch name."""

    if index < 0 or index >= result.field_t.size:
        raise InputValidationError("The selected field-point index is out of range.")
    turning_index = int(np.argmin(result.field_t))
    if index < turning_index:
        name = "descending branch"
    elif index == turning_index:
        name = "turning point"
    else:
        name = "ascending branch"
    return CriticalState(
        name=name,
        field_t=float(result.field_t[index]),
        mean_m=np.array(result.mean_m[index], dtype=np.float64, copy=True),
        gamma=np.array(result.gamma[index], dtype=np.float64, copy=True),
    )


def _crossing(
    coordinate: NDArray[np.float64],
    target: float,
) -> tuple[int, int, float]:
    shifted = coordinate - target
    exact = np.flatnonzero(np.isclose(shifted, 0.0, atol=1.0e-14))
    if exact.size:
        index = int(exact[0])
        return index, index, 0.0
    for right in range(1, coordinate.size):
        left = right - 1
        if shifted[left] * shifted[right] < 0.0:
            weight = (target - coordinate[left]) / (
                coordinate[right] - coordinate[left]
            )
            return left, right, float(weight)
    nearest = int(np.argmin(np.abs(shifted)))
    return nearest, nearest, 0.0


def _interpolate(
    values: NDArray[np.float64],
    left: int,
    right: int,
    weight: float,
) -> NDArray[np.float64]:
    if left == right:
        return np.array(values[left], dtype=np.float64, copy=True)
    return np.asarray(
        values[left] + weight * (values[right] - values[left]),
        dtype=np.float64,
    )


def critical_states(result: HysteresisResult) -> CriticalStates:
    """Extract saturation, first descending remanence, and coercivity."""

    descending_stop = int(np.argmin(result.field_t)) + 1
    field_desc = result.field_t[:descending_stop]
    mean_desc = result.mean_m[:descending_stop]
    gamma_desc = result.gamma[:descending_stop]

    rem_left, rem_right, rem_weight = _crossing(field_desc, 0.0)
    remanence = CriticalState(
        name="remanence",
        field_t=float(_interpolate(field_desc, rem_left, rem_right, rem_weight)),
        mean_m=_interpolate(mean_desc, rem_left, rem_right, rem_weight),
        gamma=_interpolate(gamma_desc, rem_left, rem_right, rem_weight),
    )

    coe_left, coe_right, coe_weight = _crossing(mean_desc[:, 2], 0.0)
    coercive = CriticalState(
        name="coercive",
        field_t=float(_interpolate(field_desc, coe_left, coe_right, coe_weight)),
        mean_m=_interpolate(mean_desc, coe_left, coe_right, coe_weight),
        gamma=_interpolate(gamma_desc, coe_left, coe_right, coe_weight),
    )

    saturation = CriticalState(
        name="saturation",
        field_t=float(result.field_t[0]),
        mean_m=np.array(result.mean_m[0], copy=True),
        gamma=np.array(result.gamma[0], copy=True),
    )
    return CriticalStates(
        saturation=saturation,
        remanence=remanence,
        coercive=coercive,
    )
