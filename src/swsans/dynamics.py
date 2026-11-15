"""Landau-Lifshitz time stepping on the unit sphere."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def normalize_vectors(vectors: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalize a batch of non-zero three-vectors."""

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise ArithmeticError("A zero magnetization vector cannot be normalized.")
    return vectors / norms


def landau_lifshitz_rhs(
    magnetization: NDArray[np.float64],
    field_t: NDArray[np.float64],
    gyromagnetic_factor: float,
    damping: float,
) -> NDArray[np.float64]:
    """Return the reduced Landau-Lifshitz right-hand side."""

    precession = np.cross(magnetization, field_t)
    damping_torque = np.cross(magnetization, precession)
    return -gyromagnetic_factor * precession - damping * damping_torque
