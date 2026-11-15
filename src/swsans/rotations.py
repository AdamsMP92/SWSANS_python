"""Low-discrepancy orientation sampling and rotation matrices."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import qmc


@dataclass(frozen=True, slots=True)
class OrientationEnsemble:
    """Euler angles and exact legacy lab-to-crystal matrices."""

    matrices: NDArray[np.float64]
    phi: NDArray[np.float64]
    theta: NDArray[np.float64]
    psi: NDArray[np.float64]


def sobol_orientations(
    n: int,
    *,
    skip: int = 998,
    scramble: bool = False,
    seed: int | None = None,
) -> OrientationEnsemble:
    """Return uniformly distributed z-y-z orientations from a Sobol sequence.

    The angle transform and rotation-matrix coefficients reproduce the legacy
    MATLAB implementation. SciPy and MATLAB do not promise identical Sobol
    direction numbers, so exact cross-language comparisons should use stored
    orientation fixtures rather than only a shared seed.
    """

    if n < 1:
        raise ValueError("n must be positive")
    if skip < 0:
        raise ValueError("skip cannot be negative")

    total = skip + n
    power = max(0, math.ceil(math.log2(max(1, total))))
    sampler = qmc.Sobol(d=3, scramble=scramble, seed=seed)
    points = sampler.random_base2(power)[skip:total]

    phi = 2.0 * np.pi * points[:, 0]
    theta = np.arccos(np.clip(2.0 * points[:, 1] - 1.0, -1.0, 1.0))
    psi = 2.0 * np.pi * points[:, 2]

    c_phi = np.cos(phi)
    c_psi = np.cos(psi)
    c_theta = np.cos(theta)
    s_phi = np.sin(phi)
    s_psi = np.sin(psi)
    s_theta = np.sin(theta)

    matrices = np.empty((n, 3, 3), dtype=np.float64)
    matrices[:, 0, 0] = -s_phi * s_psi + c_phi * c_psi * c_theta
    matrices[:, 0, 1] = c_phi * s_psi + c_psi * c_theta * s_phi
    matrices[:, 0, 2] = -c_psi * s_theta
    matrices[:, 1, 0] = -c_psi * s_phi - c_phi * c_theta * s_psi
    matrices[:, 1, 1] = c_phi * c_psi - c_theta * s_phi * s_psi
    matrices[:, 1, 2] = s_psi * s_theta
    matrices[:, 2, 0] = c_phi * s_theta
    matrices[:, 2, 1] = s_phi * s_theta
    matrices[:, 2, 2] = c_theta

    return OrientationEnsemble(
        matrices=matrices,
        phi=phi,
        theta=theta,
        psi=psi,
    )
