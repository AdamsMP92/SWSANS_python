"""Effective fields for uniaxial and cubic Stoner-Wohlfarth particles."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from swsans.models import Anisotropy


def effective_field(
    magnetization: NDArray[np.float64],
    applied_field_t: float,
    lab_to_crystal: NDArray[np.float64],
    anisotropy: Anisotropy,
    k1_t: float,
    k2_t: float,
) -> NDArray[np.float64]:
    """Evaluate the total field in the laboratory frame.

    ``k1_t`` and ``k2_t`` denote :math:`2K_1/M_0` and :math:`2K_2/M_0`.
    The applied field points along the laboratory z axis.
    """

    local_m = np.einsum("nij,nj->ni", lab_to_crystal, magnetization)
    mx = local_m[:, 0]
    my = local_m[:, 1]
    mz = local_m[:, 2]
    local_field = np.zeros_like(local_m)

    if anisotropy is Anisotropy.UNIAXIAL:
        local_field[:, 2] = k1_t * mz + 2.0 * k2_t * mz * (1.0 - mz**2)
    elif anisotropy is Anisotropy.CUBIC:
        local_field[:, 0] = -k1_t * mx * (my**2 + mz**2) - k2_t * mx * my**2 * mz**2
        local_field[:, 1] = -k1_t * my * (mx**2 + mz**2) - k2_t * mx**2 * my * mz**2
        local_field[:, 2] = -k1_t * mz * (mx**2 + my**2) - k2_t * mx**2 * my**2 * mz
    else:  # pragma: no cover - protected by the enum
        raise ValueError(f"Unsupported anisotropy: {anisotropy}")

    lab_field = np.einsum("nji,nj->ni", lab_to_crystal, local_field)
    lab_field[:, 2] += applied_field_t
    return np.asarray(lab_field, dtype=np.float64)
