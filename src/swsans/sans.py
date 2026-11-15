"""Magnetic SANS response of dilute spherical Stoner-Wohlfarth particles."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from swsans.models import SANSConfig

Geometry = Literal["perpendicular", "parallel"]
MAGNETIC_SCATTERING_LENGTH = 2.91e8  # A^-1 m^-1


def sphere_form_factor(x: NDArray[np.float64] | float) -> NDArray[np.float64]:
    """Return ``[j1(x)/x]^2`` with its analytic limit at the origin."""

    values = np.asarray(x, dtype=np.float64)
    small = np.abs(values) < 1.0e-4
    result = np.empty_like(values)
    squared = values[~small] ** 2
    result[~small] = (
        np.sin(values[~small]) / (values[~small] * squared)
        - np.cos(values[~small]) / squared
    ) ** 2
    small_squared = values[small] ** 2
    result[small] = 1.0 / 9.0 - small_squared / 45.0 + small_squared**2 / 525.0
    return result


def _prefactor(config: SANSConfig) -> float:
    radius_m = config.radius_nm * 1.0e-9
    volume_m3 = 4.0 * np.pi * radius_m**3 / 3.0
    return (
        16.0
        * np.pi**2
        * radius_m**6
        * MAGNETIC_SCATTERING_LENGTH**2
        * config.saturation_magnetization_a_per_m**2
        / volume_m3
    )


def _angular_factor(
    gamma: NDArray[np.float64],
    sine: NDArray[np.float64],
    cosine: NDArray[np.float64],
    geometry: Geometry,
) -> NDArray[np.float64]:
    if geometry == "perpendicular":
        return np.asarray(
            (
                gamma[0, 0]
                + gamma[1, 1] * cosine**2
                + gamma[2, 2] * sine**2
                - 2.0 * gamma[1, 2] * sine * cosine
            ),
            dtype=np.float64,
        )
    if geometry == "parallel":
        return np.asarray(
            (
                gamma[0, 0] * sine**2
                + gamma[1, 1] * cosine**2
                + gamma[2, 2]
                - 2.0 * gamma[0, 1] * sine * cosine
            ),
            dtype=np.float64,
        )
    raise ValueError(f"Unknown scattering geometry: {geometry}")


def _azimuthal_moment(gamma: NDArray[np.float64], geometry: Geometry) -> float:
    if geometry == "perpendicular":
        return float(gamma[0, 0] + 0.5 * gamma[1, 1] + 0.5 * gamma[2, 2])
    if geometry == "parallel":
        return float(0.5 * gamma[0, 0] + 0.5 * gamma[1, 1] + gamma[2, 2])
    raise ValueError(f"Unknown scattering geometry: {geometry}")


def sans_map(
    gamma: NDArray[np.float64],
    config: SANSConfig,
    geometry: Geometry,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Calculate a square two-dimensional magnetic SANS detector map.

    The returned intensity is in cm^-1 and both axes are in nm^-1. For the
    perpendicular geometry they correspond to q_y and q_z; for the parallel
    geometry they correspond to q_x and q_y.
    """

    axis = np.linspace(
        -config.q_max_per_nm,
        config.q_max_per_nm,
        config.map_points,
    )
    horizontal, vertical = np.meshgrid(axis, axis)
    q = np.hypot(horizontal, vertical)
    sine = np.divide(horizontal, q, out=np.zeros_like(q), where=q != 0.0)
    cosine = np.divide(vertical, q, out=np.ones_like(q), where=q != 0.0)
    angular = _angular_factor(gamma, sine, cosine, geometry)
    central = _azimuthal_moment(gamma, geometry)
    angular = np.where(q == 0.0, central, angular)
    intensity_per_m = (
        _prefactor(config) * sphere_form_factor(q * config.radius_nm) * angular
    )
    return axis, axis, np.maximum(intensity_per_m / 100.0, 0.0)


def sans_intensity_curve(
    gamma: NDArray[np.float64],
    config: SANSConfig,
    geometry: Geometry,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the 2-pi azimuthal average I(q) in cm^-1."""

    q_min = min(config.q_max_per_nm / 10.0, 0.01 / config.radius_nm)
    q_min = max(q_min, config.q_max_per_nm * 1.0e-5)
    q = np.geomspace(q_min, config.q_max_per_nm, config.curve_points)
    intensity = (
        _prefactor(config)
        * sphere_form_factor(q * config.radius_nm)
        * _azimuthal_moment(gamma, geometry)
        / 100.0
    )
    return q, np.maximum(intensity, 0.0)


def pair_distance_curves(
    gamma: NDArray[np.float64],
    config: SANSConfig,
    geometry: Geometry,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return r, p(r), and c(r)=p(r)/r^2 in nm-based units."""

    radius_m = config.radius_nm * 1.0e-9
    volume_m3 = 4.0 * np.pi * radius_m**3 / 3.0
    r_m = np.linspace(0.0, 2.0 * radius_m, config.curve_points)
    shape = 1.0 - 3.0 * r_m / (4.0 * radius_m) + r_m**3 / (16.0 * radius_m**3)
    amplitude = (
        16.0
        * np.pi**2
        * radius_m**3
        * MAGNETIC_SCATTERING_LENGTH**2
        * config.saturation_magnetization_a_per_m**2
        / volume_m3
        * np.pi
        / 6.0
        * _azimuthal_moment(gamma, geometry)
    )
    correlation_per_m4 = amplitude * shape
    pair_per_m2 = correlation_per_m4 * r_m**2
    return (
        r_m * 1.0e9,
        np.maximum(pair_per_m2 * 1.0e-18, 0.0),
        np.maximum(correlation_per_m4 * 1.0e-36, 0.0),
    )
