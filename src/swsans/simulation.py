"""Vectorized Stoner-Wohlfarth ensemble simulation."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from swsans.anisotropy import effective_field
from swsans.dynamics import landau_lifshitz_rhs, normalize_vectors
from swsans.models import (
    ConvergenceDiagnostics,
    HysteresisResult,
    Integrator,
    SimulationConfig,
)
from swsans.rotations import sobol_orientations

ProgressCallback = Callable[[int, int], None]


def field_sweep(field_max_t: float, n_field_points: int) -> NDArray[np.float64]:
    """Return a +Bmax -> -Bmax -> +Bmax triangular field cycle."""

    phase = np.linspace(0.0, 1.0, n_field_points, dtype=np.float64)
    return field_max_t * (1.0 - 4.0 * np.abs(phase - np.floor(phase + 0.5)))


def _step_euler(
    magnetization: NDArray[np.float64],
    applied_field_t: float,
    rotations: NDArray[np.float64],
    config: SimulationConfig,
) -> tuple[NDArray[np.float64], float]:
    field = effective_field(
        magnetization,
        applied_field_t,
        rotations,
        config.anisotropy,
        config.k1_t,
        config.k2_t,
    )
    rhs = landau_lifshitz_rhs(
        magnetization,
        field,
        config.gyromagnetic_factor,
        config.damping,
    )
    residual = float(np.max(np.einsum("ni,ni->n", rhs, rhs)))
    updated = normalize_vectors(magnetization + config.time_step * rhs)
    return updated, residual


def _step_heun(
    magnetization: NDArray[np.float64],
    applied_field_t: float,
    rotations: NDArray[np.float64],
    config: SimulationConfig,
) -> tuple[NDArray[np.float64], float]:
    field_0 = effective_field(
        magnetization,
        applied_field_t,
        rotations,
        config.anisotropy,
        config.k1_t,
        config.k2_t,
    )
    rhs_0 = landau_lifshitz_rhs(
        magnetization,
        field_0,
        config.gyromagnetic_factor,
        config.damping,
    )
    predictor = normalize_vectors(magnetization + config.time_step * rhs_0)
    field_1 = effective_field(
        predictor,
        applied_field_t,
        rotations,
        config.anisotropy,
        config.k1_t,
        config.k2_t,
    )
    rhs_1 = landau_lifshitz_rhs(
        predictor,
        field_1,
        config.gyromagnetic_factor,
        config.damping,
    )
    residual = float(np.max(np.einsum("ni,ni->n", rhs_1, rhs_1)))
    updated = normalize_vectors(
        magnetization + 0.5 * config.time_step * (rhs_0 + rhs_1)
    )
    return updated, residual


def _relax_at_field(
    magnetization: NDArray[np.float64],
    applied_field_t: float,
    rotations: NDArray[np.float64],
    config: SimulationConfig,
) -> tuple[NDArray[np.float64], int, float, bool]:
    stepper = (
        _step_euler if config.integrator is Integrator.LEGACY_EULER else _step_heun
    )
    residual = np.inf
    for iteration in range(1, config.max_iterations + 1):
        magnetization, residual = stepper(
            magnetization,
            applied_field_t,
            rotations,
            config,
        )
        if not np.isfinite(residual):
            raise ArithmeticError(
                "The Landau-Lifshitz relaxation produced a non-finite residual."
            )
        if residual <= config.tolerance:
            return magnetization, iteration, residual, True
    return magnetization, config.max_iterations, residual, False


def simulate_hysteresis(
    config: SimulationConfig,
    *,
    progress: ProgressCallback | None = None,
) -> HysteresisResult:
    """Relax a randomly oriented particle ensemble along a full field cycle.

    Only first and second ensemble moments are retained at every field point.
    This avoids the three dense N-by-K trajectory arrays allocated by the
    original MATLAB application and makes browser execution practical.
    """

    fields = field_sweep(config.field_max_t, config.n_field_points)
    orientations = sobol_orientations(
        config.n_orientations,
        skip=config.sobol_skip,
    )
    rng = np.random.default_rng(config.seed)
    azimuth = rng.uniform(0.0, 2.0 * np.pi, config.n_orientations)
    initial_z = 0.98
    initial_rho = np.sqrt(1.0 - initial_z**2)
    magnetization = np.column_stack(
        (
            initial_rho * np.cos(azimuth),
            initial_rho * np.sin(azimuth),
            np.full(config.n_orientations, initial_z),
        )
    )

    mean_m = np.empty((config.n_field_points, 3), dtype=np.float64)
    gamma = np.empty((config.n_field_points, 3, 3), dtype=np.float64)
    iterations = np.empty(config.n_field_points, dtype=np.int64)
    residuals = np.empty(config.n_field_points, dtype=np.float64)
    converged = np.empty(config.n_field_points, dtype=np.bool_)

    for index, applied_field_t in enumerate(fields):
        magnetization, count, residual, did_converge = _relax_at_field(
            magnetization,
            float(applied_field_t),
            orientations.matrices,
            config,
        )
        mean_m[index] = np.mean(magnetization, axis=0)
        gamma[index] = magnetization.T @ magnetization / config.n_orientations
        iterations[index] = count
        residuals[index] = residual
        converged[index] = did_converge
        if progress is not None:
            progress(index + 1, config.n_field_points)

    diagnostics = ConvergenceDiagnostics(
        iterations=iterations,
        residual=residuals,
        converged=converged,
    )
    return HysteresisResult(
        config=config,
        field_t=fields,
        mean_m=mean_m,
        gamma=gamma,
        diagnostics=diagnostics,
    )
