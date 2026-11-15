"""Validated data models shared by the numerical core and web interface."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]


class InputValidationError(ValueError):
    """Raised when a model input is outside its supported domain."""


class Anisotropy(StrEnum):
    """Supported magnetocrystalline-anisotropy symmetries."""

    UNIAXIAL = "uniaxial"
    CUBIC = "cubic"


class Integrator(StrEnum):
    """Time-stepping methods for Landau-Lifshitz relaxation."""

    LEGACY_EULER = "legacy_euler"
    HEUN = "heun"


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Inputs for a randomly oriented Stoner-Wohlfarth ensemble.

    ``k1_t`` and ``k2_t`` are the reduced anisotropy constants
    :math:`k_i=2K_i/M_0`, expressed in tesla, following the associated paper.
    The time variables are reduced numerical parameters used to relax each
    field point to a static state.
    """

    anisotropy: Anisotropy = Anisotropy.UNIAXIAL
    k1_t: float = 1.0
    k2_t: float = 0.0
    field_max_t: float = 2.0
    n_orientations: int = 256
    n_field_points: int = 101
    gyromagnetic_factor: float = 1.0
    damping: float = 6.0
    time_step: float = 0.01
    tolerance: float = 1.0e-8
    max_iterations: int = 5000
    integrator: Integrator = Integrator.LEGACY_EULER
    seed: int = 2023
    sobol_skip: int = 998

    def __post_init__(self) -> None:
        if not isinstance(self.anisotropy, Anisotropy):
            object.__setattr__(self, "anisotropy", Anisotropy(self.anisotropy))
        if not isinstance(self.integrator, Integrator):
            object.__setattr__(self, "integrator", Integrator(self.integrator))
        if not np.isfinite(self.k1_t) or not np.isfinite(self.k2_t):
            raise InputValidationError("Anisotropy constants must be finite.")
        if not np.isfinite(self.field_max_t) or self.field_max_t <= 0.0:
            raise InputValidationError("The maximum field must be positive.")
        if self.n_orientations < 8:
            raise InputValidationError("Use at least 8 particle orientations.")
        if self.n_field_points < 5 or self.n_field_points % 2 == 0:
            raise InputValidationError("Field points must be an odd integer >= 5.")
        if self.gyromagnetic_factor < 0.0 or self.damping <= 0.0:
            raise InputValidationError(
                "The gyromagnetic factor must be non-negative and damping positive."
            )
        if self.time_step <= 0.0:
            raise InputValidationError("The time step must be positive.")
        if self.tolerance <= 0.0:
            raise InputValidationError("The convergence tolerance must be positive.")
        if self.max_iterations < 1:
            raise InputValidationError("Maximum iterations must be at least one.")
        if self.sobol_skip < 0:
            raise InputValidationError("The Sobol skip cannot be negative.")


@dataclass(frozen=True, slots=True)
class ConvergenceDiagnostics:
    """Per-field convergence information."""

    iterations: IntArray
    residual: FloatArray
    converged: BoolArray

    @property
    def all_converged(self) -> bool:
        return bool(np.all(self.converged))

    @property
    def converged_fraction(self) -> float:
        return float(np.mean(self.converged))


@dataclass(frozen=True, slots=True)
class HysteresisResult:
    """Streaming ensemble result without an N-by-K trajectory allocation."""

    config: SimulationConfig
    field_t: FloatArray
    mean_m: FloatArray
    gamma: FloatArray
    diagnostics: ConvergenceDiagnostics

    @property
    def mz(self) -> FloatArray:
        return self.mean_m[:, 2]

    @property
    def anisotropy_parameter_perpendicular(self) -> FloatArray:
        numerator = self.gamma[:, 0, 0] + self.gamma[:, 1, 1]
        denominator = self.gamma[:, 0, 0] + self.gamma[:, 2, 2]
        return np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan),
            where=denominator != 0.0,
        )


@dataclass(frozen=True, slots=True)
class CriticalState:
    """Ensemble state at a sampled or interpolated field value."""

    name: str
    field_t: float
    mean_m: FloatArray
    gamma: FloatArray


@dataclass(frozen=True, slots=True)
class CriticalStates:
    """Saturation, descending remanence, and descending coercive state."""

    saturation: CriticalState
    remanence: CriticalState
    coercive: CriticalState

    def by_name(self, name: str) -> CriticalState:
        mapping = {
            "saturation": self.saturation,
            "remanence": self.remanence,
            "coercive": self.coercive,
        }
        try:
            return mapping[name]
        except KeyError as exc:
            raise InputValidationError(f"Unknown critical state: {name}") from exc


@dataclass(frozen=True, slots=True)
class SANSConfig:
    """Physical and numerical inputs for magnetic SANS observables."""

    radius_nm: float = 5.0
    saturation_magnetization_a_per_m: float = 3.1e5
    q_max_per_nm: float = 1.0
    map_points: int = 151
    curve_points: int = 320

    def __post_init__(self) -> None:
        if self.radius_nm <= 0.0:
            raise InputValidationError("Particle radius must be positive.")
        if self.saturation_magnetization_a_per_m <= 0.0:
            raise InputValidationError("Saturation magnetization must be positive.")
        if self.q_max_per_nm <= 0.0:
            raise InputValidationError("The maximum q must be positive.")
        if self.map_points < 31 or self.curve_points < 32:
            raise InputValidationError("Increase the SANS sampling resolution.")
