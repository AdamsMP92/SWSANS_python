import numpy as np

from swsans.models import (
    ConvergenceDiagnostics,
    HysteresisResult,
    SimulationConfig,
)
from swsans.observables import critical_states
from swsans.simulation import field_sweep


def test_critical_states_interpolate_descending_branch() -> None:
    config = SimulationConfig(n_orientations=8, n_field_points=9)
    field = field_sweep(2.0, 9)
    mean_m = np.zeros((9, 3))
    mean_m[:, 2] = [1.0, 0.8, 0.5, -0.5, -1.0, -0.5, 0.5, 0.8, 1.0]
    gamma = np.zeros((9, 3, 3))
    gamma[:, 0, 0] = 0.2
    gamma[:, 1, 1] = 0.3
    gamma[:, 2, 2] = 0.5
    gamma[:, 0, 1] = gamma[:, 1, 0] = 0.04
    gamma[:, 1, 2] = gamma[:, 2, 1] = 0.01
    diagnostics = ConvergenceDiagnostics(
        iterations=np.ones(9, dtype=np.int64),
        residual=np.zeros(9),
        converged=np.ones(9, dtype=np.bool_),
    )
    result = HysteresisResult(config, field, mean_m, gamma, diagnostics)
    states = critical_states(result)

    assert states.remanence.field_t == 0.0
    assert states.remanence.mean_m[2] == 0.5
    assert states.coercive.field_t == -0.5
    assert states.remanence.gamma[0, 1] == 0.04
    assert states.remanence.gamma[1, 2] == 0.01
