import numpy as np

from swsans import SimulationConfig, field_sweep, simulate_hysteresis


def test_field_sweep_contains_full_triangular_cycle() -> None:
    np.testing.assert_allclose(
        field_sweep(2.0, 9),
        [2.0, 1.0, 0.0, -1.0, -2.0, -1.0, 0.0, 1.0, 2.0],
    )


def test_small_simulation_streams_normalized_second_moments() -> None:
    result = simulate_hysteresis(
        SimulationConfig(
            n_orientations=16,
            n_field_points=9,
            tolerance=1.0e-5,
            max_iterations=1200,
        )
    )
    assert result.mean_m.shape == (9, 3)
    assert result.gamma.shape == (9, 3, 3)
    assert result.diagnostics.iterations.shape == (9,)
    np.testing.assert_allclose(
        np.trace(result.gamma, axis1=1, axis2=2),
        np.ones(9),
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        result.gamma,
        np.swapaxes(result.gamma, 1, 2),
        atol=2.0e-13,
    )
