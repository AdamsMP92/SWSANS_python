import numpy as np

from swsans import Anisotropy, SimulationConfig, field_sweep, simulate_hysteresis


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


def test_original_positional_config_api_is_unchanged() -> None:
    config = SimulationConfig(Anisotropy.CUBIC, -1.0, 9.0, 4.0, 16, 9)

    assert config.anisotropy is Anisotropy.CUBIC
    assert config.k1_t == -1.0
    assert config.k2_t == 9.0
    assert config.field_max_t == 4.0
    assert config.n_orientations == 16
    assert config.n_field_points == 9
    assert config.cubic_k1_t == 0.0
    assert config.cubic_k2_t == 0.0


def test_small_mixed_simulation_produces_finite_normalized_moments() -> None:
    result = simulate_hysteresis(
        SimulationConfig(
            anisotropy=Anisotropy.MIXED,
            k1_t=0.7,
            k2_t=0.1,
            cubic_k1_t=-0.3,
            cubic_k2_t=0.05,
            n_orientations=8,
            n_field_points=5,
            tolerance=1.0e-4,
            max_iterations=400,
        )
    )

    assert np.all(np.isfinite(result.mean_m))
    assert np.all(np.isfinite(result.gamma))
    np.testing.assert_allclose(
        np.trace(result.gamma, axis1=1, axis2=2),
        np.ones(5),
        atol=2.0e-13,
    )
