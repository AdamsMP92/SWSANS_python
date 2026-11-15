import pytest

from swsans import SimulationConfig, critical_states, simulate_hysteresis


@pytest.mark.slow
def test_uniaxial_easy_axis_is_consistent_with_published_reference() -> None:
    result = simulate_hysteresis(
        SimulationConfig(
            k1_t=1.0,
            k2_t=0.0,
            field_max_t=2.0,
            n_orientations=256,
            n_field_points=101,
            tolerance=1.0e-7,
            max_iterations=5000,
        )
    )
    states = critical_states(result)
    assert states.remanence.mean_m[2] == pytest.approx(0.5, abs=0.08)
    assert abs(states.coercive.field_t) == pytest.approx(0.482, abs=0.10)
