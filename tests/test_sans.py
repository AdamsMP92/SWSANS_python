import numpy as np

from swsans.models import SANSConfig
from swsans.sans import (
    pair_distance_curves,
    sans_intensity_curve,
    sans_map,
    sphere_form_factor,
)


def test_sphere_form_factor_has_correct_origin_limit() -> None:
    values = sphere_form_factor(np.array([0.0, 1.0e-8, 0.1]))
    np.testing.assert_allclose(values[:2], [1.0 / 9.0, 1.0 / 9.0], rtol=1.0e-13)
    assert np.all(values >= 0.0)


def test_isotropic_gamma_produces_isotropic_detector_maps() -> None:
    gamma = np.eye(3) / 3.0
    config = SANSConfig(map_points=51, curve_points=64)
    for geometry in ("perpendicular", "parallel"):
        _, _, intensity = sans_map(gamma, config, geometry)
        radius = 12
        centre = config.map_points // 2
        ring = np.array(
            [
                intensity[centre + radius, centre],
                intensity[centre - radius, centre],
                intensity[centre, centre + radius],
                intensity[centre, centre - radius],
            ]
        )
        np.testing.assert_allclose(ring, np.full(4, ring[0]), rtol=1.0e-13)


def test_sans_curves_are_nonnegative_and_have_compact_pair_support() -> None:
    gamma = np.diag([0.2, 0.3, 0.5])
    config = SANSConfig(map_points=51, curve_points=64)
    q, intensity = sans_intensity_curve(gamma, config, "perpendicular")
    r, pair, correlation = pair_distance_curves(gamma, config, "perpendicular")
    assert np.all(np.diff(q) > 0.0)
    assert np.all(intensity >= 0.0)
    assert r[0] == 0.0
    assert r[-1] == 2.0 * config.radius_nm
    assert pair[0] == 0.0
    assert abs(pair[-1]) < 1.0e-24
    assert abs(correlation[-1]) < 1.0e-24
