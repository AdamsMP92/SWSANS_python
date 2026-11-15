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


def test_detector_axes_follow_the_documented_coordinate_convention() -> None:
    config = SANSConfig(map_points=51, curve_points=64)
    centre = config.map_points // 2
    offset = 8

    perpendicular_gamma = np.diag([0.0, 1.0, 0.0])
    _, _, perpendicular = sans_map(
        perpendicular_gamma,
        config,
        "perpendicular",
    )
    qz_direction = perpendicular[centre, centre + offset]
    qy_direction = perpendicular[centre + offset, centre]
    assert qz_direction > qy_direction

    parallel_gamma = np.diag([1.0, 0.0, 0.0])
    _, _, parallel = sans_map(parallel_gamma, config, "parallel")
    qx_direction = parallel[centre, centre + offset]
    qy_direction = parallel[centre + offset, centre]
    assert qy_direction > qx_direction


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


def test_sans_intensity_curve_supports_linear_and_logarithmic_sampling() -> None:
    gamma = np.eye(3) / 3.0
    config = SANSConfig(map_points=51, curve_points=64)

    q_log, _ = sans_intensity_curve(gamma, config, "perpendicular", spacing="log")
    q_linear, _ = sans_intensity_curve(
        gamma,
        config,
        "perpendicular",
        spacing="linear",
    )

    assert q_log[0] > 0.0
    assert q_linear[0] == 0.0
    np.testing.assert_allclose(np.diff(q_linear), np.diff(q_linear)[0])
    np.testing.assert_allclose(q_log[1:] / q_log[:-1], q_log[1] / q_log[0])


def test_spin_flip_parallel_channel_excludes_longitudinal_gamma_zz() -> None:
    gamma = np.diag([0.0, 0.0, 1.0])
    config = SANSConfig(map_points=51, curve_points=64)

    _, _, parallel_map = sans_map(gamma, config, "parallel", "spin_flip")
    _, parallel_curve = sans_intensity_curve(
        gamma,
        config,
        "parallel",
        cross_section="spin_flip",
    )
    _, parallel_pair, parallel_correlation = pair_distance_curves(
        gamma,
        config,
        "parallel",
        "spin_flip",
    )

    np.testing.assert_array_equal(parallel_map, np.zeros_like(parallel_map))
    np.testing.assert_array_equal(parallel_curve, np.zeros_like(parallel_curve))
    np.testing.assert_array_equal(parallel_pair, np.zeros_like(parallel_pair))
    np.testing.assert_array_equal(
        parallel_correlation,
        np.zeros_like(parallel_correlation),
    )

    _, _, perpendicular_map = sans_map(
        gamma,
        config,
        "perpendicular",
        "spin_flip",
    )
    centre = config.map_points // 2
    offset = 7
    assert perpendicular_map[centre + offset, centre + offset] > 0.0
    assert perpendicular_map[centre, centre + offset] == 0.0
    assert perpendicular_map[centre + offset, centre] == 0.0


def test_spin_flip_parallel_map_uses_rotated_numagsans_axes() -> None:
    gamma = np.array(
        [
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    config = SANSConfig(map_points=51, curve_points=64)
    _, _, intensity = sans_map(gamma, config, "parallel", "spin_flip")
    centre = config.map_points // 2
    offset = 6

    same_sign_quadrant = intensity[centre + offset, centre + offset]
    opposite_sign_quadrant = intensity[centre + offset, centre - offset]
    assert same_sign_quadrant < 1.0e-14 * opposite_sign_quadrant
    assert opposite_sign_quadrant > 0.0


def test_spin_flip_curves_use_the_analytic_azimuthal_moments() -> None:
    gamma = np.diag([0.2, 0.3, 0.5])
    config = SANSConfig(map_points=51, curve_points=64)

    for geometry, expected_ratio in (
        ("perpendicular", 0.375 / 0.6),
        ("parallel", 0.25 / 0.75),
    ):
        _, unpolarized = sans_intensity_curve(
            gamma,
            config,
            geometry,
            spacing="linear",
        )
        _, spin_flip = sans_intensity_curve(
            gamma,
            config,
            geometry,
            spacing="linear",
            cross_section="spin_flip",
        )
        np.testing.assert_allclose(
            spin_flip,
            expected_ratio * unpolarized,
            rtol=1.0e-14,
        )
