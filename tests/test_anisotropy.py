import numpy as np

from swsans.anisotropy import effective_field
from swsans.models import Anisotropy


def test_uniaxial_field_for_aligned_easy_axis() -> None:
    magnetization = np.array([[0.0, 0.0, 1.0]])
    rotations = np.eye(3)[None, :, :]
    field = effective_field(
        magnetization,
        applied_field_t=0.2,
        lab_to_crystal=rotations,
        anisotropy=Anisotropy.UNIAXIAL,
        k1_t=1.0,
        k2_t=0.5,
    )
    np.testing.assert_allclose(field, [[0.0, 0.0, 1.2]])


def test_cubic_field_vanishes_along_100_axis_without_applied_field() -> None:
    magnetization = np.array([[1.0, 0.0, 0.0]])
    rotations = np.eye(3)[None, :, :]
    field = effective_field(
        magnetization,
        applied_field_t=0.0,
        lab_to_crystal=rotations,
        anisotropy=Anisotropy.CUBIC,
        k1_t=1.0,
        k2_t=9.0,
    )
    np.testing.assert_allclose(field, np.zeros((1, 3)))


def test_mixed_field_adds_both_anisotropies_and_applied_field_once() -> None:
    magnetization = np.array([[0.36, -0.48, 0.8]])
    rotations = np.array(
        [
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ]
        ]
    )
    applied_field_t = 0.37

    uniaxial = effective_field(
        magnetization,
        applied_field_t=applied_field_t,
        lab_to_crystal=rotations,
        anisotropy=Anisotropy.UNIAXIAL,
        k1_t=0.8,
        k2_t=-0.2,
    )
    cubic = effective_field(
        magnetization,
        applied_field_t=applied_field_t,
        lab_to_crystal=rotations,
        anisotropy=Anisotropy.CUBIC,
        k1_t=-0.6,
        k2_t=0.15,
    )
    mixed = effective_field(
        magnetization,
        applied_field_t=applied_field_t,
        lab_to_crystal=rotations,
        anisotropy=Anisotropy.MIXED,
        k1_t=0.8,
        k2_t=-0.2,
        cubic_k1_t=-0.6,
        cubic_k2_t=0.15,
    )

    applied_field = np.array([[0.0, 0.0, applied_field_t]])
    np.testing.assert_allclose(mixed, uniaxial + cubic - applied_field)


def test_mixed_limits_match_existing_pure_symmetries() -> None:
    magnetization = np.array([[0.36, -0.48, 0.8]])
    rotations = np.eye(3)[None, :, :]
    common = {
        "magnetization": magnetization,
        "applied_field_t": -0.25,
        "lab_to_crystal": rotations,
    }

    uniaxial = effective_field(
        **common,
        anisotropy=Anisotropy.UNIAXIAL,
        k1_t=0.8,
        k2_t=-0.2,
    )
    mixed_uniaxial = effective_field(
        **common,
        anisotropy=Anisotropy.MIXED,
        k1_t=0.8,
        k2_t=-0.2,
    )
    cubic = effective_field(
        **common,
        anisotropy=Anisotropy.CUBIC,
        k1_t=-0.6,
        k2_t=0.15,
    )
    mixed_cubic = effective_field(
        **common,
        anisotropy=Anisotropy.MIXED,
        k1_t=0.0,
        k2_t=0.0,
        cubic_k1_t=-0.6,
        cubic_k2_t=0.15,
    )

    np.testing.assert_allclose(mixed_uniaxial, uniaxial)
    np.testing.assert_allclose(mixed_cubic, cubic)
