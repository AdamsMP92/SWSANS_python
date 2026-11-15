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
