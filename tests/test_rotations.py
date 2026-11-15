import numpy as np

from swsans.rotations import sobol_orientations


def test_sobol_rotation_matrices_are_proper_orthogonal() -> None:
    ensemble = sobol_orientations(64, skip=32)
    identity = np.einsum(
        "nij,nkj->nik",
        ensemble.matrices,
        ensemble.matrices,
    )
    expected = np.broadcast_to(np.eye(3), identity.shape)
    np.testing.assert_allclose(identity, expected, atol=1.0e-13)
    np.testing.assert_allclose(
        np.linalg.det(ensemble.matrices),
        np.ones(64),
        atol=1.0e-13,
    )


def test_orientation_sampling_is_deterministic() -> None:
    first = sobol_orientations(16, skip=998)
    second = sobol_orientations(16, skip=998)
    np.testing.assert_array_equal(first.matrices, second.matrices)
