import numpy as np

from swsans.dynamics import landau_lifshitz_rhs, normalize_vectors


def test_landau_lifshitz_rhs_is_tangent_to_unit_sphere() -> None:
    magnetization = normalize_vectors(np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 0.25]]))
    field = np.array([[0.3, -0.2, 1.1], [0.1, 0.7, -0.4]])
    rhs = landau_lifshitz_rhs(magnetization, field, 1.0, 6.0)
    np.testing.assert_allclose(
        np.einsum("ni,ni->n", magnetization, rhs),
        np.zeros(2),
        atol=1.0e-14,
    )


def test_normalize_vectors_returns_unit_vectors() -> None:
    normalized = normalize_vectors(np.array([[3.0, 4.0, 0.0], [0.0, 0.0, -2.0]]))
    np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), 1.0)
