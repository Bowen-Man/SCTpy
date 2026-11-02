import numpy as np
import pytest

from sctransform_native._regularize import (
    _dispersion_target,
    _theta_from_dispersion,
    sheather_jones_bandwidth,
)


def test_sheather_jones_bandwidth_is_affine_equivariant() -> None:
    x = np.linspace(-1.5, 2.0, 48) + 0.15 * np.sin(np.arange(48))
    bandwidth = sheather_jones_bandwidth(x, adjust=1.0)
    transformed = sheather_jones_bandwidth(4.5 * x + 7.0, adjust=1.0)
    assert bandwidth > 0.0
    assert transformed == pytest.approx(4.5 * bandwidth, rel=1.0e-8)


def test_sheather_jones_adjustment_is_explicit_multiplier() -> None:
    x = np.linspace(-2.0, 1.0, 32) ** 3
    base = sheather_jones_bandwidth(x, adjust=1.0)
    adjusted = sheather_jones_bandwidth(x, adjust=3.0)
    assert adjusted == pytest.approx(3.0 * base, rel=1.0e-12)


@pytest.mark.parametrize("mode", ["native", "sct_v2"])
def test_dispersion_transform_round_trips_theta(mode: str) -> None:
    gene_mean = np.array([0.03, 0.15, 1.3, 6.0, 24.0])
    geometric_mean = np.array([0.02, 0.1, 1.0, 5.0, 20.0])
    theta = np.array([0.2, 0.5, 2.0, 10.0, 100.0])

    dispersion = _dispersion_target(gene_mean, geometric_mean, theta, mode=mode)
    reconstructed = _theta_from_dispersion(
        gene_mean, geometric_mean, dispersion, mode=mode
    )

    np.testing.assert_allclose(reconstructed, theta, rtol=2.0e-15)
