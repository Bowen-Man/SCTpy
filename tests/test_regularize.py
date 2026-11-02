import numpy as np
import pytest

from sctransform_native._regularize import sheather_jones_bandwidth


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
