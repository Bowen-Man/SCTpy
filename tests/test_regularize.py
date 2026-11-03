import numpy as np
import pytest

from sctpy._regularize import (
    _dispersion_target,
    _regularization_prediction_points,
    _theta_from_dispersion,
    r_stats_binned_outliers,
    r_stats_ksmooth_normal,
    r_stats_sj_bandwidth,
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


def test_r_stats_sj_matches_external_golden_values() -> None:
    small = np.array([-1.4, -0.8, -0.2, 0.1, 0.4, 0.9, 1.5, 2.2])
    large = np.linspace(-2.0, 3.0, 600) + 0.03 * np.sin(np.arange(1, 601))

    assert r_stats_sj_bandwidth(small, adjust=1.0) == pytest.approx(
        1.0067421200429481, rel=1.0e-14
    )
    assert r_stats_sj_bandwidth(large, adjust=1.0) == pytest.approx(
        0.28201580646256219, rel=1.0e-14
    )


def test_r_stats_ksmooth_matches_external_golden_values() -> None:
    x = np.array([-1.4, -0.8, -0.2, 0.1, 0.4, 0.9, 1.5, 2.2])
    y = np.array([0.2, -0.1, 0.5, 1.2, -0.4, 0.8, 1.5, -0.2])
    points = np.array([-1.0, 0.0, 0.7, 1.8])
    expected = np.array(
        [
            0.012414897251106066,
            0.631289142672135761,
            0.382229507017641301,
            0.816812664492320595,
        ]
    )

    actual = r_stats_ksmooth_normal(x, y, points, bandwidth=0.8)

    np.testing.assert_allclose(actual, expected, rtol=2.0e-15, atol=1.0e-15)


def test_r_stats_binned_outliers_matches_external_golden_mask() -> None:
    coordinates = np.linspace(-1.0, 1.0, 40)
    values = 0.2 * coordinates + np.sin(np.arange(1, 41)) / 100.0
    values[[7, 30]] = [3.0, -4.0]

    actual = r_stats_binned_outliers(coordinates, values, threshold=10.0)

    np.testing.assert_array_equal(np.flatnonzero(actual), np.array([7, 30]))


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


def test_regularization_support_policy_is_explicit() -> None:
    x_train = np.array([-1.0, 0.0, 2.0])
    x_predict = np.array([-3.0, 0.5, 4.0])

    extrapolated = _regularization_prediction_points(
        x_train, x_predict, support="extrapolate"
    )
    clamped = _regularization_prediction_points(x_train, x_predict, support="clamp")

    np.testing.assert_array_equal(extrapolated, x_predict)
    np.testing.assert_array_equal(clamped, np.array([-1.0, 0.5, 2.0]))
