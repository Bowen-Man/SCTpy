import numpy as np
import pytest

from sctpy._fit import (
    _cox_reid_score_batch,
    fit_cox_reid_offset,
    fit_cox_reid_offset_batch,
    fit_nb_offset,
    local_median_dispersion_trend,
    offset_intercept,
    theta_moment_ratio_mask,
)


def test_nb_offset_mle_recovers_simulated_parameters() -> None:
    rng = np.random.default_rng(20261102)
    exposure = rng.lognormal(mean=6.5, sigma=0.45, size=1500)
    expected_intercept = np.log(0.012)
    expected_theta = 4.0
    mu = np.exp(expected_intercept) * exposure
    probability = expected_theta / (expected_theta + mu)
    y = rng.negative_binomial(expected_theta, probability).astype(np.float64)

    fitted = fit_nb_offset(
        y,
        exposure,
        theta_min=1e-4,
        theta_max=1e8,
        maxiter=300,
        ftol=1e-11,
        gtol=1e-7,
    )

    assert fitted.converged
    assert abs(fitted.intercept - expected_intercept) < 0.08
    assert abs(np.log(fitted.theta / expected_theta)) < 0.20


def test_theta_moment_ratio_mask_handles_poisson_and_overdispersed_genes() -> None:
    mask = theta_moment_ratio_mask(
        np.array([2.0, 2.0, 2.0]),
        np.array([2.0, 6.0, 6.0]),
        np.array([1.0e8, 1.0e4, 1.0e2]),
        threshold=1.0e-3,
    )
    np.testing.assert_array_equal(mask, np.array([False, True, False]))


def test_cox_reid_fit_matches_external_golden_values() -> None:
    exposure = np.array(
        [80, 95, 110, 130, 90, 150, 105, 125, 100, 140, 115, 85],
        dtype=np.float64,
    )
    counts = np.array(
        [
            [0, 1, 0, 2, 1, 3, 0, 1, 4, 2, 1, 0],
            [0, 0, 0, 0, 1, 0, 10, 0, 0, 20, 0, 0],
            [5, 6, 4, 5, 7, 6, 5, 8, 7, 6, 5, 4],
            [1, 4, 2, 8, 1, 12, 2, 7, 3, 11, 2, 1],
            [0, 2, 1, 0, 5, 1, 0, 4, 2, 6, 1, 0],
        ],
        dtype=np.float64,
    )
    expected_theta = np.array(
        [
            9.02902784778621,
            0.07029276317158564,
            np.inf,
            3.614883952910656,
            1.114701072127238,
        ]
    )
    expected_intercept = np.array(
        [
            -4.5370171134977895,
            -3.870587188491462,
            -2.942087922391061,
            -3.3080613064152,
            -4.114199663838848,
        ]
    )

    fits = [fit_cox_reid_offset(row, exposure) for row in counts]
    actual_theta = np.asarray([fit.theta for fit in fits])
    preliminary_intercept = np.asarray([fit.intercept for fit in fits])
    dispersion = np.zeros(actual_theta.shape, dtype=np.float64)
    finite = np.isfinite(actual_theta)
    dispersion[finite] = 1.0 / actual_theta[finite]
    abundance = np.exp(preliminary_intercept) * np.mean(exposure)
    trend = local_median_dispersion_trend(abundance, dispersion)
    actual_intercept = np.asarray(
        [
            offset_intercept(row, exposure, value)[0]
            for row, value in zip(counts, trend, strict=True)
        ]
    )

    np.testing.assert_allclose(actual_theta, expected_theta, rtol=5.0e-9)
    np.testing.assert_allclose(actual_intercept, expected_intercept, rtol=1.0e-9)


def test_batched_cox_reid_matches_score_reference_and_scalar_fit() -> None:
    exposure = np.array(
        [80, 95, 110, 130, 90, 150, 105, 125, 100, 140, 115, 85],
        dtype=np.float64,
    )
    counts = np.array(
        [
            [0, 1, 0, 2, 1, 3, 0, 1, 4, 2, 1, 0],
            [0, 0, 0, 0, 1, 0, 10, 0, 0, 20, 0, 0],
            [5, 6, 4, 5, 7, 6, 5, 8, 7, 6, 5, 4],
            [1, 4, 2, 8, 1, 12, 2, 7, 3, 11, 2, 1],
        ],
        dtype=np.float64,
    )
    log_dispersion = np.log(np.array([0.1, 2.0, 1.0e-8, 0.5]))
    mu = np.array([1.0, 1.5, 5.5, 4.0])[:, None] * exposure / exposure.mean()

    expected_score = np.array(
        [
            0.03971052386506343,
            5.592784326327438,
            # Independent 80-digit integer sum; the historical oracle value
            # -1.68518843091391e-07 contains the small-dispersion error.
            -1.9020701317774512e-07,
            -1.168731328214636,
        ]
    )
    actual_score = _cox_reid_score_batch(log_dispersion, counts, mu)
    np.testing.assert_allclose(actual_score, expected_score, rtol=0.0, atol=1.0e-13)

    expected_fit = [fit_cox_reid_offset(row, exposure) for row in counts]
    actual_fit = fit_cox_reid_offset_batch(counts, exposure)
    np.testing.assert_allclose(
        [value.intercept for value in actual_fit],
        [value.intercept for value in expected_fit],
        rtol=0.0,
        atol=1.0e-14,
    )
    np.testing.assert_array_equal(
        np.isfinite([value.theta for value in actual_fit]),
        np.isfinite([value.theta for value in expected_fit]),
    )
    finite = np.isfinite([value.theta for value in expected_fit])
    np.testing.assert_allclose(
        np.log10([value.theta for value in actual_fit])[finite],
        np.log10([value.theta for value in expected_fit])[finite],
        rtol=0.0,
        atol=2.0e-11,
    )


def test_local_median_trend_matches_external_golden_values() -> None:
    abundance = np.array([0.2, 1.5, 0.7, 3.0, 2.2, 4.1, 5.0, 3.7, 6.2, 5.8, 7.0, 8.1])
    dispersion = np.array([0.4, 0.1, 0.8, 0.2, 0.0, 1.1, 0.3, 0.9, 0.5, 0.7, 1.3, 0.6])
    expected = np.array([0.1, 0.1, 0.1, 0.2, 0.0, 1.1, 0.3, 0.9, 0.5, 0.7, 0.5, 0.5])

    actual = local_median_dispersion_trend(abundance, dispersion, points=5)

    np.testing.assert_array_equal(actual, expected)


def test_local_median_trend_gene_bins_are_explicit() -> None:
    abundance = np.arange(1.0, 13.0)
    dispersion = np.array([0.4, 0.1, 0.8, 0.2, 0.0, 1.1, 0.3, 0.9, 0.5, 0.7, 1.3, 0.6])

    expected = np.concatenate(
        [
            local_median_dispersion_trend(abundance[:5], dispersion[:5]),
            local_median_dispersion_trend(abundance[5:10], dispersion[5:10]),
            local_median_dispersion_trend(abundance[10:], dispersion[10:]),
        ]
    )
    actual = local_median_dispersion_trend(abundance, dispersion, gene_bin_size=5)

    np.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError, match="gene_bin_size"):
        local_median_dispersion_trend(abundance, dispersion, gene_bin_size=0)
