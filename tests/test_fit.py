import numpy as np

from sctransform_native._fit import fit_nb_offset, theta_moment_ratio_mask


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
