import numpy as np

from sctransform_native._fit import fit_nb_offset


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
