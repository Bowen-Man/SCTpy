import numpy as np

from benchmarks.gpu_cox_reid import (
    dispersion_roots_numpy,
    prepare_root_inputs,
)
from sctransform_native._fit import fit_cox_reid_offset_batch


def test_gpu_benchmark_cpu_root_matches_validated_batch_fit() -> None:
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

    theta, poisson, ceiling, _ = dispersion_roots_numpy(
        prepare_root_inputs(counts, exposure)
    )
    expected = fit_cox_reid_offset_batch(counts, exposure)
    expected_theta = np.asarray([fit.theta for fit in expected])

    np.testing.assert_array_equal(poisson, np.isinf(expected_theta))
    np.testing.assert_array_equal(ceiling, np.zeros(theta.size, dtype=bool))
    finite = np.isfinite(expected_theta)
    np.testing.assert_allclose(
        np.log10(theta[finite]),
        np.log10(expected_theta[finite]),
        rtol=0.0,
        atol=2.0e-11,
    )
